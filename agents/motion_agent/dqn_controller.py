"""
dqn_controller.py — Controller DQN para navegação de robots.

Substitui o MotionController tabular (Q-tables) por dois DQN Dueling+Double
com replay buffer e rede target:

  dqn_route — escolha do próximo nó (quando IDLE)
  dqn_vel   — comando de velocidade dec/hold/acc (quando MOVING)

Mantém a mesma interface externa que MotionController:
  act(robot_id, obs, valid_actions, goal) → Action
  update(robot_id, prev_obs, curr_obs, events, goal) → None
  set_epsilon / reset_memory / decay_epsilon / save / load

Fluxo por tick (idêntico ao tabular):
  obs    = iface.get_motion_view(robot_id)
  valid  = iface.get_valid_actions(robot_id)
  action = controller.act(robot_id, obs, valid, goal)
  ...
  events = engine.step(all_actions)
  ...
  new_obs = iface.get_motion_view(robot_id)
  controller.update(robot_id, obs, new_obs, events, goal)
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulation_engine.core.entities import RobotState
from simulation_engine.core.graph import FactoryGraph
from simulation_engine.core.events import Events
from simulation_engine.interface.observation_builder import MotionObs

from agents.motion_agent.dqn_network import DQNAgent
from agents.motion_agent.obs_encoder import (
    build_route_obs,
    build_velocity_obs,
    MAX_NEIGHBORS,
    ROUTE_OBS_DIM,
    VEL_OBS_DIM,
)
from agents.motion_agent.reward import (
    RewardWeights,
    count_collisions,
    route_reward,
    velocity_reward,
)

_VEL_CMDS = ["dec", "hold", "acc"]
_N_VEL    = 3

# Máscara fixa para velocidade — as 3 ações são sempre válidas
_VEL_MASK = np.ones(_N_VEL, dtype=bool)

Action = Tuple[Optional[str], str]


# ---------------------------------------------------------------------------
# DQN Controller
# ---------------------------------------------------------------------------

class DQNController:
    """
    Controller DQN partilhado por todos os robots (parameter sharing).

    Um único DQNController é criado por treino; todos os robots alimentam
    os mesmos replay buffers e partilham os pesos das redes.
    """

    def __init__(
        self,
        graph:              FactoryGraph,
        vel_max:            float = 30.0,
        gamma:              float = 0.95,
        lr:                 float = 1e-3,
        epsilon_route:      float = 0.30,
        epsilon_vel:        float = 0.15,
        reward_weights:     Optional[RewardWeights] = None,
        batch_size:         int   = 128,
        buffer_capacity:    int   = 50_000,
        target_update_freq: int   = 500,
        train_freq:         int   = 4,
        replay_start:       int   = 500,
        device:             str   = "cpu",
    ) -> None:
        self.graph   = graph
        self.vel_max = vel_max
        self.weights = reward_weights or RewardWeights()

        self.dqn_route = DQNAgent(
            obs_dim            = ROUTE_OBS_DIM,
            n_actions          = MAX_NEIGHBORS,
            hidden             = (256, 128),
            lr                 = lr,
            gamma              = gamma,
            buffer_capacity    = buffer_capacity,
            batch_size         = batch_size,
            target_update_freq = target_update_freq,
            train_freq         = train_freq,
            replay_start       = replay_start,
            device             = device,
        )
        self.dqn_vel = DQNAgent(
            obs_dim            = VEL_OBS_DIM,
            n_actions          = _N_VEL,
            hidden             = (64, 32),
            lr                 = lr,
            gamma              = gamma,
            buffer_capacity    = buffer_capacity,
            batch_size         = batch_size,
            target_update_freq = target_update_freq,
            train_freq         = train_freq,
            replay_start       = replay_start,
            device             = device,
        )

        self.epsilon_route: float = epsilon_route
        self.epsilon_vel:   float = epsilon_vel

        # Memória deferred por robot
        # route: (obs_vec, action_idx, neighbors, dist_before, node_occupied, mask)
        # vel:   (obs_vec, action_idx)
        self._route_mem:       Dict[str, Optional[Tuple]] = {}
        self._vel_mem:         Dict[str, Tuple]           = {}
        self._collision_tally: Dict[str, int]             = {}

    # ------------------------------------------------------------------
    # Epsilon
    # ------------------------------------------------------------------

    def set_epsilon(self, epsilon_route: float, epsilon_vel: float) -> None:
        self.epsilon_route = epsilon_route
        self.epsilon_vel   = epsilon_vel

    def reset_memory(self) -> None:
        self._route_mem.clear()
        self._vel_mem.clear()
        self._collision_tally.clear()

    @staticmethod
    def decay_epsilon(start: float, end: float, episode: int, tau: float) -> float:
        """Decaimento exponencial: ε = end + (start - end) × e^(-episode / τ)."""
        return end + (start - end) * math.exp(-episode / tau)

    # ------------------------------------------------------------------
    # Act — chamado antes de engine.step()
    # ------------------------------------------------------------------

    def act(
        self,
        robot_id:     str,
        obs:          MotionObs,
        valid_actions: List[Action],
        goal:         str,
        world=None,
    ) -> Action:
        if obs.state == RobotState.IDLE:
            if obs.can_dispatch():
                return self._select_route(robot_id, obs, valid_actions, goal, world)
            return (None, "hold")

        if obs.state == RobotState.MOVING:
            if obs.docking_exit:
                return (None, "acc")
            return self._select_velocity(robot_id, obs, goal)

        return (None, "hold")

    # ------------------------------------------------------------------
    # Update — chamado depois de engine.step()
    # ------------------------------------------------------------------

    def update(
        self,
        robot_id: str,
        prev_obs: MotionObs,
        curr_obs: MotionObs,
        events:   Events,
        goal:     str,
    ) -> None:
        if prev_obs.state == RobotState.MOVING and not prev_obs.docking_exit:
            self._update_velocity(robot_id, curr_obs, events, goal)

        if prev_obs.state == RobotState.MOVING and curr_obs.state == RobotState.IDLE:
            self._update_route(robot_id, curr_obs, events, goal)

    # ------------------------------------------------------------------
    # Route — seleção
    # ------------------------------------------------------------------

    def _select_route(
        self,
        robot_id:     str,
        obs:          MotionObs,
        valid_actions: List[Action],
        goal:         str,
        world=None,
    ) -> Action:
        candidates = [a[0] for a in valid_actions if a[0] is not None]
        if not candidates:
            return (None, "hold")

        obs_vec, mask, neighbors = build_route_obs(obs, goal, self.graph, world)

        # Intersetar a máscara com os candidatos aprovados pelo action_builder
        cand_set = set(candidates)
        effective_mask = np.array([
            mask[i] and (i < len(neighbors) and neighbors[i] in cand_set)
            for i in range(MAX_NEIGHBORS)
        ], dtype=bool)

        if not effective_mask.any():
            # Todos os vizinhos observáveis foram filtrados pelo action_builder;
            # usar o primeiro candidato como fallback sem guardar memória.
            self._route_mem[robot_id] = None
            return (candidates[0], "acc")

        action_idx = self.dqn_route.select_action(obs_vec, effective_mask, self.epsilon_route)
        chosen     = neighbors[action_idx]

        dist_before   = _safe_dist(self.graph, obs.current_node, goal)
        node_occupied = obs.neighbor_idle.get(chosen, 0) > 0

        self._route_mem[robot_id] = (
            obs_vec, action_idx, neighbors, dist_before, node_occupied, effective_mask
        )
        return (chosen, "acc")

    # ------------------------------------------------------------------
    # Route — update (deferred, dispara em MOVING → IDLE)
    # ------------------------------------------------------------------

    def _update_route(
        self,
        robot_id: str,
        curr_obs: MotionObs,
        events:   Events,
        goal:     str,
    ) -> None:
        mem = self._route_mem.pop(robot_id, None)
        if mem is None:
            return

        obs_vec, action_idx, _neighbors, dist_before, node_occupied, _mask = mem

        dist_after = _safe_dist(self.graph, curr_obs.current_node, goal)
        n_col = (
            self._collision_tally.pop(robot_id, 0)
            + count_collisions(robot_id, events)
        )
        reward = route_reward(dist_before, dist_after, n_col, self.weights, node_occupied)

        done                      = curr_obs.current_node == goal
        next_vec, next_mask, _    = build_route_obs(curr_obs, goal, self.graph)

        self.dqn_route.push(obs_vec, action_idx, reward, next_vec, next_mask, done)

    # ------------------------------------------------------------------
    # Velocity — seleção
    # ------------------------------------------------------------------

    def _select_velocity(
        self,
        robot_id: str,
        obs:      MotionObs,
        goal:     str,
    ) -> Action:
        obs_vec    = build_velocity_obs(obs, goal, self.graph)
        action_idx = self.dqn_vel.select_action(obs_vec, _VEL_MASK, self.epsilon_vel)
        self._vel_mem[robot_id] = (obs_vec, action_idx)
        return (None, _VEL_CMDS[action_idx])

    # ------------------------------------------------------------------
    # Velocity — update
    # ------------------------------------------------------------------

    def _update_velocity(
        self,
        robot_id: str,
        curr_obs: MotionObs,
        events:   Events,
        goal:     str,
    ) -> None:
        mem = self._vel_mem.pop(robot_id, None)
        if mem is None:
            return

        obs_vec, action_idx = mem
        n_col = count_collisions(robot_id, events)
        self._collision_tally[robot_id] = (
            self._collision_tally.get(robot_id, 0) + n_col
        )

        is_blocked = (curr_obs.state == RobotState.MOVING and curr_obs.speed == 0.0)
        reward = velocity_reward(
            abs(curr_obs.speed), self.vel_max, n_col,
            self.weights, is_blocked, lead_gap=curr_obs.lead_gap,
        )

        done     = curr_obs.state == RobotState.IDLE
        next_vec = build_velocity_obs(curr_obs, goal, self.graph)
        self.dqn_vel.push(obs_vec, action_idx, reward, next_vec, _VEL_MASK, done)

    # ------------------------------------------------------------------
    # Lookahead buffer (mesma API que MotionController)
    # ------------------------------------------------------------------

    def compute_buffered_next_node(
        self,
        current_node: str,
        chosen_next:  str,
        goal:         str,
    ) -> Optional[str]:
        if chosen_next == goal:
            return None

        neighbors = list(self.graph.neighbors(chosen_next))
        if not neighbors:
            return None

        if current_node in neighbors and len(neighbors) > 1:
            neighbors = [n for n in neighbors if n != current_node]
        if not neighbors:
            return None

        try:
            return min(neighbors, key=lambda n: self.graph.shortest_path_length(n, goal))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Diagnóstico
    # ------------------------------------------------------------------

    def table_sizes(self) -> Dict[str, int]:
        return {
            "route_buf":   len(self.dqn_route.buffer),
            "vel_buf":     len(self.dqn_vel.buffer),
            "route_steps": self.dqn_route._step_count,
            "vel_steps":   self.dqn_vel._step_count,
        }

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.dqn_route.save(str(d / "dqn_route.pt"))
        self.dqn_vel.save(str(d / "dqn_vel.pt"))

    def load(self, directory: str | Path) -> None:
        d = Path(directory)
        self.dqn_route.load(str(d / "dqn_route.pt"))
        self.dqn_vel.load(str(d / "dqn_vel.pt"))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_dist(graph: FactoryGraph, node: Optional[str], goal: str) -> float:
    if node is None:
        return 0.0
    try:
        return graph.shortest_path_length(node, goal)
    except Exception:
        return 0.0
