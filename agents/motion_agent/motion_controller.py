"""
motion_controller.py — Q-learning tabular para navegação de robots.

Dois Q-tables partilhados (parameter sharing) controlam todos os robots:

  Q_route    → escolha do próximo nó (na chegada a um nó)
  Q_velocity → comando de velocidade (dec/hold/acc) por tick

Fluxo por tick:
    obs    = iface.get_motion_view(robot_id)
    valid  = iface.get_valid_actions(robot_id)
    action = controller.act(robot_id, obs, valid, goal)
    ...
    events = engine.step(all_actions)
    ...
    new_obs = iface.get_motion_view(robot_id)
    controller.update(robot_id, obs, new_obs, events, goal)

Preparação para lookahead:
    - o controller continua a escolher apenas o próximo nó
    - expõe um helper para sugerir um buffered_next_node heurístico
    - a validação/consumo desse buffer fica para o engine
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from simulation_engine.core.entities import RobotState
from simulation_engine.core.graph import FactoryGraph
from simulation_engine.core.events import Events
from simulation_engine.interface.observation_builder import MotionObs

from agents.motion_agent.q_tables import TabularQTable
from agents.motion_agent.state_encoder import encode_route_state, encode_velocity_state
from agents.motion_agent.reward import (
    RewardWeights,
    count_collisions,
    route_reward,
    velocity_reward,
)

# Índices das ações de velocidade
_VEL_DEC = 0
_VEL_HOLD = 1
_VEL_ACC = 2
_VEL_CMDS = ["dec", "hold", "acc"]
_N_VEL = 3

Action = Tuple[Optional[str], str]


class MotionController:
    """
    Controller Q-learning tabular partilhado por todos os robots.

    Um único MotionController é criado por treino; todos os robots usam
    os mesmos Q-tables (parameter sharing), o que acelera a convergência.
    """

    def __init__(
        self,
        graph: FactoryGraph,
        vel_max: float = 3.0,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_route: float = 0.30,
        epsilon_vel: float = 0.15,
        reward_weights: Optional[RewardWeights] = None,
    ) -> None:
        self.graph = graph
        self.vel_max = vel_max
        self.weights = reward_weights or RewardWeights()

        self.q_route = TabularQTable(
            alpha=alpha,
            gamma=gamma,
            default_q=0.0,
            fixed_n_actions=None,
        )
        self.q_velocity = TabularQTable(
            alpha=alpha,
            gamma=gamma,
            default_q=0.0,
            fixed_n_actions=_N_VEL,
        )

        self.epsilon_route: float = epsilon_route
        self.epsilon_vel: float = epsilon_vel

        # Memória por robot para deferred Q-updates
        # velocity: (state, action_idx)
        # route:    (state, next_node, dist_before, node_occupied)
        self._vel_mem: Dict[str, Tuple] = {}
        self._route_mem: Dict[str, Tuple] = {}

        # Colisões acumuladas por robot desde o último update de route.
        # Necessário porque _update_route só corre na chegada ao nó,
        # mas as colisões podem ocorrer em qualquer tick da jornada.
        self._collision_tally: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Epsilon
    # ------------------------------------------------------------------

    def set_epsilon(self, epsilon_route: float, epsilon_vel: float) -> None:
        self.epsilon_route = epsilon_route
        self.epsilon_vel = epsilon_vel

    def reset_memory(self) -> None:
        """Limpa memória deferred entre episódios."""
        self._vel_mem.clear()
        self._route_mem.clear()
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
        robot_id: str,
        obs: MotionObs,
        valid_actions: List[Action],
        goal: str,
    ) -> Action:
        """
        Devolve a próxima ação para robot_id.

        - IDLE + pode despachar   → Q_route seleciona próximo nó
        - MOVING (normal)         → Q_velocity seleciona dec/hold/acc
        - MOVING (docking_exit)   → sempre "acc" (dispatch_system gere)
        - caso contrário          → "hold"
        """
        if obs.state == RobotState.IDLE:
            if obs.can_dispatch():
                return self._select_route(robot_id, obs, valid_actions, goal)
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
        events: Events,
        goal: str,
    ) -> None:
        """
        Atualiza Q-tables com base na transição (prev_obs → curr_obs).

        - Se o robot estava MOVING: atualiza Q_velocity
        - Se o robot transitou MOVING → IDLE: atualiza Q_route
        """
        if prev_obs.state == RobotState.MOVING and not prev_obs.docking_exit:
            self._update_velocity(robot_id, curr_obs, events, goal)

        if prev_obs.state == RobotState.MOVING and curr_obs.state == RobotState.IDLE:
            self._update_route(robot_id, curr_obs, events, goal)

    # ------------------------------------------------------------------
    # Helpers — route candidates
    # ------------------------------------------------------------------

    def _route_candidates_from_valid_actions(self, valid_actions: List[Action]) -> List[str]:
        """Extrai next_nodes válidos de uma lista de ações."""
        return [a[0] for a in valid_actions if a[0] is not None]

    def _route_candidates_for_bootstrap(self, obs: MotionObs) -> List[str]:
        """
        Candidatos de bootstrap para Q_route.

        Importante:
        - Q_route decide apenas sobre próximos nós.
        - O estado de route NÃO codifica cooldowns (`yield_ticks`, `turn_cooldown`),
          por isso o bootstrap não deve depender de `obs.can_dispatch()`.

        Esta função reproduz a parte estrutural do action_builder:
        - anti-U-turn
        - sem tráfego oposto
        - sem hold
        """
        candidates = list(obs.neighbors)
        if not candidates:
            return []

        # anti U-turn
        if obs.prev_node is not None and len(candidates) > 1:
            no_uturn = [n for n in candidates if n != obs.prev_node]
            if no_uturn:
                candidates = no_uturn

        # sem tráfego oposto
        no_opposing = [n for n in candidates if not obs.has_opposing(n)]
        if no_opposing:
            candidates = no_opposing

        return candidates if candidates else list(obs.neighbors)

    # ------------------------------------------------------------------
    # Lookahead / buffer preparation
    # ------------------------------------------------------------------

    def compute_buffered_next_node(
        self,
        current_node: str,
        chosen_next: str,
        goal: str,
    ) -> Optional[str]:
        """
        Sugere um buffered_next_node heurístico para usar depois de `chosen_next`.

        Regras:
        - se `chosen_next` já for o goal, devolve None
        - evita voltar imediatamente a `current_node` se existirem alternativas
        - escolhe o vizinho de `chosen_next` com menor distância ao goal

        Nota:
        - isto é apenas uma sugestão heurística
        - o engine deve revalidar o buffer antes de o consumir
        - se houver colisão, reversão, cooldown ou invalidação local,
          o buffer deve ser descartado
        """
        if chosen_next == goal:
            return None

        neighbors = list(self.graph.neighbors(chosen_next))
        if not neighbors:
            return None

        # evitar U-turn imediato para o nó atual, se houver alternativas
        if current_node in neighbors and len(neighbors) > 1:
            filtered = [n for n in neighbors if n != current_node]
            if filtered:
                neighbors = filtered

        if not neighbors:
            return None

        try:
            return min(
                neighbors,
                key=lambda n: self.graph.shortest_path_length(n, goal),
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Q_route — seleção e update
    # ------------------------------------------------------------------

    def _select_route(
        self,
        robot_id: str,
        obs: MotionObs,
        valid_actions: List[Action],
        goal: str,
    ) -> Action:
        """
        ε-greedy sobre Q_route; armazena
        (state, next_node, dist_before, node_occupied) em memória.
        """
        candidates = self._route_candidates_from_valid_actions(valid_actions)
        if not candidates:
            return (None, "hold")

        state = encode_route_state(obs, goal, self.graph)

        if random.random() < self.epsilon_route:
            chosen = random.choice(candidates)
        else:
            chosen = self.q_route.best_action(state, candidates)

        dist_before = self.graph.shortest_path_length(obs.current_node, goal)

        # congestionamento local no momento da escolha
        node_occupied = obs.neighbor_idle.get(chosen, 0) > 0

        self._route_mem[robot_id] = (state, chosen, dist_before, node_occupied)
        return (chosen, "acc")

    def _update_route(
        self,
        robot_id: str,
        curr_obs: MotionObs,
        events: Events,
        goal: str,
    ) -> None:
        """Q-update para Q_route após chegada ao nó (MOVING → IDLE)."""
        mem = self._route_mem.pop(robot_id, None)
        if mem is None:
            return

        state, action, dist_before, node_occupied = mem

        dist_after = self.graph.shortest_path_length(curr_obs.current_node, goal)
        # usar total acumulado da jornada; count_collisions(events) captura só o tick de chegada
        n_col = self._collision_tally.pop(robot_id, 0) + count_collisions(robot_id, events)
        reward = route_reward(
            dist_before,
            dist_after,
            n_col,
            self.weights,
            node_occupied,
        )

        if curr_obs.current_node == goal:
            self.q_route.update(state, action, reward, state, [], done=True)
            return

        next_state = encode_route_state(curr_obs, goal, self.graph)
        next_cands = self._route_candidates_for_bootstrap(curr_obs)
        self.q_route.update(state, action, reward, next_state, next_cands)

    # ------------------------------------------------------------------
    # Q_velocity — seleção e update
    # ------------------------------------------------------------------

    def _select_velocity(
        self,
        robot_id: str,
        obs: MotionObs,
        goal: str,
    ) -> Action:
        """ε-greedy sobre Q_velocity; armazena (state, action_idx) em memória."""
        state = encode_velocity_state(obs, goal, self.vel_max, self.graph)

        if random.random() < self.epsilon_vel:
            action_idx = random.randint(0, _N_VEL - 1)
        else:
            action_idx = self.q_velocity.best_action(state, list(range(_N_VEL)))

        self._vel_mem[robot_id] = (state, action_idx)
        return (None, _VEL_CMDS[action_idx])

    def _update_velocity(
        self,
        robot_id: str,
        curr_obs: MotionObs,
        events: Events,
        goal: str,
    ) -> None:
        """Q-update para Q_velocity após cada tick MOVING."""
        mem = self._vel_mem.pop(robot_id, None)
        if mem is None:
            return

        state, action_idx = mem
        n_col = count_collisions(robot_id, events)
        # acumular para que _update_route receba o total da jornada
        self._collision_tally[robot_id] = self._collision_tally.get(robot_id, 0) + n_col
        is_blocked = curr_obs.state == RobotState.MOVING and curr_obs.speed == 0.0
        reward = velocity_reward(
            abs(curr_obs.speed),
            self.vel_max,
            n_col,
            self.weights,
            is_blocked,
            lead_gap=curr_obs.lead_gap,
        )

        done = curr_obs.state == RobotState.IDLE
        if done:
            self.q_velocity.update(state, action_idx, reward, state, [], done=True)
        else:
            next_state = encode_velocity_state(curr_obs, goal, self.vel_max, self.graph)
            self.q_velocity.update(
                state,
                action_idx,
                reward,
                next_state,
                list(range(_N_VEL)),
            )

    # ------------------------------------------------------------------
    # Diagnóstico
    # ------------------------------------------------------------------

    def table_sizes(self) -> Dict[str, int]:
        return {
            "q_route_states": self.q_route.size(),
            "q_route_entries": self.q_route.n_entries(),
            "q_vel_states": self.q_velocity.size(),
            "q_vel_entries": self.q_velocity.n_entries(),
        }

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Guarda ambas as Q-tables em `directory`."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.q_route.save(d / "q_route.pkl")
        self.q_velocity.save(d / "q_velocity.pkl")

    def load(self, directory: str | Path) -> None:
        """Carrega Q-tables de `directory`."""
        d = Path(directory)
        self.q_route = TabularQTable.load(d / "q_route.pkl")
        self.q_velocity = TabularQTable.load(d / "q_velocity.pkl")