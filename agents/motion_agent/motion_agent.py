"""
motion_agent.py — Agente de navegação baseado em caminho mínimo.

O MotionAgent recebe a cada tick:
  - MotionObs  → observação local (posição, tráfego, cooldowns)
  - valid_actions → ações filtradas pelo action_builder

E devolve a ação que aproxima o robot do seu goal_node.

Estratégia
----------
  MOVING → sempre "acc" (manter velocidade de cruzeiro)
  IDLE, aguardar cooldown → "hold"
  IDLE, sem goal ou no goal → "hold"
  IDLE, com goal → escolher o next hop ótimo disponível

Prioridade de seleção de next hop:
  1. Hops nos caminhos mínimos (all_next_hops) que estão nas valid_actions
  2. Qualquer vizinho válido que reduza a distância ao goal
  3. Hold
"""

from __future__ import annotations

from typing import Collection, List, Optional, Tuple

from simulation_engine.core.entities import RobotState
from simulation_engine.core.graph import FactoryGraph
from simulation_engine.interface.observation_builder import MotionObs
import simulation_engine.systems.routing_utils as routing

Action = Tuple[Optional[str], str]


class MotionAgent:
    """
    Agente de navegação para um único robot.

    Parameters
    ----------
    graph     : grafo da fábrica (pré-computado)
    robot_id  : ID do robot que este agente controla
    """

    def __init__(self, graph: FactoryGraph, robot_id: str):
        self.graph    = graph
        self.robot_id = robot_id
        self._goal: Optional[str] = None

    # ------------------------------------------------------------------
    # Gestão do goal
    # ------------------------------------------------------------------

    def set_goal(self, goal_node: str) -> None:
        """Define o nó de destino para navegar."""
        self._goal = goal_node

    def clear_goal(self) -> None:
        """Remove o objetivo atual (robot fica IDLE)."""
        self._goal = None

    @property
    def goal(self) -> Optional[str]:
        return self._goal

    def has_arrived(self, obs: MotionObs) -> bool:
        """True se o robot está IDLE exatamente no goal."""
        return (
            self._goal is not None
            and obs.state == RobotState.IDLE
            and obs.current_node == self._goal
        )

    # ------------------------------------------------------------------
    # Decisão por tick
    # ------------------------------------------------------------------

    def act(self, obs: MotionObs, valid_actions: List[Action]) -> Action:
        """
        Devolve a próxima ação.

        Parameters
        ----------
        obs           : observação local do robot neste tick
        valid_actions : ações filtradas pelo action_builder
        """

        # ------------------------------------------------------------------
        # MOVING: manter velocidade de cruzeiro
        # ------------------------------------------------------------------
        if obs.state == RobotState.MOVING:
            return (None, "acc")

        # ------------------------------------------------------------------
        # IDLE mas em cooldown
        # ------------------------------------------------------------------
        if not obs.can_dispatch():
            return (None, "hold")

        # ------------------------------------------------------------------
        # Sem goal, ou já no goal
        # ------------------------------------------------------------------
        if self._goal is None or obs.current_node == self._goal:
            return (None, "hold")

        # ------------------------------------------------------------------
        # Escolher next hop
        # ------------------------------------------------------------------
        return self._choose_next(obs, valid_actions)

    # ------------------------------------------------------------------
    # Seleção de next hop
    # ------------------------------------------------------------------

    def _choose_next(self, obs: MotionObs, valid_actions: List[Action]) -> Action:
        """Escolhe o melhor next node a partir das valid_actions."""

        current = obs.current_node

        # Nós disponíveis nas valid_actions (excluindo None/hold)
        valid_nodes = {a[0] for a in valid_actions if a[0] is not None}
        if not valid_nodes:
            return (None, "hold")

        # ------------------------------------------------------------------
        # Prioridade 1 — hops nos caminhos mínimos
        # ------------------------------------------------------------------
        optimal = routing.all_next_hops(self.graph, current, self._goal)
        reachable_optimal = optimal & valid_nodes
        if reachable_optimal:
            best = self._min_dist_to_goal(reachable_optimal)
            return (best, "acc")

        # ------------------------------------------------------------------
        # Prioridade 2 — qualquer vizinho que reduza a distância ao goal
        # ------------------------------------------------------------------
        current_dist = routing.route_distance(self.graph, current, self._goal)
        closer = {
            n for n in valid_nodes
            if routing.route_distance(self.graph, n, self._goal) < current_dist
        }
        if closer:
            best = self._min_dist_to_goal(closer)
            return (best, "acc")

        # ------------------------------------------------------------------
        # Sem opção melhor — aguardar (e.g., tráfego oposto em todos os hops)
        # ------------------------------------------------------------------
        return (None, "hold")

    def _min_dist_to_goal(self, nodes: Collection[str]) -> str:
        """Devolve o nó em `nodes` com menor distância ao goal."""
        return min(nodes, key=lambda n: routing.route_distance(self.graph, n, self._goal))
