"""
robot_interface.py — Ponto de entrada read-only para os agentes.

RobotInterface é instanciado pelo SimulationEngine e passado aos agentes.
Nunca modifica o WorldState; apenas constrói vistas derivadas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from simulation_engine.core.entities import BoxState
from simulation_engine.core.world_state import WorldState
from simulation_engine.core.graph import FactoryGraph

from simulation_engine.interface.observation_builder import (
    MotionObs,
    build_motion_obs,
)
from simulation_engine.interface.action_builder import (
    Action,
    get_valid_actions,
)


# ---------------------------------------------------------------------------
# TaskView
# ---------------------------------------------------------------------------

@dataclass
class TaskView:
    """
    Visão de tarefa de alto nível para um robot.

    carried_box    — ID da box transportada (None se livre)
    box_next_node  — próximo nó da pipeline desta box (None se nenhuma)
    boxes_here     — IDs de boxes disponíveis para pickup no nó atual
    """
    robot_id:      str
    carried_box:   Optional[str]
    box_next_node: Optional[str]
    boxes_here:    List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RobotInterface
# ---------------------------------------------------------------------------

class RobotInterface:
    """
    Fachada read-only para acesso ao estado do ambiente por parte dos agentes.

    Expõe:
        get_motion_view(robot_id)  → MotionObs
        get_valid_actions(robot_id) → List[Action]
        get_task_view(robot_id)    → TaskView
        robot_ids()                → List[str]
    """

    def __init__(self, world: WorldState, graph: FactoryGraph):
        self._world = world
        self._graph = graph

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def get_motion_view(self, robot_id: str) -> MotionObs:
        """Observação local para o MotionAgent."""
        return build_motion_obs(self._world, self._graph, robot_id)

    def get_valid_actions(self, robot_id: str, obs: Optional[MotionObs] = None) -> List[Action]:
        """Lista de ações filtradas para este robot neste tick.

        Pass a prior get_motion_view result as obs to avoid rebuilding.
        """
        if obs is None:
            obs = build_motion_obs(self._world, self._graph, robot_id)
        return get_valid_actions(obs)

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------

    def get_task_view(self, robot_id: str) -> TaskView:
        """Visão de alto nível sobre a tarefa atual do robot."""
        robot = self._world.robots[robot_id]

        carried_box   = robot.carried_box
        box_next_node = None

        if carried_box is not None:
            box = self._world.boxes[carried_box]
            box_next_node = box.destination_node

        boxes_here: List[str] = []
        if robot.current_node is not None:
            boxes_here = [
                b.id for b in self._world.boxes_at_node(robot.current_node)
            ]

        return TaskView(
            robot_id      = robot_id,
            carried_box   = carried_box,
            box_next_node = box_next_node,
            boxes_here    = boxes_here,
        )

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def robot_ids(self) -> List[str]:
        """Lista de IDs de todos os robots registados."""
        return list(self._world.robots.keys())
