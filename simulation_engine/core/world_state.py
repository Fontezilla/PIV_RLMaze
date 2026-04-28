from typing import Dict, List

from simulation_engine.core.entities import Robot, Box, Process, RobotState, BoxState


class WorldState:
    """
    Representa o estado global do ambiente num dado instante.

    Este é o "single source of truth" do SimulationEngine.
    Todos os systems modificam este estado.
    Agents apenas leem (via interface).
    """

    def __init__(self):
        # ------------------------------------------------------------------
        # Entidades
        # ------------------------------------------------------------------

        self.robots: Dict[str, Robot] = {}
        self.boxes: Dict[str, Box] = {}
        self.processes: Dict[str, Process] = {}

        # ------------------------------------------------------------------
        # Tempo
        # ------------------------------------------------------------------

        self.tick: int = 0

        # ------------------------------------------------------------------
        # Histórico
        # ------------------------------------------------------------------

        self.delivered_boxes: List[str] = []

    # ----------------------------------------------------------------------
    # Gestão de entidades
    # ----------------------------------------------------------------------

    def add_robot(self, robot: Robot) -> None:
        self.robots[robot.id] = robot

    def add_box(self, box: Box) -> None:
        self.boxes[box.id] = box

    def add_process(self, process: Process) -> None:
        self.processes[process.node_id] = process

    # ----------------------------------------------------------------------
    # Acesso direto
    # ----------------------------------------------------------------------

    def get_robot(self, robot_id: str) -> Robot:
        return self.robots[robot_id]

    def get_box(self, box_id: str) -> Box:
        return self.boxes[box_id]

    def get_process(self, node_id: str) -> Process:
        return self.processes[node_id]

    # ----------------------------------------------------------------------
    # Queries simples (sem lógica pesada)
    # ----------------------------------------------------------------------

    def robots_at_node(self, node_id: str) -> List[Robot]:
        """
        Devolve todos os robôs que estão num determinado nó (IDLE).
        """
        return [
            r for r in self.robots.values()
            if r.current_node == node_id
        ]

    def boxes_at_node(self, node_id: str) -> List[Box]:
        """
        Devolve todas as caixas disponíveis num nó.
        """
        return [
            b for b in self.boxes.values()
            if b.current_node == node_id and b.state == BoxState.AT_NODE
        ]

    def robots_on_edge(self, from_node: str, to_node: str) -> List[Robot]:
        """
        Devolve robôs que estão a mover-se numa determinada aresta.
        """
        return [
            r for r in self.robots.values()
            if r.state == RobotState.MOVING
            and r.from_node == from_node
            and r.to_node == to_node
        ]

    # ----------------------------------------------------------------------
    # Helpers úteis
    # ----------------------------------------------------------------------

    def is_node_free(self, node_id: str) -> bool:
        """
        Verifica se um nó está livre:
        - nenhum robô IDLE nele
        - nenhum robô MOVING a dirigir-se para ele
        """
        for r in self.robots.values():
            if r.current_node == node_id:
                return False
            if r.to_node == node_id:
                return False
        return True

    def is_box_available(self, box_id: str) -> bool:
        """
        Verifica se uma caixa está disponível para pickup.
        """
        box = self.boxes.get(box_id)
        return (
            box is not None
            and box.state == BoxState.AT_NODE
            and box.carried_by is None
        )
        
    def edge_has_opposing_traffic(self, from_node: str, to_node: str) -> bool:
        """
        Verifica se existe tráfego físico na direção oposta (to_node → from_node).

        Inclui docking exits: um robot em docking exit com
        (from_node=J, to_node=E, docking_exit=True) move-se fisicamente
        de E para J, pelo que opõe-se a um novo despacho de J para E.
        """
        for r in self.robots.values():
            if r.state != RobotState.MOVING:
                continue
            # Tráfego normal em sentido oposto
            if r.from_node == to_node and r.to_node == from_node:
                return True
            # Docking exit: armazenado com (from=J, to=E) mas move-se E→J
            # opõe-se a qualquer despacho J→E na mesma aresta
            if r.docking_exit and r.from_node == from_node and r.to_node == to_node:
                return True
        return False

    # ----------------------------------------------------------------------
    # Reset
    # ----------------------------------------------------------------------

    def reset(self) -> None:
        """
        Limpa completamente o estado do mundo (novo episódio).
        """
        self.robots.clear()
        self.boxes.clear()
        self.processes.clear()
        self.delivered_boxes.clear()
        self.tick = 0