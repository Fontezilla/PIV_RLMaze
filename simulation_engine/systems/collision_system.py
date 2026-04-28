import math
from typing import Tuple

from simulation_engine.core.world_state import WorldState
from simulation_engine.core.events import Events, CollisionEvent
from simulation_engine.core.entities import RobotState
from simulation_engine.core.graph import FactoryGraph


class CollisionSystem:
    """
    Sistema de colisões físico — deteção por sobreposição, separação por impulso.

    Cada tick, para cada par de robots:
        1. Calcula a distância entre os dois.
        2. Se dist < min_dist (sobreposição): empurra cada robot MOVING
           para trás na sua aresta pelo valor do overlap, garantindo que
           ficam fora da hitbox um do outro.
        3. Para o robot que está a avançar em direção ao outro (speed > 0,
           a aproximar-se), reseta a speed a 0 para prevenir tunneling.
        4. Marca collided=True e emite CollisionEvent.

    Sem look-ahead: a aprendizagem por reforço (penalização de colisão
    na reward + sinal collided no estado de velocidade) ensina o agente
    a travar proativamente antes do contacto.
    """

    def __init__(self, graph: FactoryGraph):
        self.graph = graph

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def update(self, world: WorldState, events: Events) -> None:
        self._reset_collision_flags(world)

        robots = list(world.robots.values())
        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                self._handle_pair(robots[i], robots[j], events)

        # Reset ticks counter for robots no longer colliding this tick
        for robot in robots:
            if not robot.collided:
                robot.collision_ticks = 0

    # ------------------------------------------------------------------
    # Reset flags
    # ------------------------------------------------------------------

    def _reset_collision_flags(self, world: WorldState) -> None:
        for robot in world.robots.values():
            robot.collided = False
            # collision_ticks NOT reset here — accumulates across consecutive ticks

    # ------------------------------------------------------------------
    # Par
    # ------------------------------------------------------------------

    def _handle_pair(self, r1, r2, events: Events) -> None:
        # Head-on tunnel detection: same physical edge, opposite directions.
        # Two robots can cross each other in a single tick without world-space
        # overlap ever being detected. Catch it by checking progress sum > 1.
        if (r1.state == RobotState.MOVING and r2.state == RobotState.MOVING
                and r1.from_node is not None and r1.to_node is not None
                and r1.from_node == r2.to_node and r1.to_node == r2.from_node):
            x1, y1 = self.graph.node_position(r1.from_node)
            x2, y2 = self.graph.node_position(r1.to_node)
            edge_len = math.hypot(x2 - x1, y2 - y1)
            min_dist_opp = r1.collision_radius + r2.collision_radius
            margin = (min_dist_opp / edge_len) if edge_len > 1e-9 else 0.0
            if r1.progress + r2.progress > 1.0 - margin:
                excess = r1.progress + r2.progress - (1.0 - margin)
                r1.progress = max(0.0, r1.progress - excess * 0.5)
                r2.progress = max(0.0, r2.progress - excess * 0.5)
                r1.speed = 0.0
                r2.speed = 0.0
                r1.collided = True
                r2.collided = True
                r1.collision_ticks += 1
                r2.collision_ticks += 1
                events.collisions.append(CollisionEvent(
                    robot_a=r1.id, robot_b=r2.id,
                    node_from=r1.from_node or "",
                    node_to=r1.to_node or "",
                ))
                return

        dx = r2.world_x - r1.world_x
        dy = r2.world_y - r1.world_y
        dist = math.hypot(dx, dy)
        min_dist = r1.collision_radius + r2.collision_radius

        if dist >= min_dist:
            return  # sem sobreposição — sem ação

        # ------------------------------------------------------------------
        # SOBREPOSIÇÃO DETETADA
        # ------------------------------------------------------------------

        r1.collided = True
        r2.collided = True
        r1.collision_ticks += 1
        r2.collision_ticks += 1

        # Parar robots que se aproximam mutuamente — evita tunneling
        if self._is_approaching(r1, r2):
            r1.speed = 0.0
        if self._is_approaching(r2, r1):
            r2.speed = 0.0

        # Empurrar robots para fora da hitbox via progress
        self._push_apart(r1, r2, dx, dy, dist, min_dist)

        events.collisions.append(CollisionEvent(
            robot_a=r1.id,
            robot_b=r2.id,
            node_from=r1.from_node or r1.current_node or "",
            node_to=r2.from_node or r2.current_node or "",
        ))

    # ------------------------------------------------------------------
    # Separação física
    # ------------------------------------------------------------------

    def _push_apart(
        self,
        r1, r2,
        dx: float, dy: float,
        dist: float, min_dist: float,
    ) -> None:
        """
        Empurra cada robot MOVING para trás na sua aresta pelo overlap.
        Robots IDLE ficam ancorados ao nó.
        """
        overlap = min_dist - dist
        if overlap <= 0.0:
            return

        r1_moving = r1.state == RobotState.MOVING
        r2_moving = r2.state == RobotState.MOVING

        if r1_moving and r2_moving:
            self._push_back(r1, overlap * 0.5)
            self._push_back(r2, overlap * 0.5)
        elif r1_moving:
            self._push_back(r1, overlap)
        elif r2_moving:
            self._push_back(r2, overlap)
        # ambos IDLE: ancorados — sem ação

    def _push_back(self, robot, distance: float) -> None:
        """Afasta o robot `distance` unidades da origem do movimento na aresta."""
        if robot.from_node is None or robot.to_node is None:
            return

        x1, y1 = self.graph.node_position(robot.from_node)
        x2, y2 = self.graph.node_position(robot.to_node)
        edge_len = math.hypot(x2 - x1, y2 - y1)

        if edge_len < 1e-9:
            return

        if robot.docking_exit:
            # docking_exit: robot move-se de to_node→from_node (progress decresce).
            # "afastar" = aumentar progress (empurrar de volta para to_node/process)
            robot.progress = min(1.0, robot.progress + distance / edge_len)
        else:
            robot.progress = max(0.0, robot.progress - distance / edge_len)

    # ------------------------------------------------------------------
    # Deteção de aproximação
    # ------------------------------------------------------------------

    def _is_approaching(self, mover, other) -> bool:
        """
        True se 'mover' está a reduzir a distância a 'other'.
        Usa produto interno da velocidade relativa com a direção mover→other.
        """
        if mover.state != RobotState.MOVING:
            return False
        if mover.from_node is None or mover.to_node is None:
            return False
        if mover.speed == 0.0:
            return False

        vx_m, vy_m = self._velocity(mover)
        vx_o, vy_o = self._velocity(other)

        rel_vx = vx_m - vx_o
        rel_vy = vy_m - vy_o

        dx = other.world_x - mover.world_x
        dy = other.world_y - mover.world_y
        dist = math.hypot(dx, dy)

        if dist == 0.0:
            return rel_vx != 0.0 or rel_vy != 0.0

        return (rel_vx * dx + rel_vy * dy) / dist > 0.0

    # ------------------------------------------------------------------
    # Helper: velocidade em coordenadas mundo (units/tick)
    # ------------------------------------------------------------------

    def _velocity(self, robot) -> Tuple[float, float]:
        if robot.state != RobotState.MOVING:
            return 0.0, 0.0
        if robot.from_node is None or robot.to_node is None:
            return 0.0, 0.0
        if robot.speed == 0.0:
            return 0.0, 0.0

        x1, y1 = self.graph.node_position(robot.from_node)
        x2, y2 = self.graph.node_position(robot.to_node)
        edge_len = math.hypot(x2 - x1, y2 - y1)

        if edge_len == 0.0:
            return 0.0, 0.0

        vx = (x2 - x1) / edge_len * robot.speed
        vy = (y2 - y1) / edge_len * robot.speed
        return vx, vy
