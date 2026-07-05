"""Física do movimento dos robots no grafo."""

from __future__ import annotations

from old.env.core.entities import Robot, RobotState
from old.env.core.graph import FactoryGraph

MAX_SPEED      = 30.0
V_STRAIGHT     = 30.0
V_CURVE        = 18.0
V_REVERSE      = 6.0
ACCEL          = 6.0
TURN_THRESHOLD = 45.0

SHORTEST_EDGE  = 150.0
CURVE_ZONE     = SHORTEST_EDGE * 0.30

TURN_TICKS_90  = 3
TURN_TICKS_180 = 5


def node_degree(graph: FactoryGraph, node: str) -> int:
    return graph.graph.degree(node)


def is_dead_end(graph: FactoryGraph, node: str) -> bool:
    return node_degree(graph, node) == 1


def turn_delay(graph: FactoryGraph, came_from: str | None, at: str, going_to: str) -> int:
    """Ticks de atraso por turn em `at`."""
    if came_from is None or came_from == at:
        return 0

    try:
        angle = graph.turn_angle(came_from, at, going_to)
    except (ValueError, KeyError):
        return 0

    if angle >= 150:
        return TURN_TICKS_180
    if angle >= TURN_THRESHOLD:
        return TURN_TICKS_90
    return 0


def _is_curve(graph: FactoryGraph, came_from: str | None, at: str, going_to: str) -> bool:
    if came_from is None or came_from == at or came_from == going_to:
        return False
    try:
        return graph.turn_angle(came_from, at, going_to) >= TURN_THRESHOLD
    except (ValueError, KeyError):
        return False


def edge_speed_limit(graph: FactoryGraph, robot: Robot) -> float:
    """Velocidade máxima para a posição actual do robot."""
    if robot.from_node is None or robot.to_node is None:
        return V_STRAIGHT

    from_node = robot.from_node
    to_node   = robot.to_node
    came_from = robot.came_from

    if robot.parked_at is not None:
        return V_CURVE

    if is_dead_end(graph, from_node) and robot.progress > 0.01:
        return V_REVERSE

    dist = graph.edge_distance(from_node, to_node)
    if dist <= 0:
        return V_STRAIGHT

    threshold_exit     = CURVE_ZONE / dist
    threshold_approach = 1.0 - (CURVE_ZONE / dist)

    if robot.progress < threshold_exit:
        if _is_curve(graph, came_from, from_node, to_node):
            return V_CURVE

    if robot.progress > threshold_approach:
        for next_node in graph.neighbors(to_node):
            if next_node == from_node:
                continue
            if _is_curve(graph, from_node, to_node, next_node):
                return V_CURVE

    return V_STRAIGHT


def compute_world_pos(graph: FactoryGraph, robot: Robot) -> tuple[float, float]:
    """Posição mundial interpolada do robot."""
    if robot.state == RobotState.MOVING and robot.from_node and robot.to_node:
        x0, y0 = graph.node_position(robot.from_node)
        x1, y1 = graph.node_position(robot.to_node)
        t = robot.progress
        return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t

    if robot.parked_at is not None:
        u, v, frac = robot.parked_at
        return graph.parking_point_position(u, v, frac)

    if robot.current_node is not None:
        return graph.node_position(robot.current_node)

    return robot.world_x, robot.world_y


def tick(graph: FactoryGraph, robot: Robot) -> bool:
    """Avança o robot um tick. True quando chega a um nó ou parking point."""
    if robot.wait_ticks > 0:
        robot.wait_ticks -= 1
        robot.world_x, robot.world_y = compute_world_pos(graph, robot)
        return False

    if robot.state != RobotState.MOVING:
        robot.world_x, robot.world_y = compute_world_pos(graph, robot)
        return False

    if robot.from_node is None or robot.to_node is None:
        robot.world_x, robot.world_y = compute_world_pos(graph, robot)
        return False

    v_limit = edge_speed_limit(graph, robot)
    target  = min(robot.target_speed, v_limit)

    if robot.speed < target:
        robot.speed = min(robot.speed + ACCEL, target)
    elif robot.speed > target:
        robot.speed = max(robot.speed - ACCEL, target)

    dist = graph.edge_distance(robot.from_node, robot.to_node)
    delta = 1.0 if dist <= 0 else robot.speed / dist

    target_progress = 1.0
    if robot.parked_at is not None:
        _, _, frac = robot.parked_at
        target_progress = frac

    robot.progress = min(robot.progress + delta, target_progress)
    robot.world_x, robot.world_y = compute_world_pos(graph, robot)

    if robot.progress < target_progress:
        return False

    if robot.parked_at is not None:
        u, v, frac = robot.parked_at
        robot.state        = RobotState.PARKED
        robot.current_node = u
        robot.from_node    = None
        robot.to_node      = None
        robot.progress     = frac
        robot.speed        = 0.0
        robot.world_x, robot.world_y = graph.parking_point_position(u, v, frac)
        return True

    robot.came_from    = robot.from_node
    robot.current_node = robot.to_node
    robot.from_node    = None
    robot.to_node      = None
    robot.progress     = 0.0
    robot.state        = RobotState.IDLE
    robot.speed        = 0.0
    robot.world_x, robot.world_y = graph.node_position(robot.current_node)
    return True
