from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from old.env.core.entities import Robot, RobotState
from old.env.core.graph import FactoryGraph
from old.env.core.physics import MAX_SPEED, tick as physics_tick, turn_delay
from old.env.core.world import World
from old.env.traffic.router import Router
from old.render.renderer import Renderer

YAML_PATH = Path(".configs/map_factory.yaml")
CACHE_PATH = Path(".configs/graph_cache.pkl")
CSV_PATH = Path("debug_ticks.csv")
N_ROBOTS = 8
SEED = 42


def random_goal(graph: FactoryGraph, current_node: str, rng: random.Random) -> str:
    """
    Sorteia um junction node alcançável a partir de current_node.

    Usa graph.reachable_from() para filtrar apenas goals válidos —
    evita atribuir goals inalcançáveis que forçam o router a gerar
    planos parciais através de dead-ends.

    Fallback para todos os junction nodes se nenhum for alcançável
    (não deve acontecer num grafo bem formado).
    """
    reachable = graph.reachable_from(current_node)
    options = [
        n for n in graph.junction_nodes()
        if n != current_node and n in reachable
    ]
    if not options:
        # Fallback de segurança — grafo mal formado ou nó isolado
        options = [n for n in graph.junction_nodes() if n != current_node]
    return rng.choice(options)


def dispatch_move(robot: Robot, next_node: str, graph: FactoryGraph) -> None:
    """
    Envia o robot para um nó.

    came_from não é alterado aqui.
    came_from representa o nó anterior ao current_node.
    Só é actualizado quando o robot chega ao próximo nó em physics.tick().
    """
    delay = (
        turn_delay(graph, robot.came_from, robot.current_node, next_node)
        if robot.current_node
        else 0
    )

    robot.from_node = robot.current_node
    robot.to_node = next_node
    robot.current_node = None
    robot.progress = 0.0
    robot.wait_ticks = delay
    robot.turn_ticks_total = delay
    robot.speed = 0.0 if delay > 0 else robot.speed
    robot.state = RobotState.MOVING
    robot.target_speed = MAX_SPEED
    robot.wait_ticks_in_junction = 0
    robot.parked_at = None


def dispatch_park(robot: Robot, park_spec: str, graph: FactoryGraph, router: Router) -> None:
    """Envia o robot para um parking point. park_spec = 'u|v|fraction'."""
    u, v, frac_str = park_spec.split("|")
    fraction = float(frac_str)

    robot.from_node              = u
    robot.to_node                = v
    robot.current_node           = None
    robot.progress               = 0.0
    robot.wait_ticks             = 0
    robot.speed                  = 0.0
    robot.state                  = RobotState.MOVING
    robot.target_speed           = MAX_SPEED
    robot.wait_ticks_in_junction = 0
    robot.parked_at              = (u, v, fraction)

    # O robot saiu fisicamente do nó u.
    # O nó fica livre, mas o segmento u-v fica reservado.
    router.node_lock.release(u, robot.id)


def dispatch_unpark(
    robot: Robot,
    target_node: str | None,
    graph: FactoryGraph,
    router: Router,
) -> None:
    """
    Robot sai do parking.

    Se target_node for o nó da frente da aresta, continua para a frente.
    Se target_node for o nó de origem, recua.
    """
    if robot.parked_at is None:
        return

    u, v, frac = robot.parked_at

    if target_node == v:
        if not router.node_lock.try_acquire(v, robot.id):
            return

        robot.from_node = u
        robot.to_node = v
        robot.current_node = None
        robot.came_from = u
        robot.progress = frac

    else:
        if not router.node_lock.try_acquire(u, robot.id):
            return

        robot.from_node = v
        robot.to_node = u
        robot.current_node = None
        robot.came_from = v
        robot.progress = 1.0 - frac

    robot.wait_ticks = 0
    robot.state = RobotState.MOVING
    robot.speed = 0.0
    robot.target_speed = MAX_SPEED
    robot.wait_ticks_in_junction = 0
    router.unparked(robot)   # liberta parking_lock ANTES de limpar parked_at


def main() -> None:
    rng = random.Random(SEED)
    graph = FactoryGraph(str(YAML_PATH), str(CACHE_PATH))
    world = World()
    router = Router(graph)
    renderer = Renderer(graph)

    spawn_nodes = rng.sample(graph.junction_nodes(), N_ROBOTS)

    for i, node in enumerate(spawn_nodes):
        robot = Robot(id=f"robot_{i}")
        robot.current_node = node
        robot.state = RobotState.IDLE
        robot.came_from = None
        robot.goal_node = random_goal(graph, node, rng)
        robot.target_speed = MAX_SPEED
        robot.speed = 0.0
        robot.wait_ticks = 0
        robot.wait_ticks_in_junction = 0
        robot.parked_at = None

        x, y = graph.node_position(node)
        robot.world_x = x
        robot.world_y = y

        world.add_robot(robot)
        router.register(robot)

    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "tick",
        "robot_id",
        "state",
        "from_node",
        "to_node",
        "progress",
        "current_node",
        "came_from",
        "goal_node",
        "wait_ticks_in_junction",
        "parked_at",
    ])

    while True:
        world.step_tick()

        # Reset WAITING -> IDLE
        for robot in world.all_robots():
            if robot.state == RobotState.WAITING:
                robot.state = RobotState.IDLE

        # Física
        for robot in world.all_robots():
            arrived = physics_tick(graph, robot)

            if arrived:
                if robot.state == RobotState.PARKED:
                    if robot.parked_at is not None:
                        u, v, frac = robot.parked_at
                        router.parked(robot, u, v, frac)
                else:
                    if robot.current_node is not None:
                        router.arrived(robot, robot.current_node)
                        robot.wait_ticks_in_junction = 0

                        if robot.reached_goal():
                            router.release_all(robot)
                            robot.goal_node = random_goal(
                                graph, robot.current_node, rng
                            )
                            router.register(robot)

        # Liberta nós de origem quando robots MOVING já saíram fisicamente deles.
        router.sync_moving_node_locks(world.all_robots())

        # Decisão
        robots = world.all_robots()

        for robot in robots:
            if robot.state == RobotState.MOVING:
                continue

            if robot.is_idle() and robot.current_node == robot.goal_node:
                continue

            decision = router.decide(robot, robots)

            if decision is None:
                robot.wait_ticks_in_junction += 1
                if robot.is_idle():
                    robot.state = RobotState.WAITING
                continue

            action, payload = decision

            if action == "move" and payload is not None:
                dispatch_move(robot, payload, graph)

            elif action == "park" and payload is not None:
                dispatch_park(robot, payload, graph, router)

            elif action == "unpark":
                dispatch_unpark(robot, payload, graph, router)

        for robot in world.all_robots():
            parked_str = ""

            if robot.parked_at:
                u, v, f = robot.parked_at
                parked_str = f"{u}|{v}|{f:.2f}"

            csv_writer.writerow([
                world.tick,
                robot.id,
                robot.state.name,
                robot.from_node or "",
                robot.to_node or "",
                f"{robot.progress:.3f}",
                robot.current_node or "",
                robot.came_from or "",
                robot.goal_node or "",
                robot.wait_ticks_in_junction,
                parked_str,
            ])

        csv_file.flush()

        info = {"tick": world.tick, "robots": world.robot_count()}
        result = renderer.render(world, info)

        if result == "quit":
            break

    csv_file.close()
    renderer.close()


if __name__ == "__main__":
    main()