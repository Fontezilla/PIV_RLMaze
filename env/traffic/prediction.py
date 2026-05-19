from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from env.core.entities import Robot, RobotState
from env.core.graph import FactoryGraph


LOOKAHEAD_STEPS = 5

PENALTY_RESERVED_EDGE = 10_000.0
PENALTY_ENTRY_EXIT_EDGE = 900.0
PENALTY_PROCESS_EDGE = 750.0
PENALTY_JUNCTION_CORRIDOR = 250.0
PENALTY_DEAD_END = 350.0
PENALTY_NEAR_GOAL_OF_OTHER = 1200.0
PENALTY_OCCUPIED_NOW = 10_000.0
PENALTY_SAME_EDGE_AS_PARKED = 10_000.0

MAX_PARKED_TICKS_SOFT = 14
MAX_PARKED_TICKS_HARD = 45


@dataclass(frozen=True)
class ParkingCandidate:
    """
    Candidato de parking.

    edge:
        Aresta direccional que o robot deve percorrer para estacionar.
        Exemplo: ("N", "L") significa sair de N em direcção a L.

    fraction:
        Fracção ao longo da aresta edge[0] -> edge[1].

    score:
        Quanto menor, melhor.
    """
    edge: tuple[str, str]
    fraction: float
    score: float


def canonical_edge(u: str, v: str) -> tuple[str, str]:
    """Chave canónica para comparar ocupação de segmentos sem direcção."""
    return (u, v) if u <= v else (v, u)


def robot_current_edge(robot: Robot) -> tuple[str, str] | None:
    """Aresta física actualmente ocupada por um robot em movimento."""
    if robot.from_node is None or robot.to_node is None:
        return None

    return canonical_edge(robot.from_node, robot.to_node)


def robot_parked_edge(robot: Robot) -> tuple[str, str] | None:
    """Aresta física ocupada por um robot estacionado."""
    if robot.parked_at is None:
        return None

    u, v, _ = robot.parked_at
    return canonical_edge(u, v)


def robot_reference_node(robot: Robot) -> str | None:
    """
    Nó lógico de referência para previsão.

    - Se está num nó, usa current_node.
    - Se está a mover, usa to_node como aproximação do próximo nó.
    - Se está estacionado, usa o nó de origem do parking.
    """
    if robot.current_node is not None:
        return robot.current_node

    if robot.to_node is not None:
        return robot.to_node

    if robot.from_node is not None:
        return robot.from_node

    if robot.parked_at is not None:
        u, _, _ = robot.parked_at
        return u

    return None


def choose_greedy_next_node(
    graph: FactoryGraph,
    current: str,
    goal: str,
    previous: str | None,
) -> str | None:
    """
    Escolhe o próximo nó por heurística simples.

    Evita voltar imediatamente para trás quando há alternativa.
    """
    candidates: list[tuple[float, str]] = []
    neighbors = graph.neighbors(current)

    for nb in neighbors:
        if previous is not None and nb == previous and len(neighbors) > 1:
            continue

        h = graph.heuristic(nb, goal)
        if h == float("inf"):
            continue

        dist = graph.edge_distance(current, nb)
        candidates.append((dist + h, nb))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def predict_robot_edges(
    graph: FactoryGraph,
    robot: Robot,
    steps: int = LOOKAHEAD_STEPS,
) -> list[tuple[str, str]]:
    """
    Estima as próximas arestas de um robot sem planeamento pesado.

    Retorna arestas canónicas, porque isto serve para detectar ocupação/conflito.
    """
    predicted: list[tuple[str, str]] = []

    current_edge = robot_current_edge(robot)
    if current_edge is not None:
        predicted.append(current_edge)

    goal = robot.goal_node
    if goal is None:
        return predicted[:steps]

    current = robot_reference_node(robot)
    if current is None:
        return predicted[:steps]

    previous = robot.came_from

    if robot.from_node is not None and robot.to_node is not None:
        previous = robot.from_node
        current = robot.to_node

    visited_edges: set[tuple[str, str]] = set(predicted)

    while len(predicted) < steps:
        if current == goal:
            break

        nxt = choose_greedy_next_node(graph, current, goal, previous)
        if nxt is None:
            break

        edge = canonical_edge(current, nxt)

        if edge in visited_edges:
            break

        predicted.append(edge)
        visited_edges.add(edge)

        previous, current = current, nxt

    return predicted[:steps]


def build_future_edge_usage(
    graph: FactoryGraph,
    robots: Iterable[Robot],
    exclude_robot_id: str | None = None,
    steps: int = LOOKAHEAD_STEPS,
) -> dict[tuple[str, str], set[str]]:
    """
    Cria mapa:
        edge -> {robot_id, ...}

    Inclui arestas ocupadas agora e próximas arestas previstas.
    """
    usage: dict[tuple[str, str], set[str]] = {}

    for robot in robots:
        if exclude_robot_id is not None and robot.id == exclude_robot_id:
            continue

        if robot.state in {RobotState.IDLE, RobotState.PARKED} and robot.goal_node is None:
            continue

        for edge in predict_robot_edges(graph, robot, steps):
            usage.setdefault(edge, set()).add(robot.id)

    return usage


def occupied_edges_now(
    robots: Iterable[Robot],
    exclude_robot_id: str | None = None,
) -> set[tuple[str, str]]:
    """Arestas fisicamente ocupadas neste tick."""
    occupied: set[tuple[str, str]] = set()

    for robot in robots:
        if exclude_robot_id is not None and robot.id == exclude_robot_id:
            continue

        moving_edge = robot_current_edge(robot)
        if moving_edge is not None:
            occupied.add(moving_edge)

        parked_edge = robot_parked_edge(robot)
        if parked_edge is not None:
            occupied.add(parked_edge)

    return occupied


def is_process_node(graph: FactoryGraph, node: str) -> bool:
    node_type = graph.node_type(node)
    return isinstance(node_type, str) and node_type.startswith("process")


def is_entry_exit_edge(graph: FactoryGraph, u: str, v: str) -> bool:
    return graph.is_entry(u) or graph.is_entry(v) or graph.is_exit(u) or graph.is_exit(v)


def is_process_edge(graph: FactoryGraph, u: str, v: str) -> bool:
    return is_process_node(graph, u) or is_process_node(graph, v)


def is_junction_corridor(graph: FactoryGraph, u: str, v: str) -> bool:
    return graph.is_junction(u) and graph.is_junction(v)


def is_dead_end_edge(graph: FactoryGraph, u: str, v: str) -> bool:
    return graph.graph.degree(u) == 1 or graph.graph.degree(v) == 1


def edge_is_near_other_goal(
    edge: tuple[str, str],
    robots: Iterable[Robot],
    exclude_robot_id: str,
) -> bool:
    u, v = edge

    for robot in robots:
        if robot.id == exclude_robot_id:
            continue

        if robot.goal_node in {u, v}:
            return True

    return False


def score_parking_edge(
    graph: FactoryGraph,
    directed_edge: tuple[str, str],
    robot: Robot,
    robots: Iterable[Robot],
    future_usage: dict[tuple[str, str], set[str]],
    occupied_now: set[tuple[str, str]],
) -> float:
    """
    Pontua uma aresta para parking.

    Usa directed_edge para manter a direcção real,
    mas compara ocupação com canonical_edge.
    """
    u, v = directed_edge
    canonical = canonical_edge(u, v)

    score = 0.0

    if canonical in occupied_now:
        score += PENALTY_OCCUPIED_NOW

    if canonical in future_usage:
        score += PENALTY_RESERVED_EDGE * len(future_usage[canonical])

    if is_entry_exit_edge(graph, u, v):
        score += PENALTY_ENTRY_EXIT_EDGE

    if is_process_edge(graph, u, v):
        score += PENALTY_PROCESS_EDGE

    if is_junction_corridor(graph, u, v):
        score += PENALTY_JUNCTION_CORRIDOR

    if is_dead_end_edge(graph, u, v):
        score += PENALTY_DEAD_END

    if edge_is_near_other_goal(canonical, robots, robot.id):
        score += PENALTY_NEAR_GOAL_OF_OTHER

    # Evita repetir parking em arestas acabadas de usar.
    # Isto reduz ciclos do tipo sair de parking -> bloquear -> voltar
    # imediatamente para a mesma zona.
    if (u, v) in robot.last_edges:
        score += 1200.0

    if (v, u) in robot.last_edges:
        score += 2200.0

    reference = robot_reference_node(robot)
    if reference is not None:
        dist_to_u = graph.heuristic(reference, u)
        dist_to_v = graph.heuristic(reference, v)
        score += min(dist_to_u, dist_to_v)

    return score


def find_safe_parking_candidate(
    graph: FactoryGraph,
    robot: Robot,
    robots: Iterable[Robot],
) -> ParkingCandidate | None:
    """
    Procura parking seguro numa aresta adjacente ao robot.

    Importante:
    - não procura em qualquer ponto do mapa;
    - só considera arestas que o robot consegue usar a partir do nó actual;
    - evita arestas previstas por outros robots;
    - penaliza entries, exits, process edges e corredores entre junctions,
      mas não as proíbe de forma absoluta.
    """
    if robot.current_node is None:
        return None

    current = robot.current_node

    future_usage = build_future_edge_usage(
        graph=graph,
        robots=robots,
        exclude_robot_id=robot.id,
        steps=LOOKAHEAD_STEPS,
    )

    occupied_now = occupied_edges_now(
        robots=robots,
        exclude_robot_id=robot.id,
    )

    candidates: list[ParkingCandidate] = []

    neighbors = list(graph.neighbors(current))

    # Primeiro tenta não voltar para o nó de onde veio.
    preferred_neighbors = [
        nb for nb in neighbors
        if robot.came_from is None or nb != robot.came_from
    ]

    fallback_neighbors = [
        nb for nb in neighbors
        if robot.came_from is not None and nb == robot.came_from
    ]

    ordered_neighbors = preferred_neighbors + fallback_neighbors

    for nb in ordered_neighbors:
        if not graph.has_edge(current, nb):
            continue

        parking_points = graph.parking_points(current, nb)
        if not parking_points:
            continue

        directed_edge = (current, nb)

        edge_score = score_parking_edge(
            graph=graph,
            directed_edge=directed_edge,
            robot=robot,
            robots=robots,
            future_usage=future_usage,
            occupied_now=occupied_now,
        )

        for fraction in parking_points:
            candidates.append(
                ParkingCandidate(
                    edge=directed_edge,
                    fraction=fraction,
                    score=edge_score,
                )
            )

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: candidate.score)
    best = candidates[0]

    # Se a melhor opção está ocupada ou reservada por outro robot,
    # é melhor esperar do que estacionar e bloquear.
    if best.score >= PENALTY_RESERVED_EDGE:
        return None

    return best


def parked_robot_blocks_someone(
    graph: FactoryGraph,
    parked_robot: Robot,
    robots: Iterable[Robot],
) -> bool:
    """True se a aresta onde o robot está estacionado aparece na previsão de outro robot."""
    parked_edge = robot_parked_edge(parked_robot)
    if parked_edge is None:
        return False

    future_usage = build_future_edge_usage(
        graph=graph,
        robots=robots,
        exclude_robot_id=parked_robot.id,
        steps=LOOKAHEAD_STEPS,
    )

    return parked_edge in future_usage


def parked_robot_should_leave(
    graph: FactoryGraph,
    robot: Robot,
    robots: Iterable[Robot],
) -> bool:
    """
    Decide se um robot PARKED deve sair do parking.

    Critérios:
    - está a bloquear uma rota prevista;
    - já está estacionado há demasiados ticks;
    - o seu próprio caminho aparenta estar livre.
    """
    if robot.state != RobotState.PARKED:
        return False

    if parked_robot_blocks_someone(graph, robot, robots):
        return True

    # Em alguns runners, robots em PARKED continuam a incrementar
    # wait_ticks_in_junction em vez de parked_ticks. Usamos o máximo
    # para evitar que um robot fique estacionado indefinidamente.
    parked_ticks = max(robot.parked_ticks, robot.wait_ticks_in_junction)

    if parked_ticks >= MAX_PARKED_TICKS_HARD:
        return True

    if parked_ticks < MAX_PARKED_TICKS_SOFT:
        return False

    own_edges = predict_robot_edges(graph, robot, LOOKAHEAD_STEPS)

    other_usage = build_future_edge_usage(
        graph=graph,
        robots=robots,
        exclude_robot_id=robot.id,
        steps=LOOKAHEAD_STEPS,
    )

    for edge in own_edges:
        if edge in other_usage:
            return False

    return True


def parking_candidate_to_target(
    candidate: ParkingCandidate,
) -> tuple[str, str, float]:
    """Converte ParkingCandidate para (u, v, fraction)."""
    u, v = candidate.edge
    return u, v, candidate.fraction