from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from env.core.graph import FactoryGraph


DEFAULT_AVOID_NODE_PENALTY = 1000.0
DEFAULT_AVOID_EDGE_PENALTY = 1400.0

REVERSE_EDGE_PENALTY = 4500.0
RECENT_NODE_MULTIPLIER = 1.0
RECENT_EDGE_MULTIPLIER = 1.0

TURN_PENALTY_90 = 80.0
TURN_PENALTY_180 = 500.0

CONGESTED_NODE_PENALTY = 2000.0
CONGESTED_EDGE_PENALTY = 3000.0


@dataclass(order=True)
class _Node:
    f: float
    g: float = field(compare=False)
    node: str = field(compare=False)
    came_from: str | None = field(compare=False)
    parent: "_Node | None" = field(compare=False, default=None)


def plan(
    graph: FactoryGraph,
    src: str,
    dst: str,
    came_from: str | None,
    blocked_nodes: set[str] | None = None,
    blocked_edges: set[tuple[str, str]] | None = None,
    avoid_nodes: set[str] | dict[str, float] | None = None,
    avoid_edges: set[tuple[str, str]] | dict[tuple[str, str], float] | None = None,
    congested_nodes: set[str] | dict[str, float] | None = None,
    congested_edges: set[tuple[str, str]] | dict[tuple[str, str], float] | None = None,
    immediate_reverse_penalty: float = REVERSE_EDGE_PENALTY,
) -> list[str] | None:
    """A* de src para dst com penalizações anti-loop e custos dinâmicos."""
    blocked_nodes = blocked_nodes or set()
    blocked_edges = blocked_edges or set()
    avoid_nodes = avoid_nodes or {}
    avoid_edges = avoid_edges or {}
    congested_nodes = congested_nodes or {}
    congested_edges = congested_edges or {}

    if src == dst:
        return [src]

    open_heap: list[_Node] = []
    best_g: dict[tuple[str, str | None], float] = {}

    root = _Node(
        f=graph.heuristic(src, dst),
        g=0.0,
        node=src,
        came_from=came_from,
    )

    heapq.heappush(open_heap, root)
    best_g[(src, came_from)] = 0.0

    while open_heap:
        current = heapq.heappop(open_heap)

        state_key = (current.node, current.came_from)
        if current.g > best_g.get(state_key, float("inf")):
            continue

        if current.node == dst:
            return _reconstruct(current)

        for neighbor in graph.neighbors(current.node):
            if neighbor in blocked_nodes and neighbor != dst:
                continue

            segment = graph.segment_id(current.node, neighbor)
            if segment in blocked_edges:
                continue

            dist = graph.edge_distance(current.node, neighbor)

            penalty = 0.0

            penalty += _node_penalty(
                node=neighbor,
                dst=dst,
                avoid_nodes=avoid_nodes,
            )

            penalty += _edge_penalty(
                graph=graph,
                u=current.node,
                v=neighbor,
                avoid_edges=avoid_edges,
            )

            penalty += _congestion_node_penalty(
                node=neighbor,
                dst=dst,
                congested_nodes=congested_nodes,
            )

            penalty += _congestion_edge_penalty(
                graph=graph,
                u=current.node,
                v=neighbor,
                congested_edges=congested_edges,
            )

            penalty += _turn_penalty(
                graph=graph,
                came_from=current.came_from,
                at=current.node,
                going_to=neighbor,
            )

            # Evita ping-pong imediato: A -> B -> A
            if (
                current.came_from is not None
                and neighbor == current.came_from
                and neighbor != dst
            ):
                penalty += immediate_reverse_penalty

            g_new = current.g + dist + penalty
            next_key = (neighbor, current.node)

            if g_new >= best_g.get(next_key, float("inf")):
                continue

            best_g[next_key] = g_new

            h_new = graph.heuristic(neighbor, dst)
            if h_new == float("inf"):
                continue

            child = _Node(
                f=g_new + h_new,
                g=g_new,
                node=neighbor,
                came_from=current.node,
                parent=current,
            )

            heapq.heappush(open_heap, child)

    return None


def _node_penalty(
    node: str,
    dst: str,
    avoid_nodes: set[str] | dict[str, float],
) -> float:
    """Penalização para nós recentemente visitados ou pouco desejáveis."""
    if node == dst:
        return 0.0

    if isinstance(avoid_nodes, dict):
        return avoid_nodes.get(node, 0.0) * RECENT_NODE_MULTIPLIER

    if node in avoid_nodes:
        return DEFAULT_AVOID_NODE_PENALTY

    return 0.0


def _edge_penalty(
    graph: FactoryGraph,
    u: str,
    v: str,
    avoid_edges: set[tuple[str, str]] | dict[tuple[str, str], float],
) -> float:
    """Penalização para arestas recentemente usadas."""
    directed = (u, v)
    reverse = (v, u)
    segment = graph.segment_id(u, v)

    if isinstance(avoid_edges, dict):
        penalty = 0.0

        penalty += avoid_edges.get(directed, 0.0)
        penalty += avoid_edges.get(segment, 0.0)

        # Inverter uma aresta recente é quase sempre sinal de loop.
        if reverse in avoid_edges:
            penalty += avoid_edges[reverse] * 1.8

        return penalty * RECENT_EDGE_MULTIPLIER

    penalty = 0.0

    if directed in avoid_edges:
        penalty += DEFAULT_AVOID_EDGE_PENALTY

    if segment in avoid_edges:
        penalty += DEFAULT_AVOID_EDGE_PENALTY

    if reverse in avoid_edges:
        penalty += DEFAULT_AVOID_EDGE_PENALTY * 1.8

    return penalty


def _congestion_node_penalty(
    node: str,
    dst: str,
    congested_nodes: set[str] | dict[str, float],
) -> float:
    """Penalização para nós congestionados."""
    if node == dst:
        return 0.0

    if isinstance(congested_nodes, dict):
        return congested_nodes.get(node, 0.0)

    if node in congested_nodes:
        return CONGESTED_NODE_PENALTY

    return 0.0


def _congestion_edge_penalty(
    graph: FactoryGraph,
    u: str,
    v: str,
    congested_edges: set[tuple[str, str]] | dict[tuple[str, str], float],
) -> float:
    """Penalização para arestas congestionadas, ocupadas ou previstas."""
    directed = (u, v)
    reverse = (v, u)
    segment = graph.segment_id(u, v)

    if isinstance(congested_edges, dict):
        return max(
            congested_edges.get(directed, 0.0),
            congested_edges.get(reverse, 0.0),
            congested_edges.get(segment, 0.0),
        )

    if (
        directed in congested_edges
        or reverse in congested_edges
        or segment in congested_edges
    ):
        return CONGESTED_EDGE_PENALTY

    return 0.0


def _turn_penalty(
    graph: FactoryGraph,
    came_from: str | None,
    at: str,
    going_to: str,
) -> float:
    """Penaliza curvas fortes."""
    if came_from is None:
        return 0.0

    if came_from == at:
        return 0.0

    if came_from == going_to:
        return TURN_PENALTY_180

    try:
        angle = graph.turn_angle(came_from, at, going_to)
    except (ValueError, KeyError):
        return 0.0

    if angle >= 150.0:
        return TURN_PENALTY_180

    if angle >= 45.0:
        return TURN_PENALTY_90

    return 0.0


def _reconstruct(node: _Node) -> list[str]:
    path: list[str] = []
    current: _Node | None = node

    while current is not None:
        path.append(current.node)
        current = current.parent

    path.reverse()
    return path