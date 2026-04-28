"""
blackboard_builder.py — Vista global do mundo para GNN.

GlobalGraph contém features normalizadas por nó e por aresta,
prontas para alimentar uma Graph Neural Network durante treino RL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from simulation_engine.core.entities import RobotState
from simulation_engine.core.world_state import WorldState
from simulation_engine.core.graph import FactoryGraph


# ---------------------------------------------------------------------------
# Dataclasses de features
# ---------------------------------------------------------------------------

@dataclass
class NodeFeatures:
    """Features de um nó (input de nó para GNN)."""
    node_id:          str
    node_type:        str
    x_norm:           float   # coordenada x normalizada [0, 1]
    y_norm:           float   # coordenada y normalizada [0, 1]
    robots_idle:      int     # robots IDLE neste nó
    robots_arriving:  int     # robots cujo to_node é este nó
    robots_departing: int     # robots cujo from_node é este nó
    boxes_available:  int     # boxes AT_NODE neste nó
    process_busy:     bool    # processo em curso (apenas nós de processo)


@dataclass
class EdgeFeatures:
    """Features de uma aresta direcional (input de aresta para GNN)."""
    from_node:     str
    to_node:       str
    distance_norm: float   # distância normalizada [0, 1]
    robots_fwd:    int     # robots a mover-se from_node → to_node
    robots_bwd:    int     # robots a mover-se to_node → from_node (oposto)


@dataclass
class GlobalGraph:
    """
    Grafo global do ambiente num dado tick.

    nodes — dict node_id → NodeFeatures
    edges — lista de EdgeFeatures (uma entrada por direção)
    tick  — instante temporal
    """
    nodes: Dict[str, NodeFeatures] = field(default_factory=dict)
    edges: List[EdgeFeatures]      = field(default_factory=list)
    tick:  int                     = 0


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_global_graph(world: WorldState, graph: FactoryGraph) -> GlobalGraph:
    """
    Constrói GlobalGraph a partir do estado global atual.

    Normaliza coordenadas pelo bounding-box do mapa
    e distâncias pela aresta mais longa.
    """

    # ------------------------------------------------------------------
    # Limites para normalização
    # ------------------------------------------------------------------
    all_coords = [graph.node_position(n) for n in graph.graph.nodes]
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    x_min, x_range = min(xs), max(xs) - min(xs) or 1.0
    y_min, y_range = min(ys), max(ys) - min(ys) or 1.0

    all_dists  = [graph.graph[u][v]["distance"] for u, v in graph.graph.edges()]
    max_dist   = max(all_dists) or 1.0

    # ------------------------------------------------------------------
    # Node features
    # ------------------------------------------------------------------
    nodes: Dict[str, NodeFeatures] = {}

    for node_id in graph.graph.nodes:
        x, y  = graph.node_position(node_id)
        ntype = graph.node_type(node_id)

        robots_idle = sum(
            1 for r in world.robots.values()
            if r.current_node == node_id
        )
        robots_arriving = sum(
            1 for r in world.robots.values()
            if r.to_node == node_id
        )
        robots_departing = sum(
            1 for r in world.robots.values()
            if r.from_node == node_id and r.state == RobotState.MOVING
        )

        boxes_available = len(world.boxes_at_node(node_id))

        proc       = world.processes.get(node_id)
        proc_busy  = proc.busy if proc is not None else False

        nodes[node_id] = NodeFeatures(
            node_id          = node_id,
            node_type        = ntype,
            x_norm           = (x - x_min) / x_range,
            y_norm           = (y - y_min) / y_range,
            robots_idle      = robots_idle,
            robots_arriving  = robots_arriving,
            robots_departing = robots_departing,
            boxes_available  = boxes_available,
            process_busy     = proc_busy,
        )

    # ------------------------------------------------------------------
    # Edge features (uma por direção)
    # ------------------------------------------------------------------
    edges: List[EdgeFeatures] = []

    for u, v in graph.graph.edges():
        dist_norm = graph.graph[u][v]["distance"] / max_dist
        fwd = len(world.robots_on_edge(u, v))
        bwd = len(world.robots_on_edge(v, u))

        edges.append(EdgeFeatures(u, v, dist_norm, fwd, bwd))
        edges.append(EdgeFeatures(v, u, dist_norm, bwd, fwd))

    return GlobalGraph(nodes=nodes, edges=edges, tick=world.tick)
