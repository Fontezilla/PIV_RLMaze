"""Grafo da fábrica: nós, arestas, distâncias e parking points geométricos."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import networkx as nx
import yaml


PARKING_MIN_DIST         = 100.0
PARKING_SPECIAL_MIN_DIST = 40.0


class FactoryGraph:
    """Grafo sem subnós, com parking points geométricos nas arestas."""

    SPECIAL_TYPES: frozenset = frozenset({
        "entry", "exit",
        "processA_entry", "processA_exit",
        "processB_entry", "processB_exit",
    })

    def __init__(self, yaml_path: str, cache_path: str | None = None):
        self.graph = nx.Graph()
        self.shortest_distances: dict = {}
        self.betweenness        : dict[str, float] = {}
        self.degree_cache       : dict[str, int]   = {}
        self.dist_to_type       : dict[str, dict[str, float]] = {}
        self._reachable_cache   : dict[str, frozenset[str]] = {}

        self._load_from_yaml(yaml_path)

        if cache_path and Path(cache_path).exists():
            self._load_cache(cache_path)
        else:
            raise FileNotFoundError(
                f"Cache não encontrado em '{cache_path}' "
                f"— corre scripts/precompute_graph.py primeiro"
            )

    def _load_from_yaml(self, path: str) -> None:
        with open(path) as f:
            data = yaml.safe_load(f)

        for node_id, info in data["nodes"].items():
            x, y = info.get("coords", [0.0, 0.0])
            self.graph.add_node(
                node_id,
                type=info.get("type"),
                x=float(x),
                y=float(y),
            )

        for edge in data["edges"]:
            self.graph.add_edge(
                edge["from"],
                edge["to"],
                distance=float(edge["distance"]),
            )

    def _load_cache(self, path: str) -> None:
        with open(path, "rb") as f:
            cache = pickle.load(f)
        self.shortest_distances = cache["shortest_distances"]
        self.betweenness        = cache.get("betweenness", {})
        self.degree_cache       = cache.get("degree", {})
        self.dist_to_type       = cache.get("dist_to_type", {})
        self._reachable_cache = {
            src: frozenset(
                dst for dst, d in dsts.items() if d < float("inf")
            )
            for src, dsts in self.shortest_distances.items()
        }

    def node_position(self, node_id: str) -> tuple[float, float]:
        n = self.graph.nodes[node_id]
        return n["x"], n["y"]

    def node_type(self, node_id: str) -> str:
        return self.graph.nodes[node_id]["type"]

    def is_junction(self, node_id: str) -> bool:
        return self.node_type(node_id) == "junction"

    def is_special(self, node_id: str) -> bool:
        return self.node_type(node_id) in self.SPECIAL_TYPES

    def is_entry(self, node_id: str) -> bool:
        return self.node_type(node_id) == "entry"

    def is_exit(self, node_id: str) -> bool:
        return self.node_type(node_id) == "exit"

    def is_process_node(self, node_id: str) -> bool:
        return self.node_type(node_id).startswith("process")

    def is_special_edge(self, u: str, v: str) -> bool:
        """True se u ou v é entry/exit/process."""
        return self.is_special(u) or self.is_special(v)

    def is_evasion_node(self, node_id: str) -> bool:
        """Nó válido para evasão temporária."""
        return self.is_special(node_id)

    def all_nodes(self) -> list[str]:
        return list(self.graph.nodes)

    def junction_nodes(self) -> list[str]:
        return [n for n in self.graph.nodes if self.is_junction(n)]

    def neighbors(self, node_id: str) -> list[str]:
        return list(self.graph.neighbors(node_id))

    def has_edge(self, u: str, v: str) -> bool:
        return self.graph.has_edge(u, v)

    def edge_distance(self, u: str, v: str) -> float:
        return self.graph[u][v]["distance"]

    def edge_angle(self, u: str, v: str, w: str) -> float:
        """Ângulo (graus) entre as arestas u→v e v→w."""
        if u == w:
            raise ValueError(f"edge_angle: u e w iguais ({u})")
        if not self.has_edge(u, v):
            raise ValueError(f"edge_angle: aresta {u}→{v} não existe")
        if not self.has_edge(v, w):
            raise ValueError(f"edge_angle: aresta {v}→{w} não existe")

        ux, uy = self.node_position(u)
        vx, vy = self.node_position(v)
        wx, wy = self.node_position(w)

        dx1, dy1 = vx - ux, vy - uy
        dx2, dy2 = wx - vx, wy - vy

        dot = dx1 * dx2 + dy1 * dy2
        mag1 = math.hypot(dx1, dy1)
        mag2 = math.hypot(dx2, dy2)

        if mag1 < 1e-9 or mag2 < 1e-9:
            return 0.0

        cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_a))

    def turn_angle(self, came_from: str | None, at: str, going_to: str) -> float:
        """Ângulo do turn em `at` vindo de `came_from` para `going_to`."""
        if came_from is None:
            return 0.0
        return self.edge_angle(came_from, at, going_to)

    def segment_id(self, u: str, v: str) -> tuple[str, str]:
        """Identificador canónico de aresta (ordenado lexicograficamente)."""
        return min(u, v), max(u, v)

    def heuristic(self, node: str, goal: str) -> float:
        """Distância real pré-computada (usada como heurística no A*)."""
        return self.shortest_distances.get(node, {}).get(goal, float("inf"))

    def shortest_distance(self, src: str, dst: str) -> float:
        """Distância real pré-computada entre src e dst."""
        return self.shortest_distances.get(src, {}).get(dst, float("inf"))

    def reachable_from(self, node: str) -> frozenset[str]:
        """Conjunto de nós alcançáveis a partir de `node`."""
        return self._reachable_cache.get(node, frozenset())

    def parking_points(self, u: str, v: str) -> list[float]:
        """Fracções onde existem parking points na aresta u-v."""
        dist = self.edge_distance(u, v)

        if self.is_special_edge(u, v):
            if dist < PARKING_SPECIAL_MIN_DIST:
                return []
            return [0.25]

        if dist < PARKING_MIN_DIST:
            return []

        if dist <= 300.0:
            return [0.5]

        return [0.333, 0.667]

    def parking_point_position(self, u: str, v: str, fraction: float) -> tuple[float, float]:
        """Posição geográfica de um parking point na aresta u→v."""
        ux, uy = self.node_position(u)
        vx, vy = self.node_position(v)
        return (
            ux + (vx - ux) * fraction,
            uy + (vy - uy) * fraction,
        )

    def __repr__(self) -> str:
        return (
            f"FactoryGraph("
            f"nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )
