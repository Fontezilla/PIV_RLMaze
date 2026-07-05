"""Grafo da fábrica: nós, arestas, distâncias e sub-nós de junction.

Os sub-nós substituem o antigo mecanismo de parking: cada junction ganha um
sub-nó a JUNCTION_SUBNODE_DIST de distância em cada uma das suas arestas
originais (tipo "junction", nós normais do grafo). Uma aresta entre duas
junctions fica com um sub-nó de cada lado (3 troços); uma aresta
junction→outro tipo fica só com o sub-nó do lado da junction (2 troços).
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import networkx as nx
import yaml


JUNCTION_SUBNODE_DIST = 50.0


class FactoryGraph:
    """Grafo da fábrica com sub-nós de junction gerados automaticamente e
    distâncias/features pré-computadas carregadas de uma cache."""

    SPECIAL_TYPES: frozenset = frozenset({
        "entry", "exit",
        "processA_entry", "processA_exit",
        "processB_entry", "processB_exit",
    })

    def __init__(self, yaml_path: str, cache_path: str | None = None):
        """Carrega o mapa do YAML, gera os sub-nós e (se dado) a cache de
        distâncias. `cache_path=None` constrói o grafo sem cache (usado pelo
        precompute que vai calcular a própria cache)."""
        self.graph = nx.Graph()
        self.shortest_distances: dict = {}
        self.betweenness        : dict[str, float] = {}
        self.degree_cache       : dict[str, int]   = {}
        self.dist_to_type       : dict[str, dict[str, float]] = {}
        self._reachable_cache   : dict[str, frozenset[str]] = {}
        self._subnodes          : set[str] = set()

        self._load_from_yaml(yaml_path)
        self._generate_junction_subnodes()

        if cache_path is None:
            pass
        elif Path(cache_path).exists():
            self._load_cache(cache_path)
        else:
            raise FileNotFoundError(
                f"Cache não encontrado em '{cache_path}' "
                f"— corre scripts/precompute_graph.py primeiro"
            )

    def _load_from_yaml(self, path: str) -> None:
        """Lê nós (tipo + coordenadas) e arestas (distância) do ficheiro YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)
        for node_id, info in data["nodes"].items():
            x, y = info.get("coords", [0.0, 0.0])
            self.graph.add_node(node_id, type=info.get("type"), x=float(x), y=float(y))
        for edge in data["edges"]:
            self.graph.add_edge(edge["from"], edge["to"], distance=float(edge["distance"]))

    def _generate_junction_subnodes(self) -> None:
        """Insere sub-nós em todas as arestas incidentes a junctions,
        dividindo cada aresta nos troços correspondentes."""
        for u, v, data in list(self.graph.edges(data=True)):
            dist = data["distance"]
            u_junction = self.graph.nodes[u]["type"] == "junction"
            v_junction = self.graph.nodes[v]["type"] == "junction"
            if not u_junction and not v_junction:
                continue
            self.graph.remove_edge(u, v)

            if u_junction and v_junction:
                self._require_min_length(u, v, dist, 2 * JUNCTION_SUBNODE_DIST)
                sub_u = self._add_subnode(u, v)
                sub_v = self._add_subnode(v, u)
                self.graph.add_edge(u, sub_u, distance=JUNCTION_SUBNODE_DIST)
                self.graph.add_edge(sub_u, sub_v, distance=dist - 2 * JUNCTION_SUBNODE_DIST)
                self.graph.add_edge(sub_v, v, distance=JUNCTION_SUBNODE_DIST)
            elif u_junction:
                self._require_min_length(u, v, dist, JUNCTION_SUBNODE_DIST)
                sub_u = self._add_subnode(u, v)
                self.graph.add_edge(u, sub_u, distance=JUNCTION_SUBNODE_DIST)
                self.graph.add_edge(sub_u, v, distance=dist - JUNCTION_SUBNODE_DIST)
            else:
                self._require_min_length(u, v, dist, JUNCTION_SUBNODE_DIST)
                sub_v = self._add_subnode(v, u)
                self.graph.add_edge(u, sub_v, distance=dist - JUNCTION_SUBNODE_DIST)
                self.graph.add_edge(sub_v, v, distance=JUNCTION_SUBNODE_DIST)

    def _add_subnode(self, junction: str, other: str) -> str:
        """Cria e devolve o sub-nó de `junction` na direcção de `other`, a
        JUNCTION_SUBNODE_DIST de distância."""
        sub_id = f"{junction}__sub__{other}"
        jx, jy = self.node_position(junction)
        ox, oy = self.node_position(other)
        total = math.hypot(ox - jx, oy - jy)
        frac = 0.0 if total <= 0 else JUNCTION_SUBNODE_DIST / total
        self.graph.add_node(sub_id, type="junction",
                            x=jx + (ox - jx) * frac, y=jy + (oy - jy) * frac)
        self._subnodes.add(sub_id)
        return sub_id

    def _require_min_length(self, u: str, v: str, dist: float, minimum: float) -> None:
        """Rebenta se a aresta for demasiado curta para caberem os sub-nós."""
        if dist < minimum:
            raise ValueError(
                f"Aresta {u}-{v} (dist={dist}) demasiado curta para sub-nós "
                f"de junction (mínimo {minimum})."
            )

    def _load_cache(self, path: str) -> None:
        """Carrega distâncias par-a-par, betweenness, grau e distância-a-tipo
        de um pickle, e deriva o cache de alcançabilidade."""
        with open(path, "rb") as f:
            cache = pickle.load(f)
        self.shortest_distances = cache["shortest_distances"]
        self.betweenness        = cache.get("betweenness", {})
        self.degree_cache       = cache.get("degree", {})
        self.dist_to_type       = cache.get("dist_to_type", {})
        self._reachable_cache = {
            src: frozenset(dst for dst, d in dsts.items() if d < float("inf"))
            for src, dsts in self.shortest_distances.items()
        }

    def node_position(self, node_id: str) -> tuple[float, float]:
        """Coordenadas (x, y) do nó."""
        n = self.graph.nodes[node_id]
        return n["x"], n["y"]

    def node_type(self, node_id: str) -> str:
        """Tipo do nó (entry/exit/process.../junction)."""
        return self.graph.nodes[node_id]["type"]

    def is_junction(self, node_id: str) -> bool:
        """True se o tipo é 'junction' (inclui sub-nós, que têm este tipo)."""
        return self.node_type(node_id) == "junction"

    def is_subnode(self, node_id: str) -> bool:
        """True se é um sub-nó de junction gerado automaticamente."""
        return node_id in self._subnodes

    def is_turn_node(self, node_id: str) -> bool:
        """True se o robot pode RODAR aqui — uma junction real do mapa (não um
        sub-nó). Não usar o grau: há junctions reais de grau 2 (cantos) onde
        se roda, e sub-nós de grau 2 onde não se roda."""
        return self.is_junction(node_id) and not self.is_subnode(node_id)

    def is_special(self, node_id: str) -> bool:
        """True se o nó é entry/exit/process (estação, não corredor)."""
        return self.node_type(node_id) in self.SPECIAL_TYPES

    def is_entry(self, node_id: str) -> bool:
        """True se o nó é um entry."""
        return self.node_type(node_id) == "entry"

    def is_exit(self, node_id: str) -> bool:
        """True se o nó é um exit."""
        return self.node_type(node_id) == "exit"

    def is_process_node(self, node_id: str) -> bool:
        """True se o nó é uma estação de processo (processA*/processB*)."""
        return self.node_type(node_id).startswith("process")

    def is_special_edge(self, u: str, v: str) -> bool:
        """True se qualquer extremo da aresta é entry/exit/process."""
        return self.is_special(u) or self.is_special(v)

    def all_nodes(self) -> list[str]:
        """Todos os nós do grafo (incluindo sub-nós)."""
        return list(self.graph.nodes)

    def junction_nodes(self) -> list[str]:
        """Todos os nós de tipo junction (inclui sub-nós)."""
        return [n for n in self.graph.nodes if self.is_junction(n)]

    def neighbors(self, node_id: str) -> list[str]:
        """Vizinhos directos do nó."""
        return list(self.graph.neighbors(node_id))

    def has_edge(self, u: str, v: str) -> bool:
        """True se existe aresta entre u e v."""
        return self.graph.has_edge(u, v)

    def edge_distance(self, u: str, v: str) -> float:
        """Comprimento da aresta u-v."""
        return self.graph[u][v]["distance"]

    def edge_angle(self, u: str, v: str, w: str) -> float:
        """Ângulo (graus) entre as direcções u→v e v→w."""
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
        mag1 = math.hypot(dx1, dy1)
        mag2 = math.hypot(dx2, dy2)
        if mag1 < 1e-9 or mag2 < 1e-9:
            return 0.0
        cos_a = max(-1.0, min(1.0, (dx1 * dx2 + dy1 * dy2) / (mag1 * mag2)))
        return math.degrees(math.acos(cos_a))

    def turn_angle(self, came_from: str | None, at: str, going_to: str) -> float:
        """Ângulo do turn em `at` vindo de `came_from` para `going_to`
        (0 se não há nó anterior)."""
        if came_from is None:
            return 0.0
        return self.edge_angle(came_from, at, going_to)

    def segment_id(self, u: str, v: str) -> tuple[str, str]:
        """Identificador canónico (não-direccional) de uma aresta."""
        return min(u, v), max(u, v)

    def heuristic(self, node: str, goal: str) -> float:
        """Distância real pré-computada node→goal (heurística admissível)."""
        return self.shortest_distances.get(node, {}).get(goal, float("inf"))

    def shortest_distance(self, src: str, dst: str) -> float:
        """Distância real pré-computada entre src e dst."""
        return self.shortest_distances.get(src, {}).get(dst, float("inf"))

    def reachable_from(self, node: str) -> frozenset[str]:
        """Conjunto de nós alcançáveis a partir de `node`."""
        return self._reachable_cache.get(node, frozenset())

    def __repr__(self) -> str:
        """Resumo com contagens de nós, arestas e sub-nós."""
        return (f"FactoryGraph(nodes={self.graph.number_of_nodes()}, "
                f"edges={self.graph.number_of_edges()}, subnodes={len(self._subnodes)})")
