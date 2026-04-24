import math
import pickle
from pathlib import Path

import networkx as nx
import yaml


class FactoryGraph:
    """
    Grafo da fábrica.

    - Bidirecional (nx.Graph)
    - Pré-computação de shortest paths, next hops e k-shortest paths
    - Utilitários de ângulo entre arestas para o rules_system
    - Utilitários de adjacência para o action_masks

    Cache:
        Passa cache_path para carregar caminhos pré-computados em vez
        de os calcular em runtime. Gerar o cache com precompute_graph.py.
    """

    def __init__(self, yaml_path: str = "../.configs/map_factory.yaml", cache_path: str = "../.configs/graph_cache.pkl"):
        self.graph = nx.Graph()

        self.all_shortest_paths: dict = {}
        self.next_hops: dict = {}
        self.shortest_distances: dict = {}
        self.k_shortest_paths: dict = {}

        self._load_from_yaml(yaml_path)

        if cache_path and Path(cache_path).exists():
            self._load_cache(cache_path)
        else:
            raise FileNotFoundError(
                f"FactoryGraph: cache não encontrado em '{cache_path}' — corre precompute_graph.py primeiro"
    )

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    def _load_from_yaml(self, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        for node_id, node_data in data["nodes"].items():
            coords = node_data.get("coords", [0.0, 0.0])
            self.graph.add_node(
                node_id,
                type=node_data.get("type"),
                x=float(coords[0]),
                y=float(coords[1]),
            )

        for edge in data["edges"]:
            u = edge["from"]
            v = edge["to"]
            d = edge["distance"]
            self.graph.add_edge(u, v, distance=d)


    # ------------------------------------------------------------------
    # CACHE
    # ------------------------------------------------------------------

    def _load_cache(self, path: str):
        with open(path, "rb") as f:
            cache = pickle.load(f)
        self.shortest_distances = cache["shortest_distances"]
        self.all_shortest_paths = cache["all_shortest_paths"]
        self.next_hops          = cache["next_hops"]
        self.k_shortest_paths   = cache["k_shortest_paths"]

    def save_cache(self, path: str):
        cache = {
            "shortest_distances": self.shortest_distances,
            "all_shortest_paths": self.all_shortest_paths,
            "next_hops":          self.next_hops,
            "k_shortest_paths":   self.k_shortest_paths,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(cache, f)

    # ------------------------------------------------------------------
    # BASIC QUERIES
    # ------------------------------------------------------------------

    def neighbors(self, node_id: str) -> list[str]:
        return list(self.graph.neighbors(node_id))

    def distance(self, u: str, v: str) -> float:
        return self.graph[u][v]["distance"]

    def has_edge(self, u: str, v: str) -> bool:
        return self.graph.has_edge(u, v)

    # ------------------------------------------------------------------
    # ADJACENCY — usado pelo action_masks
    # ------------------------------------------------------------------

    def get_adjacents_of(self, node_id: str, exclude: str = None) -> list[str]:
        """
        Vizinhos de node_id excluindo opcionalmente um nó.

        Usado pelo action_masks para calcular as opções válidas
        de next_next_node dado um next_node escolhido.

        Exemplo:
            robot está em A, escolhe next_node=B
            get_adjacents_of("B", exclude="A") → opções para next_next_node
        """
        return [
            n for n in self.graph.neighbors(node_id)
            if n != exclude
        ]

    # ------------------------------------------------------------------
    # SHORTEST PATH ACCESS
    # ------------------------------------------------------------------

    def get_all_shortest_paths(self, start: str, goal: str) -> list:
        return self.all_shortest_paths[start][goal]

    def get_next_hops(self, start: str, goal: str) -> set:
        """
        Conjunto de nós possíveis para o próximo passo em caminhos mínimos.
        Usado pela reward para calcular delta_dist_goal.
        """
        return self.next_hops[start][goal]

    def shortest_path(self, start: str, goal: str) -> list:
        """Retorna um caminho — útil para debug e routing_utils."""
        return nx.shortest_path(
            self.graph,
            source=start,
            target=goal,
            weight="distance",
        )

    def shortest_path_length(self, start: str, goal: str) -> float:
        try:
            return self.shortest_distances[start][goal]
        except KeyError:
            return float("inf")

    # ------------------------------------------------------------------
    # ANGLE — usado pelo rules_system para validar velocidade em curvas
    # ------------------------------------------------------------------

    def edge_angle(self, u: str, v: str, w: str) -> float:
        """
        Ângulo em graus entre aresta u→v e v→w.

        Retorna 0.0 se é reta, 90.0 se é curva a 90°, etc.
        Usado pelo rules_system para determinar a velocidade máxima
        permitida na aproximação ao nó v vindo de u em direção a w.

        Raises ValueError se u==w (reversão) ou se as arestas não existem.
        """
        if u == w:
            raise ValueError(f"edge_angle: u e w são iguais ({u}) — reversão não é curva")
        if not self.has_edge(u, v):
            raise ValueError(f"edge_angle: aresta {u}→{v} não existe")
        if not self.has_edge(v, w):
            raise ValueError(f"edge_angle: aresta {v}→{w} não existe")

        ux, uy = self.node_position(u)
        vx, vy = self.node_position(v)
        wx, wy = self.node_position(w)

        # vetor de entrada u→v
        dx1 = vx - ux
        dy1 = vy - uy

        # vetor de saída v→w
        dx2 = wx - vx
        dy2 = wy - vy

        # ângulo entre os dois vetores
        dot   = dx1 * dx2 + dy1 * dy2
        mag1  = math.hypot(dx1, dy1)
        mag2  = math.hypot(dx2, dy2)

        if mag1 < 1e-9 or mag2 < 1e-9:
            return 0.0

        cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        angle = math.degrees(math.acos(cos_a))

        # desvio em relação à reta — 0° = reta, 90° = curva a 90°
        return abs(180.0 - angle)

    def is_straight(self, u: str, v: str, w: str, threshold_deg: float = 10.0) -> bool:
        """
        True se a transição u→v→w é praticamente reta.
        Threshold em graus — abaixo disto considera-se reta.
        """
        return self.edge_angle(u, v, w) < threshold_deg

    # ------------------------------------------------------------------
    # NODE INFO
    # ------------------------------------------------------------------

    def node_type(self, node_id: str) -> str:
        return self.graph.nodes[node_id]["type"]

    def node_position(self, node_id: str) -> tuple[float, float]:
        node = self.graph.nodes[node_id]
        return node["x"], node["y"]

    # ------------------------------------------------------------------
    # NODE CLASSIFICATION
    # ------------------------------------------------------------------

    SPECIAL_TYPES: frozenset = frozenset({
        "entry",
        "exit",
        "processA_entry",
        "processA_exit",
        "processB_entry",
        "processB_exit",
    })

    def is_special(self, node_id: str) -> bool:
        """Nó de docking (entry / exit / process)."""
        return self.node_type(node_id) in self.SPECIAL_TYPES

    def is_junction(self, node_id: str) -> bool:
        """Nó de junção (cruzamento / curva)."""
        return self.node_type(node_id) == "junction"

    def is_docking_edge(self, u: str, v: str) -> bool:
        """Aresta adjacente a pelo menos um nó especial."""
        return self.is_special(u) or self.is_special(v)

    # ------------------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"FactoryGraph(nodes={len(self.graph.nodes)}, edges={len(self.graph.edges)})"