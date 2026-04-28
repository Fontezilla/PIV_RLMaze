import networkx as nx
import yaml


class FactoryGraph:
    """
    Grafo da fábrica.

    - Bidirecional (nx.Graph)
    - Pré-computação de shortest paths, next hops e k-shortest paths
    """

    def __init__(self, yaml_path: str):
        self.graph = nx.Graph()

        self.all_shortest_paths = {}
        self.next_hops = {}

        self._load_from_yaml(yaml_path)
        self._precompute_shortest_paths()
        self._precompute_k_shortest_paths()

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    def _load_from_yaml(self, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # ------------------------
        # Nodes
        # ------------------------

        for node_id, node_data in data["nodes"].items():
            coords = node_data.get("coords", [0.0, 0.0])
            self.graph.add_node(
                node_id,
                type=node_data.get("type"),
                x=float(coords[0]),
                y=float(coords[1]),
            )

        # ------------------------
        # Edges (bidirecionais)
        # ------------------------

        for edge in data["edges"]:
            u = edge["from"]
            v = edge["to"]
            d = edge["distance"]

            self.graph.add_edge(u, v, distance=d)

    # ------------------------------------------------------------------
    # PRECOMPUTE SHORTEST PATHS
    # ------------------------------------------------------------------

    def _precompute_shortest_paths(self):
        nodes = list(self.graph.nodes)

        self.shortest_distances: dict = {}

        for start in nodes:
            self.all_shortest_paths[start] = {}
            self.next_hops[start] = {}
            self.shortest_distances[start] = {}

            for goal in nodes:

                if start == goal:
                    self.all_shortest_paths[start][goal] = [[start]]
                    self.next_hops[start][goal] = set()
                    self.shortest_distances[start][goal] = 0.0
                    continue

                try:
                    paths = list(nx.all_shortest_paths(
                        self.graph,
                        source=start,
                        target=goal,
                        weight="distance"
                    ))

                    self.all_shortest_paths[start][goal] = paths

                    # extrair next hops
                    hops = set()
                    for path in paths:
                        if len(path) > 1:
                            hops.add(path[1])

                    self.next_hops[start][goal] = hops

                    # distância pré-computada (soma dos pesos da primeira path)
                    p = paths[0]
                    self.shortest_distances[start][goal] = sum(
                        self.graph[p[i]][p[i + 1]]["distance"]
                        for i in range(len(p) - 1)
                    )

                except nx.NetworkXNoPath:
                    self.all_shortest_paths[start][goal] = []
                    self.next_hops[start][goal] = set()
                    self.shortest_distances[start][goal] = float("inf")

    # ------------------------------------------------------------------
    # K-SHORTEST PATHS (Yen's algorithm via NetworkX)
    # ------------------------------------------------------------------

    K_PATHS = 6

    def _precompute_k_shortest_paths(self):
        """
        Para cada par (start, goal) guarda até K_PATHS caminhos simples
        ordenados por distância estática crescente.
        Armazena (path, static_length) para evitar recálculo em runtime.
        """
        self.k_shortest_paths: dict = {}
        for start in self.graph.nodes:
            self.k_shortest_paths[start] = {}
            for goal in self.graph.nodes:
                if start == goal:
                    self.k_shortest_paths[start][goal] = [([start], 0.0)]
                    continue
                entries = []
                try:
                    gen = nx.shortest_simple_paths(
                        self.graph, start, goal, weight="distance"
                    )
                    for path in gen:
                        length = sum(
                            self.graph[path[i]][path[i + 1]]["distance"]
                            for i in range(len(path) - 1)
                        )
                        entries.append((path, length))
                        if len(entries) >= self.K_PATHS:
                            break
                except nx.NetworkXNoPath:
                    pass
                self.k_shortest_paths[start][goal] = entries

    # ------------------------------------------------------------------
    # BASIC QUERIES
    # ------------------------------------------------------------------

    def neighbors(self, node_id: str):
        return list(self.graph.neighbors(node_id))

    def distance(self, u: str, v: str) -> float:
        return self.graph[u][v]["distance"]

    # ------------------------------------------------------------------
    # SHORTEST PATH ACCESS
    # ------------------------------------------------------------------

    def get_all_shortest_paths(self, start: str, goal: str):
        return self.all_shortest_paths[start][goal]

    def get_next_hops(self, start: str, goal: str):
        """
        Retorna conjunto de nós possíveis para o próximo passo
        em caminhos mínimos.
        """
        return self.next_hops[start][goal]

    def shortest_path(self, start: str, goal: str):
        """
        Retorna um caminho (qualquer) — útil para debug.
        """
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
    # NODE INFO
    # ------------------------------------------------------------------

    def node_type(self, node_id: str) -> str:
        return self.graph.nodes[node_id]["type"]

    def node_position(self, node_id: str):
        node = self.graph.nodes[node_id]
        return node["x"], node["y"]

    # ------------------------------------------------------------------
    # NODE CLASSIFICATION
    # ------------------------------------------------------------------

    # Tipos de nós com zona de docking (Regra 1)
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
        """Nó de junção (cruzamento / curva) — Regras 2 e 3."""
        return self.node_type(node_id) == "junction"

    def is_docking_edge(self, u: str, v: str) -> bool:
        """Aresta adjacente a pelo menos um nó especial."""
        return self.is_special(u) or self.is_special(v)

    # ------------------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------------------

    def __str__(self):
        return f"FactoryGraph(nodes={len(self.graph.nodes)}, edges={len(self.graph.edges)})"