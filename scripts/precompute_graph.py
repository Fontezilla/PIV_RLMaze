"""
precompute_graph.py

Script de pré-computação do grafo da fábrica.

Corre UMA VEZ antes do treino para gerar o cache dos caminhos mínimos.
O cache é carregado pelo FactoryGraph em runtime, evitando recalcular
shortest paths a cada início de treino.

Uso:
    python precompute_graph.py

Output:
    .configs/graph_cache.pkl
"""

import pickle
import time
from pathlib import Path

import networkx as nx
import yaml


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

YAML_PATH  = Path("../.configs/map_factory.yaml")
CACHE_PATH = Path("../.configs/graph_cache.pkl")

K_PATHS = 6


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------

def load_graph(yaml_path: Path) -> nx.Graph:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    G = nx.Graph()

    for node_id, node_data in data["nodes"].items():
        coords = node_data.get("coords", [0.0, 0.0])
        G.add_node(
            node_id,
            type=node_data.get("type"),
            x=float(coords[0]),
            y=float(coords[1]),
        )

    for edge in data["edges"]:
        G.add_edge(edge["from"], edge["to"], distance=edge["distance"])

    return G


# ------------------------------------------------------------------
# Precompute
# ------------------------------------------------------------------

def precompute(G: nx.Graph) -> dict:
    nodes = list(G.nodes)
    n = len(nodes)

    shortest_distances  = {}
    all_shortest_paths  = {}
    next_hops           = {}
    k_shortest_paths    = {}

    print(f"Grafo: {n} nós, {len(G.edges)} arestas")
    print(f"Pares a calcular: {n * n}")
    print()

    for i, start in enumerate(nodes):
        shortest_distances[start]  = {}
        all_shortest_paths[start]  = {}
        next_hops[start]           = {}
        k_shortest_paths[start]    = {}

        for goal in nodes:

            # -------------------------------------------------------
            # Mesmo nó
            # -------------------------------------------------------
            if start == goal:
                all_shortest_paths[start][goal] = [[start]]
                next_hops[start][goal]          = set()
                shortest_distances[start][goal] = 0.0
                k_shortest_paths[start][goal]   = [([start], 0.0)]
                continue

            # -------------------------------------------------------
            # Shortest paths
            # -------------------------------------------------------
            try:
                paths = list(nx.all_shortest_paths(
                    G, source=start, target=goal, weight="distance"
                ))

                all_shortest_paths[start][goal] = paths

                hops = set()
                for path in paths:
                    if len(path) > 1:
                        hops.add(path[1])
                next_hops[start][goal] = hops

                p = paths[0]
                shortest_distances[start][goal] = sum(
                    G[p[i]][p[i + 1]]["distance"]
                    for i in range(len(p) - 1)
                )

            except nx.NetworkXNoPath:
                all_shortest_paths[start][goal] = []
                next_hops[start][goal]          = set()
                shortest_distances[start][goal] = float("inf")

            # -------------------------------------------------------
            # K-shortest paths
            # -------------------------------------------------------
            entries = []
            try:
                gen = nx.shortest_simple_paths(G, start, goal, weight="distance")
                for path in gen:
                    length = sum(
                        G[path[j]][path[j + 1]]["distance"]
                        for j in range(len(path) - 1)
                    )
                    entries.append((path, length))
                    if len(entries) >= K_PATHS:
                        break
            except nx.NetworkXNoPath:
                pass

            k_shortest_paths[start][goal] = entries

        print(f"  [{i + 1:>3}/{n}] {start} ✓")

    return {
        "shortest_distances": shortest_distances,
        "all_shortest_paths": all_shortest_paths,
        "next_hops":          next_hops,
        "k_shortest_paths":   k_shortest_paths,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"YAML não encontrado: {YAML_PATH}")

    print("=" * 50)
    print("Pré-computação do grafo da fábrica")
    print("=" * 50)
    print(f"Input:  {YAML_PATH}")
    print(f"Output: {CACHE_PATH}")
    print()

    t0 = time.perf_counter()

    G     = load_graph(YAML_PATH)
    cache = precompute(G)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    elapsed = time.perf_counter() - t0

    print()
    print("=" * 50)
    print(f"Cache guardado em: {CACHE_PATH}")
    print(f"Tempo total: {elapsed:.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()