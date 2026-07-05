"""Pré-computa a cache do grafo (distâncias, betweenness, grau) — INCLUINDO
os sub-nós de junction gerados por `env.core.graph.FactoryGraph`.

A cache antiga (`old/.configs/graph_cache.pkl` ou `.configs/graph_cache.pkl`
na raiz) é do grafo sem sub-nós — não serve para o novo projecto.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from env.core.graph import FactoryGraph

YAML_PATH  = Path(__file__).parent.parent.parent / ".configs" / "map_factory.yaml"
CACHE_PATH = Path(__file__).parent.parent / ".configs" / "graph_cache.pkl"


def precompute(G: nx.Graph) -> dict:
    """Distâncias par-a-par, betweenness e grau — para todos os nós,
    incluindo os sub-nós de junction."""
    nodes = list(G.nodes)
    n = len(nodes)
    shortest_distances: dict = {}

    print(f"Grafo: {n} nós ({G.number_of_edges()} arestas)")

    for i, src in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(G, src, weight="distance")
        shortest_distances[src] = {dst: lengths.get(dst, float("inf")) for dst in nodes}
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"  [{i + 1:>3}/{n}] {src} OK")

    print("Betweenness centrality...")
    betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)
    degree = {node: G.degree(node) for node in nodes}

    nodes_by_type: dict[str, list[str]] = {}
    for node, attrs in G.nodes(data=True):
        nodes_by_type.setdefault(attrs.get("type", ""), []).append(node)

    target_types = ["exit", "processA_entry", "processB_entry"]
    dist_to_type: dict[str, dict[str, float]] = {}
    for node in nodes:
        dist_to_type[node] = {}
        for t in target_types:
            candidates = nodes_by_type.get(t, [])
            dist_to_type[node][t] = min(
                (shortest_distances[node].get(c, float("inf")) for c in candidates),
                default=float("inf"),
            )

    return {
        "shortest_distances": shortest_distances,
        "betweenness": betweenness,
        "degree": degree,
        "dist_to_type": dist_to_type,
    }


def main() -> None:
    """Gera e grava a cache do grafo (distâncias, betweenness, grau)."""
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"YAML não encontrado: {YAML_PATH}")

    print(f"Input:  {YAML_PATH}")
    print(f"Output: {CACHE_PATH}")

    t0 = time.perf_counter()
    graph = FactoryGraph(str(YAML_PATH), cache_path=None)
    cache = precompute(graph.graph)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    print(f"Cache guardada em: {CACHE_PATH}")
    print(f"Tempo total: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
