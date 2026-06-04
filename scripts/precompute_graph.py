from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import yaml

YAML_PATH  = Path(".configs/map_factory.yaml")
CACHE_PATH = Path(".configs/graph_cache.pkl")


def load_graph(yaml_path: Path) -> nx.Graph:
    """Carrega o grafo directamente do YAML sem subnós."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    G = nx.Graph()

    for node_id, info in data["nodes"].items():
        x, y = info.get("coords", [0.0, 0.0])
        G.add_node(
            node_id,
            type=info.get("type"),
            x=float(x),
            y=float(y),
        )

    for edge in data["edges"]:
        G.add_edge(
            edge["from"], edge["to"],
            distance=float(edge["distance"])
        )

    return G


def precompute(G: nx.Graph) -> dict:
    """Pré-computa distâncias e features estruturais do grafo.

    Devolve:
      - shortest_distances: distâncias mínimas par-a-par.
      - betweenness:        centralidade de cada nó (gargalo).
      - degree:             grau (nº de vizinhos) por nó.
      - dist_to_type:       distância mínima de cada nó ao tipo mais próximo,
                            para tipos {exit, processA_entry, processB_entry}.
    """
    nodes = list(G.nodes)
    n     = len(nodes)
    shortest_distances: dict = {}

    print(f"Grafo: {n} nós, {G.number_of_edges()} arestas")
    print()

    # Distâncias par-a-par
    for i, src in enumerate(nodes):
        shortest_distances[src] = {}
        lengths = nx.single_source_dijkstra_path_length(G, src, weight="distance")
        for dst in nodes:
            shortest_distances[src][dst] = lengths.get(dst, float("inf"))

        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"  [{i + 1:>3}/{n}] {src} OK")

    # Betweenness centrality (normalizada [0, 1])
    print("Calculando betweenness centrality...")
    betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)

    # Degree por nó
    degree = {node: G.degree(node) for node in nodes}

    # Distância ao tipo mais próximo (exit, processA_entry, processB_entry)
    nodes_by_type: dict[str, list[str]] = {}
    for node, attrs in G.nodes(data=True):
        t = attrs.get("type", "")
        nodes_by_type.setdefault(t, []).append(node)

    target_types = ["exit", "processA_entry", "processB_entry"]
    dist_to_type: dict[str, dict[str, float]] = {}
    for node in nodes:
        dist_to_type[node] = {}
        for t in target_types:
            candidates = nodes_by_type.get(t, [])
            if not candidates:
                dist_to_type[node][t] = float("inf")
                continue
            dist_to_type[node][t] = min(
                shortest_distances[node].get(c, float("inf"))
                for c in candidates
            )

    return {
        "shortest_distances": shortest_distances,
        "betweenness":        betweenness,
        "degree":             degree,
        "dist_to_type":       dist_to_type,
    }


def main() -> None:
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"YAML não encontrado: {YAML_PATH}")

    print("=" * 50)
    print(f"Input:  {YAML_PATH}")
    print(f"Output: {CACHE_PATH}")
    print("Sem subnós — grafo original do YAML")
    print("=" * 50)

    t0    = time.perf_counter()
    G     = load_graph(YAML_PATH)
    cache = precompute(G)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    print()
    print(f"Cache guardado em: {CACHE_PATH}")
    print(f"Tempo total: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()