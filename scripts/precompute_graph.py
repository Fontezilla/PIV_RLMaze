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
    """Pré-computa distâncias mínimas entre todos os pares de nós."""
    nodes = list(G.nodes)
    n     = len(nodes)
    shortest_distances: dict = {}

    print(f"Grafo: {n} nós, {G.number_of_edges()} arestas")
    print()

    for i, src in enumerate(nodes):
        shortest_distances[src] = {}
        lengths = nx.single_source_dijkstra_path_length(G, src, weight="distance")
        for dst in nodes:
            shortest_distances[src][dst] = lengths.get(dst, float("inf"))

        if (i + 1) % 10 == 0 or i + 1 == n:
            print(f"  [{i + 1:>3}/{n}] {src} ✓")

    return {"shortest_distances": shortest_distances}


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