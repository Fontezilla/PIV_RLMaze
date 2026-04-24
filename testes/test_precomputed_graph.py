import pickle

with open("../.configs/graph_cache.pkl", "rb") as f:
    cache = pickle.load(f)

start, goal = "entryA", "exitD"
path = cache["all_shortest_paths"][start][goal][0]
dist = cache["shortest_distances"][start][goal]

print(f"{start} → {goal}")
print(f"Distância: {dist:.1f}")
print(f"Caminho:   {' → '.join(path)}")

hops = cache["next_hops"][start][goal]
print(f"Next hops: {hops}")

print(f"\nTop {len(cache['k_shortest_paths'][start][goal])} caminhos:")
for i, (p, d) in enumerate(cache["k_shortest_paths"][start][goal]):
    print(f"  {i+1}. {' → '.join(p)}  ({d:.1f})")