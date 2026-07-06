"""Corre o mesmo episódio do `demo_factory.py` (política aleatória, seed=3,
3 robots, 6 caixas) e exporta para JSON em vez de abrir o renderer pygame —
para dar ao render 3D em Godot. Correr da raiz do projecto."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.wrapper.factory_env import FactoryEnv
from env.render.godot_export import export_episode

YAML_PATH = Path(__file__).parent.parent / ".configs" / "map_factory.yaml"
CACHE_PATH = Path(__file__).parent.parent / ".configs" / "graph_cache.pkl"
PIPE_PATH = Path(__file__).parent.parent / ".configs" / "box_pipeline.yaml"
OUT_PATH = Path(__file__).parent.parent / "Visual" / "data" / "demo_episode.json"


def main() -> None:
    """Corre um episódio com política aleatória e exporta o JSON do episódio."""
    env = FactoryEnv(n_robots=3, n_boxes=6, tick_limit=8000,
                     map_path=str(YAML_PATH), cache_path=str(CACHE_PATH),
                     pipeline_path=str(PIPE_PATH), seed=3, record=True)
    rng = random.Random(3)
    state, _ = env.reset()

    for step in range(2000):
        used, assign = set(), {}
        for rid in state["pending_robot_ids"]:
            opts = [(b, t) for (b, t) in state["available_box_targets"] if b not in used]
            if not opts:
                break
            b, t = rng.choice(opts)
            used.add(b)
            assign[rid] = (b, t)
        state, reward, term, trunc, _ = env.step(assign)
        if term or trunc:
            print(f"episódio terminou: entregue={state['delivered']}/{env.n_boxes} "
                  f"clock={state['clock']:.0f} term={term}")
            break

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    export_episode(env, OUT_PATH)
    print(f"exportado para: {OUT_PATH}")


if __name__ == "__main__":
    main()
