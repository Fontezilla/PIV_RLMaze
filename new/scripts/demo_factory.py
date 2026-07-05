"""Corre um episódio do FactoryEnv com política aleatória e reproduz no
renderer novo. Correr da pasta new/ com o python que tem pygame."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.wrapper.factory_env import FactoryEnv
from env.render import renderer

YAML_PATH = Path(__file__).parent.parent.parent / ".configs" / "map_factory.yaml"
CACHE_PATH = Path(__file__).parent.parent / ".configs" / "graph_cache.pkl"
PIPE_PATH = Path(__file__).parent.parent.parent / ".configs" / "box_pipeline.yaml"


def main() -> None:
    """Corre um episódio com política aleatória e reproduz no renderer."""
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

    renderer.play(env.graph, env.render_players, env.box_log,
                  title="Factory RL — FactoryEnv (3 robots, 6 caixas)", speed=20.0)


if __name__ == "__main__":
    main()
