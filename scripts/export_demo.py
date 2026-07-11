"""Corre um episódio com o AGENTE TREINADO (determinístico) e exporta para
JSON em vez de abrir o renderer pygame — para o render 3D em Godot mostrar a
política aprendida (ex. distribuição pelos exits). Correr da raiz do projecto.

Uso:
    python scripts/export_demo.py                       # best.pt, layout fixo
    python scripts/export_demo.py --checkpoint checkpoints/checkpoint_ep01008.pt
    python scripts/export_demo.py --box-layout "BB RG GG B" --robots 2
    python scripts/export_demo.py --random              # política aleatória (sem treino)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from env.wrapper.factory_env import FactoryEnv
from env.render.godot_export import export_episode
from agent.policy import Policy

ROOT = Path(__file__).parent.parent
YAML_PATH = ROOT / ".configs" / "map_factory.yaml"
CACHE_PATH = ROOT / ".configs" / "graph_cache.pkl"
PIPE_PATH = ROOT / ".configs" / "box_pipeline.yaml"
CFG_PATH = ROOT / ".configs" / "train_config.yaml"
OUT_PATH = ROOT / "render 3D" / "data" / "demo_episode.json"


def main() -> None:
    """Corre um episódio (agente treinado ou aleatório) e exporta o JSON."""
    parser = argparse.ArgumentParser(description="Exporta um episódio para o render 3D (Godot)")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "best.pt"),
                        help="Checkpoint (.pt) do agente treinado")
    parser.add_argument("--box-layout", type=str, default="BB RG GG B",
                        help="Layout fixo de caixas (ex.: \"BB RG GG B\")")
    parser.add_argument("--robots", type=int, default=2, help="Número de robots")
    parser.add_argument("--seed", type=int, default=1, help="Seed do cenário")
    parser.add_argument("--random", action="store_true",
                        help="Usa política aleatória (ignora o checkpoint)")
    args = parser.parse_args()

    torch.set_num_threads(4)
    import yaml
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = FactoryEnv(n_robots=args.robots, box_layout=args.box_layout, tick_limit=8000,
                     map_path=str(YAML_PATH), cache_path=str(CACHE_PATH),
                     pipeline_path=str(PIPE_PATH), seed=args.seed, record=True)
    state, _ = env.reset()

    policy = None
    if not args.random:
        policy = Policy(hidden_dim=cfg["agent"]["hidden_dim"], n_layers=cfg["agent"]["n_layers"],
                        dropout=cfg["agent"]["dropout"], device="cpu")
        policy.load(args.checkpoint)

    rng = random.Random(args.seed)
    for step in range(4000):
        if policy is not None:
            assign, _ = policy.act(state, deterministic=True)
        else:
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
            src = "aleatória" if args.random else Path(args.checkpoint).name
            print(f"episódio terminou ({src}): entregue={state['delivered']}/{env.n_boxes} "
                  f"clock={state['clock']:.0f} exits={dict(env._box_manager._exit_deliveries)}")
            break

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    export_episode(env, OUT_PATH)
    print(f"exportado para: {OUT_PATH}")


if __name__ == "__main__":
    main()
