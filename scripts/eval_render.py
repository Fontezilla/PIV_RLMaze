"""Corre um episódio com o AGENTE TREINADO (congelado, determinístico) e
reproduz no renderer pygame — para ver visualmente a política aprendida.

Uso (da raiz do projecto):
    python scripts/eval_render.py                    # best.pt, seed aleatório
    python scripts/eval_render.py --seed 1007        # cenário fixo
    python scripts/eval_render.py --checkpoint checkpoints/checkpoint_ep00600.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from env.wrapper.factory_env import FactoryEnv
from env.render import renderer
from agent.policy import Policy

ROOT = Path(__file__).parent.parent
CFG_PATH = ROOT / ".configs" / "train_config.yaml"


def load_cfg() -> dict:
    """Lê a config de treino e resolve os caminhos do env para absolutos."""
    import yaml
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ("map_path", "cache_path", "pipeline_path"):
        cfg["env"][k] = str(ROOT / cfg["env"][k])
    return cfg


def main() -> None:
    """Carrega o checkpoint, corre um episódio determinístico e renderiza."""
    parser = argparse.ArgumentParser(description="Render de um episódio com o agente treinado")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "best.pt"),
                        help="Caminho do checkpoint (.pt) a carregar")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed do cenário (fixo). Omite para aleatório.")
    parser.add_argument("--speed", type=float, default=20.0,
                        help="Ticks de sim por segundo real (20 = 1x)")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 10_000)
    cfg = load_cfg()
    torch.set_num_threads(4)

    env = FactoryEnv(
        n_robots=cfg["env"]["n_robots"], n_boxes=cfg["env"]["n_boxes"],
        tick_limit=cfg["env"]["tick_limit"],
        map_path=cfg["env"]["map_path"], cache_path=cfg["env"]["cache_path"],
        pipeline_path=cfg["env"]["pipeline_path"], seed=seed, record=True,
        box_layout=cfg["env"].get("box_layout"),
    )
    policy = Policy(
        hidden_dim=cfg["agent"]["hidden_dim"], n_layers=cfg["agent"]["n_layers"],
        dropout=cfg["agent"]["dropout"], device="cpu",
    )
    policy.load(args.checkpoint)

    state, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        assignments, _ = policy.act(state, deterministic=True)
        state, reward, term, trunc, _ = env.step(assignments)
        total_reward += reward
        done = term or trunc

    print(f"checkpoint: {Path(args.checkpoint).name}  seed={seed}")
    print(f"entregue={state['delivered']}/{env.n_boxes}  "
          f"clock={state['clock']:.0f}  reward={total_reward:.1f}  "
          f"all_done={term}")

    renderer.play(
        env.graph, env.render_players, env.box_log,
        title=f"Factory RL — agente treinado (seed {seed})", speed=args.speed,
    )


if __name__ == "__main__":
    main()
