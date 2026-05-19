"""
train.py
~~~~~~~~
Loop de treino PPO para o agente GNN de gestão de fábrica.

Uso
---
  # Treino com config por defeito
  python train.py

  # Config personalizada
  python train.py --config train_config.yaml

  # Sem wandb
  python train.py --no-wandb

  # Retomar de checkpoint
  python train.py --resume checkpoints/latest.pt

  # Só avaliação com render
  python train.py --eval --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml
import torch
from tqdm import tqdm

from env.wrapper.factory_env import FactoryEnv
from agent.policy import Policy


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _ensure_graph_cache(cfg: dict) -> None:
    """Verifica se o graph_cache.pkl existe; se não, gera-o automaticamente."""
    cache_path = Path(cfg["env"]["cache_path"])
    if cache_path.exists():
        return
    print(f"Cache do grafo não encontrada ({cache_path}). A gerar (pode demorar ~30 s)...")
    result = subprocess.run(
        [sys.executable, "scripts/precompute_graph.py"],
        check=False,
    )
    if result.returncode != 0 or not cache_path.exists():
        raise RuntimeError(
            f"Falha ao gerar cache do grafo. Corre manualmente:\n"
            f"  python scripts/precompute_graph.py"
        )
    print(f"Cache gerada em {cache_path}.")


def make_env(cfg: dict, render: bool = False) -> FactoryEnv:
    env_cfg = cfg["env"]
    return FactoryEnv(
        n_robots      = env_cfg["n_robots"],
        n_boxes       = env_cfg["n_boxes"],
        tick_limit    = env_cfg["tick_limit"],
        map_path      = env_cfg["map_path"],
        cache_path    = env_cfg["cache_path"],
        pipeline_path = env_cfg["pipeline_path"],
        render_mode   = "human" if render else None,
    )


def make_policy(cfg: dict) -> Policy:
    agent_cfg = cfg["agent"]
    return Policy(
        hidden_dim = agent_cfg["hidden_dim"],
        n_layers   = agent_cfg["n_layers"],
        dropout    = agent_cfg["dropout"],
        device     = agent_cfg["device"],
    )


# ---------------------------------------------------------------------------
# Episódio
# ---------------------------------------------------------------------------

def run_episode(
    env         : FactoryEnv,
    policy      : Policy,
    deterministic: bool = False,
    record      : bool  = True,
) -> dict[str, float]:
    """
    Corre um episódio completo.

    Parâmetros
    ----------
    env           : ambiente de fábrica
    policy        : política GNN + PPO
    deterministic : se True, usa argmax (avaliação)
    record        : se True, guarda trajectória para update

    Devolve
    -------
    dict com métricas do episódio:
      total_reward, n_steps, delivered, ticks, duration_s
    """
    state, info = env.reset()
    done        = False

    total_reward = 0.0
    n_steps      = 0
    t_start      = time.time()

    while not done:
        assignments, transition = policy.act(state, deterministic=deterministic)

        state, reward, terminated, truncated, info = env.step(assignments)
        done = terminated or truncated

        total_reward += reward
        n_steps      += 1

        if record:
            policy.record(transition, reward=reward, done=done)

    return {
        "total_reward" : total_reward,
        "n_steps"      : n_steps,
        "delivered"    : info.get("delivered", 0),
        "n_boxes"      : env.n_boxes,
        "ticks"        : info.get("tick", 0),
        "all_delivered": info.get("all_delivered", False),
        "duration_s"   : time.time() - t_start,
    }


# ---------------------------------------------------------------------------
# Avaliação com render
# ---------------------------------------------------------------------------

def evaluate(
    cfg          : dict,
    policy       : Policy,
    n_episodes   : int,
    render       : bool = True,
) -> dict[str, float]:
    """
    Corre N episódios de avaliação em modo determinístico.
    Se render=True, abre a interface gráfica.

    Devolve métricas médias.
    """
    env = make_env(cfg, render=render)

    rewards    = []
    deliveries = []
    ticks      = []
    all_done   = []

    for ep in range(n_episodes):
        metrics = run_episode(env, policy, deterministic=True, record=False)
        rewards.append(metrics["total_reward"])
        deliveries.append(metrics["delivered"])
        ticks.append(metrics["ticks"])
        all_done.append(float(metrics["all_delivered"]))

        if render:
            print(
                f"  [eval {ep+1}/{n_episodes}] "
                f"reward={metrics['total_reward']:.1f}  "
                f"delivered={metrics['delivered']}/{metrics['n_boxes']}  "
                f"ticks={metrics['ticks']}"
            )

    env.close()

    n = max(len(rewards), 1)
    return {
        "eval/reward_mean"    : sum(rewards)    / n,
        "eval/reward_min"     : min(rewards),
        "eval/reward_max"     : max(rewards),
        "eval/delivered_mean" : sum(deliveries) / n,
        "eval/all_done_rate"  : sum(all_done)   / n,
        "eval/ticks_mean"     : sum(ticks)      / n,
    }


# ---------------------------------------------------------------------------
# Treino principal
# ---------------------------------------------------------------------------

def train(cfg: dict, args: argparse.Namespace) -> None:
    _ensure_graph_cache(cfg)

    train_cfg      = cfg["train"]
    eval_cfg       = cfg["eval"]
    ckpt_cfg       = cfg["checkpoints"]
    wandb_cfg      = cfg.get("wandb", {})

    use_wandb      = not args.no_wandb
    ckpt_dir       = Path(ckpt_cfg["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Wandb ---
    if use_wandb:
        import wandb
        wandb.init(
            project = wandb_cfg.get("project", "factory-rl"),
            entity  = wandb_cfg.get("entity")  or None,
            tags    = wandb_cfg.get("tags",   []),
            notes   = wandb_cfg.get("notes",  ""),
            config  = {
                **cfg["env"],
                **cfg["agent"],
                **cfg["train"],
            },
        )

    # --- Ambiente e política ---
    env    = make_env(cfg, render=False)
    policy = make_policy(cfg)

    if args.resume:
        print(f"A retomar de {args.resume}")
        policy.load(args.resume)

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr = train_cfg["lr"],
    )

    best_eval_reward = float("-inf")

    print(f"\n{'='*60}")
    print(f"  Factory RL — GNN + PPO")
    print(f"  n_robots={cfg['env']['n_robots']}  "
          f"n_boxes={cfg['env']['n_boxes']}  "
          f"hidden_dim={cfg['agent']['hidden_dim']}")
    print(f"{'='*60}\n")

    # --- Loop de treino ---
    pbar = tqdm(
        range(1, train_cfg["n_episodes"] + 1),
        desc     = "Treino",
        unit     = "ep",
        dynamic_ncols = True,
    )

    try:
        for episode in pbar:

            # Treino
            ep_metrics = run_episode(env, policy, deterministic=False, record=True)

            update_metrics = policy.update(
                optimizer            = optimizer,
                gamma                = train_cfg["gamma"],
                lam                  = train_cfg["lam"],
                clip_eps             = train_cfg["clip_eps"],
                value_coef           = train_cfg["value_coef"],
                entropy_coef         = train_cfg["entropy_coef"],
                n_epochs             = train_cfg["n_epochs"],
                normalize_advantages = train_cfg["normalize_advantages"],
            )
            policy.clear()

            # Actualiza barra de progresso
            pbar.set_postfix({
                "R"       : f"{ep_metrics['total_reward']:.1f}",
                "del"     : f"{ep_metrics['delivered']}/{env.n_boxes}",
                "p_loss"  : f"{update_metrics.get('policy_loss', 0):.3f}",
                "entropy" : f"{update_metrics.get('entropy', 0):.3f}",
            })

            # Log por episódio
            log = {
                "train/reward"       : ep_metrics["total_reward"],
                "train/delivered"    : ep_metrics["delivered"],
                "train/all_done"     : float(ep_metrics["all_delivered"]),
                "train/ticks"        : ep_metrics["ticks"],
                "train/steps"        : ep_metrics["n_steps"],
                "train/duration_s"   : ep_metrics["duration_s"],
                **{f"train/{k}": v for k, v in update_metrics.items()},
                "episode"            : episode,
            }

            if use_wandb:
                import wandb
                wandb.log(log)

            # Checkpoint periódico
            if episode % ckpt_cfg["save_every"] == 0:
                ckpt_path = ckpt_dir / f"checkpoint_ep{episode:05d}.pt"
                policy.save(str(ckpt_path))
                policy.save(str(ckpt_dir / "latest.pt"))
                tqdm.write(f"  → checkpoint guardado: {ckpt_path}")

            # Avaliação periódica com render
            if episode % eval_cfg["every_n_episodes"] == 0:
                tqdm.write(f"\n--- Avaliação ep {episode} ---")
                eval_metrics = evaluate(
                    cfg        = cfg,
                    policy     = policy,
                    n_episodes = eval_cfg["n_eval_episodes"],
                    render     = eval_cfg["render"],
                )

                if use_wandb:
                    import wandb
                    wandb.log({**eval_metrics, "episode": episode})

                tqdm.write(
                    f"  reward_mean={eval_metrics['eval/reward_mean']:.1f}  "
                    f"delivered_mean={eval_metrics['eval/delivered_mean']:.1f}  "
                    f"all_done_rate={eval_metrics['eval/all_done_rate']:.2f}\n"
                )

                # Guarda melhor modelo
                if eval_metrics["eval/reward_mean"] > best_eval_reward:
                    best_eval_reward = eval_metrics["eval/reward_mean"]
                    policy.save(str(ckpt_dir / "best.pt"))
                    tqdm.write(f"  → melhor modelo actualizado "
                               f"(reward={best_eval_reward:.1f})")

    finally:
        pbar.close()
        env.close()
        if use_wandb:
            import wandb
            wandb.finish()

    print("\nTreino concluído.")
    print(f"Melhor eval reward: {best_eval_reward:.1f}")
    print(f"Checkpoints em: {ckpt_dir}/")


# ---------------------------------------------------------------------------
# Avaliação standalone
# ---------------------------------------------------------------------------

def eval_only(cfg: dict, args: argparse.Namespace) -> None:
    """Corre só avaliação a partir de um checkpoint."""
    _ensure_graph_cache(cfg)
    policy = make_policy(cfg)

    ckpt = args.checkpoint or str(Path(cfg["checkpoints"]["dir"]) / "best.pt")
    print(f"A carregar checkpoint: {ckpt}")
    policy.load(ckpt)

    eval_cfg = cfg["eval"]
    metrics  = evaluate(
        cfg        = cfg,
        policy     = policy,
        n_episodes = eval_cfg["n_eval_episodes"],
        render     = True,
    )

    print("\n--- Resultados ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Factory RL — GNN + PPO")

    parser.add_argument(
        "--config",
        type    = str,
        default = ".configs/train_config.yaml",
        help    = "Caminho para o ficheiro de configuração YAML",
    )
    parser.add_argument(
        "--no-wandb",
        action  = "store_true",
        help    = "Desactiva o logging para wandb",
    )
    parser.add_argument(
        "--resume",
        type    = str,
        default = None,
        help    = "Caminho para checkpoint para retomar treino",
    )
    parser.add_argument(
        "--eval",
        action  = "store_true",
        help    = "Corre só avaliação (sem treino)",
    )
    parser.add_argument(
        "--checkpoint",
        type    = str,
        default = None,
        help    = "Checkpoint para avaliação (usado com --eval)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)

    if args.eval:
        eval_only(cfg, args)
    else:
        train(cfg, args)