"""train.py."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml
import torch
from tqdm import tqdm

from old.env.wrapper.factory_env import FactoryEnv
from old.agent.policy import Policy
from old.agent.explain import explain_step


# ── Workers persistentes ──────────────────────────────────────────────────
# O env e a policy são criados UMA VEZ por processo worker (_init_worker) e
# reutilizados em todos os episódios desse worker. Isto evita recarregar o
# pickle do grafo, reconstruir BoxManager, etc. a cada episódio.
_WORKER_ENV    = None
_WORKER_POLICY = None


def _init_worker(cfg: dict) -> None:
    """Inicializa env e policy no processo worker (executado uma única vez)."""
    global _WORKER_ENV, _WORKER_POLICY
    cfg_w = copy.deepcopy(cfg)
    cfg_w["agent"]["device"] = "cpu"   # GPU não é partilhável entre processos
    _WORKER_ENV    = make_env(cfg_w)
    _WORKER_POLICY = make_policy(cfg_w)


def _collect_episode(args: tuple) -> tuple[dict, list]:
    """Worker para recolha paralela de episódios (usa env/policy persistentes)."""
    state_dict, _seed = args
    assert _WORKER_ENV    is not None, "Worker não foi inicializado."
    assert _WORKER_POLICY is not None, "Worker não foi inicializado."

    _WORKER_POLICY.net.load_state_dict(state_dict)
    _WORKER_POLICY.clear()

    metrics = run_episode(_WORKER_ENV, _WORKER_POLICY, deterministic=False, record=True)
    traj    = list(_WORKER_POLICY.trajectory)
    return metrics, traj


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
            "Falha ao gerar cache do grafo. Corre manualmente:\n"
            "  python scripts/precompute_graph.py"
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


def run_episode(
    env         : FactoryEnv,
    policy      : Policy,
    deterministic: bool = False,
    record      : bool  = True,
    debug       : bool  = False,
    explain     : bool  = False,
) -> dict[str, float]:
    """Corre um episódio completo.

    debug   — log compacto por step (tick + escolhas).
    explain — log detalhado por step (scores de todos os candidatos + masking).
    """
    state, info = env.reset()
    done        = False

    total_reward = 0.0
    n_steps      = 0
    t_start      = time.time()

    while not done:
        # Guarda o state que originou as decisões (será usado no explain).
        state_before = state

        if explain:
            assignments, transition, decisions = policy.act_verbose(
                state, deterministic=deterministic
            )
        else:
            assignments, transition = policy.act(state, deterministic=deterministic)
            decisions = None

        if debug and not explain:
            tick    = state.get("tick", "?")
            pending = [r.split("_")[-1] for r in state.get("pending_robot_ids", [])]
            boxes   = {b["box_id"]: b for b in state.get("boxes", [])}
            avail   = state.get("available_box_ids", [])
            box_str = ", ".join(
                f"b{bid}({boxes[bid]['pipeline'][0]})@{boxes[bid].get('current_node') or 'transit'}→{boxes[bid].get('next_waypoint') or '?'}"
                for bid in avail if bid in boxes
            ) or "—"
            asgn_str = ", ".join(
                f"r{k.split('_')[-1]}→{'b'+str(v) if isinstance(v, int) else (v or 'idle')}"
                for k, v in assignments.items()
            ) if assignments else "—"
            print(f"  t={tick:4d} | robots={pending} | avail=[{box_str}]  → {asgn_str}")

        state, reward, terminated, truncated, info = env.step(assignments)
        done = terminated or truncated

        if explain and decisions is not None:
            print(explain_step(
                state          = state_before,
                decisions      = decisions,
                action_indices = transition.action_indices,
                reward         = reward,
            ))

        total_reward += reward
        n_steps      += 1

        if record:
            policy.record(
                transition,
                reward     = reward,
                terminated = terminated,
                truncated  = truncated,
            )

    return {
        "total_reward" : total_reward,
        "n_steps"      : n_steps,
        "delivered"    : info.get("delivered", 0),
        "n_boxes"      : env.n_boxes,
        "ticks"        : info.get("tick", 0),
        "all_delivered": info.get("all_delivered", False),
        "duration_s"   : time.time() - t_start,
    }


def _entropy_coef(episode: int, n_episodes: int, cfg: dict) -> float:
    """Decaimento linear de entropy_coef_start → entropy_coef_end.

    Suporta a chave legada 'entropy_coef' como fallback (valor fixo).
    """
    fallback = cfg.get("entropy_coef", 0.01)
    start    = cfg.get("entropy_coef_start", fallback)
    end      = cfg.get("entropy_coef_end",   fallback)
    t        = min(episode / max(n_episodes, 1), 1.0)
    return start + (end - start) * t


def evaluate(
    cfg          : dict,
    policy       : Policy,
    n_episodes   : int,
    render       : bool = True,
    explain      : bool = False,
) -> dict[str, float]:
    """Corre N episódios de avaliação em modo determinístico."""
    env = make_env(cfg, render=render)

    rewards    = []
    deliveries = []
    ticks      = []
    all_done   = []

    for ep in range(n_episodes):
        if render or explain:
            print(f"\n  [ep {ep+1}/{n_episodes}]")
        metrics = run_episode(
            env, policy,
            deterministic = True,
            record        = False,
            debug         = render and not explain,
            explain       = explain,
        )
        rewards.append(metrics["total_reward"])
        deliveries.append(metrics["delivered"])
        ticks.append(metrics["ticks"])
        all_done.append(float(metrics["all_delivered"]))

        if render:
            print(
                f"  [eval {ep+1}/{n_episodes}] "
                f"reward={metrics['total_reward']:.1f}  "
                f"delivered={metrics['delivered']}/{metrics['n_boxes']}  "
                f"ticks={metrics['ticks']}  "
                f"all_done={metrics['all_delivered']}"
            )

    env.close()

    n = max(len(rewards), 1)

    # Métrica primária de eficiência: ticks médios nos episódios onde entregou tudo.
    # float("inf") quando o agente ainda não completou nenhum episódio.
    ticks_complete    = [t for t, d in zip(ticks, all_done) if d > 0.5]
    ticks_to_all_done = (
        sum(ticks_complete) / len(ticks_complete)
        if ticks_complete else float("inf")
    )

    return {
        "eval/reward_mean"       : sum(rewards)    / n,
        "eval/reward_min"        : min(rewards),
        "eval/reward_max"        : max(rewards),
        "eval/delivered_mean"    : sum(deliveries) / n,
        "eval/all_done_rate"     : sum(all_done)   / n,
        "eval/ticks_mean"        : sum(ticks)      / n,
        "eval/ticks_to_all_done" : ticks_to_all_done,
    }


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

    # Critério de melhor modelo: minimizar ticks_to_all_done (eficiência).
    # Fase inicial (all_done_rate < 0.5): fallback para maximizar all_done_rate.
    best_ticks_to_all_done = float("inf")
    best_all_done_rate     = 0.0

    print(f"\n{'='*60}")
    print("  Factory RL — GNN + PPO")
    print(f"  n_robots={cfg['env']['n_robots']}  "
          f"n_boxes={cfg['env']['n_boxes']}  "
          f"hidden_dim={cfg['agent']['hidden_dim']}")
    print(f"{'='*60}\n")

    # --- Loop de treino ---
    n_workers    = train_cfg.get("n_workers", 1)
    use_parallel = n_workers > 1
    executor     = (
        ProcessPoolExecutor(
            max_workers = n_workers,
            initializer = _init_worker,   # env+policy criados 1x por worker
            initargs    = (cfg,),
        )
        if use_parallel else None
    )

    pbar    = tqdm(total=train_cfg["n_episodes"], desc="Treino", unit="ep", dynamic_ncols=True)
    episode = 0

    try:
        while episode < train_cfg["n_episodes"]:
            prev_ep = episode

            if use_parallel:
                # ── Recolha paralela ──────────────────────────────────────
                batch  = min(n_workers, train_cfg["n_episodes"] - episode)
                cpu_sd = {k: v.cpu() for k, v in policy.net.state_dict().items()}
                results = list(executor.map(
                    _collect_episode,
                    [(cpu_sd, episode + i) for i in range(batch)],
                ))
                combined_traj = [t for _, traj in results for t in traj]
                metrics_list  = [m for m, _ in results]

                entropy_coef = _entropy_coef(episode, train_cfg["n_episodes"], train_cfg)
                update_metrics = policy.update(
                    optimizer            = optimizer,
                    gamma                = train_cfg["gamma"],
                    lam                  = train_cfg["lam"],
                    clip_eps             = train_cfg["clip_eps"],
                    value_coef           = train_cfg["value_coef"],
                    entropy_coef         = entropy_coef,
                    n_epochs             = train_cfg["n_epochs"],
                    normalize_advantages = train_cfg["normalize_advantages"],
                    max_grad_norm        = train_cfg["max_grad_norm"],
                    trajectory           = combined_traj,
                )
                policy.clear()

                n = len(metrics_list)
                ep_metrics = {
                    "total_reward" : sum(m["total_reward"] for m in metrics_list) / n,
                    "delivered"    : sum(m["delivered"]    for m in metrics_list) / n,
                    "n_boxes"      : env.n_boxes,
                    "all_delivered": any(m["all_delivered"] for m in metrics_list),
                    "ticks"        : sum(m["ticks"]        for m in metrics_list) / n,
                    "n_steps"      : sum(m["n_steps"]      for m in metrics_list) / n,
                    "duration_s"   : sum(m["duration_s"]   for m in metrics_list) / n,
                }
                episode += batch
                pbar.update(batch)

            else:
                # ── Recolha single-worker ─────────────────────────────────
                ep_metrics   = run_episode(env, policy, deterministic=False, record=True)
                entropy_coef = _entropy_coef(episode, train_cfg["n_episodes"], train_cfg)
                update_metrics = policy.update(
                    optimizer            = optimizer,
                    gamma                = train_cfg["gamma"],
                    lam                  = train_cfg["lam"],
                    clip_eps             = train_cfg["clip_eps"],
                    value_coef           = train_cfg["value_coef"],
                    entropy_coef         = entropy_coef,
                    n_epochs             = train_cfg["n_epochs"],
                    normalize_advantages = train_cfg["normalize_advantages"],
                    max_grad_norm        = train_cfg["max_grad_norm"],
                )
                policy.clear()
                episode += 1
                pbar.update(1)

            # Barra de progresso
            pbar.set_postfix({
                "R"    : f"{ep_metrics['total_reward']:.1f}",
                "del"  : f"{ep_metrics['delivered']:.1f}/{env.n_boxes}",
                "H"    : f"{update_metrics.get('entropy', 0):.2f}",
                "H_c"  : f"{entropy_coef:.3f}",
                "p_L"  : f"{update_metrics.get('policy_loss', 0):.3f}",
            })

            # Log wandb
            log = {
                "train/reward"       : ep_metrics["total_reward"],
                "train/delivered"    : ep_metrics["delivered"],
                "train/all_done"     : float(ep_metrics["all_delivered"]),
                "train/ticks"        : ep_metrics["ticks"],
                "train/steps"        : ep_metrics["n_steps"],
                "train/duration_s"   : ep_metrics["duration_s"],
                "train/entropy_coef" : entropy_coef,
                **{f"train/{k}": v for k, v in update_metrics.items()},
                "episode"            : episode,
            }
            if use_wandb:
                import wandb
                wandb.log(log)

            # Checkpoint — dispara ao cruzar múltiplo de save_every
            save_every = ckpt_cfg["save_every"]
            if (episode // save_every) > (prev_ep // save_every):
                ckpt_path = ckpt_dir / f"checkpoint_ep{episode:05d}.pt"
                policy.save(str(ckpt_path))
                policy.save(str(ckpt_dir / "latest.pt"))
                tqdm.write(f"  → checkpoint: {ckpt_path}")

            # Avaliação — dispara ao cruzar múltiplo de every_n_episodes
            eval_every = eval_cfg["every_n_episodes"]
            if (episode // eval_every) > (prev_ep // eval_every):
                tqdm.write(f"\n--- Avaliação ep {episode} ---")
                eval_metrics = evaluate(
                    cfg        = cfg,
                    policy     = policy,
                    n_episodes = eval_cfg["n_eval_episodes"],
                    render     = eval_cfg["render"],
                )
                if use_wandb:
                    import wandb
                    # Filtra inf para evitar erros de serialização JSON no wandb.
                    wandb.log({
                        **{k: v for k, v in eval_metrics.items() if v != float("inf")},
                        "episode": episode,
                    })

                t2ad = eval_metrics.get("eval/ticks_to_all_done", float("inf"))
                t2ad_str = f"{t2ad:.0f}" if t2ad != float("inf") else "N/A"
                tqdm.write(
                    f"  all_done_rate={eval_metrics['eval/all_done_rate']:.2f}  "
                    f"ticks_to_all_done={t2ad_str}  "
                    f"reward_mean={eval_metrics['eval/reward_mean']:.1f}\n"
                )

                # Critério: minimizar ticks_to_all_done (eficiência).
                # Fase inicial (all_done_rate < 0.5): maximizar all_done_rate.
                cur_rate = eval_metrics["eval/all_done_rate"]
                is_better = False
                if cur_rate >= 0.5 and t2ad < best_ticks_to_all_done:
                    best_ticks_to_all_done = t2ad
                    is_better = True
                elif cur_rate < 0.5 and cur_rate > best_all_done_rate:
                    best_all_done_rate = cur_rate
                    is_better = True

                if is_better:
                    policy.save(str(ckpt_dir / "best.pt"))
                    label = (
                        f"ticks_all_done={best_ticks_to_all_done:.0f}"
                        if cur_rate >= 0.5 else
                        f"all_done_rate={best_all_done_rate:.2f}"
                    )
                    tqdm.write(f"  → melhor modelo ({label})")

    finally:
        pbar.close()
        env.close()
        if executor is not None:
            executor.shutdown(wait=True)
        if use_wandb:
            import wandb
            wandb.finish()

    print("\nTreino concluído.")
    if best_ticks_to_all_done != float("inf"):
        print(f"Melhor ticks_to_all_done: {best_ticks_to_all_done:.0f}")
    else:
        print(f"Melhor all_done_rate: {best_all_done_rate:.2f}")
    print(f"Checkpoints em: {ckpt_dir}/")


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
        explain    = args.explain,
    )

    print("\n--- Resultados ---")
    for k, v in metrics.items():
        val_str = "N/A" if v == float("inf") else f"{v:.3f}"
        print(f"  {k}: {val_str}")


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
    parser.add_argument(
        "--explain",
        action  = "store_true",
        help    = "Imprime explicação detalhada de cada decisão durante eval",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)

    if args.eval:
        eval_only(cfg, args)
    else:
        train(cfg, args)