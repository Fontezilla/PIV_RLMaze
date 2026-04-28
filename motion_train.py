import argparse
import os
import random
import sys
import time
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm as _tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

sys.path.insert(0, os.path.dirname(__file__))

from simulation_engine.simulation_engine import SimulationEngine
from simulation_engine.core.entities import RobotState
from agents.motion_agent import DQNController, RewardWeights

# ---------------------------------------------------------------------------
# Configuração base
# ---------------------------------------------------------------------------

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "configs", "map_factory.yaml")
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), "configs", "box_pipeline.yaml")

VEL_MAX = 30.0
SEED = None  # None = seed aleatório; passar --seed N para reprodutibilidade
EPISODE_TICKS = 1000

ROBOT_IDS = [f"R{i}" for i in range(8)]

# ---------------------------------------------------------------------------
# Configuração de treino flat (sem curriculum)
# ---------------------------------------------------------------------------

N_ROBOTS_DEFAULT  = 8
N_EPISODES_DEFAULT = 1080

EPS_ROUTE_START = 0.35
EPS_ROUTE_END   = 0.03
EPS_VEL_START   = 0.18
EPS_VEL_END     = 0.02

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engine() -> SimulationEngine:
    return SimulationEngine(
        graph_path=GRAPH_PATH,
        pipeline_config_path=PIPELINE_PATH,
        vel_max=VEL_MAX,
    )


def all_nodes(engine: SimulationEngine) -> List[str]:
    return sorted(list(engine.graph.graph.nodes()))


def sample_spawn_nodes(rng: random.Random, nodes: List[str], n_robots: int) -> List[str]:
    """
    Escolhe nós de spawn aleatórios sem repetição.
    """
    if n_robots > len(nodes):
        raise ValueError(f"n_robots={n_robots} excede nº de nós disponíveis={len(nodes)}")
    return rng.sample(nodes, n_robots)


def add_random_robots(
    engine: SimulationEngine,
    rng: random.Random,
    n_robots: int,
) -> List[Tuple[str, str]]:
    """
    Adiciona n_robots em nós aleatórios.
    Devolve lista [(robot_id, spawn_node), ...].
    """
    nodes = all_nodes(engine)
    spawn_nodes = sample_spawn_nodes(rng, nodes, n_robots)

    assignments: List[Tuple[str, str]] = []
    for robot_id, node in zip(ROBOT_IDS[:n_robots], spawn_nodes):
        engine.add_robot(robot_id, node)
        assignments.append((robot_id, node))

    return assignments


def pick_goal(rng: random.Random, nodes: List[str], exclude: str) -> str:
    """
    Escolhe destino aleatório diferente do nó atual.
    """
    choices = [n for n in nodes if n != exclude]
    return rng.choice(choices) if choices else exclude


def update_lookahead_buffers(
    engine: SimulationEngine,
    controller: DQNController,
    prev_obs: Dict[str, object],
    actions: Dict[str, Tuple[str | None, str]],
    goals: Dict[str, str],
    dispatch_results: Dict,
    events,
) -> None:
    """
    Mantém o buffered_next_node encadeado para movimento contínuo em linha reta.

    Caso A — robot estava IDLE, Q-agent despachado:
        Define buffer para o salto após o nó destino do dispatch.

    Caso B — robot estava MOVING, buffered dispatch disparou no step 5:
        Detetado por mudança de aresta (from_node/to_node diferentes).
        Define buffer para o salto após o NOVO nó destino, encadeando
        o lookahead indefinidamente enquanto o caminho estiver livre.

    Nota sobre turn_cooldown:
        Quando o robot está IDLE com turn_cooldown > 0, o agente envia
        (None, "hold") e o engine preserva o buffer (não o consome).
        Nesse caso não limpamos o buffer aqui — ele será consumido assim
        que o cooldown chegar a zero.
    """
    collided_ids = {
        c.robot_a for c in events.collisions
    } | {
        c.robot_b for c in events.collisions
    }

    for rid, obs in prev_obs.items():
        robot = engine.world.robots.get(rid)
        if robot is None:
            continue

        # ------------------------------------------------------------------
        # Caso A: robot estava IDLE — dispatch do Q-agent
        # ------------------------------------------------------------------
        if obs.state == RobotState.IDLE:
            next_node, _speed_cmd = actions.get(rid, (None, "hold"))
            result = dispatch_results.get(rid)

            if next_node is None:
                # Sem dispatch do Q-agent neste tick.
                # Se há turn_cooldown, o engine está a preservar o buffer
                # → não limpar; o buffer será consumido após a curva.
                if robot.turn_cooldown == 0:
                    robot.buffered_next_node = None
                continue

            if result is None:
                robot.buffered_next_node = None
                continue

            if not getattr(result, "accepted", False) or not getattr(result, "dispatched", False):
                robot.buffered_next_node = None
                continue

            if rid in collided_ids or robot.collided:
                robot.buffered_next_node = None
                continue

            if obs.current_node is None:
                robot.buffered_next_node = None
                continue

            robot.buffered_next_node = controller.compute_buffered_next_node(
                current_node=obs.current_node,
                chosen_next=next_node,
                goal=goals[rid],
            )

        # ------------------------------------------------------------------
        # Caso B: robot estava MOVING e a aresta mudou este tick
        #         → buffered dispatch disparou no step 5
        #         → definir buffer para o próximo salto (encadeamento)
        # ------------------------------------------------------------------
        elif (obs.state == RobotState.MOVING
              and robot.state == RobotState.MOVING
              and (robot.from_node != obs.from_node or robot.to_node != obs.to_node)):

            if rid in collided_ids or robot.collided:
                robot.buffered_next_node = None
                continue

            if robot.from_node is None or robot.to_node is None:
                robot.buffered_next_node = None
                continue

            # from_node é o nó intermédio acabado de atravessar
            # to_node   é o destino atual (resultado do buffered dispatch)
            robot.buffered_next_node = controller.compute_buffered_next_node(
                current_node=robot.from_node,
                chosen_next=robot.to_node,
                goal=goals[rid],
            )


def run_eval(
    engine,
    controller,
    rng,
    n_robots: int,
    n_ticks: int,
    fps: int,
    sub_steps: int,
    global_episode: int,
) -> str:
    """
    Episódio de avaliação com pygame (epsilon=0, sem aprendizagem).
    Devolve 'quit' se o utilizador fechar a janela, senão 'ok'.
    """
    try:
        from renderer.pygame_renderer import PygameRenderer
    except ImportError:
        print("  [eval] pygame não disponível — ignorado.")
        return "ok"

    saved_eps_r = controller.epsilon_route
    saved_eps_v = controller.epsilon_vel
    controller.set_epsilon(0.0, 0.0)

    engine.reset()
    controller.reset_memory()
    assignments = add_random_robots(engine, rng, n_robots)
    robot_ids = [rid for rid, _ in assignments]
    map_nodes = all_nodes(engine)
    goals: Dict[str, str] = {
        rid: pick_goal(rng, map_nodes, spawn)
        for rid, spawn in assignments
    }

    iface = engine.interface
    renderer = PygameRenderer(
        graph=engine.graph,
        window_width=1400,
        window_height=900,
        graph_fraction=0.72,
        fps=fps,
        sub_steps=sub_steps,
    )

    ep_cols = 0
    ep_arrivals = 0
    result = "ok"

    try:
        for _tick in range(n_ticks):
            prev_obs = {rid: iface.get_motion_view(rid) for rid in robot_ids}
            actions = {}

            for rid in robot_ids:
                obs = prev_obs[rid]
                valid = iface.get_valid_actions(rid, obs)

                if obs.state == RobotState.IDLE and obs.current_node == goals[rid]:
                    goals[rid] = pick_goal(rng, map_nodes, obs.current_node)
                    ep_arrivals += 1

                actions[rid] = controller.act(rid, obs, valid, goals[rid], engine.world)

            events, dispatch_results = engine.step(actions)
            ep_cols += len(events.collisions)

            update_lookahead_buffers(
                engine=engine,
                controller=controller,
                prev_obs=prev_obs,
                actions=actions,
                goals=goals,
                dispatch_results=dispatch_results,
                events=events,
            )

            info = {
                "mode": f"eval @ ep {global_episode}",
                "robots": len(robot_ids),
                "ep_cols": ep_cols,
                "arrivals": ep_arrivals,
                "eroute": "0.000",
                "evel": "0.000",
                "goals": goals,
            }
            if renderer.render(engine.world, info=info) == "quit":
                result = "quit"
                break
    finally:
        renderer.close()

    controller.set_epsilon(saved_eps_r, saved_eps_v)
    controller.reset_memory()
    return result


def run_episode_train(
    engine,
    controller,
    robot_ids,
    goals,
    rng,
    renderer=None,
):
    """Corre um episódio completo. Devolve (ep_cols, ep_arrivals) ou 'quit'."""
    iface = engine.interface
    map_nodes = all_nodes(engine)

    ep_cols = 0
    ep_arrivals = 0

    for _tick in range(EPISODE_TICKS):
        prev_obs = {rid: iface.get_motion_view(rid) for rid in robot_ids}

        actions = {}
        for rid in robot_ids:
            obs = prev_obs[rid]
            valid = iface.get_valid_actions(rid)

            if obs.state == RobotState.IDLE and obs.current_node == goals[rid]:
                goals[rid] = pick_goal(rng, map_nodes, obs.current_node)
                engine.world.robots[rid].carried_box = f"box_{rid}" if rng.random() < 0.4 else None
                ep_arrivals += 1

            actions[rid] = controller.act(rid, obs, valid, goals[rid], engine.world)

        events, dispatch_results = engine.step(actions)
        ep_cols += len(events.collisions)

        update_lookahead_buffers(
            engine=engine,
            controller=controller,
            prev_obs=prev_obs,
            actions=actions,
            goals=goals,
            dispatch_results=dispatch_results,
            events=events,
        )

        for rid in robot_ids:
            curr_obs = iface.get_motion_view(rid)
            controller.update(rid, prev_obs[rid], curr_obs, events, goals[rid])

        if renderer is not None:
            info = {
                "mode": "train",
                "robots": len(robot_ids),
                "ep_cols": ep_cols,
                "arrivals": ep_arrivals,
                "eroute": f"{controller.epsilon_route:.3f}",
                "evel": f"{controller.epsilon_vel:.3f}",
                "goals": goals,
            }
            if renderer.render(engine.world, info=info) == "quit":
                return "quit"

    return ep_cols, ep_arrivals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Sem render pygame (treino rápido)")
    parser.add_argument("--resume", default=None, help="Diretório de checkpoint para retomar")
    parser.add_argument("--save-every", type=int, default=50, dest="save_every",
                        help="Guardar checkpoint a cada N episódios globais")
    parser.add_argument("--log-every", type=int, default=10, dest="log_every",
                        help="Print stats a cada N episódios globais")
    parser.add_argument("--fps", type=int, default=60, help="FPS do renderer")
    parser.add_argument("--sub-steps", type=int, default=6, dest="sub_steps",
                        help="Sub-steps do renderer")
    parser.add_argument("--device", default="auto",
                        help="Dispositivo PyTorch: 'cpu', 'cuda', ou 'auto' (usa cuda se disponível)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Nº de processos worker paralelos (1 = sequencial)")
    parser.add_argument("--n-robots", type=int, default=N_ROBOTS_DEFAULT, dest="n_robots",
                        help=f"Nº de robots ativos (default: {N_ROBOTS_DEFAULT})")
    parser.add_argument("--episodes", type=int, default=N_EPISODES_DEFAULT,
                        help=f"Total de episódios de treino (default: {N_EPISODES_DEFAULT})")
    parser.add_argument("--eval-every", type=int, default=0, dest="eval_every",
                        help="Correr eval pygame a cada N episódios (0 = desativado)")
    parser.add_argument("--eval-ticks", type=int, default=500, dest="eval_ticks",
                        help="Duração do eval em ticks (default: 500)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed para reprodutibilidade (default: aleatório)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    import torch as _torch
    if args.device == "auto":
        _device = "cuda" if _torch.cuda.is_available() else "cpu"
    else:
        _device = args.device

    tau_route = max(1.0, args.episodes / 2.0)
    tau_vel   = max(1.0, args.episodes / 2.0)

    engine = make_engine()
    controller = DQNController(
        graph=engine.graph,
        vel_max=VEL_MAX,
        gamma=0.95,
        lr=3e-4,
        epsilon_route=EPS_ROUTE_START,
        epsilon_vel=EPS_VEL_START,
        reward_weights=RewardWeights(alpha=0.4, beta=0.2, gamma=0.3, delta=0.1, arrival_bonus=50.0),
        batch_size=128,
        buffer_capacity=50_000,
        target_update_freq=500,
        train_freq=4,
        replay_start=500,
        device=_device,
    )

    if args.resume:
        controller.load(args.resume)
        print(f"Retomado de {args.resume}")

    renderer = None
    if not args.headless:
        from renderer.pygame_renderer import PygameRenderer
        renderer = PygameRenderer(
            graph=engine.graph,
            window_width=1400,
            window_height=900,
            graph_fraction=0.72,
            fps=args.fps,
            sub_steps=args.sub_steps,
        )

    total_cols = 0
    t_start = time.time()

    mode = "headless" if args.headless else "render"
    print(f"[{mode}] DQN flat training  robots={args.n_robots}  episodes={args.episodes}")
    print(f"ticks/ep={EPISODE_TICKS}  seed={args.seed}  workers={args.workers}")
    print(f"eps_route {EPS_ROUTE_START:.2f}→{EPS_ROUTE_END:.2f}  "
          f"eps_vel {EPS_VEL_START:.2f}→{EPS_VEL_END:.2f}  tau={tau_route:.0f}")
    print()

    label = f"{args.n_robots}R x {EPISODE_TICKS}t x {args.episodes}ep"

    # ------------------------------------------------------------------
    # Parallel training path
    # ------------------------------------------------------------------
    if args.workers > 1:
        from parallel_env import ParallelTrainer, get_weights_numpy, push_transitions

        _REWARD_WEIGHTS = dict(alpha=0.4, beta=0.2, gamma=0.3, delta=0.1, arrival_bonus=50.0)
        _base_cfg = dict(
            project_dir=os.path.dirname(os.path.abspath(__file__)),
            graph_path=GRAPH_PATH,
            pipeline_path=PIPELINE_PATH,
            vel_max=VEL_MAX,
            episode_ticks=EPISODE_TICKS,
            reward_weights=_REWARD_WEIGHTS,
        )
        trainer = ParallelTrainer(n_workers=args.workers, base_config=_base_cfg)

        pbar = None
        if _TQDM and args.headless:
            pbar = _tqdm(
                total=args.episodes, desc=label, unit="ep",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ep [{elapsed}<{remaining} {postfix}]",
            )
        else:
            print(f"== {label}")

        # Pre-warm: force all workers to initialise before the main loop.
        # On Windows (spawn), each worker imports torch + builds SimulationEngine;
        # without this the first run_batch() silently blocks for 1-2 min.
        print(f"  Initializing {args.workers} worker processes...", flush=True)
        _warmup_seeds = [0] * args.workers
        trainer.run_batch(controller, args.n_robots, EPS_ROUTE_START, EPS_VEL_START, _warmup_seeds)
        print(f"  Workers ready.", flush=True)

        global_episode = 0
        try:
            ep_done = 0
            while ep_done < args.episodes:
                batch = min(args.workers, args.episodes - ep_done)

                eps_r = DQNController.decay_epsilon(EPS_ROUTE_START, EPS_ROUTE_END, ep_done, tau_route)
                eps_v = DQNController.decay_epsilon(EPS_VEL_START,   EPS_VEL_END,   ep_done, tau_vel)
                controller.set_epsilon(eps_r, eps_v)

                seeds = [rng.randrange(2 ** 31) for _ in range(batch)]
                results = trainer.run_batch(controller, args.n_robots, eps_r, eps_v, seeds)

                batch_cols = 0
                batch_arrivals = 0
                for r in results:
                    push_transitions(controller, r["route_transitions"], r["vel_transitions"])
                    total_cols     += r["ep_cols"]
                    batch_cols     += r["ep_cols"]
                    batch_arrivals += r["ep_arrivals"]
                    global_episode += 1

                ep_done += batch
                col_rate = batch_cols / (batch * EPISODE_TICKS)
                sizes = controller.table_sizes()

                if pbar is not None:
                    pbar.update(batch)
                    pbar.set_postfix({
                        "col/t": f"{col_rate:.3f}",
                        "arr":   batch_arrivals,
                        "er":    f"{eps_r:.3f}",
                        "ev":    f"{eps_v:.3f}",
                        "rbuf":  sizes["route_buf"],
                        "vbuf":  sizes["vel_buf"],
                    })
                elif global_episode % args.log_every < batch:
                    elapsed = time.time() - t_start
                    print(
                        f"ep={global_episode:5d}  col/t={col_rate:.3f}  arr={batch_arrivals:3d}"
                        f"  er={eps_r:.3f}  ev={eps_v:.3f}"
                        f"  rbuf={sizes['route_buf']}  vbuf={sizes['vel_buf']}"
                        f"  {elapsed:.0f}s"
                    )

                if global_episode % args.save_every < batch:
                    ckpt = os.path.join("checkpoints", "motion", f"ep_{global_episode:05d}")
                    controller.save(ckpt)
                    msg = f"  -> checkpoint: {ckpt}"
                    if pbar is not None:
                        pbar.write(msg)
                    else:
                        print(msg)

                if args.eval_every > 0 and global_episode % args.eval_every < batch:
                    msg = f"  [eval] ep={global_episode}  {args.eval_ticks} ticks (epsilon=0)"
                    if pbar is not None:
                        pbar.write(msg)
                    else:
                        print(msg)
                    eval_result = run_eval(
                        engine=engine, controller=controller, rng=rng,
                        n_robots=args.n_robots, n_ticks=args.eval_ticks,
                        fps=args.fps, sub_steps=args.sub_steps,
                        global_episode=global_episode,
                    )
                    if eval_result == "quit":
                        raise KeyboardInterrupt

        except KeyboardInterrupt:
            print("\nTreino interrompido.")
        finally:
            if pbar is not None:
                pbar.close()
            trainer.close()

        ckpt = os.path.join("checkpoints", "motion", "final")
        controller.save(ckpt)
        if renderer:
            renderer.close()
        elapsed = time.time() - t_start
        sizes = controller.table_sizes()
        print(f"\nTreino terminado  ep={global_episode}  total_cols={total_cols}  {elapsed:.1f}s")
        print(
            f"route_buf={sizes['route_buf']}  vel_buf={sizes['vel_buf']}"
            f"  route_steps={sizes['route_steps']}  vel_steps={sizes['vel_steps']}"
        )
        print(f"Checkpoint final: {ckpt}")
        return

    # ------------------------------------------------------------------
    # Sequential training path (workers=1)
    # ------------------------------------------------------------------
    pbar = None
    if _TQDM and args.headless:
        pbar = _tqdm(
            total=args.episodes, desc=label, unit="ep",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ep [{elapsed}<{remaining} {postfix}]",
        )
    else:
        print(f"== {label}")

    global_episode = 0
    try:
        for episode in range(1, args.episodes + 1):
            global_episode += 1

            engine.reset()
            controller.reset_memory()
            assignments = add_random_robots(engine, rng, args.n_robots)
            robot_ids = [rid for rid, _ in assignments]

            map_nodes = all_nodes(engine)
            goals: Dict[str, str] = {
                rid: pick_goal(rng, map_nodes, spawn_node)
                for rid, spawn_node in assignments
            }

            for rid in robot_ids:
                if rng.random() < 0.4:
                    engine.world.robots[rid].carried_box = f"box_{rid}"

            eps_r = DQNController.decay_epsilon(EPS_ROUTE_START, EPS_ROUTE_END, episode, tau_route)
            eps_v = DQNController.decay_epsilon(EPS_VEL_START,   EPS_VEL_END,   episode, tau_vel)
            controller.set_epsilon(eps_r, eps_v)

            result = run_episode_train(
                engine=engine,
                controller=controller,
                robot_ids=robot_ids,
                goals=goals,
                rng=rng,
                renderer=renderer,
            )

            if result == "quit":
                raise KeyboardInterrupt

            ep_cols, ep_arrivals = result
            total_cols += ep_cols
            col_rate = ep_cols / EPISODE_TICKS

            if pbar is not None:
                pbar.update(1)
                sizes = controller.table_sizes()
                pbar.set_postfix({
                    "col/t": f"{col_rate:.3f}",
                    "arr":   ep_arrivals,
                    "er":    f"{eps_r:.3f}",
                    "ev":    f"{eps_v:.3f}",
                    "rbuf":  sizes["route_buf"],
                    "vbuf":  sizes["vel_buf"],
                })
            elif global_episode % args.log_every == 0:
                elapsed = time.time() - t_start
                sizes = controller.table_sizes()
                print(
                    f"ep={global_episode:5d}  col/t={col_rate:.3f}  arr={ep_arrivals:3d}"
                    f"  er={eps_r:.3f}  ev={eps_v:.3f}"
                    f"  rbuf={sizes['route_buf']}  vbuf={sizes['vel_buf']}"
                    f"  {elapsed:.0f}s"
                )

            if global_episode % args.save_every == 0:
                ckpt = os.path.join("checkpoints", "motion", f"ep_{global_episode:05d}")
                controller.save(ckpt)
                msg = f"  -> checkpoint: {ckpt}"
                if pbar is not None:
                    pbar.write(msg)
                else:
                    print(msg)

            if args.eval_every > 0 and global_episode % args.eval_every == 0:
                msg = f"  [eval] ep={global_episode}  {args.eval_ticks} ticks (epsilon=0)"
                if pbar is not None:
                    pbar.write(msg)
                else:
                    print(msg)
                eval_result = run_eval(
                    engine=engine, controller=controller, rng=rng,
                    n_robots=args.n_robots, n_ticks=args.eval_ticks,
                    fps=args.fps, sub_steps=args.sub_steps,
                    global_episode=global_episode,
                )
                if eval_result == "quit":
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\nTreino interrompido.")
    finally:
        if pbar is not None:
            pbar.close()

    ckpt = os.path.join("checkpoints", "motion", "final")
    controller.save(ckpt)

    if renderer:
        renderer.close()

    elapsed = time.time() - t_start
    sizes = controller.table_sizes()
    print(f"\nTreino terminado  ep={global_episode}  total_cols={total_cols}  {elapsed:.1f}s")
    print(
        f"route_buf={sizes['route_buf']}  vel_buf={sizes['vel_buf']}"
        f"  route_steps={sizes['route_steps']}  vel_steps={sizes['vel_steps']}"
    )
    print(f"Checkpoint final: {ckpt}")


if __name__ == "__main__":
    main()