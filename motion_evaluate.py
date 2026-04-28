import argparse
import csv
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

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
SEED = 42
EPISODE_TICKS = 1000

ROBOT_IDS = [f"R{i}" for i in range(8)]

_LOG_FIELDS = [
    "episode", "tick",
    "robot_id", "state",
    "current_node", "from_node", "to_node",
    "progress", "speed",
    "goal",
    "collided", "collision_this_tick",
    "carried_box", "arrived_this_tick",
    "idle_stuck_ticks",
]


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
    Ver motion_train.py para documentação completa.
    """
    collided_ids = {c.robot_a for c in events.collisions} | {c.robot_b for c in events.collisions}

    for rid, obs in prev_obs.items():
        robot = engine.world.robots.get(rid)
        if robot is None:
            continue

        # Caso A: robot estava IDLE — dispatch do Q-agent
        if obs.state == RobotState.IDLE:
            next_node, _speed_cmd = actions.get(rid, (None, "hold"))
            result = dispatch_results.get(rid)

            if next_node is None:
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

        # Caso B: robot estava MOVING e a aresta mudou — buffered dispatch disparou
        elif (obs.state == RobotState.MOVING
              and robot.state == RobotState.MOVING
              and (robot.from_node != obs.from_node or robot.to_node != obs.to_node)):

            if rid in collided_ids or robot.collided:
                robot.buffered_next_node = None
                continue

            if robot.from_node is None or robot.to_node is None:
                robot.buffered_next_node = None
                continue

            robot.buffered_next_node = controller.compute_buffered_next_node(
                current_node=robot.from_node,
                chosen_next=robot.to_node,
                goal=goals[rid],
            )


def run_episode_evaluate(
    engine,
    controller,
    robot_ids,
    goals,
    rng,
    episode_idx: int = 0,
    renderer=None,
    log_writer: Optional[csv.DictWriter] = None,
):
    """Corre um episódio de avaliação. Devolve (ep_cols, ep_arrivals) ou 'quit'."""
    iface = engine.interface
    map_nodes = all_nodes(engine)

    ep_cols = 0
    ep_arrivals = 0

    for tick in range(EPISODE_TICKS):
        prev_obs = {rid: iface.get_motion_view(rid) for rid in robot_ids}

        actions = {}
        arrived_this_tick: set = set()
        for rid in robot_ids:
            obs = prev_obs[rid]
            valid = iface.get_valid_actions(rid, obs)

            if obs.state == RobotState.IDLE and obs.current_node == goals[rid]:
                goals[rid] = pick_goal(rng, map_nodes, obs.current_node)
                ep_arrivals += 1
                arrived_this_tick.add(rid)

            actions[rid] = controller.act(rid, obs, valid, goals[rid], engine.world)

        events, dispatch_results = engine.step(actions)
        collided_ids = {c.robot_a for c in events.collisions} | {c.robot_b for c in events.collisions}
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

        if log_writer is not None:
            for rid in robot_ids:
                obs = prev_obs[rid]
                log_writer.writerow({
                    "episode":             episode_idx,
                    "tick":                tick,
                    "robot_id":            rid,
                    "state":               obs.state.name,
                    "current_node":        obs.current_node or "",
                    "from_node":           obs.from_node or "",
                    "to_node":             obs.to_node or "",
                    "progress":            f"{obs.progress:.4f}",
                    "speed":               f"{obs.speed:.2f}",
                    "goal":                goals[rid],
                    "collided":            int(obs.collided),
                    "collision_this_tick": int(rid in collided_ids),
                    "carried_box":         int(obs.carried_box is not None),
                    "arrived_this_tick":   int(rid in arrived_this_tick),
                    "idle_stuck_ticks":    obs.idle_stuck_ticks,
                })

        if renderer is not None:
            info = {
                "mode": "eval",
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
    parser.add_argument("--checkpoint", required=True, help="Diretório do checkpoint a carregar")
    parser.add_argument("--headless", action="store_true", help="Sem render pygame")
    parser.add_argument("--episodes", type=int, default=10, help="Nº de episódios")
    parser.add_argument("--n-robots", type=int, default=8, dest="n_robots",
                        help="Número de robots por episódio")
    parser.add_argument("--fps", type=int, default=60, help="FPS do renderer")
    parser.add_argument("--sub-steps", type=int, default=6, dest="sub_steps",
                        help="Sub-steps do renderer")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed da avaliação")
    parser.add_argument("--device", default="auto",
                        help="Dispositivo PyTorch: 'cpu', 'cuda', ou 'auto' (usa cuda se disponível)")
    parser.add_argument("--log", default=None, metavar="FILE",
                        help="Ficheiro CSV para gravar detalhe por robot/tick (ex: eval_log.csv)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    import torch as _torch
    if args.device == "auto":
        _device = "cuda" if _torch.cuda.is_available() else "cpu"
    else:
        _device = args.device

    engine = make_engine()
    controller = DQNController(
        graph=engine.graph,
        vel_max=VEL_MAX,
        gamma=0.95,
        epsilon_route=0.0,
        epsilon_vel=0.0,
        reward_weights=RewardWeights(alpha=0.4, beta=0.2, gamma=0.3, delta=0.1, arrival_bonus=50.0),
        device=_device,
    )
    controller.load(args.checkpoint)
    controller.set_epsilon(0.0, 0.0)

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
    total_arrivals = 0
    t_start = time.time()
    episodes_done = 0

    mode = "headless" if args.headless else "render"
    print(f"[{mode}] DQNController evaluation")
    print(f"checkpoint={args.checkpoint}  ticks/ep={EPISODE_TICKS}  robots={args.n_robots}")
    if args.log:
        print(f"log → {args.log}")
    print()

    log_file = None
    log_writer = None
    if args.log:
        log_file = open(args.log, "w", newline="", encoding="utf-8")
        log_writer = csv.DictWriter(log_file, fieldnames=_LOG_FIELDS)
        log_writer.writeheader()

    try:
        for episode in range(1, args.episodes + 1):
            engine.reset()
            controller.reset_memory()
            assignments = add_random_robots(engine, rng, args.n_robots)
            robot_ids = [rid for rid, _ in assignments]

            map_nodes = all_nodes(engine)
            goals: Dict[str, str] = {
                rid: pick_goal(rng, map_nodes, spawn_node)
                for rid, spawn_node in assignments
            }

            result = run_episode_evaluate(
                engine=engine,
                controller=controller,
                robot_ids=robot_ids,
                goals=goals,
                rng=rng,
                episode_idx=episode,
                renderer=renderer,
                log_writer=log_writer,
            )

            if result == "quit":
                break

            ep_cols, ep_arrivals = result
            total_cols += ep_cols
            total_arrivals += ep_arrivals
            episodes_done += 1

            col_rate = ep_cols / EPISODE_TICKS
            print(
                f"ep={episode:5d}  cols/tick={col_rate:.3f}  arrivals={ep_arrivals:3d}"
            )

    finally:
        if renderer:
            renderer.close()
        if log_file:
            log_file.close()

    elapsed = time.time() - t_start
    print(f"\nAvaliação terminada  ep={episodes_done}  {elapsed:.1f}s")
    print(f"Total collisions: {total_cols}")
    print(f"Total arrivals:   {total_arrivals}")
    if episodes_done > 0:
        print(f"Média arrivals/ep: {total_arrivals / episodes_done:.2f}")
        print(f"Média cols/ep:     {total_cols / episodes_done:.2f}")


if __name__ == "__main__":
    main()