"""
eval_checkpoints.py — Avalia checkpoints guardados e compara col/t e arrivals.

Uso:
    python eval_checkpoints.py [--checkpoints checkpoints/motion] [--episodes 20] [--n-robots 4]
    python eval_checkpoints.py --log eval_log.csv   # grava detalhe por tick em CSV
"""
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

from simulation_engine.simulation_engine import SimulationEngine
from simulation_engine.core.entities import RobotState
from agents.motion_agent import DQNController, RewardWeights

GRAPH_PATH    = os.path.join(os.path.dirname(__file__), "configs", "map_factory.yaml")
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), "configs", "box_pipeline.yaml")
VEL_MAX       = 30.0
EPISODE_TICKS = 1000
ROBOT_IDS     = [f"R{i}" for i in range(8)]

_LOG_FIELDS = [
    "checkpoint", "episode", "tick",
    "robot_id", "state",
    "current_node", "from_node", "to_node",
    "progress", "speed",
    "goal",
    "collided", "collision_this_tick",
    "carried_box", "arrived_this_tick",
]


def run_eval_episodes(controller, engine, n_robots, n_episodes, seed, log_writer=None, ckpt_name=""):
    rng = random.Random(seed)
    nodes = sorted(engine.graph.graph.nodes())
    iface = engine.interface

    total_cols     = 0
    total_arrivals = 0

    for ep in range(n_episodes):
        engine.reset()
        controller.reset_memory()

        spawn_nodes = rng.sample(nodes, n_robots)
        robot_ids = []
        for rid, node in zip(ROBOT_IDS[:n_robots], spawn_nodes):
            engine.add_robot(rid, node)
            robot_ids.append(rid)

        goals = {
            rid: rng.choice([n for n in nodes if n != sp])
            for rid, sp in zip(robot_ids, spawn_nodes)
        }

        for tick in range(EPISODE_TICKS):
            obs_map = {rid: iface.get_motion_view(rid) for rid in robot_ids}
            actions = {}
            arrived_this_tick = set()

            for rid in robot_ids:
                obs = obs_map[rid]
                valid = iface.get_valid_actions(rid, obs)
                if obs.state == RobotState.IDLE and obs.current_node == goals[rid]:
                    goals[rid] = rng.choice([n for n in nodes if n != obs.current_node])
                    total_arrivals += 1
                    arrived_this_tick.add(rid)
                actions[rid] = controller.act(rid, obs, valid, goals[rid], engine.world)

            events, _ = engine.step(actions)
            collided_ids = {c.robot_a for c in events.collisions} | {c.robot_b for c in events.collisions}
            total_cols += len(events.collisions)

            if log_writer is not None:
                for rid in robot_ids:
                    obs = obs_map[rid]
                    robot = engine.world.robots.get(rid)
                    log_writer.writerow({
                        "checkpoint":          ckpt_name,
                        "episode":             ep,
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
                    })

    col_per_tick = total_cols / (n_episodes * EPISODE_TICKS)
    arr_per_ep   = total_arrivals / n_episodes
    return col_per_tick, arr_per_ep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default="checkpoints/motion",
                        help="Diretório com checkpoints (default: checkpoints/motion)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episódios de eval por checkpoint (default: 20)")
    parser.add_argument("--n-robots", type=int, default=4, dest="n_robots")
    parser.add_argument("--log", default=None, metavar="FILE",
                        help="Ficheiro CSV para gravar detalhe por tick (ex: eval_log.csv)")
    parser.add_argument("--log-checkpoints", default=None, metavar="NAMES",
                        help="Subset de checkpoints a logar, separados por vírgula "
                             "(default: todos). Ex: ep_00250,ep_00500")
    args = parser.parse_args()

    ckpt_dir = args.checkpoints
    if not os.path.isdir(ckpt_dir):
        print(f"Diretório não encontrado: {ckpt_dir}")
        return

    ckpts = sorted([
        d for d in os.listdir(ckpt_dir)
        if os.path.isdir(os.path.join(ckpt_dir, d))
    ])

    if not ckpts:
        print("Nenhum checkpoint encontrado.")
        return

    log_filter = None
    if args.log_checkpoints:
        log_filter = set(args.log_checkpoints.split(","))

    engine = SimulationEngine(
        graph_path=GRAPH_PATH,
        pipeline_config_path=PIPELINE_PATH,
        vel_max=VEL_MAX,
    )
    controller = DQNController(
        graph=engine.graph,
        vel_max=VEL_MAX,
        gamma=0.95,
        buffer_capacity=1000,
        replay_start=999_999_999,
        device="cpu",
        reward_weights=RewardWeights(alpha=0.4, beta=0.2, gamma=0.3, delta=0.1, arrival_bonus=50.0),
    )
    controller.set_epsilon(0.0, 0.0)

    print(f"\n{'Checkpoint':<20} {'col/t':>8} {'arr/ep':>8}")
    print("-" * 40)

    log_file = None
    log_writer = None
    if args.log:
        log_file = open(args.log, "w", newline="", encoding="utf-8")
        log_writer = csv.DictWriter(log_file, fieldnames=_LOG_FIELDS)
        log_writer.writeheader()
        print(f"Logging detalhado → {args.log}")

    try:
        for ckpt_name in ckpts:
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            controller.load(ckpt_path)

            use_log = log_writer if (log_filter is None or ckpt_name in log_filter) else None

            col_t, arr_ep = run_eval_episodes(
                controller, engine,
                n_robots=args.n_robots,
                n_episodes=args.episodes,
                seed=999,
                log_writer=use_log,
                ckpt_name=ckpt_name,
            )
            print(f"{ckpt_name:<20} {col_t:>8.3f} {arr_ep:>8.1f}")
    finally:
        if log_file:
            log_file.close()

    print()
    if args.log:
        print(f"Log gravado em: {args.log}")


if __name__ == "__main__":
    main()
