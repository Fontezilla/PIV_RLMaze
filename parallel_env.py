"""
parallel_env.py — Parallel episode workers for DQN training.

Each worker process:
  1. Initialises SimulationEngine + DQNController ONCE (via executor initializer)
  2. Per episode: loads current weights, resets state, runs N ticks, returns transitions
  3. Returns raw transitions + episode stats — no training happens in workers

The main process:
  1. Dispatches a batch of N_WORKERS episodes simultaneously
  2. Receives transitions and pushes them into its own (GPU) replay buffers
  3. DQNAgent.push() triggers training as usual
  4. Updated weights are re-serialised to workers at the start of each batch

Only numpy arrays and plain Python types cross process boundaries —
no live torch tensors, no shared memory.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def get_weights_numpy(controller) -> Dict[str, Dict[str, "np.ndarray"]]:
    """Serialise online-network weights to CPU numpy arrays (picklable)."""
    def _to_np(sd):
        return {k: v.detach().cpu().numpy() for k, v in sd.items()}
    return {
        "route": _to_np(controller.dqn_route.online.state_dict()),
        "vel":   _to_np(controller.dqn_vel.online.state_dict()),
    }


def push_transitions(controller, route_transitions: list, vel_transitions: list) -> None:
    """Ingest worker-collected transitions into the main controller's replay buffers."""
    for t in route_transitions:
        controller.dqn_route.push(*t)
    for t in vel_transitions:
        controller.dqn_vel.push(*t)


# ---------------------------------------------------------------------------
# Lookahead buffer update — replicated from motion_train.py
# ---------------------------------------------------------------------------

def _update_lookahead_buffers(engine, ctrl, prev_obs, actions, goals, dispatch_results, events):
    from simulation_engine.core.entities import RobotState

    collided_ids = (
        {c.robot_a for c in events.collisions} |
        {c.robot_b for c in events.collisions}
    )

    for rid, obs in prev_obs.items():
        robot = engine.world.robots.get(rid)
        if robot is None:
            continue

        if obs.state == RobotState.IDLE:
            next_node, _ = actions.get(rid, (None, "hold"))
            result = dispatch_results.get(rid)

            if next_node is None:
                if robot.turn_cooldown == 0:
                    robot.buffered_next_node = None
                continue
            if (result is None
                    or not getattr(result, "accepted", False)
                    or not getattr(result, "dispatched", False)):
                robot.buffered_next_node = None
                continue
            if rid in collided_ids or robot.collided or obs.current_node is None:
                robot.buffered_next_node = None
                continue

            robot.buffered_next_node = ctrl.compute_buffered_next_node(
                current_node=obs.current_node,
                chosen_next=next_node,
                goal=goals[rid],
            )

        elif (obs.state == RobotState.MOVING
              and robot.state == RobotState.MOVING
              and (robot.from_node != obs.from_node or robot.to_node != obs.to_node)):

            if rid in collided_ids or robot.collided:
                robot.buffered_next_node = None
                continue
            if robot.from_node is None or robot.to_node is None:
                robot.buffered_next_node = None
                continue

            robot.buffered_next_node = ctrl.compute_buffered_next_node(
                current_node=robot.from_node,
                chosen_next=robot.to_node,
                goal=goals[rid],
            )


# ---------------------------------------------------------------------------
# Worker process — module-level state (initialised once per process)
# ---------------------------------------------------------------------------

_worker_engine = None
_worker_ctrl   = None
_worker_nodes  = None


def _worker_init(base_config: dict) -> None:
    """
    Called once when the worker process starts.
    Initialises SimulationEngine and DQNController so they are reused
    across all episodes assigned to this worker.
    """
    global _worker_engine, _worker_ctrl, _worker_nodes

    import sys
    sys.path.insert(0, base_config["project_dir"])

    from simulation_engine.simulation_engine import SimulationEngine
    from agents.motion_agent.dqn_controller import DQNController
    from agents.motion_agent.reward import RewardWeights

    _worker_engine = SimulationEngine(
        graph_path=base_config["graph_path"],
        pipeline_config_path=base_config["pipeline_path"],
        vel_max=base_config["vel_max"],
    )
    _worker_nodes = sorted(_worker_engine.graph.graph.nodes())

    rw = base_config["reward_weights"]
    _worker_ctrl = DQNController(
        graph=_worker_engine.graph,
        vel_max=base_config["vel_max"],
        gamma=0.95,
        buffer_capacity=20_000,
        replay_start=999_999_999,   # never trains in workers
        device="cpu",
        reward_weights=RewardWeights(
            alpha=rw["alpha"], beta=rw["beta"],
            gamma=rw["gamma"], delta=rw["delta"],
            arrival_bonus=rw["arrival_bonus"],
        ),
    )


# ---------------------------------------------------------------------------
# Worker — top-level function (must be a module-level def for pickling)
# ---------------------------------------------------------------------------

def _worker_run_episode(config: dict) -> dict:
    """
    Runs one training episode reusing the pre-initialised engine/controller.

    config keys
    -----------
    n_robots, epsilon_route, epsilon_vel
    weights_route, weights_vel  — dict[str, np.ndarray]  (online net only)
    episode_ticks, seed

    Returns
    -------
    dict with route_transitions, vel_transitions, ep_cols, ep_arrivals
    """
    import random as _rnd
    import torch
    from simulation_engine.core.entities import RobotState

    engine = _worker_engine
    ctrl   = _worker_ctrl
    nodes  = _worker_nodes

    # -------------------------------------------------- load current weights
    def _load_weights(agent, np_sd):
        agent.online.load_state_dict({k: torch.tensor(v) for k, v in np_sd.items()})

    _load_weights(ctrl.dqn_route, config["weights_route"])
    _load_weights(ctrl.dqn_vel,   config["weights_vel"])
    ctrl.set_epsilon(config["epsilon_route"], config["epsilon_vel"])

    # -------------------------------------------------- reset for new episode
    engine.reset()
    ctrl.reset_memory()
    ctrl.dqn_route.buffer._buf.clear()
    ctrl.dqn_vel.buffer._buf.clear()

    rng = _rnd.Random(config["seed"])
    n_robots = config["n_robots"]
    ROBOT_IDS = [f"R{i}" for i in range(8)]

    spawn_nodes = rng.sample(nodes, n_robots)
    robot_ids = []
    for rid, node in zip(ROBOT_IDS[:n_robots], spawn_nodes):
        engine.add_robot(rid, node)
        robot_ids.append(rid)

    goals = {
        rid: rng.choice([n for n in nodes if n != spawn]) if len(nodes) > 1 else spawn
        for rid, spawn in zip(robot_ids, spawn_nodes)
    }

    for rid in robot_ids:
        if rng.random() < 0.4:
            engine.world.robots[rid].carried_box = f"box_{rid}"

    # ---------------------------------------------------------------- tick loop
    iface = engine.interface
    ep_cols     = 0
    ep_arrivals = 0

    # build initial obs once; reuse curr_obs as prev_obs next tick
    curr_obs = {rid: iface.get_motion_view(rid) for rid in robot_ids}

    for _tick in range(config["episode_ticks"]):
        prev_obs = curr_obs

        actions = {}
        for rid in robot_ids:
            obs = prev_obs[rid]
            valid = iface.get_valid_actions(rid, obs)
            if obs.state == RobotState.IDLE and obs.current_node == goals[rid]:
                choices = [n for n in nodes if n != obs.current_node]
                goals[rid] = rng.choice(choices) if choices else obs.current_node
                engine.world.robots[rid].carried_box = f"box_{rid}" if rng.random() < 0.4 else None
                ep_arrivals += 1
            actions[rid] = ctrl.act(rid, obs, valid, goals[rid], engine.world)

        events, dispatch_results = engine.step(actions)
        ep_cols += len(events.collisions)

        _update_lookahead_buffers(
            engine, ctrl, prev_obs, actions, goals, dispatch_results, events
        )

        curr_obs = {rid: iface.get_motion_view(rid) for rid in robot_ids}

        for rid in robot_ids:
            ctrl.update(rid, prev_obs[rid], curr_obs[rid], events, goals[rid])

    # --------------------------------------------------- extract transitions
    return {
        "route_transitions": list(ctrl.dqn_route.buffer._buf),
        "vel_transitions":   list(ctrl.dqn_vel.buffer._buf),
        "ep_cols":           ep_cols,
        "ep_arrivals":       ep_arrivals,
    }


# ---------------------------------------------------------------------------
# Parallel Trainer
# ---------------------------------------------------------------------------

class ParallelTrainer:
    """
    Manages a pool of worker processes for parallel episode collection.

    Workers are initialised once per process (heavy resources: SimulationEngine,
    DQNController, shortest-path precomputation). Each call to run_batch()
    dispatches up to n_workers episodes simultaneously and blocks until all
    results are ready. Weights are re-synced at the start of every batch.
    """

    def __init__(self, n_workers: int, base_config: dict) -> None:
        self.n_workers   = n_workers
        self.base_config = base_config
        self._executor   = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(base_config,),
        )

    def run_batch(
        self,
        controller,
        n_robots:      int,
        epsilon_route: float,
        epsilon_vel:   float,
        seeds:         List[int],
    ) -> List[dict]:
        """Submit len(seeds) episodes in parallel and collect all results."""
        weights = get_weights_numpy(controller)
        configs = [
            {
                "n_robots":      n_robots,
                "epsilon_route": epsilon_route,
                "epsilon_vel":   epsilon_vel,
                "weights_route": weights["route"],
                "weights_vel":   weights["vel"],
                "episode_ticks": self.base_config["episode_ticks"],
                "seed":          seed,
            }
            for seed in seeds
        ]
        futures = [self._executor.submit(_worker_run_episode, c) for c in configs]
        return [f.result() for f in futures]

    def close(self) -> None:
        self._executor.shutdown(wait=True)
