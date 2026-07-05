"""Orquestrador do ambiente de fábrica — corre simulação e expõe estado ao agente RL."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from old.env.core.entities import Box, Robot, RobotState
from old.env.core.graph import FactoryGraph
from old.env.core.physics import MAX_SPEED, tick as physics_tick, turn_delay
from old.env.core.world import World
from old.env.core.box_manager import BoxManager
from old.env.traffic.router import Router, DEADLOCK_THRESHOLD


SPAWN_NODE    = "N"
SPAWN_STAGGER = 10
TICK_LIMIT    = 5_000

REWARD_PER_TICK     = -0.001
REWARD_DEADLOCK     = -0.5
REWARD_WAITING      = -0.02
REWARD_IDLE         = -0.005
# Distribuição: bónus por entregar num exit pouco usado, penalty se já saturado.
REWARD_DISTRIBUTION = 2.0
# Penalty quando o agent muda de assignment a meio do trajeto (anti-thrashing).
REWARD_ABORT        = -0.5


DEFAULT_MAP      = Path(".configs/map_factory.yaml")
DEFAULT_CACHE    = Path(".configs/graph_cache.pkl")
DEFAULT_PIPELINE = Path(".configs/box_pipeline.yaml")


def _dispatch_move(robot: Robot, next_node: str, graph: FactoryGraph) -> None:
    """Despacha um robot para se mover para `next_node`."""
    delay = (
        turn_delay(graph, robot.came_from, robot.current_node, next_node)
        if robot.current_node else 0
    )
    robot.from_node              = robot.current_node
    robot.to_node                = next_node
    robot.current_node           = None
    robot.progress               = 0.0
    robot.wait_ticks             = delay
    robot.turn_ticks_total       = delay
    robot.speed                  = 0.0 if delay > 0 else robot.speed
    robot.state                  = RobotState.MOVING
    robot.target_speed           = MAX_SPEED
    robot.wait_ticks_in_junction = 0
    robot.parked_at              = None


def _dispatch_park(robot: Robot, park_spec: str, router: Router) -> None:
    """Despacha um robot para parking point geométrico u|v|fraction."""
    u, v, frac_str = park_spec.split("|")
    fraction = float(frac_str)
    robot.from_node              = u
    robot.to_node                = v
    robot.current_node           = None
    robot.progress               = 0.0
    robot.wait_ticks             = 0
    robot.speed                  = 0.0
    robot.state                  = RobotState.MOVING
    robot.target_speed           = MAX_SPEED
    robot.wait_ticks_in_junction = 0
    robot.parked_at              = (u, v, fraction)
    router.node_lock.release(u, robot.id)


def _dispatch_unpark(
    robot       : Robot,
    target_node : str | None,
    router      : Router,
) -> None:
    """Tira o robot do parking em direcção a `target_node`."""
    if robot.parked_at is None:
        return
    u, v, frac = robot.parked_at
    if target_node == v:
        if not router.node_lock.try_acquire(v, robot.id):
            return
        robot.from_node    = u
        robot.to_node      = v
        robot.current_node = None
        robot.came_from    = u
        robot.progress     = frac
    else:
        if not router.node_lock.try_acquire(u, robot.id):
            return
        robot.from_node    = v
        robot.to_node      = u
        robot.current_node = None
        robot.came_from    = v
        robot.progress     = 1.0 - frac
    robot.wait_ticks             = 0
    robot.state                  = RobotState.MOVING
    robot.speed                  = 0.0
    robot.target_speed           = MAX_SPEED
    robot.wait_ticks_in_junction = 0
    router.unparked(robot)


class FactoryEnv:
    """Orquestrador do ambiente da fábrica (sim tick-a-tick + estado RL)."""

    def __init__(
        self,
        n_robots      : int           = 4,
        n_boxes       : int           = 8,
        tick_limit    : int           = TICK_LIMIT,
        map_path      : str | Path    = DEFAULT_MAP,
        cache_path    : str | Path    = DEFAULT_CACHE,
        pipeline_path : str | Path    = DEFAULT_PIPELINE,
        render_mode   : Optional[str] = None,
        seed          : Optional[int] = None,
    ) -> None:
        self.n_robots    = n_robots
        self.n_boxes     = n_boxes
        self.tick_limit  = tick_limit
        self.render_mode = render_mode
        self._rng        = random.Random(seed)

        self.graph = FactoryGraph(str(map_path), str(cache_path))

        # ── Caches imutáveis (topologia não muda entre episódios) ──────────
        # _graph_edge_features: lista completa, calculada uma vez.
        self._cached_graph_edges: list[dict] = [
            {"from": u, "to": v, "distance": data.get("distance", 1.0)}
            for u, v, data in self.graph.graph.edges(data=True)
        ]
        # _graph_node_features: partes estáticas por nó (type, pos, betweenness…).
        # Só n_robots_here e n_boxes_waiting são dinâmicas e calculadas por step.
        self._cached_node_static: dict[str, dict] = {}
        for _n in self.graph.all_nodes():
            _x, _y = self.graph.node_position(_n)
            _d2t   = self.graph.dist_to_type.get(_n, {})
            self._cached_node_static[_n] = {
                "type":            self.graph.node_type(_n),
                "x":               _x,
                "y":               _y,
                "betweenness":     self.graph.betweenness.get(_n, 0.0),
                "degree":          self.graph.degree_cache.get(_n, 1),
                "dist_to_exit":    _d2t.get("exit",            float("inf")),
                "dist_to_processA":_d2t.get("processA_entry",  float("inf")),
                "dist_to_processB":_d2t.get("processB_entry",  float("inf")),
            }

        self._box_manager = BoxManager(
            pipeline_path = pipeline_path,
            n_boxes       = n_boxes,
            seed          = seed,
        )

        self.world        : World  | None = None
        self.router       : Router | None = None
        self._spawn_ticks : list[int]     = []
        self._robots_list : list[Robot]   = []
        self._pending_robots: list[Robot] = []
        self._renderer = None

    def reset(
        self,
        seed    : Optional[int]  = None,
        options : Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """Reinicia o episódio; devolve (state, info)."""
        if seed is not None:
            self._rng = random.Random(seed)

        self.world  = World()
        self.router = Router(self.graph)

        self._box_manager.reset(seed=seed)

        self._spawn_ticks   = [i * SPAWN_STAGGER for i in range(self.n_robots)]
        self._robots_list   = [Robot(id=f"robot_{i}") for i in range(self.n_robots)]
        self._pending_robots = []

        self._run_until_decision_needed(accumulate_reward=False)

        state = self._build_state()
        info  = self._build_info()
        return state, info

    def step(
        self,
        assignments: dict[str, tuple[int, str] | None],
    ) -> tuple[dict, float, bool, bool, dict]:
        """Aplica assignments e avança até ao próximo evento de decisão."""
        assert self.world  is not None, "Chama reset() antes de step()."
        assert self.router is not None

        reward = self._apply_assignments(assignments)
        self._pending_robots.clear()
        reward += self._run_until_decision_needed(accumulate_reward=True)

        terminated = self._box_manager.all_delivered()
        truncated  = self.world.tick >= self.tick_limit

        state = self._build_state()
        info  = self._build_info(reward=reward)

        if self.render_mode == "human":
            self._do_render()

        return state, float(reward), terminated, truncated, info

    def _run_until_decision_needed(
        self,
        accumulate_reward: bool = True,
    ) -> float:
        """Corre ticks até haver robots pendentes ou terminação."""
        reward = 0.0
        while True:
            if self._pending_robots:
                break
            if self._box_manager.all_delivered():
                break
            if self.world.tick >= self.tick_limit:
                break

            tick_reward = self._tick()
            if accumulate_reward:
                reward += tick_reward

            if self.render_mode == "human":
                self._do_render()
        return reward

    def _tick(self) -> float:
        """Avança um tick da simulação; devolve o reward do tick."""
        assert self.world  is not None
        assert self.router is not None

        self.world.step_tick()
        self.router.begin_tick()   # limpa cache de congestionamento do tick anterior
        reward = 0.0

        self._maybe_spawn_robots()
        newly_spawned = self._box_manager.tick_spawn()

        # Atualiza priority de cada robot — quem carrega caixa quase a delivery
        # tem priority mais alta, e cede menos.
        self._update_priorities()

        for robot in self.world.all_robots():
            if robot.state == RobotState.WAITING:
                robot.state = RobotState.IDLE

        newly_available_boxes: list[int] = []
        any_pick_happened = False

        for robot in self.world.all_robots():
            arrived = physics_tick(self.graph, robot)
            if not arrived:
                continue

            if robot.state == RobotState.PARKED:
                if robot.parked_at is not None:
                    u, v, frac = robot.parked_at
                    self.router.parked(robot, u, v, frac)
                continue

            if robot.current_node is None:
                continue

            self.router.arrived(robot, robot.current_node)
            robot.wait_ticks_in_junction = 0

            drop_reward, delivered, box_id = self._box_manager.on_drop(
                robot, robot.current_node
            )
            reward += drop_reward

            if delivered:
                reward += self._delivery_distribution_bonus(robot.current_node)

            if box_id is not None:
                if not delivered:
                    newly_available_boxes.append(box_id)
                self._mark_robot_pending(robot)
                continue

            if robot.carrying_box is None and robot.current_node is not None:
                pick_reward = self._try_auto_pick(robot)
                reward += pick_reward
                if pick_reward > 0:
                    any_pick_happened = True

            if robot.reached_goal() and robot.carrying_box is None:
                self.router.release_all(robot)
                robot.last_visited.clear()
                robot.last_edges.clear()
                self.router.register(robot)
                # Limpa assigned_box_id stuck: chegou ao goal mas a box já não está
                # pickable cá (foi processada, picked por outro, delivered, etc).
                if robot.assigned_box_id is not None:
                    self._release_stuck_assignment(robot)
                if robot.assigned_box_id is None:
                    self._mark_robot_pending(robot)

        for box_id in newly_available_boxes:
            box = self._box_manager.get_box(box_id)
            if box is None or not box.is_available or box.current_node is None:
                continue
            for robot in self.world.all_robots():
                if (
                    robot.assigned_box_id == box_id
                    and robot.current_node == box.current_node
                    and robot.carrying_box is None
                    and (robot.is_idle() or robot.is_waiting())
                ):
                    pick_reward = self._try_auto_pick(robot)
                    reward += pick_reward
                    if pick_reward > 0:
                        any_pick_happened = True
                        break

        self.router.sync_moving_node_locks(self.world.all_robots())

        robots = self.world.all_robots()
        for robot in robots:
            if robot.state == RobotState.MOVING:
                continue
            # Robot IDLE em descanso (sem goal, ou já no goal) fica quieto —
            # não invocar router (sem goal devolve None e acumula deadlock).
            if robot.is_idle() and (
                robot.goal_node is None
                or robot.current_node == robot.goal_node
            ):
                continue

            decision = self.router.decide(robot, robots)
            if decision is None:
                robot.wait_ticks_in_junction += 1
                if robot.is_idle():
                    robot.state = RobotState.WAITING
                continue

            action, payload = decision
            if action == "move" and payload is not None:
                _dispatch_move(robot, payload, self.graph)
            elif action == "park" and payload is not None:
                _dispatch_park(robot, payload, self.router)
            elif action == "unpark":
                _dispatch_unpark(robot, payload, self.router)

        active = self.world.all_robots()
        reward += REWARD_PER_TICK * len(active)

        needed_nodes: set[str] = set()
        for r in active:
            if r.state == RobotState.WAITING:
                blocked_next = self.router.next_planned_node(r.id)
                if blocked_next is not None:
                    needed_nodes.add(blocked_next)

        for robot in active:
            truly_idle = (
                robot.is_idle()
                and robot.carrying_box is None
                and robot.assigned_box_id is None
                and robot.current_node is not None
            )
            blocking = truly_idle and robot.current_node in needed_nodes

            if blocking:
                reward += REWARD_IDLE

            if robot.state == RobotState.WAITING:
                reward += REWARD_WAITING
            if robot.wait_ticks_in_junction >= DEADLOCK_THRESHOLD:
                reward += REWARD_DEADLOCK

        # Apanha robots que já estavam no goal quando a box assigned mudou de
        # current_node (auto-advance, picked por outro, delivered).
        self._sweep_stuck_assignments(active)

        # Auto-park: robots IDLE blocking sem trabalho são deslocados pelo env
        # para parking points geométricos, libertando o nó.
        self._auto_park_blocking(active, needed_nodes)

        if newly_available_boxes or any_pick_happened or newly_spawned:
            self._replan_idle_robots()

        return reward

    def _update_priorities(self) -> None:
        """Define robot.priority por tick.

        - Sem caixa: 0.0
        - Com caixa: 1.0 + 0.5 * (steps_done/steps_total)
        Quanto mais perto do delivery, maior priority.
        """
        for robot in self.world.all_robots():
            if robot.carrying_box is None:
                robot.priority = 0.0
                continue
            box = self._box_manager.get_box(robot.carrying_box)
            if box is None:
                robot.priority = 1.0
                continue
            total = max(box.pipeline_total_steps, 1)
            progress = box.steps_done / total
            robot.priority = 1.0 + 0.5 * progress

    def _delivery_distribution_bonus(self, exit_node: str) -> float:
        """Bónus/penalty por equilíbrio de entregas pelos exits.

        Conta entregas por exit; compara com a média esperada. Sub-utilizado dá
        bónus positivo, saturado dá penalty. Chamado *depois* da box estar DONE,
        portanto subtrai 1 do exit actual para representar o "antes do delivery".
        """
        all_exits = [
            n for n in self.graph.all_nodes()
            if self.graph.node_type(n) == "exit"
        ]
        if not all_exits:
            return 0.0

        counts = {n: 0 for n in all_exits}
        for box in self._box_manager.boxes():
            if box.is_done and box.current_node in counts:
                counts[box.current_node] += 1

        # Antes do delivery actual
        counts[exit_node] = max(0, counts[exit_node] - 1)
        total = sum(counts.values())

        if total == 0:
            # Primeiro delivery do episódio → bónus pleno.
            return REWARD_DISTRIBUTION

        expected = total / len(all_exits)
        actual   = counts[exit_node]
        diff_norm = (expected - actual) / max(expected, 1.0)
        return REWARD_DISTRIBUTION * diff_norm

    def _auto_park_blocking(
        self,
        robots       : list[Robot],
        needed_nodes : set[str],
    ) -> None:
        """Auto-parka robots IDLE sem goal que bloqueiam ou ocupam nodes críticos.

        Parka se:
          (a) O nó está em needed_nodes (outro robot WAITING precisa dele), OU
          (b) O nó é especial (entry/exit/process) — ocupá-lo IDLE bloqueia
              futuras entregas/pickups. Reação imediata, sem esperar conflito.
        """
        for robot in robots:
            if not robot.is_idle():
                continue
            if robot.carrying_box is not None or robot.assigned_box_id is not None:
                continue
            if robot.goal_node is not None:
                continue
            if robot.current_node is None:
                continue

            is_blocking   = robot.current_node in needed_nodes
            is_in_special = self.graph.is_special(robot.current_node)
            if not (is_blocking or is_in_special):
                continue

            park_spec = self.router.find_parking_spec(robot, robots)
            if park_spec is None:
                continue
            _dispatch_park(robot, park_spec, self.router)
            robot.idle_acknowledged = False

    def _release_stuck_assignment(self, robot: Robot) -> None:
        """Se a box assigned ao robot já não está pickable no nó actual, limpa."""
        if robot.assigned_box_id is None or robot.current_node is None:
            return
        box = self._box_manager.get_box(robot.assigned_box_id)
        if (
            box is None
            or not box.is_available
            or box.current_node != robot.current_node
        ):
            robot.assigned_box_id = None

    def _sweep_stuck_assignments(self, robots: list[Robot]) -> None:
        """Detecta robots IDLE no goal com assigned stuck e re-pendê-los."""
        for robot in robots:
            if (
                robot.is_idle()
                and robot.carrying_box is None
                and robot.assigned_box_id is not None
                and robot.current_node is not None
                and robot.current_node == robot.goal_node
            ):
                self._release_stuck_assignment(robot)
                if robot.assigned_box_id is None:
                    self._mark_robot_pending(robot)

    def _try_auto_pick(self, robot: Robot) -> float:
        """Pick automático: caixa reservada se está cá, ou oportunista sem reserva."""
        if robot.current_node is None:
            return 0.0

        if robot.assigned_box_id is not None:
            box = self._box_manager.get_box(robot.assigned_box_id)
            if (
                box is not None
                and box.is_available
                and box.current_node == robot.current_node
            ):
                reward = self._box_manager.on_pick(robot, robot.assigned_box_id)
                if reward > 0:
                    robot.assigned_box_id = None
                    self._sync_goal_to_waypoint(robot)
                    return reward
            return 0.0

        if robot.carrying_box is not None:
            return 0.0

        all_robots = self.world.all_robots()
        assigned_to_others = {
            r.assigned_box_id
            for r in all_robots
            if r.id != robot.id and r.assigned_box_id is not None
        }

        for box in self._box_manager.available_boxes_at(robot.current_node):
            if box.box_id in assigned_to_others:
                continue
            next_wp = box.next_waypoint
            if next_wp is None:
                continue
            if not self._box_manager.is_node_drop_available(next_wp):
                continue

            reward = self._box_manager.on_pick(robot, box.box_id)
            if reward > 0:
                self._sync_goal_to_waypoint(robot)
                return reward

        return 0.0

    def _sync_goal_to_waypoint(self, robot: Robot) -> None:
        """Força goal_node para o próximo waypoint da caixa transportada."""
        if robot.carrying_box is None:
            return
        box = self._box_manager.get_box(robot.carrying_box)
        if box is None or box.next_waypoint is None:
            return
        target = box.next_waypoint
        if robot.goal_node != target:
            self.router.release_all(robot)
            robot.goal_node = target
            robot.last_visited.clear()
            robot.last_edges.clear()
            self.router.register(robot)

    def _mark_robot_pending(self, robot: Robot) -> None:
        """Marca robot como pendente de decisão."""
        if robot not in self._pending_robots:
            self._pending_robots.append(robot)

    def _replan_idle_robots(self) -> None:
        """Marca robots livres + em trânsito p/ pickup como pendentes.

        Chamado quando há novidade no mundo (newly_available_boxes, picks,
        spawns). Só reseta idle_acknowledged para robots que efectivamente
        se tornam pendentes — robots ocupados não são perturbados.
        """
        for robot in self.world.all_robots():
            if robot in self._pending_robots:
                continue
            if robot.is_free():
                robot.idle_acknowledged = False
                self._pending_robots.append(robot)
                continue
            # Em trânsito para pickup: tem assignment, ainda não tem caixa.
            # Pode ser necessário reatribuir se a box já não estiver disponível.
            if (
                robot.state == RobotState.MOVING
                and robot.carrying_box is None
                and robot.assigned_box_id is not None
            ):
                robot.idle_acknowledged = False
                self._pending_robots.append(robot)

    def _apply_assignments(self, assignments: dict[str, int | str]) -> float:
        """Aplica assignments {robot_id: (box_id, drop_node) | None}.

        2 passes:
          1. Box pickups — valida e define (box, waypoint, goal).
          2. Idle        — limpa estado para robots sem assignment.
        """
        reward = 0.0

        assigned_boxes          : set[int] = set()
        assigned_next_waypoints : set[str] = set()

        # Snapshot pre-reset por robot pendente, para detectar abort/changes
        # de assignment a meio do trajeto. Capturamos (box_id, target) onde
        # target = box.next_waypoint do prior assignment.
        prior_full: dict[str, tuple[int | None, str | None]] = {}
        for robot in self._pending_robots:
            if not self.world.has_robot(robot.id):
                continue
            prior_target: str | None = None
            if robot.assigned_box_id is not None:
                box = self._box_manager.get_box(robot.assigned_box_id)
                if box is not None:
                    prior_target = box.next_waypoint
            prior_full[robot.id] = (robot.assigned_box_id, prior_target)

        # Pre-limpa o assignment anterior em todos os pendentes (necessário para
        # had_box_assignment e para que o pass de boxes possa re-atribuir).
        # Se o robot estava a ir buscar uma box (sem ter pegado ainda), liberta
        # box.next_waypoint para a box voltar a ser candidate com novas opções.
        prior_had_assignment: dict[str, bool] = {}
        valid_pending: list[Robot] = []
        for robot in self._pending_robots:
            if not self.world.has_robot(robot.id):
                continue
            prior_had_assignment[robot.id] = robot.assigned_box_id is not None
            if (
                robot.assigned_box_id is not None
                and robot.carrying_box is None
            ):
                box = self._box_manager.get_box(robot.assigned_box_id)
                if box is not None and box.is_available:
                    box.next_waypoint = None
            robot.assigned_box_id = None
            valid_pending.append(robot)

        # ── Pass 1: box pickups (target = (box_id, drop_node)) ───────────
        for robot in valid_pending:
            target = assignments.get(robot.id)
            if not isinstance(target, tuple) or len(target) != 2:
                continue
            box_id, drop_node = target
            if not isinstance(box_id, int) or not isinstance(drop_node, str):
                robot.state = RobotState.IDLE
                continue

            box = self._box_manager.get_box(box_id)
            if box is None or not box.is_available or box_id in assigned_boxes:
                robot.state = RobotState.IDLE
                continue

            # Valida que drop_node é uma opção válida da pipeline da box.
            valid_targets = box.target_options()
            if drop_node not in valid_targets:
                robot.state = RobotState.IDLE
                continue

            if drop_node in assigned_next_waypoints:
                robot.state = RobotState.IDLE
                continue
            if not self._box_manager.is_node_drop_available(drop_node):
                robot.state = RobotState.IDLE
                continue
            assigned_next_waypoints.add(drop_node)

            # Set o destino escolhido pelo agent.
            box.next_waypoint = drop_node

            assigned_boxes.add(box_id)
            robot.assigned_box_id = box_id
            goal = box.current_node
            self._set_robot_goal(robot, goal)

            if robot.current_node == goal:
                reward += self._try_auto_pick(robot)

        # ── Pass 2: idle — robots sem box assignment ─────────────────────
        for robot in valid_pending:
            if robot.assigned_box_id is not None:
                continue
            # Preserva PARKED e MOVING (estados com física activa).
            if robot.state not in (RobotState.PARKED, RobotState.MOVING):
                robot.state = RobotState.IDLE
            robot.idle_acknowledged = True
            if prior_had_assignment.get(robot.id) or robot.goal_node is not None:
                self.router.release_all(robot)
                robot.goal_node = None
                robot.last_visited.clear()
                robot.last_edges.clear()
                self.router.register(robot)

        # Robots com box assignment saem do idle.
        for robot in valid_pending:
            if robot.assigned_box_id is not None:
                robot.idle_acknowledged = False

        # REWARD_ABORT: por cada robot que tinha assignment prior e mudou para
        # algo diferente (outra box, outro target, ou idle).
        for robot in valid_pending:
            prior = prior_full.get(robot.id)
            if prior is None:
                continue
            prior_box, prior_target = prior
            if prior_box is None:
                continue
            new_box    = robot.assigned_box_id
            new_target = None
            if new_box is not None:
                box = self._box_manager.get_box(new_box)
                if box is not None:
                    new_target = box.next_waypoint
            if (prior_box, prior_target) != (new_box, new_target):
                reward += REWARD_ABORT

        return reward

    def _set_robot_goal(self, robot: Robot, goal: str) -> None:
        """Define o goal do robot e regista no router.

        Preserva state=PARKED (router faz unpark) e state=MOVING (robot
        continua o trajeto e o router replan automaticamente para o novo goal).
        Só força IDLE se o robot estava em WAITING/IDLE.
        """
        self.router.release_all(robot)
        robot.goal_node = goal
        robot.last_visited.clear()
        robot.last_edges.clear()
        if robot.state not in (RobotState.PARKED, RobotState.MOVING):
            robot.state = RobotState.IDLE
        self.router.register(robot)

    def _maybe_spawn_robots(self) -> None:
        """Spawn de robots ainda não activos cujo tick chegou."""
        tick = self.world.tick
        for i, robot in enumerate(self._robots_list):
            if self.world.has_robot(robot.id):
                continue
            if tick < self._spawn_ticks[i]:
                continue

            robot.current_node           = SPAWN_NODE
            robot.state                  = RobotState.IDLE
            robot.came_from              = None
            robot.goal_node              = None
            robot.target_speed           = MAX_SPEED
            robot.speed                  = 0.0
            robot.wait_ticks             = 0
            robot.wait_ticks_in_junction = 0
            robot.parked_at              = None
            robot.carrying_box           = None

            x, y = self.graph.node_position(SPAWN_NODE)
            robot.world_x = x
            robot.world_y = y

            self.world.add_robot(robot)
            self.router.register(robot)
            self._mark_robot_pending(robot)

    def _build_state(self) -> dict:
        """Estado bruto para o agente construir o HeteroGraph."""
        all_robots = self.world.all_robots() if self.world else []
        pending_ids: set[str] = {r.id for r in self._pending_robots}

        already_assigned_boxes = {
            r.assigned_box_id
            for r in all_robots
            if r.assigned_box_id is not None and r.id not in pending_ids
        }

        # Candidatos (box_id, target_node) para o agent.
        in_transit_targets = {
            b.next_waypoint for b in self._box_manager.active_boxes()
            if b.is_in_transit and b.next_waypoint is not None
        }
        available_box_targets: list[tuple[int, str]] = []
        for box in self._box_manager.available_boxes():
            if box.box_id in already_assigned_boxes:
                continue
            for target in box.target_options():
                if target in in_transit_targets:
                    continue
                if not self._box_manager.is_node_drop_available(target):
                    continue
                available_box_targets.append((box.box_id, target))

        # Future paths dos robots (próximos K nodes planeados). O agent vê para
        # onde os outros vão para coordenar espacialmente.
        future_paths: dict[str, list[str]] = {}
        if self.router is not None:
            for r in all_robots:
                fp = self.router.peek_plan(r.id, k_steps=5)
                if fp:
                    future_paths[r.id] = fp

        return {
            "tick":               self.world.tick if self.world else 0,
            "robots":             self.world.snapshot()["robots"] if self.world else [],
            "boxes":              self._box_manager.snapshot(),
            "pending_robot_ids":  [r.id for r in self._pending_robots],
            "available_box_ids":  [
                b.box_id for b in self._box_manager.available_boxes()
                if b.box_id not in already_assigned_boxes
            ],
            "available_box_targets": available_box_targets,
            "future_paths":       future_paths,
            "graph_nodes":        self._graph_node_features(),
            "graph_edges":        self._graph_edge_features(),
        }

    def _graph_node_features(self) -> dict[str, dict]:
        """Features de cada nó do mapa.

        Partes estáticas (type, pos, betweenness, degree, dist_to_*)
        são lidas do cache pré-computado no __init__. Só as duas propriedades
        dinâmicas (n_robots_here, n_boxes_waiting) são calculadas por step.
        """
        world    = self.world
        features = {}
        for node, static in self._cached_node_static.items():
            features[node] = {
                **static,
                "n_robots_here":   len(world.robots_on_node(node)) if world else 0,
                "n_boxes_waiting": len(self._box_manager.available_boxes_at(node)),
            }
        return features

    def _graph_edge_features(self) -> list[dict]:
        """Arestas com distância (topologia estática — devolvido do cache)."""
        return self._cached_graph_edges

    def _build_info(self, reward: float = 0.0) -> dict:
        """Info dict para debug/logging."""
        if self.world is None:
            return {}
        return {
            **self.world.snapshot(),
            "boxes":              self._box_manager.snapshot(),
            "delivered":          self._box_manager.delivered_count(),
            "queued_boxes":       self._box_manager.queue_count(),
            "all_delivered":      self._box_manager.all_delivered(),
            "pending_robot_ids":  [r.id for r in self._pending_robots],
            "step_reward":        reward,
        }

    def render(self) -> None:
        if self.render_mode == "human":
            self._do_render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _do_render(self) -> None:
        if self._renderer is None:
            from old.render.renderer import Renderer
            self._renderer = Renderer(self.graph)

        info = {
            "tick":   self.world.tick,
            "robots": self.world.robot_count(),
            "boxes":  self._box_manager.snapshot(),
        }
        result = self._renderer.render(self.world, info)
        if result == "quit":
            self.close()
