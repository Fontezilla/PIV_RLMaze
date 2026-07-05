"""FactoryEnv — ambiente RL orientado a EVENTOS sobre o Space-Time A* (SIPP).

Ciclo: o agente atribui a cada robot livre uma tarefa (caixa, nó-destino);
o env planeia o percurso `posição → nó-da-caixa (pick) → destino (drop)` com
o SIPP, reserva-o, e avança o relógio até ao próximo robot ficar livre,
aplicando pick/drop e recompensas pelo caminho. É a "verdade" exacta (com
tráfego); treina-se directamente aqui (sem tabela de ETA — ver memória).

v1: pick/drop instantâneos; contrato de estado ainda mínimo (será alinhado
ao contrato do agente GNN quando o ligarmos).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from env.core.graph import FactoryGraph
from env.core.box_manager import BoxManager
from env.physics import rules, engine
from env.physics.rules import Heading
from env.traffic import cooperative_astar as cts
from env.traffic.reservations import Interval, ReservationTable


REWARD_PER_TICK = -0.001
"""Pressão temporal: penalização por robot activo e por tick."""

BUFFER = 5.0
"""Folga espaço-temporal do SIPP (traduz-se em ~45 unidades entre centros;
ver renderer para o raio físico de 12)."""


@dataclass
class RobotState:
    """Estado de um robot no env. `node` é o nó actual quando livre (None em
    trânsito); `carrying` a caixa que leva; `assigned` a tarefa (box_id,
    target) em execução; `busy_until` o instante em que fica livre; `events`
    a lista de (tempo, 'pick'|'drop', box_id, nó) do percurso; `end_state` o
    (nó, orientação) em que fica ao terminar."""
    robot_id: str
    node: str | None
    heading: Heading
    speed: float = 0.0
    carrying: int | None = None
    assigned: tuple[int, str] | None = None
    busy_until: float = 0.0
    player: engine.SchedulePlayer | None = None
    events: list = field(default_factory=list)
    end_state: tuple[str, Heading] | None = None

    @property
    def free(self) -> bool:
        """True se o robot não tem tarefa atribuída (pode receber uma)."""
        return self.assigned is None


class FactoryEnv:
    """Ambiente RL orientado a eventos: o agente atribui robots a caixas, o
    env planeia (SIPP) e simula, e devolve estado/recompensa/terminação."""

    def __init__(
        self,
        n_robots: int = 2,
        n_boxes: int = 4,
        tick_limit: int = 3000,
        map_path: str | Path = "../.configs/map_factory.yaml",
        cache_path: str | Path = ".configs/graph_cache.pkl",
        pipeline_path: str | Path = "../.configs/box_pipeline.yaml",
        seed: Optional[int] = None,
        record: bool = False,
    ) -> None:
        """Carrega grafo e pipelines. `record=True` guarda trajectórias e um
        log de caixas para reprodução no renderer."""
        self.n_robots = n_robots
        self.n_boxes = n_boxes
        self.tick_limit = tick_limit
        self.graph = FactoryGraph(str(map_path), str(cache_path))
        self._box_manager = BoxManager(pipeline_path, n_boxes=n_boxes, seed=seed)
        self._seed = seed
        self._start_nodes = [n for n in ("E", "G", "J", "M", "O") if self.graph.has_edge(n, self.graph.neighbors(n)[0])]

        self.clock: float = 0.0
        self.robots: list[RobotState] = []
        self.reservations = ReservationTable()

        self.record = record
        self.render_players: dict[str, engine.SchedulePlayer] = {}
        self.box_log: list[tuple[float, list[dict]]] = []

    def reset(self, seed: Optional[int] = None) -> tuple[dict, dict]:
        """Reinicia caixas, reservas e robots (nas posições de partida) e
        devolve o estado inicial."""
        self._box_manager.reset(seed if seed is not None else self._seed)
        self.reservations.reset()
        self.clock = 0.0

        real = [n for n in self.graph.all_nodes() if self.graph.is_turn_node(n)]
        starts = self._start_nodes if len(self._start_nodes) >= self.n_robots else real
        self.robots = []
        for i in range(self.n_robots):
            node = starts[i % len(starts)]
            self.robots.append(RobotState(robot_id=f"robot_{i}", node=node, heading=Heading.initial()))
            self.reservations.reserve_node(node, Interval(0.0, float("inf")), f"robot_{i}")

        if self.record:
            self.render_players = {
                r.robot_id: engine.SchedulePlayer(
                    r.robot_id, self.graph,
                    [cts.ScheduleEntry(r.node, 0.0, 0.0, 0.0)], carrying_box=False,
                )
                for r in self.robots
            }
            self.box_log = [(0.0, self._box_manager.snapshot())]

        return self._build_state(), {}

    def step(self, assignments: dict[str, tuple[int, str] | None]) -> tuple[dict, float, bool, bool, dict]:
        """Aplica as atribuições aos robots livres, avança até ao próximo
        robot ficar livre, e devolve (estado, recompensa, terminated,
        truncated, info). `assignments` mapeia robot_id → (box_id, target)."""
        reward = 0.0

        for robot in self.robots:
            if not robot.free:
                continue
            action = assignments.get(robot.robot_id)
            if action is None:
                continue
            box_id, target = action
            if not self._start_task(robot, box_id, target):
                continue

        reward += self._advance()

        terminated = self._box_manager.all_delivered()
        truncated = self.clock >= self.tick_limit
        return self._build_state(), reward, terminated, truncated, {}

    def _start_task(self, robot: RobotState, box_id: int, target: str) -> bool:
        """Planeia e reserva o percurso pick+carry de um robot: perna 1
        (posição→caixa, vazio, paragem intermédia) e perna 2 (caixa→destino,
        com caixa, estaciona no fim). Devolve False se não houver caminho."""
        box = self._box_manager.get_box(box_id)
        if box is None or not box.is_available or target not in box.target_options():
            return False
        pick_node = box.current_node
        if pick_node is None:
            return False

        self.reservations.clear_robot(robot.robot_id)

        seg1 = cts.plan(self.graph, robot.node, pick_node, carrying_box=False,
                        start_time=self.clock, initial_heading=robot.heading,
                        initial_speed=robot.speed, reservations=self.reservations,
                        robot_id=robot.robot_id, buffer=BUFFER, park=False)
        if seg1 is None:
            self._rehold(robot)
            return False
        path1, t_pick, sched1 = seg1
        heading1 = self._final_heading(path1, robot.heading)

        seg2 = cts.plan(self.graph, pick_node, target, carrying_box=True,
                        start_time=t_pick, initial_heading=heading1,
                        initial_speed=0.0, reservations=self.reservations,
                        robot_id=robot.robot_id, buffer=BUFFER, park=True)
        if seg2 is None:
            self.reservations.clear_robot(robot.robot_id)
            self._rehold(robot)
            return False
        path2, t_drop, sched2 = seg2

        box.next_waypoint = target
        box.carried_by = robot.robot_id

        robot.assigned = (box_id, target)
        robot.busy_until = t_drop
        robot.node = None
        robot.events = [(t_pick, "pick", box_id, pick_node), (t_drop, "drop", box_id, target)]
        robot.player = engine.SchedulePlayer(robot.robot_id, self.graph, sched1, carrying_box=False)
        robot.player.extend(sched2)
        robot.end_state = (target, self._final_heading(path2, heading1))

        if self.record:
            rp = self.render_players[robot.robot_id]
            rp.extend(sched1)
            rp.extend(sched2)
        return True

    def _advance(self) -> float:
        """Avança o relógio até o próximo robot ficar livre, aplicando os
        eventos (pick/drop) e a pressão temporal. Devolve a recompensa."""
        busy = [r for r in self.robots if not r.free]
        if not busy:
            self._box_manager.tick_spawn()
            return 0.0

        next_t = min(r.busy_until for r in busy)
        dt = next_t - self.clock
        reward = REWARD_PER_TICK * len(busy) * max(dt, 0.0)

        due = sorted(
            ((t, r, kind, box_id, node)
             for r in busy for (t, kind, box_id, node) in r.events if t <= next_t + 1e-9),
            key=lambda e: e[0],
        )
        for (t, robot, kind, box_id, node) in due:
            reward += self._apply_event(robot, kind, box_id, node)
            if self.record:
                self.box_log.append((t, self._box_manager.snapshot()))
        for robot in busy:
            robot.events = [e for e in robot.events if e[0] > next_t + 1e-9]

        self.clock = next_t

        for robot in busy:
            if robot.busy_until <= self.clock + 1e-9 and not robot.events:
                end_node, end_heading = robot.end_state
                robot.node = end_node
                robot.heading = end_heading
                robot.speed = 0.0
                robot.assigned = None
                robot.carrying = None

        self._box_manager.tick_spawn()
        return reward

    def _apply_event(self, robot: RobotState, kind: str, box_id: int, node: str) -> float:
        """Aplica um evento de pick ou drop e devolve a recompensa dele."""
        if kind == "pick":
            r = self._box_manager.on_pick(robot.robot_id, box_id, node)
            robot.carrying = box_id
            return r
        r, _delivered = self._box_manager.on_drop(box_id, node)
        robot.carrying = None
        return r

    def _rehold(self, robot: RobotState) -> None:
        """Volta a reservar (indefinidamente) o nó de um robot que ficou
        idle por não ter conseguido planear."""
        if robot.node is not None:
            self.reservations.reserve_node(robot.node, Interval(self.clock, float("inf")), robot.robot_id)

    def _final_heading(self, path: list[str], initial: Heading) -> Heading:
        """Orientação do robot no fim de um caminho, partindo de `initial`."""
        h = initial
        for a, b in zip(path, path[1:]):
            h = rules.advance_heading(self.graph, h, a, b)
        return h

    def _build_state(self) -> dict:
        """Constrói o dicionário de estado devolvido ao agente (robots livres
        pendentes, candidatos caixa→destino, snapshot das caixas)."""
        available = [
            (b.box_id, opt)
            for b in self._box_manager.available_boxes()
            for opt in b.target_options()
        ]
        pending = [r.robot_id for r in self.robots if r.free] if available else []
        return {
            "clock": self.clock,
            "robots": [
                {"id": r.robot_id, "node": r.node, "carrying": r.carrying,
                 "assigned": r.assigned, "free": r.free}
                for r in self.robots
            ],
            "boxes": self._box_manager.snapshot(),
            "pending_robot_ids": pending,
            "available_box_targets": available,
            "delivered": self._box_manager.delivered_count(),
        }
