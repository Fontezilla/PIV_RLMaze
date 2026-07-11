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
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from env.core.graph import FactoryGraph
from env.core.box_manager import BoxManager
from env.core.entities import BoxStatus
from env.physics import rules, engine
from env.physics.rules import Heading
from env.traffic import cooperative_astar as cts
from env.traffic.reservations import Interval, ReservationTable


REWARD_PER_TICK = -0.001
"""Pressão temporal: penalização por robot activo e por tick."""

BUFFER = 5.0
"""Folga espaço-temporal do SIPP (traduz-se em ~45 unidades entre centros;
ver renderer para o raio físico de 12)."""

PRIORITY_IDLE = 0.0
PRIORITY_NO_BOX = 1.0
PRIORITY_WITH_BOX = 2.0
PRIORITY_SPAWNING = float("inf")
"""Pesos da prioridade de conflito: idle sempre cede primeiro; a nascer
sempre vence; robots activos pesam pela distância restante até ao próximo
evento (pick/drop), com o dobro do peso se transportam caixa."""

SPAWN_NODE = "N"
"""Único ponto de spawn: todos os robots nascem aqui, um de cada vez."""

SPAWN_BUFFER = 20.0
"""Cooldown depois de `SPAWN_NODE` ficar livre — aplica-se sempre que
QUALQUER robot passa por lá (nascimento ou trânsito normal), não só ao
nascer. O primeiro robot do episódio também espera este tempo (uma reserva
"fantasma" em t=0 força isso, ver `reset`)."""


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
        map_path: str | Path = ".configs/map_factory.yaml",
        cache_path: str | Path = ".configs/graph_cache.pkl",
        pipeline_path: str | Path = ".configs/box_pipeline.yaml",
        seed: Optional[int] = None,
        record: bool = False,
        box_layout: str | None = None,
        robots_min: Optional[int] = None,
        robots_max: Optional[int] = None,
        boxes_min: Optional[int] = None,
        boxes_max: Optional[int] = None,
    ) -> None:
        """Carrega grafo e pipelines. `record=True` guarda trajectórias e um
        log de caixas para reprodução no renderer.

        Dois modos, decididos por `box_layout`:
        - **específico** (`box_layout` dado, ex. "BB RG GG B"): entry+cor de
          cada caixa fixos por slot, `n_robots` fixo, total de caixas vem do
          layout.
        - **aleatório** (`box_layout` vazio/None): cores aleatórias e, a cada
          `reset`, o nº de robots é amostrado em [robots_min, robots_max] e o
          de caixas em [boxes_min, boxes_max] (domain randomization). Se os
          limites não forem dados, degeneram para `n_robots`/`n_boxes` fixos."""
        self.n_robots = n_robots
        self.tick_limit = tick_limit
        self.graph = FactoryGraph(str(map_path), str(cache_path))
        self._box_manager = BoxManager(pipeline_path, n_boxes=n_boxes, seed=seed,
                                       layout=box_layout)
        self.n_boxes = self._box_manager.n_boxes
        self._seed = seed

        self._random_mode = not box_layout
        self._robots_min = robots_min if robots_min is not None else n_robots
        self._robots_max = robots_max if robots_max is not None else n_robots
        self._boxes_min  = boxes_min  if boxes_min  is not None else n_boxes
        self._boxes_max  = boxes_max  if boxes_max  is not None else n_boxes
        self._sample_rng = random.Random(seed)

        self.clock: float = 0.0
        self.robots: list[RobotState] = []
        self.reservations = ReservationTable()
        self._spawn_queue: list[str] = []

        self.record = record
        self.render_players: dict[str, engine.SchedulePlayer] = {}
        self.box_log: list[tuple[float, list[dict]]] = []

        # Topologia "real" (sem sub-nós de corredor) para o agente — estática,
        # calculada uma só vez, já que o mapa nunca muda durante o episódio.
        self._real_nodes = self.graph.real_nodes()
        self._real_edges = [{"from": u, "to": v} for u, v in self.graph.real_edges()]

    def reset(self, seed: Optional[int] = None) -> tuple[dict, dict]:
        """Reinicia caixas e reservas. Nenhum robot existe ainda — todos
        ficam numa fila de spawn em `SPAWN_NODE`, um de cada vez, com
        `SPAWN_BUFFER` ticks de intervalo desde o último uso do nó (ver
        `_next_spawn_time`) — `_n_last_used=0.0` força mesmo o primeiro
        robot a esperar esse intervalo. Avança logo o primeiro nascimento
        possível, para o estado devolvido já ter pelo menos um robot
        pendente."""
        # Modo aleatório: amostra nº de robots e caixas por episódio.
        if self._random_mode:
            self.n_robots = self._sample_rng.randint(self._robots_min, self._robots_max)
            n_boxes = self._sample_rng.randint(self._boxes_min, self._boxes_max)
            self._box_manager.reset(seed if seed is not None else self._seed, n_boxes=n_boxes)
        else:
            self._box_manager.reset(seed if seed is not None else self._seed)
        self.n_boxes = self._box_manager.n_boxes
        self.reservations.reset()
        self.clock = 0.0

        self._spawn_queue = [f"robot_{i}" for i in range(self.n_robots)]
        self._n_last_used = 0.0
        self.robots = []

        if self.record:
            self.render_players = {}
            self.box_log = [(0.0, self._box_manager.snapshot())]

        self._advance()
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

    def close(self) -> None:
        """Sem recursos externos a libertar — existe por compatibilidade
        com a convenção `gym.Env`."""
        return None

    def _start_task(self, robot: RobotState, box_id: int, target: str) -> bool:
        """Planeia e reserva o percurso pick+carry de um robot: perna 1
        (posição→caixa, vazio, paragem intermédia) e perna 2 (caixa→destino,
        com caixa, estaciona no fim). Tenta primeiro FCFS puro (mais barato,
        cobre o caso comum); só recorre a prioridade (podendo atropelar
        robots PARADOS de prioridade menor) se isso falhar — ex.: o destino
        está ocupado por um robot idle a descansar ali. Devolve False se
        impossível de planear mesmo com prioridade."""
        box = self._box_manager.get_box(box_id)
        if box is None or not box.is_available or target not in box.target_options():
            return False
        pick_node = box.current_node
        if pick_node is None:
            return False

        self.reservations.clear_robot(robot.robot_id)

        victims = self._commit_journey(
            robot, robot.node, self.clock, robot.speed, robot.heading,
            pick_node, target, box_id, use_priority=False,
        )
        if victims is None:
            victims = self._commit_journey(
                robot, robot.node, self.clock, robot.speed, robot.heading,
                pick_node, target, box_id, use_priority=True,
            )
        if victims is None:
            self._rehold(robot)
            return False

        for victim_id in victims:
            self._replan_victim(victim_id)
        return True

    def _commit_journey(
        self,
        robot: RobotState,
        src: str,
        start_time: float,
        start_speed: float,
        start_heading: Heading,
        pick_node: str,
        target: str,
        box_id: int,
        use_priority: bool,
    ) -> set[str] | None:
        """Planeia e comete a viagem completa de `robot`: perna vazia
        `src`→`pick_node` (paragem intermédia) + perna com caixa
        `pick_node`→`target` (estaciona no fim). Actualiza o RobotState e
        (se `record`) o histórico de render. Devolve o conjunto de robots
        atropelados — sempre robots PARADOS num nó, nunca a meio de uma
        aresta (arestas são sempre FCFS puro, ver `cooperative_astar.plan`)
        — ou None se impossível de planear."""
        priorities = self._all_priorities(exclude={robot.robot_id}) if use_priority else None

        req_p1 = (PRIORITY_NO_BOX * self.graph.shortest_distance(src, pick_node)
                  if use_priority else float("inf"))
        seg1 = cts.plan(self.graph, src, pick_node, carrying_box=False,
                        start_time=start_time, initial_heading=start_heading,
                        initial_speed=start_speed, reservations=self.reservations,
                        robot_id=robot.robot_id, buffer=BUFFER, park=False,
                        priorities=priorities, requester_priority=req_p1, now=self.clock)
        if seg1 is None:
            return None
        path1, t_pick, sched1, victims1 = seg1
        heading1 = self._final_heading(path1, start_heading)

        req_p2 = (PRIORITY_WITH_BOX * self.graph.shortest_distance(pick_node, target)
                  if use_priority else float("inf"))
        seg2 = cts.plan(self.graph, pick_node, target, carrying_box=True,
                        start_time=t_pick, initial_heading=heading1,
                        initial_speed=0.0, reservations=self.reservations,
                        robot_id=robot.robot_id, buffer=BUFFER, park=True,
                        priorities=priorities, requester_priority=req_p2, now=self.clock)
        if seg2 is None:
            self.reservations.clear_robot(robot.robot_id)
            return None
        path2, t_drop, sched2, victims2 = seg2

        box = self._box_manager.get_box(box_id)
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
            rp = self.render_players.get(robot.robot_id)
            if rp is not None:
                rp.truncate(start_time)
                rp.extend(sched1)
                rp.extend(sched2)

        return victims1 | victims2

    def _resting_state(self, robot: RobotState) -> tuple[str, Heading] | None:
        """Nó onde `robot` está parado agora, e a sua orientação. Só faz
        sentido chamar isto sobre uma vítima de atropelamento — que está
        sempre parada num nó, nunca a meio de uma aresta (arestas são
        sempre FCFS puro). Devolve None no caso inesperado de, por engano,
        estar em movimento."""
        if robot.player is None:
            return robot.node, robot.heading
        state = robot.player.state_at(self.clock)
        if state["status"] == "MOVING":
            return None
        return state["from_node"], self._heading_at(robot, self.clock)

    def _replan_victim(self, robot_id: str, depth: int = 1) -> None:
        """Robot `robot_id` foi atropelado por uma reserva de prioridade
        maior num nó onde estava parado (idle, ou só uma pausa breve entre
        troços de uma tarefa). Aborta de forma limpa qualquer tarefa em
        curso — a caixa que levava volta a ficar disponível — e foge por
        BFS para o nó seguro mais próximo, respeitando quem tem prioridade
        igual ou maior (nunca atropela ninguém: a sua prioridade durante a
        manobra é sempre 0, a mais baixa). Fica idle ao chegar lá. Se não
        houver nenhum refúgio alcançável, fica onde está."""
        if depth > 8:
            return
        robot = next((r for r in self.robots if r.robot_id == robot_id), None)
        if robot is None:
            return
        resting = self._resting_state(robot)
        if resting is None:
            return
        src, start_heading = resting

        if robot.assigned is not None and robot.assigned[0] is not None:
            box_id, _target = robot.assigned
            box = self._box_manager.get_box(box_id)
            if box is not None:
                box.carried_by = None
                box.next_waypoint = None
                if robot.carrying is not None:
                    box.status = BoxStatus.WAITING
                    box.current_node = src
            robot.carrying = None

        priorities = self._all_priorities(exclude={robot.robot_id})
        self.reservations.clear_robot(robot_id)

        result = None
        visited = {src}
        frontier = [src]
        for _ in range(8):
            next_frontier: list[str] = []
            for node in frontier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
                    result = cts.plan(self.graph, src, neighbor, carrying_box=False,
                                      start_time=self.clock, initial_heading=start_heading,
                                      initial_speed=0.0, reservations=self.reservations,
                                      robot_id=robot_id, buffer=BUFFER, park=True,
                                      priorities=priorities, requester_priority=PRIORITY_IDLE,
                                      now=self.clock)
                    if result is not None:
                        break
                if result is not None:
                    break
            if result is not None or not next_frontier:
                break
            frontier = next_frontier

        if result is None:
            # Sem fuga possível — fica em `src`. Verifica em FCFS puro (sem
            # atropelar ninguém, já não há prioridade a aplicar aqui) se
            # `src` está mesmo livre para sempre a partir de agora; se
            # outro robot já tiver um plano futuro comprometido para esse
            # nó, o "ficar aqui" tem de esperar por essa janela + buffer,
            # em vez de reservar `self.clock` às cegas (colidiria). Se há
            # de esperar (`settle_t` > agora), fica "ocupado à espera"
            # (como uma evacuação normal) até lá — nunca livre demasiado
            # cedo, senão uma tarefa nova partiria com `self.clock`
            # desalinhado da reserva.
            intervals = self.reservations.safe_intervals(src, robot_id, buffer=BUFFER, now=self.clock)
            settle_t = max(intervals[-1][0], self.clock)
            self.reservations.reserve_node(src, Interval(settle_t, float("inf")), robot_id)
            robot.assigned = None if settle_t <= self.clock + 1e-9 else (None, src)
            robot.node = src if settle_t <= self.clock + 1e-9 else None
            robot.heading = start_heading
            robot.speed = 0.0
            robot.events = []
            robot.busy_until = settle_t
            robot.end_state = (src, start_heading)
            if self.record:
                # corta o histórico de render em `self.clock` (AGORA — não
                # em `settle_t`, que pode ser mais tarde; o plano antigo
                # abandonado nunca chega a acontecer) e regista a espera
                # em `src` até `settle_t` como uma pausa explícita.
                rp = self.render_players.get(robot_id)
                if rp is not None:
                    rp.truncate(self.clock)
                    rp.extend([cts.ScheduleEntry(src, self.clock, settle_t, 0.0)])
            return

        path, t_end, sched, victims = result
        new_heading = self._final_heading(path, start_heading)

        robot.assigned = (None, path[-1])
        robot.busy_until = t_end
        robot.node = None
        robot.events = []
        robot.player = engine.SchedulePlayer(robot_id, self.graph, sched, carrying_box=False)
        robot.end_state = (path[-1], new_heading)

        if self.record:
            rp = self.render_players.get(robot_id)
            if rp is not None:
                rp.truncate(self.clock)
                rp.extend(sched)

        for victim_id in victims:
            self._replan_victim(victim_id, depth + 1)

    def _advance(self) -> float:
        """Avança o relógio até o próximo robot ficar livre, aplicando os
        eventos (pick/drop), o próximo nascimento em `SPAWN_NODE` (se
        houver fila), o fim de processamento (par entry->exit, se alguma
        caixa estiver a processar) e a pressão temporal. Devolve a
        recompensa."""
        busy = [r for r in self.robots if not r.free]
        spawn_ready = self._next_spawn_time() if self._spawn_queue else None
        process_ready = self._box_manager.next_process_ready()

        candidates = [r.busy_until for r in busy]
        if spawn_ready is not None:
            candidates.append(spawn_ready)
        if process_ready is not None:
            candidates.append(process_ready)
        if not candidates:
            self._box_manager.tick_spawn()
            return 0.0

        next_t = min(candidates)

        due = sorted(
            ((t, r, kind, box_id, node)
             for r in busy for (t, kind, box_id, node) in r.events if t <= next_t + 1e-9),
            key=lambda e: e[0],
        )
        reward = 0.0
        for (t, robot, kind, box_id, node) in due:
            reward += self._apply_event(robot, kind, box_id, node, t)
            if self.record:
                self.box_log.append((t, self._box_manager.snapshot()))
        for robot in busy:
            robot.events = [e for e in robot.events if e[0] > next_t + 1e-9]

        # Um "drop" agora aplicado pode ter começado uma nova janela de
        # PROCESSING cujo fim (`t + PROCESS_DELAY_TICKS`) é mais cedo que
        # `next_t` (decidido acima, antes de sabermos disso) — reencurta
        # para não passar por cima dela; senão o relógio saltaria à frente
        # e a chamada seguinte tentaria recuar até esse fim (impossível,
        # o tempo não anda para trás — ficaria preso ou corrompia o histórico).
        new_process_ready = self._box_manager.next_process_ready()
        if new_process_ready is not None:
            next_t = min(next_t, new_process_ready)

        dt = next_t - self.clock
        reward += REWARD_PER_TICK * len(busy) * max(dt, 0.0)

        self.clock = next_t

        for robot in busy:
            if robot.busy_until <= self.clock + 1e-9 and not robot.events:
                end_node, end_heading = robot.end_state
                robot.node = end_node
                robot.heading = end_heading
                robot.speed = 0.0
                robot.assigned = None
                robot.carrying = None

        if spawn_ready is not None and next_t >= spawn_ready - 1e-9:
            actual_spawn_t = self._spawn_next_robot(next_t)
            # `_spawn_next_robot` pode ter empurrado o instante para lá de
            # `next_t` (ex.: para dar margem física a quem evacuou) — o
            # relógio TEM de reflectir isso, senão planeamento seguinte usa
            # um "agora" desactualizado e corrompe o histórico do render.
            self.clock = max(self.clock, actual_spawn_t)

        if new_process_ready is not None and next_t >= new_process_ready - 1e-9:
            self._box_manager.tick_processing(self.clock)
            if self.record:
                self.box_log.append((self.clock, self._box_manager.snapshot()))

        self._box_manager.tick_spawn()
        return reward

    def _next_spawn_time(self) -> float | None:
        """Próximo instante (>= clock) em que o robot da frente da fila de
        spawn pode nascer em `SPAWN_NODE`. Nascer é um PARQUEAMENTO
        indefinido (`Interval(t, ∞)`) — por isso usa-se `safe_intervals`
        (o mesmo conceito do `park=True` no SIPP: um intervalo que chega a
        ∞), não `earliest_free_node_instant` (que só testa um instante
        pontual e ignoraria uma reserva futura distante de outro robot
        no mesmo nó). O último intervalo devolvido estende-se sempre a ∞.

        Vale o mais tardio entre:
        1. Esse intervalo estar livre (atropelando prioridade menor —
           pedido de nascimento tem `PRIORITY_SPAWNING`, infinita).
        2. `SPAWN_BUFFER` ticks desde `_n_last_used` (o cooldown do nó em
           si — NÃO pode ser saltado por prioridade, senão o robot recém-
           -nascido, idle e de prioridade 0, seria atropelado de imediato
           pelo pedido seguinte, anulando o cooldown)."""
        robot_id = self._spawn_queue[0]
        priorities = self._all_priorities(exclude=set())
        intervals = self.reservations.safe_intervals(
            SPAWN_NODE, robot_id, buffer=BUFFER,
            priorities=priorities, requester_priority=PRIORITY_SPAWNING,
            now=self.clock,
        )
        lo, _hi = intervals[-1]
        return max(lo, self.clock, self._n_last_used + SPAWN_BUFFER)

    def _spawn_next_robot(self, t: float) -> float:
        """Activa o próximo robot da fila em `SPAWN_NODE`. `t` (de
        `_next_spawn_time`) é só um candidato — se a reserva indefinida
        atropela alguém parado ali, evacua-o PRIMEIRO (tabela ainda sem
        este conflito, evita um "auto-conflito" no instante do handoff) e
        só depois comprometemos o instante FINAL: a partida real da
        evacuação (`schedule[0].depart`) + `BUFFER`, nunca antes disso —
        senão o evacuado e o novo robot ficavam ambos em `SPAWN_NODE` no
        mesmíssimo instante (zero margem física). Devolve o instante FINAL
        efectivamente usado — o chamador TEM de actualizar `self.clock`
        para este valor (pode ser maior que `t`)."""
        robot_id = self._spawn_queue.pop(0)
        claim = Interval(t, float("inf"))

        priorities = self._all_priorities(exclude=set())
        victims = self.reservations.conflicts_below_priority(
            [(SPAWN_NODE, claim)], [], robot_id, priorities, PRIORITY_SPAWNING,
            buffer=BUFFER, now=self.clock,
        )
        for victim_id in victims:
            self._replan_victim(victim_id)
            victim = next((r for r in self.robots if r.robot_id == victim_id), None)
            if victim is not None and victim.player is not None and victim.player.schedule:
                t = max(t, victim.player.schedule[0].depart + BUFFER)

        self._n_last_used = t
        claim = Interval(t, float("inf"))
        robot = RobotState(robot_id=robot_id, node=SPAWN_NODE, heading=Heading.initial())
        self.robots.append(robot)
        self.reservations.reserve_node(SPAWN_NODE, claim, robot_id)
        if self.record:
            self.render_players[robot_id] = engine.SchedulePlayer(
                robot_id, self.graph,
                [cts.ScheduleEntry(SPAWN_NODE, t, t, 0.0)], carrying_box=False,
            )
        return t

    def _apply_event(self, robot: RobotState, kind: str, box_id: int, node: str, t: float) -> float:
        """Aplica um evento de pick ou drop (no instante `t` real do evento —
        NÃO `self.clock`, que só é actualizado depois de todos os eventos
        `due` desta chamada serem aplicados) e devolve a recompensa dele."""
        if kind == "pick":
            r = self._box_manager.on_pick(robot.robot_id, box_id, node, t)
            robot.carrying = box_id
            return r
        r, _delivered = self._box_manager.on_drop(box_id, node, t)
        robot.carrying = None
        return r

    def _rehold(self, robot: RobotState) -> None:
        """Volta a reservar (indefinidamente) o nó de um robot que ficou
        idle por não ter conseguido planear."""
        if robot.node is not None:
            self.reservations.reserve_node(robot.node, Interval(self.clock, float("inf")), robot.robot_id)

    def _remaining_distance(self, robot: RobotState, now: float, until_t: float) -> float:
        """Distância física por percorrer entre `now` e `until_t`, ao longo
        do horário já comprometido do robot (soma dos troços, com o troço
        actual proporcional ao que falta nele)."""
        if robot.player is None:
            return 0.0
        total = 0.0
        for leg in robot.player.legs:
            if leg.arrive <= now + 1e-9:
                continue
            dist = self.graph.edge_distance(leg.from_node, leg.to_node)
            if leg.depart < now:
                span = leg.arrive - leg.depart
                frac_done = (now - leg.depart) / span if span > 1e-9 else 1.0
                total += dist * max(0.0, 1.0 - frac_done)
            else:
                total += dist
            if leg.arrive >= until_t - 1e-9:
                break
        return total

    def _priority(self, robot: RobotState) -> float:
        """Prioridade do robot num conflito de recursos: idle cede sempre
        primeiro (0), robots activos pesam pela distância restante até ao
        próximo evento (1x sem caixa, 2x com caixa). A prioridade "a
        nascer" (infinito) é tratada à parte, na fila de spawn."""
        if robot.free or not robot.events:
            return PRIORITY_IDLE
        next_event_t = robot.events[0][0]
        remaining = self._remaining_distance(robot, self.clock, next_event_t)
        weight = PRIORITY_WITH_BOX if robot.carrying is not None else PRIORITY_NO_BOX
        return weight * remaining

    def _all_priorities(self, exclude: set[str]) -> dict[str, float]:
        """Prioridade actual de todos os robots activos, excepto os
        indicados (tipicamente o próprio robot a planear)."""
        return {r.robot_id: self._priority(r) for r in self.robots if r.robot_id not in exclude}

    def _heading_at(self, robot: RobotState, now: float) -> Heading:
        """Orientação do robot no instante `now`: reproduz as pernas já
        concluídas da tarefa actual a partir de `robot.heading` (a
        orientação com que começou esta tarefa)."""
        if robot.player is None:
            return robot.heading
        h = robot.heading
        for leg in robot.player.legs:
            if leg.arrive > now + 1e-9:
                break
            h = rules.advance_heading(self.graph, h, leg.from_node, leg.to_node)
        return h

    def _final_heading(self, path: list[str], initial: Heading) -> Heading:
        """Orientação do robot no fim de um caminho, partindo de `initial`."""
        h = initial
        for a, b in zip(path, path[1:]):
            h = rules.advance_heading(self.graph, h, a, b)
        return h

    def _robot_position(self, robot: RobotState) -> tuple[float, float]:
        """Posição física (x, y) do robot — interpolada se em trânsito,
        exacta se parado num nó."""
        if robot.node is not None:
            return self.graph.node_position(robot.node)
        if robot.player is not None:
            return robot.player.state_at(self.clock)["pos"]
        return self.graph.node_position(SPAWN_NODE)

    def _robot_status(self, robot: RobotState) -> str:
        """Estado do robot para o agente: "idle" (livre, pode receber
        tarefa), "evacuating" (a fugir de um atropelamento, sem tarefa
        real), ou "busy" (a executar uma tarefa de caixa)."""
        if robot.free:
            return "idle"
        if robot.assigned is not None and robot.assigned[0] is None:
            return "evacuating"
        return "busy"

    def _future_path(self, robot: RobotState, limit: int = 6) -> list[str]:
        """Próximos nós REAIS (sem sub-nós) do horário comprometido do
        robot, a partir de agora — até `limit` nós, para o agente perceber
        para onde o robot vai sem se afogar em sub-nós de corredor."""
        if robot.player is None:
            return []
        path: list[str] = []
        for leg in robot.player.legs:
            if leg.arrive <= self.clock + 1e-9:
                continue
            if not self.graph.is_subnode(leg.to_node) and (not path or path[-1] != leg.to_node):
                path.append(leg.to_node)
            if len(path) >= limit:
                break
        return path

    def _build_graph_nodes(self) -> dict[str, dict]:
        """Features por nó REAL do mapa: tipo, posição, ocupação actual
        (robots/caixas), e estrutura pré-computada (betweenness, grau,
        distância às estações-alvo) — estática excepto a ocupação."""
        n_robots_here: dict[str, int] = {}
        for r in self.robots:
            if r.node is not None:
                n_robots_here[r.node] = n_robots_here.get(r.node, 0) + 1
        n_boxes_here: dict[str, int] = {}
        for b in self._box_manager.boxes():
            if b.is_waiting and b.current_node is not None:
                n_boxes_here[b.current_node] = n_boxes_here.get(b.current_node, 0) + 1

        nodes: dict[str, dict] = {}
        for n in self._real_nodes:
            x, y = self.graph.node_position(n)
            dist = self.graph.dist_to_type.get(n, {})
            nodes[n] = {
                "type": self.graph.node_type(n),
                "x": x, "y": y,
                "n_robots_here": n_robots_here.get(n, 0),
                "n_boxes_waiting": n_boxes_here.get(n, 0),
                "betweenness": self.graph.betweenness.get(n, 0.0),
                "degree": self.graph.degree_cache.get(n, len(self.graph.neighbors(n))),
                "dist_to_exit": dist.get("exit", float("inf")),
                "dist_to_processA": dist.get("processA_entry", float("inf")),
                "dist_to_processB": dist.get("processB_entry", float("inf")),
            }
        return nodes

    def _build_state(self) -> dict:
        """Constrói o dicionário de estado devolvido ao agente: robots
        livres pendentes, candidatos caixa→destino, snapshot das caixas, e
        o grafo do mapa (só nós reais — sub-nós de corredor não interessam
        à decisão). Um nó já reservado como `next_waypoint` de outra caixa
        em trânsito nunca é oferecido como candidato — evita duas entregas
        simultâneas ao mesmo nó (só um robot pode estacionar lá de cada vez)."""
        available = [
            (b.box_id, opt)
            for b in self._box_manager.available_boxes()
            for opt in b.target_options()
            if self._box_manager.is_node_drop_available(opt)
        ]
        pending = [r.robot_id for r in self.robots if r.free] if available else []
        return {
            "clock": self.clock,
            "robots": [
                {
                    "id": r.robot_id,
                    "node": r.node,
                    "carrying": r.carrying,
                    "assigned": r.assigned,
                    "free": r.free,
                    "status": self._robot_status(r),
                    "world_x": self._robot_position(r)[0],
                    "world_y": self._robot_position(r)[1],
                    "future_path": self._future_path(r),
                    "ticks_until_free": max(0.0, r.busy_until - self.clock) if not r.free else 0.0,
                }
                for r in self.robots
            ],
            "boxes": self._box_manager.snapshot(),
            "pending_robot_ids": pending,
            "available_box_targets": available,
            "delivered": self._box_manager.delivered_count(),
            "graph_nodes": self._build_graph_nodes(),
            "graph_edges": self._real_edges,
        }
