"""Cooperative A* (Space-Time A*) sequencial.

Procura no espaço (nó, tempo), usando o tempo físico real (`kinematics.py`)
como custo de aresta. A decisão "este nó exige paragem" usa sempre a janela
de 3 nós (veio_de, nó, candidato) — disponível exactamente no momento da
expansão, tal como o ângulo em `rules.turn_angle`.

Correcção retroactiva: quando ao expandir (nó->candidato) descobrimos que
`nó` afinal exigia parar, o tempo de chegada a `nó` (calculado
optimisticamente ao ser posto na fila, assumindo que não parava) é
corrigido em `zona_decel/velocidade_cruzeiro` — a diferença exacta entre
desacelerar e cruzar nos últimos `CURVE_ZONE`. Ver kinematics.py para a
mesma lógica aplicada a um caminho já conhecido (`path_time`).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from env.core.graph import FactoryGraph
from env.physics import kinematics, rules
from env.physics.rules import Heading
from env.traffic.reservations import Interval, ReservationTable


MAX_EXPANSIONS = 100000
"""Rede de segurança contra bugs — com SIPP a procura fica muito abaixo."""


@dataclass(order=True)
class _Node:
    """Estado da procura SIPP: nó, intervalo-seguro em que lá está, tempos e
    orientação. `g` = chegada; `si_lo/si_hi` = intervalo seguro do nó;
    `depart_time` = quando partiu do pai; `entry_reverse` = chegou a recuar.
    Só `f` é comparável (ordem na fila de prioridade)."""
    f          : float
    g          : float = field(compare=False)
    node       : str = field(compare=False)
    came_from  : str | None = field(compare=False)
    heading    : Heading = field(compare=False)
    speed      : float = field(compare=False)
    si_lo      : float = field(compare=False, default=0.0)
    si_hi      : float = field(compare=False, default=float("inf"))
    si_idx     : int = field(compare=False, default=0)
    depart_time: float = field(compare=False, default=0.0)
    entry_reverse: bool = field(compare=False, default=False)
    parent     : "_Node | None" = field(compare=False, default=None)


def _stop_correction(
    graph: FactoryGraph,
    came_from: str | None,
    node: str,
    carrying_box: bool,
    entry_reverse: bool = False,
) -> float:
    """Delta de tempo entre ter cruzado (assumido) e ter desacelerado
    (afinal necessário) nos últimos CURVE_ZONE da aresta came_from->node."""
    if came_from is None:
        return 0.0
    distance = graph.edge_distance(came_from, node)
    decel_zone = min(rules.CURVE_ZONE, distance)
    cruise = rules.effective_max_speed(carrying_box, entry_reverse)
    return decel_zone / cruise if cruise > 0 else 0.0


def _reconstruct(node: _Node) -> list[str]:
    """Reconstrói a lista de nós do caminho, do início ao `node` final."""
    path: list[str] = []
    current: _Node | None = node
    while current is not None:
        path.append(current.node)
        current = current.parent
    path.reverse()
    return path


@dataclass(frozen=True)
class ScheduleEntry:
    """Um nó do caminho com os tempos exactos de chegada/partida
    calculados durante a procura (inclui esperas por reservas)."""
    node: str
    arrival: float
    depart: float
    speed: float


def _extract_schedule(goal_node: "_Node", final_time: float) -> list[ScheduleEntry]:
    """Horário exacto por-nó, a partir da cadeia da procura — para quem
    precisar de animar/inspeccionar o caminho sem recalcular do zero
    (isso podia divergir das esperas que a procura decidiu)."""
    entries: list[ScheduleEntry] = []
    current: _Node | None = goal_node
    depart_from_current = final_time

    while current is not None:
        entries.append(ScheduleEntry(current.node, current.g, depart_from_current, current.speed))
        depart_from_current = current.depart_time
        current = current.parent

    entries.reverse()
    return entries


def _commit_chain(
    reservations: ReservationTable,
    robot_id: str,
    goal_node: "_Node",
    final_time: float,
    park: bool = True,
    buffer: float = 0.0,
    priorities: dict[str, float] | None = None,
    requester_priority: float = float("inf"),
    now: float | None = None,
) -> set[str]:
    """Regista as reservas usando os valores exactos calculados durante a
    procura (não recalcula nada) — evita qualquer divergência entre o que
    foi validado contra a tabela e o que fica lá reservado.

    `park=True`: o robot fica estacionado no destino até nova tarefa —
    reserva até ∞ (impede outro de planear para o pouso final deste).
    `park=False`: paragem intermédia (ex.: ir buscar uma caixa) — reserva só
    a estadia curta [chegada, final_time], libertando o nó a seguir para a
    perna seguinte da viagem.

    Se `priorities` for dado, antes de reservar verifica que robots de
    prioridade menor ficam "atropelados" pela cadeia agora comprometida —
    devolve o conjunto desses robot_ids (o chamador tem de os replanear).
    Só reservas de NÓS podem ser atropeladas (um robot parado, fácil de
    redireccionar); arestas são sempre FCFS puro (movimento em curso não se
    desfaz), por isso nunca geram vítimas — ver `plan`.
    """
    hold_end = float("inf") if park else final_time
    node_claims: list[tuple[str, Interval]] = [(goal_node.node, Interval(goal_node.g, hold_end))]
    edge_claims: list[tuple[str, str, Interval]] = []

    child = goal_node
    parent = child.parent
    while parent is not None:
        edge_claims.append((parent.node, child.node, Interval(child.depart_time, child.g)))
        node_claims.append((parent.node, Interval(parent.g, child.depart_time)))
        child = parent
        parent = child.parent

    victims: set[str] = set()
    if priorities is not None:
        victims = reservations.conflicts_below_priority(
            node_claims, [], robot_id, priorities, requester_priority, buffer, now
        )

    for node, interval in node_claims:
        reservations.reserve_node(node, interval, robot_id)
    for u, v, interval in edge_claims:
        reservations.reserve_edge(u, v, interval, robot_id)

    return victims


def plan(
    graph: FactoryGraph,
    src: str,
    dst: str,
    carrying_box: bool,
    start_time: float = 0.0,
    initial_heading: Heading | None = None,
    initial_speed: float = 0.0,
    reservations: ReservationTable | None = None,
    robot_id: str | None = None,
    buffer: float = 1.0,
    park: bool = True,
    priorities: dict[str, float] | None = None,
    requester_priority: float = float("inf"),
    now: float | None = None,
) -> tuple[list[str], float, list[ScheduleEntry], set[str]] | None:
    """Caminho mais rápido de src a dst, via SIPP (Safe Interval Path
    Planning) sobre a física real (`kinematics`).

    O estado é (nó, intervalo-seguro) em vez de (nó, tempo): para cada nó,
    o tempo parte-se nas janelas livres de outros robots (`safe_intervals`)
    e guarda-se um estado por janela, não um por tick — é isto que evita a
    explosão da procura em tempo contínuo. A espera está embutida (ficar
    dentro do intervalo seguro do nó actual = ceder passagem), por isso
    resolve-se rota + espera + dar-a-volta sem heurísticas.

    Um robot só TERMINA num intervalo seguro do destino que chega a ∞
    (pode estacionar lá para sempre) — senão outro usa o nó depois e haveria
    colisão. Ao terminar, reserva o caminho (`_commit_chain`) com os tempos
    exactos da procura.

    Se `priorities` for dado, reservas de robots com prioridade menor que
    `requester_priority` são ignoradas (tratadas como livres) durante a
    procura — este robot "atropela-os". Só reservas que já COMEÇARAM em
    `now` (o relógio real da simulação — CUIDADO: não é o mesmo que
    `start_time`, que para a perna com caixa é um instante futuro
    projectado, a chegada estimada ao pick_node) podem ser atropeladas;
    uma reserva ainda no futuro nunca é, seja qual for a prioridade — ver
    `ReservationTable._blocks`. Devolve o conjunto de atropelados (o
    chamador tem de os replanear) como 4º elemento do resultado.

    Devolve (caminho, tempo_de_chegada, horário, atropelados) ou None se
    inatingível.
    """
    if src == dst:
        return ([src], start_time, [ScheduleEntry(src, start_time, start_time, initial_speed)], set())

    coordinated = reservations is not None and robot_id is not None

    _si_cache: dict[str, list[tuple[float, float]]] = {}

    def safe_intervals(node: str) -> list[tuple[float, float]]:
        """Intervalos seguros de `node` (com cache local por chamada)."""
        cached = _si_cache.get(node)
        if cached is None:
            cached = (reservations.safe_intervals(node, robot_id, buffer, priorities, requester_priority, now)
                      if coordinated else [(0.0, float("inf"))])
            _si_cache[node] = cached
        return cached

    open_heap: list[_Node] = []
    best_g: dict[tuple[str, str | None, int], float] = {}

    src_intervals = safe_intervals(src)
    root_idx = next(
        (i for i, (lo, hi) in enumerate(src_intervals) if lo - 1e-9 <= start_time <= hi + 1e-9),
        len(src_intervals) - 1,
    )
    r_lo, r_hi = src_intervals[root_idx]
    root = _Node(
        f=start_time + graph.heuristic(src, dst) / rules.V_MAX,
        g=start_time,
        node=src,
        came_from=None,
        heading=initial_heading or Heading.initial(),
        speed=initial_speed,
        si_lo=r_lo, si_hi=r_hi, si_idx=root_idx,
        depart_time=start_time,
    )
    heapq.heappush(open_heap, root)
    best_g[(src, None, root_idx)] = start_time

    expansions = 0
    while open_heap:
        expansions += 1
        if expansions > MAX_EXPANSIONS:
            return None
        current = heapq.heappop(open_heap)

        state_key = (current.node, current.came_from, current.si_idx)
        if current.g > best_g.get(state_key, float("inf")) + 1e-9:
            continue

        if current.node == dst:
            final_time = current.g + _stop_correction(
                graph, current.came_from, current.node, carrying_box, current.entry_reverse
            )
            if park and current.si_hi != float("inf"):
                continue
            if not park and final_time > current.si_hi + 1e-9:
                continue
            victims: set[str] = set()
            if coordinated:
                victims = _commit_chain(
                    reservations, robot_id, current, final_time, park=park,
                    buffer=buffer, priorities=priorities, requester_priority=requester_priority,
                    now=now,
                )
            return _reconstruct(current), final_time, _extract_schedule(current, final_time), victims

        for neighbor in graph.neighbors(current.node):
            reverse = rules.is_reverse_move(graph, current.heading, current.node, neighbor)
            rotate_ticks = rules.rotation_ticks_for(graph, current.heading, current.node, neighbor)

            stops = rotate_ticks > 0 or rules.is_dead_end(graph, current.node)
            node_time = 0.0
            entry_speed = current.speed
            if stops:
                node_time = (
                    _stop_correction(graph, current.came_from, current.node,
                                     carrying_box, current.entry_reverse)
                    + rotate_ticks
                )
                entry_speed = 0.0

            distance = graph.edge_distance(current.node, neighbor)
            cruise_speed = rules.effective_max_speed(carrying_box, reverse)
            edge_time, exit_speed = kinematics.segment_time(
                distance, entry_speed, cruise_speed, must_stop=False
            )
            new_heading = rules.advance_heading(graph, current.heading, current.node, neighbor)

            earliest_depart = current.g + node_time
            latest_depart = current.si_hi
            if earliest_depart > latest_depart + 1e-9:
                continue

            h = graph.heuristic(neighbor, dst)
            if h == float("inf"):
                continue

            for si_idx, (nlo, nhi) in enumerate(safe_intervals(neighbor)):
                lo = max(earliest_depart, nlo - edge_time)
                hi = min(latest_depart, nhi - edge_time)
                if lo > hi + 1e-9:
                    continue

                if coordinated:
                    # Arestas são SEMPRE FCFS puro (nunca ignoram uma reserva
                    # por prioridade) — um robot já a meio de uma aresta é
                    # movimento em curso, que nenhuma prioridade desfaz. Só
                    # reservas de NÓS (robot parado) podem ser atropeladas.
                    depart = reservations.earliest_free_edge_start(
                        current.node, neighbor, edge_time, lo, robot_id, buffer,
                    )
                    if depart is None or depart > hi + 1e-9:
                        continue
                else:
                    depart = lo

                arrival = depart + edge_time
                nkey = (neighbor, current.node, si_idx)
                if arrival >= best_g.get(nkey, float("inf")) - 1e-9:
                    continue
                best_g[nkey] = arrival

                heapq.heappush(open_heap, _Node(
                    f=arrival + h / rules.V_MAX,
                    g=arrival,
                    node=neighbor,
                    came_from=current.node,
                    heading=new_heading,
                    speed=exit_speed,
                    si_lo=nlo, si_hi=nhi, si_idx=si_idx,
                    depart_time=depart,
                    entry_reverse=reverse,
                    parent=current,
                ))

    return None
