"""Executor de horário (schedule player) — reproduz fielmente o horário
que o Space-Time A* (`cooperative_astar.plan`) já decidiu e reservou.

Arquitectura planeador-autoridade: o Cooperative A* é a única autoridade
sobre tráfego (rota, espera, dar-a-volta) e reserva a trajetória
espaço-tempo completa. Este módulo NÃO faz reservas nem decide conflitos —
só converte o horário (`ScheduleEntry` por nó) na posição/velocidade/ângulo
reais em qualquer instante, para render e para o loop do ambiente.

Como o tempo é ditado pelo horário (e as reservas vieram do mesmo horário),
a execução é consistente com as reservas por construção — não há a
divergência que existia quando o motor recalculava tempos por conta
própria.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from env.core.graph import FactoryGraph
from env.physics import kinematics, rules
from env.physics.rules import Heading
from env.traffic.cooperative_astar import ScheduleEntry


@dataclass
class _Leg:
    """Um troço do horário: aresta from_node->to_node percorrida em
    [depart, arrive], com o perfil físico para interpolar a posição.
    `face_angle` é a direcção para onde o robot APONTA (não a do movimento:
    numa marcha-atrás aponta ao contrário do sentido de deslocamento)."""
    from_node: str
    to_node: str
    depart: float
    arrive: float
    profile: kinematics.SegmentProfile
    face_angle: float = 0.0


def _edge_angle(graph: FactoryGraph, u: str, v: str) -> float:
    """Ângulo (radianos) da direcção u→v, para orientar o triângulo do robot."""
    x0, y0 = graph.node_position(u)
    x1, y1 = graph.node_position(v)
    return math.atan2(y1 - y0, x1 - x0) if (x1 != x0 or y1 != y0) else -math.pi / 2


def build_legs(
    graph: FactoryGraph,
    schedule: list[ScheduleEntry],
    carrying_box: bool,
    initial_heading: Heading | None = None,
) -> tuple[list[_Leg], Heading]:
    """Converte um horário (nós + tempos) na lista de troços com perfil
    físico. O tempo de cada troço vem do horário (é o mesmo que o A* usou
    para reservar), por isso a execução fica sincronizada com as reservas.

    Devolve também a `Heading` no fim (para encadear pernas no replaneamento).
    A orientação (`face_angle`) só muda em movimentos para a frente; numa
    marcha-atrás (sair de beco sem saída) o robot mantém a orientação e
    recua, por isso aponta ao contrário do sentido de deslocamento.
    """
    legs: list[_Leg] = []
    heading = initial_heading or Heading.initial()
    n = len(schedule)
    for i in range(n - 1):
        a, b = schedule[i], schedule[i + 1]
        if b.node == a.node:
            continue
        reverse = rules.is_reverse_move(graph, heading, a.node, b.node)
        stopped_at_a = a.depart > a.arrival + 1e-9
        entry_speed = 0.0 if stopped_at_a else a.speed
        cruise = rules.effective_max_speed(carrying_box, reverse)
        distance = graph.edge_distance(a.node, b.node)
        stop_at_b = (b.depart > b.arrival + 1e-9) or (i + 1 == n - 1)
        profile = kinematics.segment_profile(distance, entry_speed, cruise, must_stop=stop_at_b)

        if reverse and heading.from_node is not None:
            face = _edge_angle(graph, heading.from_node, heading.to_node)
        else:
            face = _edge_angle(graph, a.node, b.node)
        heading = rules.advance_heading(graph, heading, a.node, b.node)

        legs.append(_Leg(a.node, b.node, a.depart, b.arrival, profile, face))
    return legs, heading


class SchedulePlayer:
    """Reprodutor do horário de um robot — dá posição/estado em qualquer t."""

    def __init__(
        self,
        robot_id: str,
        graph: FactoryGraph,
        schedule: list[ScheduleEntry],
        carrying_box: bool,
    ) -> None:
        """Prepara os troços físicos a partir do horário do plano."""
        self.robot_id = robot_id
        self.graph = graph
        self.carrying_box = carrying_box
        self.schedule = schedule
        self.legs, self._heading = build_legs(graph, schedule, carrying_box)

    @property
    def start_time(self) -> float:
        """Instante em que o robot começa a mover-se."""
        return self.schedule[0].arrival if self.schedule else 0.0

    @property
    def arrival_time(self) -> float:
        """Instante em que o robot termina o horário."""
        return self.schedule[-1].depart if self.schedule else 0.0

    def extend(self, schedule: list[ScheduleEntry]) -> None:
        """Concatena o horário de uma nova perna (replaneamento contínuo).

        Constrói só os legs NOVOS e acrescenta-os (incremental — evita
        reconstruir tudo a cada perna, que seria O(N²) ao longo de milhares
        de pernas).

        Cuidado no nó de junção: a perna anterior chegou a esse nó a um
        certo instante (`arrival`) e a leg de entrada, já construída, usa
        essa chegada. A nova perna começa nesse nó mas com `start_time =
        final_time` da anterior (chegada + paragem), o que daria uma
        chegada diferente e um GAP entre a leg de entrada e a entrada do
        nó. Por isso funde-se: mantém-se a chegada ORIGINAL do nó de
        junção e toma-se só a partida (e velocidade) da nova perna.
        """
        if not self.schedule:
            self.schedule = list(schedule)
            self.legs, self._heading = build_legs(self.graph, schedule, self.carrying_box)
            return

        old_last = self.schedule[-1]
        new_first = schedule[0]
        merged = ScheduleEntry(
            old_last.node, old_last.arrival, new_first.depart, new_first.speed
        )
        tail = [merged] + list(schedule[1:])
        self.schedule = self.schedule[:-1] + tail
        new_legs, self._heading = build_legs(
            self.graph, tail, self.carrying_box, initial_heading=self._heading
        )
        self.legs.extend(new_legs)

    def state_at(self, t: float) -> dict:
        """Estado do robot no instante `t`: posição, ângulo, velocidade,
        status ('AGUARDA'/'MOVING'/'WAITING'/'DONE') e a aresta actual."""
        graph = self.graph
        if not self.schedule:
            return dict(pos=(0.0, 0.0), angle=-math.pi / 2, speed=0.0,
                        status="—", from_node=None, to_node=None)

        if t <= self.schedule[0].depart:
            node = self.schedule[0].node
            return dict(pos=graph.node_position(node),
                        angle=self.legs[0].face_angle if self.legs else -math.pi / 2,
                        speed=0.0, status="AGUARDA", from_node=node, to_node=None)

        for leg in self.legs:
            if leg.depart <= t <= leg.arrive:
                dur = leg.arrive - leg.depart
                if dur <= 1e-9:
                    frac = 1.0
                else:
                    local = (t - leg.depart) * leg.profile.total_time / dur
                    frac = (leg.profile.distance_at(local) / leg.profile.total_dist
                            if leg.profile.total_dist > 0 else 1.0)
                x0, y0 = graph.node_position(leg.from_node)
                x1, y1 = graph.node_position(leg.to_node)
                dur_scale = leg.profile.total_time / dur if dur > 0 else 1.0
                speed = leg.profile.velocity_at((t - leg.depart) * dur_scale)
                return dict(pos=(x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac),
                            angle=leg.face_angle,
                            speed=speed, status="MOVING",
                            from_node=leg.from_node, to_node=leg.to_node)

        for entry in self.schedule:
            if entry.arrival <= t <= entry.depart:
                return dict(pos=graph.node_position(entry.node),
                            angle=self._rotating_angle(t),
                            speed=0.0, status="WAITING",
                            from_node=entry.node, to_node=None)

        last = self.schedule[-1]
        return dict(pos=graph.node_position(last.node),
                    angle=self.legs[-1].face_angle if self.legs else -math.pi / 2,
                    speed=0.0, status="DONE", from_node=last.node, to_node=None)

    def _rotating_angle(self, t: float) -> float:
        """Ângulo durante uma paragem: interpola da orientação do troço que
        acabou (`face_angle` de entrada) para a do troço seguinte, ao longo
        da paragem — anima a rotação. Numa marcha-atrás os dois `face_angle`
        coincidem (a orientação não muda), logo NÃO roda; numa curva diferem
        e a rotação aparece."""
        inc = None
        out = None
        for leg in self.legs:
            if leg.arrive <= t + 1e-9:
                inc = leg
            if leg.depart >= t - 1e-9 and out is None:
                out = leg
        if inc is None:
            return out.face_angle if out else -math.pi / 2
        if out is None or abs(out.face_angle - inc.face_angle) < 1e-9:
            return inc.face_angle
        span = out.depart - inc.arrive
        frac = 1.0 if span <= 1e-9 else max(0.0, min((t - inc.arrive) / span, 1.0))
        diff = (out.face_angle - inc.face_angle + math.pi) % (2 * math.pi) - math.pi
        return inc.face_angle + diff * frac
