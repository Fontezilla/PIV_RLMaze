"""Fórmula fechada de tempo de travessia (cinemática), sem simulação tick-a-tick.

Combina a classificação de `rules.py` (recta/curva/marcha-atrás, ângulo,
velocidade máxima efectiva) com um modelo cinemático de aceleração
constante para calcular quanto tempo (em ticks) demora a percorrer uma
aresta ou um caminho completo — usado como custo de aresta pelo Cooperative
A* e, mais tarde, para orientar o motor de física tick-a-tick (`engine.py`).

Perfil de uma aresta:
  1. Acelerar (se a velocidade de entrada < velocidade de cruzeiro efectiva)
     a `rules.ACCEL`, até à velocidade de cruzeiro.
  2. Cruzeiro, se sobrar distância.
  3. Desacelerar (só se o troço termina em paragem obrigatória — curva ou
     fim do caminho) nos últimos `rules.CURVE_ZONE` até 0, exactamente no nó.

A decisão "este troço termina em paragem" usa sempre a janela de 3 nós
(anterior, actual, seguinte) — a mesma que o `turn_angle` já usa — por isso
nunca precisa de adivinhar o que vem depois do nó seguinte.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from env.core.graph import FactoryGraph
from env.physics import rules
from env.physics.rules import Heading


@dataclass(frozen=True)
class SegmentProfile:
    """Perfil de velocidade dentro de uma aresta: acelerar -> cruzeiro ->
    desacelerar. `distance_at(t)` dá a distância percorrida ao fim de `t`
    ticks desde o início do troço — usado para animar a posição real
    (não uma interpolação linear no tempo, que esconderia a aceleração).
    """
    entry_speed: float
    accel: float
    accel_time: float
    accel_dist: float
    cruise_time: float
    cruise_dist: float
    decel_time: float
    decel_dist: float
    peak_speed: float
    exit_speed: float

    @property
    def total_time(self) -> float:
        """Duração total do troço (acelerar + cruzeiro + desacelerar)."""
        return self.accel_time + self.cruise_time + self.decel_time

    @property
    def total_dist(self) -> float:
        """Distância total do troço."""
        return self.accel_dist + self.cruise_dist + self.decel_dist

    def distance_at(self, t: float) -> float:
        """Distância percorrida ao fim de `t` ticks desde o início do troço."""
        t = max(0.0, min(t, self.total_time))

        if t <= self.accel_time:
            return self.entry_speed * t + 0.5 * self.accel * t * t

        t -= self.accel_time
        if t <= self.cruise_time:
            return self.accel_dist + self.peak_speed * t

        t -= self.cruise_time
        t = min(t, self.decel_time)
        decel_rate = self.peak_speed / self.decel_time if self.decel_time > 0 else 0.0
        return self.accel_dist + self.cruise_dist + (
            self.peak_speed * t - 0.5 * decel_rate * t * t
        )

    def velocity_at(self, t: float) -> float:
        """Velocidade instantânea ao fim de `t` ticks desde o início do troço."""
        t = max(0.0, min(t, self.total_time))

        if t <= self.accel_time:
            return self.entry_speed + self.accel * t

        t -= self.accel_time
        if t <= self.cruise_time:
            return self.peak_speed

        t -= self.cruise_time
        t = min(t, self.decel_time)
        decel_rate = self.peak_speed / self.decel_time if self.decel_time > 0 else 0.0
        return max(self.peak_speed - decel_rate * t, 0.0)


def segment_profile(
    distance: float,
    entry_speed: float,
    cruise_speed: float,
    must_stop: bool,
    accel: float = rules.ACCEL,
    curve_zone: float = rules.CURVE_ZONE,
) -> SegmentProfile:
    """Perfil completo (fases + durações) para percorrer `distance`.

    Entra a `entry_speed`, acelera (se necessário) até `cruise_speed`, e
    desacelera até 0 nos últimos `curve_zone` se `must_stop`. Se a aresta
    não tiver espaço para acelerar até `cruise_speed` (comum nos primeiros
    50 unidades a seguir a uma paragem — ver `graph.JUNCTION_SUBNODE_DIST`),
    acelera o que couber e sai com velocidade parcial, a continuar a
    acelerar na aresta seguinte (não é um erro, é o comportamento normal).

    Caso raro (arranque do zero mesmo a precisar de parar já na aresta
    seguinte — tipicamente em desvios forçados por reservas): não há
    espaço nem para acelerar até `cruise_speed` nem ainda para desacelerar
    nos `curve_zone` completos a partir daí. Nesse caso usa-se um perfil
    triangular (acelera e trava à mesma taxa `accel`) que pára exactamente
    ao fim de `distance`, em vez do modelo normal de "desacelera sempre
    nos últimos `curve_zone` vindo de `cruise_speed`".
    """
    decel_zone = min(curve_zone, distance) if must_stop else 0.0
    available_for_accel = distance - decel_zone

    accel_dist_needed = 0.0
    if entry_speed < cruise_speed:
        accel_dist_needed = (cruise_speed ** 2 - entry_speed ** 2) / (2.0 * accel)

    if accel_dist_needed <= available_for_accel + 1e-9:
        accel_time = (cruise_speed - entry_speed) / accel if entry_speed < cruise_speed else 0.0
        cruise_dist = available_for_accel - accel_dist_needed
        cruise_time = cruise_dist / cruise_speed if cruise_speed > 0 else 0.0
        peak_speed = cruise_speed

        decel_time = 0.0
        exit_speed = peak_speed
        if must_stop and decel_zone > 0:
            decel_time = 2.0 * decel_zone / peak_speed if peak_speed > 0 else 0.0
            exit_speed = 0.0

        return SegmentProfile(
            entry_speed=entry_speed, accel=accel,
            accel_time=accel_time, accel_dist=accel_dist_needed,
            cruise_time=cruise_time, cruise_dist=cruise_dist,
            decel_time=decel_time, decel_dist=decel_zone if must_stop else 0.0,
            peak_speed=peak_speed, exit_speed=exit_speed,
        )

    if not must_stop:
        peak_speed = math.sqrt(entry_speed ** 2 + 2.0 * accel * distance)
        accel_time = (peak_speed - entry_speed) / accel
        return SegmentProfile(
            entry_speed=entry_speed, accel=accel,
            accel_time=accel_time, accel_dist=distance,
            cruise_time=0.0, cruise_dist=0.0,
            decel_time=0.0, decel_dist=0.0,
            peak_speed=peak_speed, exit_speed=peak_speed,
        )

    peak_sq = accel * distance + (entry_speed ** 2) / 2.0
    peak_speed = math.sqrt(max(peak_sq, entry_speed ** 2))
    accel_time = (peak_speed - entry_speed) / accel
    decel_time = peak_speed / accel
    accel_dist = (peak_speed ** 2 - entry_speed ** 2) / (2.0 * accel)
    return SegmentProfile(
        entry_speed=entry_speed, accel=accel,
        accel_time=accel_time, accel_dist=accel_dist,
        cruise_time=0.0, cruise_dist=0.0,
        decel_time=decel_time, decel_dist=distance - accel_dist,
        peak_speed=peak_speed, exit_speed=0.0,
    )


def segment_time(
    distance: float,
    entry_speed: float,
    cruise_speed: float,
    must_stop: bool,
    accel: float = rules.ACCEL,
    curve_zone: float = rules.CURVE_ZONE,
) -> tuple[float, float]:
    """Tempo (ticks) e velocidade de saída para percorrer `distance`.

    Atalho sobre `segment_profile` para quem só precisa do total — ver aí
    a descrição completa do modelo (accel/cruzeiro/decel, casos-limite).
    """
    profile = segment_profile(distance, entry_speed, cruise_speed, must_stop, accel, curve_zone)
    return profile.total_time, profile.exit_speed


def transition_time(
    graph: FactoryGraph,
    heading: Heading,
    from_node: str,
    to_node: str,
    next_node: str | None,
    carrying_box: bool,
    entry_speed: float,
) -> tuple[float, float, Heading]:
    """Tempo (ticks) para from_node->to_node, incluindo a paragem/rotação em
    to_node se o troço seguinte (to_node->next_node) o exigir.

    `next_node=None` significa que to_node é o destino final do caminho —
    implica sempre paragem (é lá que a acção, pick/drop, acontece).

    Devolve (tempo_total, velocidade_de_saída, nova_orientação).
    """
    reverse = rules.is_reverse_move(graph, heading, from_node, to_node)
    cruise_speed = rules.effective_max_speed(carrying_box, reverse)
    distance = graph.edge_distance(from_node, to_node)

    new_heading = rules.advance_heading(graph, heading, from_node, to_node)

    if next_node is None:
        rotate_time = 0.0
        must_stop = True
    else:
        rotate_time = rules.rotation_ticks_for(graph, new_heading, to_node, next_node)
        must_stop = rotate_time > 0 or rules.is_dead_end(graph, to_node)

    travel_time, exit_speed = segment_time(distance, entry_speed, cruise_speed, must_stop)

    return travel_time + rotate_time, exit_speed, new_heading


def path_time(
    graph: FactoryGraph,
    path: list[str],
    carrying_box: bool,
    initial_heading: Heading | None = None,
    initial_speed: float = 0.0,
) -> tuple[float, list[float]]:
    """Tempo total e tempos cumulativos de chegada a cada nó de `path`.

    `path` é uma lista de nós já decidida (ex.: resultado do Cooperative
    A*) — cada troço já sabe o nó seguinte, por isso o cálculo é exacto,
    sem aproximações.
    """
    if len(path) < 2:
        return 0.0, [0.0] * len(path)

    heading = initial_heading or Heading.initial()
    speed = initial_speed
    cumulative = [0.0]
    total = 0.0

    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]
        next_node = path[i + 2] if i + 2 < len(path) else None

        seg_time, speed, heading = transition_time(
            graph, heading, from_node, to_node, next_node, carrying_box, speed
        )
        total += seg_time
        cumulative.append(total)

    return total, cumulative


@dataclass(frozen=True)
class PathSegment:
    """Um troço de `path_profile`: aresta from_node->to_node, com o perfil
    de velocidade completo e o tempo de rotação (se houver) antes do
    troço seguinte. `start_time` é quando o robot começa a percorrer esta
    aresta (não quando chega ao fim)."""
    from_node: str
    to_node: str
    start_time: float
    profile: SegmentProfile
    rotate_time: float


def path_profile(
    graph: FactoryGraph,
    path: list[str],
    carrying_box: bool,
    initial_heading: Heading | None = None,
    initial_speed: float = 0.0,
    start_time: float = 0.0,
) -> list[PathSegment]:
    """Detalhe por-troço de `path` (perfil de velocidade + rotação), para
    animação exacta — ao contrário de `path_time`, que só dá os totais.
    """
    segments: list[PathSegment] = []
    if len(path) < 2:
        return segments

    heading = initial_heading or Heading.initial()
    speed = initial_speed
    t = start_time

    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]
        next_node = path[i + 2] if i + 2 < len(path) else None

        reverse = rules.is_reverse_move(graph, heading, from_node, to_node)
        cruise_speed = rules.effective_max_speed(carrying_box, reverse)
        distance = graph.edge_distance(from_node, to_node)
        new_heading = rules.advance_heading(graph, heading, from_node, to_node)

        if next_node is None:
            rotate_time = 0.0
            must_stop = True
        else:
            rotate_time = rules.rotation_ticks_for(graph, new_heading, to_node, next_node)
            must_stop = rotate_time > 0 or rules.is_dead_end(graph, to_node)

        profile = segment_profile(distance, speed, cruise_speed, must_stop)

        segments.append(PathSegment(from_node, to_node, t, profile, rotate_time))

        t += profile.total_time + rotate_time
        speed = profile.exit_speed
        heading = new_heading

    return segments
