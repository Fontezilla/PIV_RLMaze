"""Regras de física do movimento: velocidades, curvas e marcha-atrás.

Define apenas a CLASSIFICAÇÃO de uma transição (recta/curva/marcha-atrás) e
as suas consequências (ângulo de rotação, velocidade máxima efectiva). O
cálculo do tempo de travessia vive em `kinematics.py`; a simulação tick-a-tick
vive em `engine.py`.

Regras (confirmadas com o utilizador):
- Recta (0°): sem perda de velocidade.
- Curva (o env só produz curvas em múltiplos de 90°): desacelera nos
  últimos CURVE_ZONE (= distância dos sub-nós de junction) até parar em
  velocidade 0 no nó, roda ROTATE_TICKS_PER_90 ticks por cada 90° de
  ângulo, depois acelera de novo gradualmente.
- Nó de grau 1 (dead-end): ao APROXIMAR-SE segue a regra normal; só ao
  AFASTAR-SE (marcha-atrás, por não conseguir dar a volta) a velocidade
  máxima é limitada a REVERSE_SPEED_FACTOR.
- A transportar caixa: velocidade máxima reduzida a BOX_SPEED_FACTOR.
  Acumula multiplicativamente com a marcha-atrás.
- Marcha-atrás NÃO reorienta o robot — a orientação (`Heading`) só
  actualiza em movimentos para a frente. É isto que faz um robot que sai
  de um dead-end e continua no mesmo eixo precisar de rodar 180° (2x
  ROTATE_TICKS_PER_90), não 0°.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from env.core.graph import FactoryGraph, JUNCTION_SUBNODE_DIST


V_MAX = 30.0
ACCEL = 6.0

CURVE_ZONE = JUNCTION_SUBNODE_DIST

ROTATE_TICKS_PER_90 = 10
TICK_DURATION        = 0.05

BOX_SPEED_FACTOR     = 0.85
REVERSE_SPEED_FACTOR = 0.60


@dataclass(frozen=True)
class Heading:
    """Orientação efectiva do robot: a última aresta percorrida para a
    frente (não o último nó visitado — divergem depois de uma marcha-atrás).
    """
    from_node: str | None
    to_node  : str | None

    @classmethod
    def initial(cls) -> "Heading":
        """Sem orientação definida — o primeiro movimento de sempre nunca
        implica rotação."""
        return cls(None, None)


def is_dead_end(graph: FactoryGraph, node: str) -> bool:
    """True se o nó tem uma só saída (grau 1) — só se sai de lá a recuar."""
    return graph.graph.degree(node) == 1


def is_reverse_move(
    graph: FactoryGraph,
    heading: Heading,
    from_node: str,
    to_node: str,
) -> bool:
    """True se o robot RECUA neste movimento (marcha-atrás).

    Regra: a rotação só acontece em JUNCTIONS. Num corredor (nó não-junction,
    grau ≤ 2, incluindo becos) o robot não roda — se o próximo troço vai na
    direcção oposta à sua orientação, ele RECUA (mantendo a orientação, a
    60%). É isto que faz um robot sair de um beco a recuar o corredor todo
    até à junction, sem "dar voltas" nos sub-nós pelo caminho.
    """
    if heading.from_node is None:
        return False
    if graph.is_turn_node(from_node):
        return False
    return turn_angle(graph, heading, from_node, to_node) > 90.0 + 1e-6


def rotation_ticks_for(
    graph: FactoryGraph,
    heading: Heading,
    from_node: str,
    to_node: str,
) -> float:
    """Ticks de rotação ao partir de `from_node` para `to_node`. Só há
    rotação em junctions REAIS (nos corredores/sub-nós o robot segue a
    direito ou recua)."""
    if not graph.is_turn_node(from_node):
        return 0.0
    return rotation_ticks(turn_angle(graph, heading, from_node, to_node))


def advance_heading(
    graph: FactoryGraph,
    heading: Heading,
    from_node: str,
    to_node: str,
) -> Heading:
    """Nova orientação após um movimento from_node -> to_node.

    A orientação só muda quando o robot RODA, e o robot só roda em junctions
    (ou ao seguir a direito num corredor, o que dá a mesma direcção). Numa
    marcha-atrás (recuar num corredor) a orientação NÃO muda.
    """
    if heading.from_node is None:
        return Heading(from_node, to_node)
    if is_reverse_move(graph, heading, from_node, to_node):
        return heading
    return Heading(from_node, to_node)


def _direction(graph: FactoryGraph, u: str, v: str) -> tuple[float, float]:
    """Vector direcção de u para v."""
    ux, uy = graph.node_position(u)
    vx, vy = graph.node_position(v)
    return vx - ux, vy - uy


def _angle_between(d1: tuple[float, float], d2: tuple[float, float]) -> float:
    """Ângulo (graus) entre dois vectores direcção."""
    dx1, dy1 = d1
    dx2, dy2 = d2
    mag1 = math.hypot(dx1, dy1)
    mag2 = math.hypot(dx2, dy2)
    if mag1 < 1e-9 or mag2 < 1e-9:
        return 0.0
    cos_a = max(-1.0, min(1.0, (dx1 * dx2 + dy1 * dy2) / (mag1 * mag2)))
    return math.degrees(math.acos(cos_a))


def turn_angle(
    graph: FactoryGraph,
    heading: Heading,
    current: str,
    next_node: str,
) -> float:
    """Ângulo (graus) entre a orientação actual e a direcção current->next_node.

    0.0 se ainda não há orientação definida (primeiro movimento do robot).
    """
    if heading.from_node is None or heading.to_node is None:
        return 0.0
    d1 = _direction(graph, heading.from_node, heading.to_node)
    d2 = _direction(graph, current, next_node)
    return _angle_between(d1, d2)


def is_curve(angle_degrees: float) -> bool:
    """True se há mudança de direcção (o env só produz 0°, 90° ou 180°)."""
    return angle_degrees > 1e-6


def rotation_ticks(angle_degrees: float) -> int:
    """Ticks de rotação parada, proporcional ao ângulo em múltiplos de 90°."""
    return round(ROTATE_TICKS_PER_90 * (angle_degrees / 90.0))


def effective_max_speed(carrying_box: bool, reverse: bool) -> float:
    """Velocidade de cruzeiro máxima: reduzida a 85% com caixa e a 60% em
    marcha-atrás (acumulam). O chamador determina `reverse` via
    `is_reverse_move` (precisa da orientação do robot)."""
    speed = V_MAX
    if carrying_box:
        speed *= BOX_SPEED_FACTOR
    if reverse:
        speed *= REVERSE_SPEED_FACTOR
    return speed
