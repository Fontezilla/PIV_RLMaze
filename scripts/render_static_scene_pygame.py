"""Abre o renderer pygame REAL do projecto (env/render/renderer.py) com
robots colocados em posições/orientações escolhidas, parados — para tirar
screenshots do render real (não uma imagem à parte) para os slides. Não há
episódio nenhum a correr; os robots ficam estáticos, com o mesmo estilo
visual do render usado no resto da apresentação (aparecem a laranja,
estado "WAITING" — apropriado para ilustrar um cenário de conflito/espera).

Uso (da raiz do projecto, precisa de pygame instalado):
    python scripts/render_static_scene_pygame.py --robot r1:B->C --robot r2:C->B

Cada `--robot` é "ID:NÓ_ATUAL->NÓ_PARA_ONDE_APONTA" — o ângulo é calculado
directamente entre as posições dos dois nós (não têm de ser vizinhos no
grafo). Abre a janela já parada nessa pose — usa Q para sair depois do
screenshot.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.core.graph import FactoryGraph
from env.physics import engine, kinematics
from env.traffic.cooperative_astar import ScheduleEntry
from env.render import renderer

YAML_PATH = Path(__file__).parent.parent / ".configs" / "map_factory.yaml"
CACHE_PATH = Path(__file__).parent.parent / ".configs" / "graph_cache.pkl"

SETTLE_TIME = 10.0  # instante a partir do qual o robot já está "assente" na pose

_ZERO_PROFILE = kinematics.SegmentProfile(
    entry_speed=0.0, accel=0.0, accel_time=0.0, accel_dist=0.0,
    cruise_time=0.0, cruise_dist=0.0, decel_time=0.0, decel_dist=0.0,
    peak_speed=0.0, exit_speed=0.0,
)


def _build_parked_player(graph: FactoryGraph, robot_id: str, at: str, facing: str) -> engine.SchedulePlayer:
    """`SchedulePlayer` parado em `at`, com o ângulo calculado directamente
    na direcção de `facing` (qualquer nó real, não precisa de ser vizinho).

    Constrói a perna manualmente (auto-loop `at`→`at`, distância zero) em
    vez de usar `build_legs` — este calcula o ângulo a partir da aresta
    PERCORRIDA (chegada), que fica invertido face ao que se quer aqui
    (apontar PARA `facing`, não vir DE lá)."""
    x0, y0 = graph.node_position(at)
    x1, y1 = graph.node_position(facing)
    angle = math.atan2(y1 - y0, x1 - x0) if (x0, y0) != (x1, y1) else 0.0

    leg = engine._Leg(from_node=at, to_node=at, depart=0.0, arrive=SETTLE_TIME,
                      profile=_ZERO_PROFILE, face_angle=angle)
    schedule = [
        ScheduleEntry(node=at, arrival=0.0, depart=0.0, speed=0.0),
        ScheduleEntry(node=at, arrival=SETTLE_TIME, depart=float("inf"), speed=0.0),
    ]

    player = engine.SchedulePlayer.__new__(engine.SchedulePlayer)
    player.robot_id = robot_id
    player.graph = graph
    player.carrying_box = False
    player.schedule = schedule
    player.legs = [leg]
    player._heading = None
    return player


def main() -> None:
    """Lê os `--robot ID:NO->NO`, monta os players parados e abre o renderer."""
    parser = argparse.ArgumentParser(description="Render pygame real com robots em poses fixas")
    parser.add_argument("--robot", action="append", default=[],
                        help="ID:NÓ->NÓ_PARA_ONDE_APONTA (repetível)")
    parser.add_argument("--title", type=str, default="Factory RL — cena estática")
    args = parser.parse_args()

    if not args.robot:
        raise SystemExit("Dá pelo menos um --robot ID:NO->NO")

    graph = FactoryGraph(str(YAML_PATH), str(CACHE_PATH))

    players: dict[str, engine.SchedulePlayer] = {}
    for spec in args.robot:
        if ":" not in spec or "->" not in spec:
            raise ValueError(f"--robot inválido (esperado ID:NO->NO): {spec!r}")
        robot_id, rest = spec.split(":", 1)
        at, facing = rest.split("->", 1)
        players[robot_id.strip()] = _build_parked_player(
            graph, robot_id.strip(), at.strip(), facing.strip()
        )

    box_log = [(0.0, [])]
    renderer.play(graph, players, box_log, title=args.title,
                  speed=0.0, start_time=SETTLE_TIME + 1.0)


if __name__ == "__main__":
    main()
