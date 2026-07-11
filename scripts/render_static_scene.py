"""Gera uma imagem estática (PNG) do mapa real da fábrica com robots
colocados em posições/orientações escolhidas — para ilustrações em slides
(ex. cenário de conflito/segurança), sem depender do pygame nem de um
episódio corrido. Usa o mesmo grafo (`FactoryGraph`) e cores do resto do
projecto, para ficar visualmente consistente com o render 3D e o pygame.

Uso (da raiz do projecto):
    python scripts/render_static_scene.py --robot N->A --robot E->G \
        --out scratchpad/safety_scene.png --label "Safety"

Cada `--robot` é "NÓ_ATUAL->NÓ_PARA_ONDE_APONTA" (a seta desenha-se na
direcção desse vizinho; o robot fica centrado no primeiro nó).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, RegularPolygon

from env.core.graph import FactoryGraph

YAML_PATH = Path(__file__).parent.parent / ".configs" / "map_factory.yaml"
CACHE_PATH = Path(__file__).parent.parent / ".configs" / "graph_cache.pkl"

EDGE_COLOR = "black"
EDGE_WIDTH = 3.5
SQUARE_SIZE = 22.0
SQUARE_COLOR = "#9a9a9a"
ROBOT_COLOR = "#ff9800"
ROBOT_RADIUS = 26.0
BORDER_COLOR = "#ff9800"
BG_COLOR = "white"

# Tipos de nó desenhados como quadrado oco (estações), o resto é só o
# cruzamento das linhas (sem marcador), como no mapa original.
SQUARE_TYPES = {"entry", "exit", "processA_entry", "processA_exit",
                "processB_entry", "processB_exit"}


def _draw_robot(ax, graph: FactoryGraph, at: str, facing: str) -> None:
    """Desenha um robot (círculo laranja + seta) no nó `at`, apontando para
    o vizinho `facing`."""
    x, y = graph.node_position(at)
    fx, fy = graph.node_position(facing)
    angle = math.atan2(fy - y, fx - x)

    ax.add_patch(plt.Circle((x, y), ROBOT_RADIUS, facecolor="white",
                            edgecolor=ROBOT_COLOR, linewidth=2.5, zorder=5))
    tri = RegularPolygon((x, y), numVertices=3, radius=ROBOT_RADIUS * 0.55,
                        orientation=-angle + math.pi / 2,
                        facecolor=ROBOT_COLOR, edgecolor="none", zorder=6)
    ax.add_patch(tri)


def render_scene(robots: list[tuple[str, str]], out_path: Path, label: str | None) -> None:
    """Desenha o mapa real + robots dados e grava em `out_path`."""
    graph = FactoryGraph(str(YAML_PATH), str(CACHE_PATH))

    xs = [graph.node_position(n)[0] for n in graph.real_nodes()]
    ys = [graph.node_position(n)[1] for n in graph.real_nodes()]

    fig, ax = plt.subplots(figsize=(11, 8), dpi=150)
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    for u, v in graph.real_edges():
        x0, y0 = graph.node_position(u)
        x1, y1 = graph.node_position(v)
        ax.plot([x0, x1], [y0, y1], color=EDGE_COLOR, linewidth=EDGE_WIDTH,
                solid_capstyle="butt", zorder=1)

    for n in graph.real_nodes():
        if graph.node_type(n) not in SQUARE_TYPES:
            continue
        x, y = graph.node_position(n)
        ax.add_patch(Rectangle(
            (x - SQUARE_SIZE / 2, y - SQUARE_SIZE / 2), SQUARE_SIZE, SQUARE_SIZE,
            facecolor="none", edgecolor=SQUARE_COLOR, linewidth=1.8, zorder=2,
        ))

    for at, facing in robots:
        _draw_robot(ax, graph, at, facing)

    pad = 90
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(max(ys) + pad, min(ys) - pad)  # y invertido (mesma convenção do mapa)
    ax.set_aspect("equal")
    ax.axis("off")

    # Moldura laranja à volta de toda a cena, com o rótulo opcional
    # (canto superior direito, como nas caixas de destaque dos slides).
    fig_border = FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98, transform=fig.transFigure,
        boxstyle="round,pad=0.005,rounding_size=0.01",
        facecolor="none", edgecolor=BORDER_COLOR, linewidth=3,
    )
    fig.patches.append(fig_border)

    if label:
        fig.text(0.97, 0.965, label, ha="right", va="top", fontsize=22,
                weight="bold", family="sans-serif",
                bbox=dict(boxstyle="square,pad=0.6", facecolor="white",
                          edgecolor=BORDER_COLOR, linewidth=3))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG_COLOR, bbox_inches="tight", pad_inches=0.15)
    print(f"gravado em: {out_path}")


def main() -> None:
    """Lê os argumentos `--robot NO->PARA` e gera a imagem."""
    parser = argparse.ArgumentParser(description="Cena estática do mapa real, para slides")
    parser.add_argument("--robot", action="append", default=[],
                        help="NÓ->NÓ_PARA_ONDE_APONTA (repetível, um por robot)")
    parser.add_argument("--out", type=str, default="scratchpad/static_scene.png")
    parser.add_argument("--label", type=str, default=None,
                        help="Rótulo no canto superior direito (ex.: 'Safety')")
    args = parser.parse_args()

    robots: list[tuple[str, str]] = []
    for spec in args.robot:
        if "->" not in spec:
            raise ValueError(f"--robot inválido (esperado NO->NO): {spec!r}")
        at, facing = spec.split("->", 1)
        robots.append((at.strip(), facing.strip()))

    render_scene(robots, Path(args.out), args.label)


if __name__ == "__main__":
    main()
