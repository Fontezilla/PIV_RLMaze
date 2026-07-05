"""Renderer pygame para o novo env — adaptado do `old/render/renderer.py`.

Diferenças face ao antigo: é conduzido pelos `SchedulePlayer` (posição/
ângulo/velocidade contínuos via `state_at(t)`) e por um log de caixas do
`FactoryEnv` (gravação com `record=True`), em vez do `World`/`Robot`
tick-a-tick. Sem parking (já não existe). Mantém o estilo visual: painel
lateral, caixas (diamante/ponto), badges de entrega, robots como triângulos
orientados com footprint físico.
"""

from __future__ import annotations

import math

import pygame
import pygame.gfxdraw

from env.core.graph import FactoryGraph


BG_COLOR      = (18, 18, 24)
PANEL_BORDER  = (55, 55, 70)
EDGE_COLOR    = (55, 65, 80)
SUBEDGE_COLOR = (36, 40, 50)
TEXT_COLOR    = (220, 220, 235)
DIM_TEXT      = (120, 120, 140)
HEADER_COLOR  = (255, 220, 80)
GOAL_COLOR    = (255, 220, 50)

NODE_COLORS = {
    "entry": (76, 175, 80), "junction": (96, 125, 139),
    "processA_entry": (255, 152, 0), "processA_exit": (255, 204, 128),
    "processB_entry": (33, 150, 243), "processB_exit": (144, 202, 249),
    "exit": (244, 67, 54),
}
BOX_COLORS = {"BLUE": (80, 160, 255), "GREEN": (80, 210, 100), "RED": (255, 90, 90)}

ROBOT_MOVING = (100, 220, 130)
ROBOT_WAIT   = (255, 160, 40)
ROBOT_IDLE   = (149, 61, 168)
ROBOT_DONE   = (180, 180, 210)
STATUS_COLOR = {"MOVING": ROBOT_MOVING, "WAITING": ROBOT_WAIT,
                "AGUARDA": ROBOT_IDLE, "DONE": ROBOT_DONE}

WINDOW_W, WINDOW_H = 1280, 900
GRAPH_FRACTION = 0.75
PADDING = 45
FPS = 60
PHYS_RADIUS = 12.0
NODE_RADIUS = 9
SUBNODE_RADIUS = 3
ROBOT_RADIUS = 11
BOX_ICON = 6


class _Transform:
    """Conversão de coordenadas do mundo para pixels do ecrã (com escala e
    centragem calculadas a partir da caixa envolvente do grafo)."""

    def __init__(self, graph, width, height, padding):
        """Calcula escala e offsets para caber o grafo na área dada."""
        coords = [graph.node_position(n) for n in graph.all_nodes()]
        xs, ys = [c[0] for c in coords], [c[1] for c in coords]
        gxmin, gxmax, gymin, gymax = min(xs), max(xs), min(ys), max(ys)
        aw, ah = width - 2 * padding, height - 2 * padding
        self.scale = min(aw / max(gxmax - gxmin, 1), ah / max(gymax - gymin, 1))
        self.ox = padding + (aw - (gxmax - gxmin) * self.scale) / 2
        self.oy = padding + (ah - (gymax - gymin) * self.scale) / 2
        self.gxmin, self.gymin = gxmin, gymin

    def to_screen(self, x, y):
        """Converte coordenadas de mundo (x, y) em pixels."""
        return (int(self.ox + (x - self.gxmin) * self.scale),
                int(self.oy + (y - self.gymin) * self.scale))

    def node(self, graph, n):
        """Pixels do nó `n`."""
        return self.to_screen(*graph.node_position(n))


def _boxes_at(box_log, t):
    """Snapshot de caixas mais recente com tempo <= t."""
    snap = box_log[0][1]
    for (bt, s) in box_log:
        if bt <= t + 1e-9:
            snap = s
        else:
            break
    return snap


def _draw_triangle(screen, center, size, color, angle):
    """Desenha o triângulo do robot, orientado por `angle`."""
    cx, cy = center
    pts_local = [(size * 1.1, 0.0), (-size * 0.7, size * 0.7), (-size * 0.7, -size * 0.7)]
    ca, sa = math.cos(angle), math.sin(angle)
    pts = [(int(cx + p[0] * ca - p[1] * sa), int(cy + p[0] * sa + p[1] * ca)) for p in pts_local]
    pygame.gfxdraw.filled_polygon(screen, pts, color)
    pygame.gfxdraw.aapolygon(screen, pts, (230, 230, 250))


def _draw_graph(screen, graph, tf):
    """Desenha arestas e nós do mapa (sub-nós mais pequenos e escuros)."""
    for u, v in graph.graph.edges():
        c = SUBEDGE_COLOR if (graph.is_subnode(u) or graph.is_subnode(v)) else EDGE_COLOR
        pygame.draw.line(screen, c, tf.node(graph, u), tf.node(graph, v), 1)
    for n in graph.all_nodes():
        px, py = tf.node(graph, n)
        if graph.is_subnode(n):
            pygame.gfxdraw.filled_circle(screen, px, py, SUBNODE_RADIUS, (70, 75, 90))
            continue
        color = NODE_COLORS.get(graph.node_type(n), (96, 96, 96))
        pygame.gfxdraw.filled_circle(screen, px, py, NODE_RADIUS, color)
        pygame.gfxdraw.aacircle(screen, px, py, NODE_RADIUS, (200, 210, 220))


def _draw_boxes(screen, graph, tf, boxes, robot_pos):
    """Caixas: diamante no nó (WAITING) ou ponto no robot (IN_TRANSIT)."""
    s = BOX_ICON
    for box in boxes:
        if box["status"] == "DONE":
            continue
        color = BOX_COLORS.get(box["pipeline"], (180, 180, 180))
        if box["status"] == "WAITING" and box["current_node"]:
            px, py = tf.node(graph, box["current_node"])
            cy = py - NODE_RADIUS - 4 - s
            pts = [(px, cy - s), (px + s, cy), (px, cy + s), (px - s, cy)]
            pygame.gfxdraw.filled_polygon(screen, pts, (*color, 220))
            pygame.gfxdraw.aapolygon(screen, pts, color)
        elif box["status"] == "IN_TRANSIT":
            carrier = box.get("carried_by")
            if carrier in robot_pos:
                rx, ry = robot_pos[carrier]
                pygame.gfxdraw.filled_circle(screen, rx + ROBOT_RADIUS - 2, ry - ROBOT_RADIUS + 2, 5, (*color, 230))


def _draw_goal_lines(screen, graph, tf, boxes, robot_pos, graph_w, window_h):
    """Linha tracejada do robot para o destino da caixa que transporta."""
    surf = pygame.Surface((graph_w, window_h), pygame.SRCALPHA)
    for box in boxes:
        if box["status"] != "IN_TRANSIT" or not box.get("next_waypoint"):
            continue
        carrier = box.get("carried_by")
        if carrier not in robot_pos:
            continue
        rx, ry = robot_pos[carrier]
        gx, gy = tf.node(graph, box["next_waypoint"])
        dx, dy = gx - rx, gy - ry
        length = math.hypot(dx, dy)
        steps = max(1, int(length / 8))
        for i in range(0, steps, 2):
            t0, t1 = i / steps, min((i + 1) / steps, 1.0)
            pygame.draw.line(surf, (*GOAL_COLOR, 90),
                             (int(rx + dx * t0), int(ry + dy * t0)),
                             (int(rx + dx * t1), int(ry + dy * t1)), 1)
    screen.blit(surf, (0, 0))


def _draw_delivery_badges(screen, graph, tf, font, boxes):
    """Desenha um badge dourado com a contagem de caixas entregues em cada exit."""
    counts = {}
    for box in boxes:
        if box["status"] == "DONE" and box["current_node"]:
            counts[box["current_node"]] = counts.get(box["current_node"], 0) + 1
    for node, n in counts.items():
        try:
            nx, ny = tf.node(graph, node)
        except Exception:
            continue
        bx, by = nx + NODE_RADIUS + 10, ny - 2
        pygame.gfxdraw.filled_circle(screen, bx, by, 10, (30, 30, 35))
        pygame.gfxdraw.aacircle(screen, bx, by, 10, HEADER_COLOR)
        lbl = font.render(str(n), True, HEADER_COLOR)
        screen.blit(lbl, (bx - lbl.get_width() // 2, by - lbl.get_height() // 2))


def play(graph: FactoryGraph, players: dict, box_log: list, title: str = "Factory RL",
         speed: float = 20.0) -> None:
    """Reproduz um episódio gravado: `players` (robot_id -> SchedulePlayer),
    `box_log` (lista (t, snapshot)). `speed` = ticks de sim por segundo real
    (20 = 1x, pois 1 tick = 0.05s)."""
    pygame.init()
    pygame.display.set_caption(title)
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()
    f_sm = pygame.font.SysFont("consolas", 12)
    f_md = pygame.font.SysFont("consolas", 14)
    f_lg = pygame.font.SysFont("consolas", 16, bold=True)
    f_hdr = pygame.font.SysFont("consolas", 18, bold=True)

    graph_w = int(WINDOW_W * GRAPH_FRACTION)
    tf = _Transform(graph, graph_w, WINDOW_H, PADDING)
    foot_px = max(int(PHYS_RADIUS * tf.scale), ROBOT_RADIUS)

    max_time = max((p.arrival_time for p in players.values()), default=0.0)
    max_time = max(max_time, box_log[-1][0] if box_log else 0.0) + 5.0

    t = 0.0
    paused = False
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(speed * 1.5, 400.0)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(speed / 1.5, 1.0)
                elif e.key == pygame.K_r:
                    t = 0.0
        if not paused:
            t = min(t + dt * speed, max_time)

        states = {rid: p.state_at(t) for rid, p in players.items()}
        robot_pos = {rid: tf.to_screen(*s["pos"]) for rid, s in states.items()}
        boxes = _boxes_at(box_log, t)

        screen.fill(BG_COLOR)
        pygame.draw.line(screen, PANEL_BORDER, (graph_w, 0), (graph_w, WINDOW_H), 2)
        _draw_graph(screen, graph, tf)
        _draw_goal_lines(screen, graph, tf, boxes, robot_pos, graph_w, WINDOW_H)
        _draw_delivery_badges(screen, graph, tf, f_md, boxes)
        _draw_boxes(screen, graph, tf, boxes, robot_pos)

        for rid, s in states.items():
            px, py = robot_pos[rid]
            color = STATUS_COLOR.get(s["status"], ROBOT_MOVING)
            pygame.draw.circle(screen, (90, 90, 110), (px, py), foot_px, 1)
            _draw_triangle(screen, (px, py), ROBOT_RADIUS, color, s["angle"])
            lbl = f_sm.render(rid.replace("robot_", "r"), True, (230, 210, 255))
            screen.blit(lbl, (px - lbl.get_width() // 2, py - foot_px - 14))

        _draw_panel(screen, (f_sm, f_md, f_lg, f_hdr), graph_w, WINDOW_W, WINDOW_H,
                    t, speed, states, boxes)
        pygame.display.flip()

    pygame.quit()


def _draw_panel(screen, fonts, panel_x, window_w, window_h, t, speed, states, boxes):
    """Desenha o painel lateral: relógio, entregas, estado de robots e caixas."""
    f_sm, f_md, f_lg, f_hdr = fonts
    px, py = panel_x + 14, [18]

    def txt(s, color=TEXT_COLOR, font=None):
        """Escreve uma linha de texto e avança o cursor vertical."""
        screen.blit((font or f_md).render(s, True, color), (px, py[0]))
        py[0] += 20

    def hline():
        """Desenha um separador horizontal."""
        pygame.draw.line(screen, PANEL_BORDER, (panel_x + 8, py[0]), (window_w - 8, py[0]), 1)
        py[0] += 8

    txt("Factory RL", HEADER_COLOR, f_hdr)
    txt("novo env — SIPP + caixas", DIM_TEXT, f_sm)
    py[0] += 6
    hline()
    txt(f"t        {t:>8.1f}", TEXT_COLOR, f_lg)
    txt(f"veloc.   x{speed / 20:.1f}", DIM_TEXT, f_sm)
    delivered = sum(1 for b in boxes if b["status"] == "DONE")
    txt(f"entregue {delivered}/{len(boxes)}", HEADER_COLOR, f_md)
    py[0] += 6
    hline()

    txt("ROBOTS", HEADER_COLOR, f_lg)
    for rid, s in states.items():
        c = STATUS_COLOR.get(s["status"], TEXT_COLOR)
        txt(f"  {rid.replace('robot_', 'r')}  {s['status']}", c, f_sm)
        if s["status"] == "MOVING":
            txt(f"    {s['from_node']} -> {s['to_node']}  v={s['speed']:.0f}", DIM_TEXT, f_sm)
        else:
            txt(f"    @ {s['from_node']}", DIM_TEXT, f_sm)
    py[0] += 8
    hline()

    txt("CAIXAS", HEADER_COLOR, f_lg)
    for box in boxes:
        if box["status"] == "DONE":
            continue
        c = BOX_COLORS.get(box["pipeline"], TEXT_COLOR)
        bid, st = box["box_id"], box["status"][:4]
        loc = box.get("carried_by") or box.get("current_node") or "?"
        nxt = box.get("next_waypoint") or "?"
        txt(f"  b{bid} {box['pipeline'][:3]} {st} @{loc}->{nxt}", c, f_sm)

    for i, hint in enumerate(["SPACE pausa", "+/- velocidade", "R reinicia", "Q sai"]):
        screen.blit(f_sm.render(hint, True, DIM_TEXT), (px, window_h - 72 + i * 16))
