"""Pygame renderer para visualização em tempo real da simulação v0.4.

Layout
------
  ┌──────────────────────────────────────────┬────────────────────────┐
  │                                          │  Factory RL  v0.4      │
  │             Graph area                   │  Tick     1 234        │
  │    (nodes, edges, robots, boxes)         │  Delivered  3 / 4      │
  │                                          │                        │
  │                                          │  BOXES                 │
  │                                          │  ■ B1  blue  @ entryA  │
  │                                          │  ■ B2  red   ▲ R1      │
  │                                          │                        │
  │                                          │  ROBOTS                │
  │                                          │  ▲ R1  MOVING          │
  │                                          │    A → B  (72 %)       │
  │                                          │  ▲ R2  IDLE @ A        │
  │                                          │                        │
  │                                          │  SPACE  pause          │
  │                                          │  +/-    sub-steps      │
  │                                          │  Q      quit           │
  └──────────────────────────────────────────┴────────────────────────┘

Controlos
---------
  SPACE   Pausa / retoma.
  +/-     Aumenta / diminui sub-steps (animação mais suave ↔ mais rápida).
  Q / ✕   Sai.
"""

from __future__ import annotations

import math
import sys
import os
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation_engine.core.world_state import WorldState
from simulation_engine.core.entities import RobotState, BoxState
from simulation_engine.core.graph import FactoryGraph

try:
    import pygame
    import pygame.gfxdraw
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

# ---------------------------------------------------------------------------
# Paleta de cores
# ---------------------------------------------------------------------------

BG_COLOR        = (18,  18,  24)
PANEL_BG        = (28,  28,  36)
PANEL_BORDER    = (55,  55,  70)
EDGE_COLOR      = (55,  65,  80)
EDGE_OCC_COLOR  = (100, 130, 180)
EDGE_GLOW_COLOR = (50,  80, 130)

NODE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "entry":          ( 76, 175,  80),
    "junction":       ( 96, 125, 139),
    "processA_entry": (255, 152,   0),
    "processA_exit":  (255, 204, 128),
    "processB_entry": ( 33, 150, 243),
    "processB_exit":  (144, 202, 249),
    "exit":           (244,  67,  54),
}

BOX_COLORS: Dict[str, Tuple[int, int, int]] = {
    "blue":  ( 41, 182, 246),
    "red":   (239,  83,  80),
    "green": (102, 187, 106),
}

ROBOT_COLOR      = (186,  85, 211)
ROBOT_IDLE_COLOR = (149,  61, 168)
TEXT_COLOR       = (220, 220, 235)
DIM_TEXT         = (120, 120, 140)
HEADER_COLOR     = (255, 220,  80)
DELIVERED_COLOR  = ( 80, 200, 120)
WARNING_COLOR    = (255, 160,  40)

NODE_RADIUS    = 9
BOX_HALF       = 6
ROBOT_RADIUS   = 11   # raio visual (px)
PHYSICS_RADIUS = 20.0 # raio físico de colisão (world units) — deve coincidir com collision_system

MIN_SUB_STEPS = 1
MAX_SUB_STEPS = 12

_PixelPos = Tuple[float, float]


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class PygameRenderer:
    """Renderer pygame para o SimulationEngine v0.4.

    Sub-tick interpolation
    ----------------------
    Cada chamada a :meth:`render` desenha ``sub_steps`` frames intermédios
    que interpolam linearmente as posições dos robôs entre o tick anterior
    e o atual. Desacopla a frame rate visual da taxa de tick física.

    Parameters
    ----------
    graph:
        Grafo da fábrica (topologia + coordenadas).
    window_width, window_height:
        Tamanho da janela em pixels.
    graph_fraction:
        Fração da largura da janela dedicada à área do grafo.
    fps:
        Target frame rate por sub-step.
    sub_steps:
        Número de frames interpolados por tick. Ajustável com +/-.
    """

    def __init__(
        self,
        graph: FactoryGraph,
        window_width: int = 1280,
        window_height: int = 900,
        graph_fraction: float = 0.75,
        fps: int = 60,
        sub_steps: int = 6,
    ) -> None:
        if not _HAS_PYGAME:
            raise ImportError(
                "pygame é necessário para o PygameRenderer.\n"
                "Instala com: pip install pygame"
            )

        self.graph = graph
        self.window_width = window_width
        self.window_height = window_height
        self.fps = fps
        self.sub_steps = max(MIN_SUB_STEPS, min(MAX_SUB_STEPS, sub_steps))

        self._graph_w = int(window_width * graph_fraction)
        self._panel_x = self._graph_w

        pygame.init()
        pygame.display.set_caption("Factory RL — v0.4")
        self._screen = pygame.display.set_mode((window_width, window_height))
        self._clock = pygame.time.Clock()

        pygame.font.init()
        self._font_sm  = pygame.font.SysFont("consolas", 12)
        self._font_md  = pygame.font.SysFont("consolas", 14)
        self._font_lg  = pygame.font.SysFont("consolas", 16, bold=True)
        self._font_hdr = pygame.font.SysFont("consolas", 18, bold=True)

        self._transform = self._build_transform(padding=45)
        self._paused = False
        self._quit   = False

        # Posições pixel do tick anterior para interpolação
        self._prev_robot_pos: Dict[str, _PixelPos] = {}

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def render(self, world: WorldState, info: Optional[Dict] = None) -> str:
        """Desenha um tick da simulação como ``sub_steps`` frames interpolados.

        Parameters
        ----------
        world:
            Estado atual do mundo.
        info:
            Dicionário opcional com informação extra para o painel
            (ex: ``{"episode": 3, "reward": 12.5}``).

        Returns
        -------
        ``"quit"`` / ``"ok"``
        """
        # Honrar pausa
        while True:
            action = self._process_events()
            if action == "quit":
                return "quit"
            if not self._paused:
                break
            self._draw_frame(world, info, robot_overrides=None)
            self._draw_pause_overlay()
            pygame.display.flip()
            self._clock.tick(self.fps)

        curr_robot_pos = self._compute_robot_positions(world)
        n = self.sub_steps if self._prev_robot_pos else 1

        for step in range(n):
            action = self._process_events()
            if action == "quit":
                self._prev_robot_pos = curr_robot_pos
                return "quit"

            t = (step + 1) / n
            interp = self._lerp_positions(self._prev_robot_pos, curr_robot_pos, t)

            self._draw_frame(world, info, robot_overrides=interp)
            self._draw_substep_indicator(step + 1, n)
            pygame.display.flip()
            self._clock.tick(self.fps)

        self._prev_robot_pos = curr_robot_pos
        return "ok"

    def close(self) -> None:
        """Fecha a janela pygame."""
        pygame.display.quit()

    # ------------------------------------------------------------------
    # Eventos
    # ------------------------------------------------------------------

    def _process_events(self) -> str:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return "quit"
                if event.key == pygame.K_SPACE:
                    self._paused = not self._paused
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.sub_steps = min(MAX_SUB_STEPS, self.sub_steps + 1)
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.sub_steps = max(MIN_SUB_STEPS, self.sub_steps - 1)
        return "ok"

    # ------------------------------------------------------------------
    # Frame principal
    # ------------------------------------------------------------------

    def _draw_frame(
        self,
        world: WorldState,
        info: Optional[Dict],
        robot_overrides: Optional[Dict[str, _PixelPos]],
    ) -> None:
        self._screen.fill(BG_COLOR)
        pygame.draw.line(self._screen, PANEL_BORDER,
                         (self._panel_x, 0), (self._panel_x, self.window_height), 2)

        goals = (info or {}).get("goals", {})

        self._draw_edges(world)
        self._draw_goal_lines(world, goals, robot_overrides)
        self._draw_nodes(world, goals)
        self._draw_boxes(world, robot_overrides)
        self._draw_robots(world, robot_overrides)
        self._draw_info_panel(world, info)

    def _draw_pause_overlay(self) -> None:
        surf = pygame.Surface((self._graph_w, self.window_height), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 120))
        self._screen.blit(surf, (0, 0))
        txt = self._font_hdr.render("  PAUSED  —  SPACE para retomar", True, HEADER_COLOR)
        x = (self._graph_w - txt.get_width()) // 2
        y = (self.window_height - txt.get_height()) // 2
        self._screen.blit(txt, (x, y))

    def _draw_substep_indicator(self, step: int, total: int) -> None:
        for i in range(total):
            color = (180, 180, 255) if i < step else (50, 50, 70)
            pygame.gfxdraw.filled_circle(self._screen, 10 + i * 8, 8, 3, color)

    # ------------------------------------------------------------------
    # Área do grafo
    # ------------------------------------------------------------------

    def _draw_edges(self, world: WorldState) -> None:
        occupied: set = set()
        for r in world.robots.values():
            if r.state == RobotState.MOVING and r.from_node and r.to_node:
                occupied.add((r.from_node, r.to_node))
                occupied.add((r.to_node, r.from_node))

        drawn: set = set()
        for u, v in self.graph.graph.edges():
            key = tuple(sorted([u, v]))
            if key in drawn:
                continue
            drawn.add(key)
            p0 = self._to_screen(u)
            p1 = self._to_screen(v)

            if (u, v) in occupied or (v, u) in occupied:
                pygame.draw.line(self._screen, EDGE_GLOW_COLOR, p0, p1, 5)
                pygame.draw.line(self._screen, EDGE_OCC_COLOR,  p0, p1, 2)
            else:
                pygame.draw.aaline(self._screen, EDGE_COLOR, p0, p1)

    def _draw_goal_lines(
        self,
        world: WorldState,
        goals: Dict[str, str],
        overrides: Optional[Dict[str, _PixelPos]],
    ) -> None:
        """Linhas amarelas semi-transparentes de cada robot ao seu nó destino."""
        if not goals:
            return

        surf = pygame.Surface((self._graph_w, self.window_height), pygame.SRCALPHA)

        for rid, goal_node in goals.items():
            robot = world.robots.get(rid)
            if robot is None:
                continue

            if overrides and rid in overrides:
                rx, ry = int(overrides[rid][0]), int(overrides[rid][1])
            else:
                pos = self._robot_screen_pos(robot)
                if pos is None:
                    continue
                rx, ry = pos

            try:
                gx, gy = self._to_screen(goal_node)
            except Exception:
                continue

            # linha pontilhada — segmentos alternados
            dx, dy = gx - rx, gy - ry
            length = math.hypot(dx, dy)
            if length < 1:
                continue
            seg = 8
            steps = max(1, int(length / seg))
            for i in range(0, steps, 2):
                t0, t1 = i / steps, min((i + 1) / steps, 1.0)
                x0 = int(rx + dx * t0)
                y0 = int(ry + dy * t0)
                x1 = int(rx + dx * t1)
                y1 = int(ry + dy * t1)
                pygame.draw.line(surf, (255, 220, 50, 90), (x0, y0), (x1, y1), 1)

        self._screen.blit(surf, (0, 0))

    def _draw_nodes(self, world: WorldState, goals: Optional[Dict[str, str]] = None) -> None:
        goal_nodes: set = set((goals or {}).values())

        for node in self.graph.graph.nodes:
            ntype = self.graph.node_type(node)
            color = NODE_COLORS.get(ntype, (96, 96, 96))
            px, py = self._to_screen(node)

            # anel amarelo nos nós que são destino de algum robot
            if node in goal_nodes:
                pygame.gfxdraw.aacircle(self._screen, px, py, NODE_RADIUS + 5, (255, 220, 50))
                pygame.gfxdraw.aacircle(self._screen, px, py, NODE_RADIUS + 6, (200, 160, 30))

            pygame.gfxdraw.filled_circle(self._screen, px, py, NODE_RADIUS, color)
            pygame.gfxdraw.aacircle(self._screen, px, py, NODE_RADIUS, (200, 210, 220))

            lbl = self._font_sm.render(node, True, DIM_TEXT)
            self._screen.blit(lbl, (px - lbl.get_width() // 2, py + NODE_RADIUS + 2))

    def _draw_robots(
        self,
        world: WorldState,
        overrides: Optional[Dict[str, _PixelPos]],
    ) -> None:
        for rid, r in world.robots.items():
            if overrides and rid in overrides:
                fx, fy = overrides[rid]
                pos = (int(fx), int(fy))
            else:
                pos = self._robot_screen_pos(r)
            if pos is None:
                continue

            angle = self._robot_direction_angle(r)

            waiting_at_boundary = r.state == RobotState.MOVING and r.progress >= 1.0

            if waiting_at_boundary:
                color = (255, 200, 50)   # amarelo — à espera na fronteira do nó
            elif r.is_turning:
                color = (255, 160, 40)   # laranja — a rodar na junção
            elif r.state == RobotState.MOVING:
                color = (100, 220, 130)  # verde — em movimento
            else:
                color = ROBOT_IDLE_COLOR if r.carried_box else ROBOT_COLOR  # roxo — idle

            # Radius físico (semitransparente)
            r_px = max(1, int(PHYSICS_RADIUS * self._transform["scale"]))
            if r_px > 1:
                surf = pygame.Surface((r_px * 2 + 2, r_px * 2 + 2), pygame.SRCALPHA)
                if waiting_at_boundary:
                    ring_fill, ring_border = (255, 200, 50, 50), (255, 220, 80, 120)
                elif r.is_turning:
                    ring_fill, ring_border = (255, 160, 40, 50), (255, 180, 60, 100)
                else:
                    ring_fill, ring_border = (180, 180, 255, 40), (160, 160, 255, 90)
                pygame.gfxdraw.filled_circle(surf, r_px + 1, r_px + 1, r_px, ring_fill)
                pygame.gfxdraw.aacircle(surf, r_px + 1, r_px + 1, r_px, ring_border)
                self._screen.blit(surf, (pos[0] - r_px - 1, pos[1] - r_px - 1))

            self._draw_triangle(pos, ROBOT_RADIUS, color, angle)
            pygame.gfxdraw.aacircle(self._screen, pos[0], pos[1], ROBOT_RADIUS, (230, 230, 250))

            lbl = self._font_sm.render(rid, True, (230, 180, 255))
            self._screen.blit(lbl, (pos[0] - lbl.get_width() // 2, pos[1] - ROBOT_RADIUS - 14))

    def _draw_boxes(
        self,
        world: WorldState,
        overrides: Optional[Dict[str, _PixelPos]],
    ) -> None:
        for bid, b in world.boxes.items():
            color = BOX_COLORS.get(b.pipeline_type, (180, 180, 180))

            if b.state == BoxState.DELIVERED:
                # Boxes entregues não têm current_node — omitir do grafo
                continue

            pos = self._box_screen_pos(b, world, overrides)
            if pos is None:
                continue
            px, py = int(pos[0]), int(pos[1])

            rect = pygame.Rect(px - BOX_HALF, py - BOX_HALF, BOX_HALF * 2, BOX_HALF * 2)
            pygame.draw.rect(self._screen, color, rect, border_radius=2)
            pygame.draw.rect(self._screen, (220, 220, 240), rect, 1, border_radius=2)

            lbl = self._font_sm.render(bid[-1], True, (20, 20, 20))
            self._screen.blit(lbl, (px - lbl.get_width() // 2, py - lbl.get_height() // 2))

    def _draw_triangle(
        self,
        center: Tuple[int, int],
        size: int,
        color: Tuple[int, int, int],
        angle_rad: float = -math.pi / 2,
    ) -> None:
        cx, cy = center
        tip_len   = size * 1.1
        base_half = size * 0.7
        local_pts = [
            ( tip_len,        0.0),
            (-base_half,  base_half),
            (-base_half, -base_half),
        ]
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        pts = [
            (int(cx + p[0] * cos_a - p[1] * sin_a),
             int(cy + p[0] * sin_a + p[1] * cos_a))
            for p in local_pts
        ]
        pygame.gfxdraw.filled_polygon(self._screen, pts, color)
        pygame.gfxdraw.aapolygon(self._screen, pts, color)

    # ------------------------------------------------------------------
    # Painel lateral
    # ------------------------------------------------------------------

    def _draw_info_panel(self, world: WorldState, info: Optional[Dict]) -> None:
        px = self._panel_x + 14
        py = 18
        lh = 20

        def txt(text: str, color=TEXT_COLOR, font=None) -> None:
            nonlocal py
            f = font or self._font_md
            surf = f.render(text, True, color)
            self._screen.blit(surf, (px, py))
            py += lh

        def sep(h: int = 8) -> None:
            nonlocal py
            py += h

        def hline() -> None:
            nonlocal py
            pygame.draw.line(self._screen, PANEL_BORDER,
                             (self._panel_x + 8, py), (self.window_width - 8, py), 1)
            sep(8)

        txt("Factory RL", HEADER_COLOR, self._font_hdr)
        txt("v0.4", DIM_TEXT, self._font_sm)
        sep(10); hline()

        txt(f"Tick      {world.tick:,}", TEXT_COLOR, self._font_lg)

        delivered = len(world.delivered_boxes)
        total = len(world.boxes)
        txt(f"Delivered {delivered} / {total}",
            DELIVERED_COLOR if delivered == total and total > 0 else TEXT_COLOR)

        if info:
            sep(4)
            for k, v in info.items():
                if k == "goals":
                    continue  # renderizado no grafo, não no painel
                if isinstance(v, float):
                    txt(f"{k:<10} {v:+.2f}", DIM_TEXT, self._font_sm)
                else:
                    txt(f"{k:<10} {v}", DIM_TEXT, self._font_sm)

        txt(f"SubSteps  {self.sub_steps}", DIM_TEXT, self._font_sm)
        sep(12); hline()

        txt("BOXES", HEADER_COLOR, self._font_lg); sep(2)
        for bid, b in world.boxes.items():
            color = BOX_COLORS.get(b.pipeline_type, TEXT_COLOR)
            if b.state == BoxState.DELIVERED:
                status, lc = "✓ delivered", DELIVERED_COLOR
            elif b.state == BoxState.IN_TRANSPORT:
                status, lc = f"▲ {b.carried_by or '?'}", color
            elif b.state == BoxState.AT_NODE:
                status, lc = f"@ {b.current_node}", color
            else:
                status, lc = b.state.value, DIM_TEXT
            txt(f"  {bid:<7} {b.pipeline_type:<6}  {status}", lc, self._font_sm)

        sep(12); hline()

        txt("ROBOTS", HEADER_COLOR, self._font_lg); sep(2)
        for rid, r in world.robots.items():
            if r.state == RobotState.MOVING:
                pct = int(r.progress * 100)
                if r.progress >= 1.0:
                    txt(f"  {rid}  WAIT @ {r.to_node}", (255, 200, 50), self._font_sm)
                    txt(f"    {r.from_node} -> {r.to_node}  (aguarda)", DIM_TEXT, self._font_sm)
                else:
                    txt(f"  {rid}  MOVING", (160, 200, 255), self._font_sm)
                    txt(f"    {r.from_node} -> {r.to_node}  ({pct}%)", DIM_TEXT, self._font_sm)
            else:
                cargo = f" [{r.carried_box}]" if r.carried_box else ""
                yield_tag = f" yield:{r.yield_ticks}" if r.yield_ticks > 0 else ""
                txt(f"  {rid}  IDLE{cargo}{yield_tag}", (160, 220, 160), self._font_sm)
                txt(f"    @ {r.current_node}", DIM_TEXT, self._font_sm)

        hint_y = self.window_height - 72
        for hint in ["SPACE  pause / retoma", "+/-    sub-steps", "Q      sair"]:
            self._screen.blit(self._font_sm.render(hint, True, DIM_TEXT), (px, hint_y))
            hint_y += 16

    # ------------------------------------------------------------------
    # Helpers de posição
    # ------------------------------------------------------------------

    def _compute_robot_positions(self, world: WorldState) -> Dict[str, _PixelPos]:
        result: Dict[str, _PixelPos] = {}
        for rid, r in world.robots.items():
            pos = self._robot_screen_pos(r)
            if pos is not None:
                result[rid] = (float(pos[0]), float(pos[1]))
        return result

    @staticmethod
    def _lerp_positions(
        prev: Dict[str, _PixelPos],
        curr: Dict[str, _PixelPos],
        t: float,
    ) -> Dict[str, _PixelPos]:
        result: Dict[str, _PixelPos] = {}
        for rid, c in curr.items():
            if rid in prev:
                p = prev[rid]
                result[rid] = (p[0] + (c[0] - p[0]) * t, p[1] + (c[1] - p[1]) * t)
            else:
                result[rid] = c
        return result

    def _robot_direction_angle(self, r) -> float:
        # Animação de rotação durante turn_cooldown
        if r.is_turning and r.rotation_ticks_total > 0:
            t = 1.0 - r.turn_cooldown / r.rotation_ticks_total
            diff = (r.rotation_angle_to - r.rotation_angle_from + math.pi) % (2 * math.pi) - math.pi
            return r.rotation_angle_from + diff * t

        if r.state == RobotState.MOVING and r.from_node and r.to_node:
            fx, fy = self._to_screen(r.from_node)
            tx, ty = self._to_screen(r.to_node)
            dx, dy = tx - fx, ty - fy
            if dx != 0 or dy != 0:
                return math.atan2(dy, dx)

        # Robot parado no nó: manter direção de chegada se disponível
        if r.prev_node and r.current_node:
            fx, fy = self._to_screen(r.prev_node)
            tx, ty = self._to_screen(r.current_node)
            dx, dy = tx - fx, ty - fy
            if dx != 0 or dy != 0:
                return math.atan2(dy, dx)

        return -math.pi / 2  # apontar para cima por defeito

    # ------------------------------------------------------------------
    # Helpers de coordenadas
    # ------------------------------------------------------------------

    def _build_transform(self, padding: int) -> Dict:
        coords = [self.graph.node_position(n) for n in self.graph.graph.nodes]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        gx_min, gx_max = min(xs), max(xs)
        gy_min, gy_max = min(ys), max(ys)

        avail_w = self._graph_w - 2 * padding
        avail_h = self.window_height - 2 * padding
        scale_x = avail_w / max(gx_max - gx_min, 1)
        scale_y = avail_h / max(gy_max - gy_min, 1)
        scale = min(scale_x, scale_y)

        rendered_w = (gx_max - gx_min) * scale
        rendered_h = (gy_max - gy_min) * scale
        ox = padding + (avail_w - rendered_w) / 2
        oy = padding + (avail_h - rendered_h) / 2

        return {"scale": scale, "gx_min": gx_min, "gy_min": gy_min, "ox": ox, "oy": oy}

    def _to_screen(self, node: str) -> Tuple[int, int]:
        gx, gy = self.graph.node_position(node)
        t = self._transform
        return (
            int(t["ox"] + (gx - t["gx_min"]) * t["scale"]),
            int(t["oy"] + (gy - t["gy_min"]) * t["scale"]),
        )

    def _interp(self, node_a: str, node_b: str, progress: float) -> Tuple[int, int]:
        ax, ay = self._to_screen(node_a)
        bx, by = self._to_screen(node_b)
        return int(ax + (bx - ax) * progress), int(ay + (by - ay) * progress)

    def _robot_screen_pos(self, r) -> Optional[Tuple[int, int]]:
        if r.state == RobotState.MOVING and r.from_node and r.to_node:
            return self._interp(r.from_node, r.to_node, r.progress)
        if r.current_node:
            return self._to_screen(r.current_node)
        return None

    def _box_screen_pos(
        self,
        b,
        world: WorldState,
        overrides: Optional[Dict[str, _PixelPos]],
    ) -> Optional[_PixelPos]:
        if b.state == BoxState.AT_NODE and b.current_node:
            bx, by = self._to_screen(b.current_node)
            return float(bx + NODE_RADIUS + BOX_HALF + 2), float(by)
        if b.state == BoxState.IN_TRANSPORT and b.carried_by:
            r = world.robots.get(b.carried_by)
            if r is None:
                return None
            if overrides and b.carried_by in overrides:
                return overrides[b.carried_by]
            pos = self._robot_screen_pos(r)
            if pos is None:
                return None
            return float(pos[0] + BOX_HALF + 2), float(pos[1])
        return None
