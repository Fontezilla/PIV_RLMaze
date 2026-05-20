from __future__ import annotations

import math
from typing import Optional

try:
    import pygame
    import pygame.gfxdraw
    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

from env.core.entities import Robot, RobotState
from env.core.graph import FactoryGraph
from env.core.world import World

BG_COLOR = (18, 18, 24)
PANEL_BORDER = (55, 55, 70)
EDGE_COLOR = (55, 65, 80)
EDGE_OCC_COLOR = (100, 130, 180)
EDGE_GLOW = (50, 80, 130)

NODE_COLORS: dict[str, tuple[int, int, int]] = {
    "entry": (76, 175, 80),
    "junction": (96, 125, 139),
    "processA_entry": (255, 152, 0),
    "processA_exit": (255, 204, 128),
    "processB_entry": (33, 150, 243),
    "processB_exit": (144, 202, 249),
    "exit": (244, 67, 54),
}

ROBOT_MOVING_COLOR = (100, 220, 130)
ROBOT_IDLE_COLOR = (149, 61, 168)
ROBOT_WAIT_COLOR = (255, 160, 40)
ROBOT_PARKED_COLOR = (180, 180, 210)

GOAL_RING_COLOR = (255, 220, 50)
TEXT_COLOR = (220, 220, 235)
DIM_TEXT = (120, 120, 140)
HEADER_COLOR = (255, 220, 80)

PARKING_POINT_COLOR = (185, 185, 195)
PARKING_POINT_BORDER = (90, 90, 105)

BOX_PIPELINE_COLORS: dict[str, tuple[int, int, int]] = {
    "BLUE":  (80, 160, 255),
    "GREEN": (80, 210, 100),
    "RED":   (255, 90,  90),
}
BOX_ICON_SIZE = 5

NODE_RADIUS = 9
ROBOT_RADIUS = 11

# 10% exacto do NODE_RADIUS ficava quase invisível.
# 2 px é o mínimo prático para se ver no pygame.
PARKING_POINT_RADIUS = max(2, round(NODE_RADIUS * 0.15))

MIN_SUB_STEPS = 1
MAX_SUB_STEPS = 12

_PixelPos = tuple[float, float]


class Renderer:
    """Renderer pygame para a v0.6 — com visualização de parking points geométricos."""

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
            raise ImportError("pygame não instalado — pip install pygame")

        self.graph = graph
        self.window_width = window_width
        self.window_height = window_height
        self.fps = fps
        self.sub_steps = max(MIN_SUB_STEPS, min(MAX_SUB_STEPS, sub_steps))

        self._graph_w = int(window_width * graph_fraction)
        self._panel_x = self._graph_w

        pygame.init()
        pygame.display.set_caption("Factory RL — v0.6")
        self._screen = pygame.display.set_mode((window_width, window_height))
        self._clock = pygame.time.Clock()

        pygame.font.init()
        self._font_sm = pygame.font.SysFont("consolas", 12)
        self._font_md = pygame.font.SysFont("consolas", 14)
        self._font_lg = pygame.font.SysFont("consolas", 16, bold=True)
        self._font_hdr = pygame.font.SysFont("consolas", 18, bold=True)

        self._transform = self._build_transform(padding=45)
        self._paused = False
        self._prev_pos: dict[str, _PixelPos] = {}

    def render(self, world: World, info: dict | None = None) -> str:
        """Desenha um tick. Retorna 'quit' se o utilizador fechar a janela."""
        while True:
            action = self._process_events()
            if action == "quit":
                return "quit"
            if not self._paused:
                break
            self._draw_frame(world, info, overrides=None)
            self._draw_pause_overlay()
            pygame.display.flip()
            self._clock.tick(self.fps)

        curr_pos = self._compute_positions(world)
        n = self.sub_steps if self._prev_pos else 1

        for step in range(n):
            if self._process_events() == "quit":
                self._prev_pos = curr_pos
                return "quit"

            t = (step + 1) / n
            interp = self._lerp(self._prev_pos, curr_pos, t)

            self._draw_frame(world, info, overrides=interp)
            self._draw_substep_indicator(step + 1, n)
            pygame.display.flip()
            self._clock.tick(self.fps)

        self._prev_pos = curr_pos
        return "ok"

    def close(self) -> None:
        pygame.display.quit()

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

    def _draw_frame(self, world: World, info: dict | None, overrides: dict | None) -> None:
        self._screen.fill(BG_COLOR)

        pygame.draw.line(
            self._screen,
            PANEL_BORDER,
            (self._panel_x, 0),
            (self._panel_x, self.window_height),
            2,
        )

        self._draw_edges(world)
        self._draw_parking_points()
        self._draw_goal_lines(world, overrides)
        self._draw_nodes(world)
        self._draw_boxes(world, info, overrides)
        self._draw_robots(world, overrides)
        self._draw_panel(world, info)

    def _draw_pause_overlay(self) -> None:
        surf = pygame.Surface((self._graph_w, self.window_height), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 120))
        self._screen.blit(surf, (0, 0))

        txt = self._font_hdr.render("PAUSED — SPACE para retomar", True, HEADER_COLOR)
        x = (self._graph_w - txt.get_width()) // 2
        y = (self.window_height - txt.get_height()) // 2
        self._screen.blit(txt, (x, y))

    def _draw_substep_indicator(self, step: int, total: int) -> None:
        for i in range(total):
            color = (180, 180, 255) if i < step else (50, 50, 70)
            pygame.gfxdraw.filled_circle(self._screen, 10 + i * 8, 8, 3, color)

    def _draw_edges(self, world: World) -> None:
        """Desenha todas as arestas do grafo."""
        occupied: set[tuple[str, str]] = set()

        for r in world.all_robots():
            if r.state != RobotState.MOVING or not r.from_node or not r.to_node:
                continue

            occupied.add((r.from_node, r.to_node))
            occupied.add((r.to_node, r.from_node))

        for u, v in self.graph.graph.edges():
            p0 = self._to_screen(u)
            p1 = self._to_screen(v)

            if (u, v) in occupied or (v, u) in occupied:
                pygame.draw.line(self._screen, EDGE_GLOW, p0, p1, 5)
                pygame.draw.line(self._screen, EDGE_OCC_COLOR, p0, p1, 2)
            else:
                pygame.draw.aaline(self._screen, EDGE_COLOR, p0, p1)

    def _draw_parking_points(self) -> None:
        """Desenha os parking points geométricos existentes nas arestas."""
        for u, v in self.graph.graph.edges():
            for fraction in self.graph.parking_points(u, v):
                px, py = self._parking_point_to_screen(u, v, fraction)

                pygame.gfxdraw.filled_circle(
                    self._screen,
                    px,
                    py,
                    PARKING_POINT_RADIUS,
                    PARKING_POINT_COLOR,
                )

                pygame.gfxdraw.aacircle(
                    self._screen,
                    px,
                    py,
                    PARKING_POINT_RADIUS,
                    PARKING_POINT_BORDER,
                )

    def _draw_goal_lines(self, world: World, overrides: dict | None) -> None:
        surf = pygame.Surface((self._graph_w, self.window_height), pygame.SRCALPHA)

        for r in world.all_robots():
            if r.goal_node is None:
                continue

            pos = overrides.get(r.id) if overrides else None
            if pos is None:
                pos = self._robot_screen_pos(r)
            if pos is None:
                continue

            rx, ry = int(pos[0]), int(pos[1])
            gx, gy = self._to_screen(r.goal_node)

            dx, dy = gx - rx, gy - ry
            length = math.hypot(dx, dy)

            if length < 1:
                continue

            steps = max(1, int(length / 8))

            for i in range(0, steps, 2):
                t0 = i / steps
                t1 = min((i + 1) / steps, 1.0)

                pygame.draw.line(
                    surf,
                    (255, 220, 50, 90),
                    (int(rx + dx * t0), int(ry + dy * t0)),
                    (int(rx + dx * t1), int(ry + dy * t1)),
                    1,
                )

        self._screen.blit(surf, (0, 0))

    def _draw_nodes(self, world: World) -> None:
        """Desenha todos os nós do grafo."""
        goal_nodes = {r.goal_node for r in world.all_robots() if r.goal_node}

        for node in self.graph.graph.nodes:
            ntype = self.graph.node_type(node)
            color = NODE_COLORS.get(ntype, (96, 96, 96))
            px, py = self._to_screen(node)

            if node in goal_nodes:
                pygame.gfxdraw.aacircle(
                    self._screen,
                    px,
                    py,
                    NODE_RADIUS + 5,
                    GOAL_RING_COLOR,
                )

            pygame.gfxdraw.filled_circle(self._screen, px, py, NODE_RADIUS, color)
            pygame.gfxdraw.aacircle(self._screen, px, py, NODE_RADIUS, (200, 210, 220))

            lbl = self._font_sm.render(node, True, DIM_TEXT)
            self._screen.blit(
                lbl,
                (px - lbl.get_width() // 2, py + NODE_RADIUS + 2),
            )

    def _draw_robots(self, world: World, overrides: dict | None) -> None:
        for r in world.all_robots():
            pos = overrides.get(r.id) if overrides else None
            if pos is None:
                pos = self._robot_screen_pos(r)
            if pos is None:
                continue

            pos = (int(pos[0]), int(pos[1]))

            if r.state == RobotState.WAITING:
                color = ROBOT_WAIT_COLOR
            elif r.state == RobotState.MOVING:
                color = ROBOT_MOVING_COLOR
            elif r.state == RobotState.PARKED:
                color = ROBOT_PARKED_COLOR
            else:
                color = ROBOT_IDLE_COLOR

            angle = self._robot_angle(r)
            self._draw_triangle(pos, ROBOT_RADIUS, color, angle)

            pygame.gfxdraw.aacircle(
                self._screen,
                pos[0],
                pos[1],
                ROBOT_RADIUS,
                (230, 230, 250),
            )

            lbl = self._font_sm.render(r.id, True, (230, 180, 255))
            self._screen.blit(
                lbl,
                (pos[0] - lbl.get_width() // 2, pos[1] - ROBOT_RADIUS - 14),
            )

    def _draw_boxes(self, world: World, info: dict | None, overrides: dict | None) -> None:
        """Desenha caixas WAITING (diamante no nó) e IN_TRANSIT (ponto no robot)."""
        if not info:
            return
        boxes = info.get("boxes") or []

        robot_pos: dict[str, tuple[int, int]] = {}
        for r in world.all_robots():
            pos = overrides.get(r.id) if overrides else None
            if pos is None:
                pos = self._robot_screen_pos(r)
            if pos is not None:
                robot_pos[r.id] = (int(pos[0]), int(pos[1]))

        s = BOX_ICON_SIZE
        for box in boxes:
            status   = box.get("status", "")
            pipeline = box.get("pipeline", "BLUE")
            color    = BOX_PIPELINE_COLORS.get(pipeline, (180, 180, 180))

            if status == "WAITING":
                node = box.get("current_node")
                if node is None:
                    continue
                try:
                    px, py = self._to_screen(node)
                except Exception:
                    continue
                cy = py - NODE_RADIUS - 4 - s
                pts = [(px, cy - s), (px + s, cy), (px, cy + s), (px - s, cy)]
                pygame.gfxdraw.filled_polygon(self._screen, pts, (*color, 210))
                pygame.gfxdraw.aapolygon(self._screen, pts, color)

            elif status == "IN_TRANSIT":
                carrier = box.get("carried_by")
                if carrier is None or carrier not in robot_pos:
                    continue
                rx, ry = robot_pos[carrier]
                ox, oy = rx + ROBOT_RADIUS - 2, ry - ROBOT_RADIUS + 2
                pygame.gfxdraw.filled_circle(self._screen, ox, oy, 4, (*color, 230))
                pygame.gfxdraw.aacircle(self._screen, ox, oy, 4, color)

    def _draw_triangle(
        self,
        center: tuple[int, int],
        size: int,
        color: tuple[int, int, int],
        angle_rad: float = -math.pi / 2,
    ) -> None:
        cx, cy = center

        pts_local = [
            (size * 1.1, 0.0),
            (-size * 0.7, size * 0.7),
            (-size * 0.7, -size * 0.7),
        ]

        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        pts = [
            (
                int(cx + p[0] * cos_a - p[1] * sin_a),
                int(cy + p[0] * sin_a + p[1] * cos_a),
            )
            for p in pts_local
        ]

        pygame.gfxdraw.filled_polygon(self._screen, pts, color)
        pygame.gfxdraw.aapolygon(self._screen, pts, color)

    def _draw_panel(self, world: World, info: dict | None) -> None:
        px = self._panel_x + 14
        py = 18
        lh = 20

        def txt(text: str, color: tuple = TEXT_COLOR, font=None) -> None:
            nonlocal py
            f = font or self._font_md
            self._screen.blit(f.render(text, True, color), (px, py))
            py += lh

        def sep(h: int = 8) -> None:
            nonlocal py
            py += h

        def hline() -> None:
            nonlocal py
            pygame.draw.line(
                self._screen,
                PANEL_BORDER,
                (self._panel_x + 8, py),
                (self.window_width - 8, py),
                1,
            )
            sep(8)

        txt("Factory RL", HEADER_COLOR, self._font_hdr)
        txt("v0.6  reactivo", DIM_TEXT, self._font_sm)
        sep(10)
        hline()

        txt(f"Tick     {world.tick:>6,}", TEXT_COLOR, self._font_lg)
        txt(f"Robots   {world.robot_count()}", DIM_TEXT, self._font_sm)

        if info:
            sep(4)
            for k, v in info.items():
                if k == "boxes":
                    continue
                txt(f"{k:<12} {v}", DIM_TEXT, self._font_sm)

        txt(f"SubSteps {self.sub_steps}", DIM_TEXT, self._font_sm)
        sep(12)
        hline()

        txt("ROBOTS", HEADER_COLOR, self._font_lg)
        sep(2)

        for r in world.all_robots():
            if r.state == RobotState.WAITING:
                rc = ROBOT_WAIT_COLOR
            elif r.state == RobotState.MOVING:
                rc = ROBOT_MOVING_COLOR
            elif r.state == RobotState.PARKED:
                rc = ROBOT_PARKED_COLOR
            else:
                rc = ROBOT_IDLE_COLOR

            if r.state == RobotState.MOVING:
                pct = int(r.progress * 100)
                txt(f"  {r.id}  MOVING", rc, self._font_sm)
                txt(f"    {r.from_node} → {r.to_node}  ({pct}%)", DIM_TEXT, self._font_sm)
                txt(f"    spd:{r.speed:.1f}", DIM_TEXT, self._font_sm)

            elif r.state == RobotState.WAITING:
                w = r.wait_ticks_in_junction
                txt(f"  {r.id}  WAITING @ {r.current_node} ({w}t)", rc, self._font_sm)

            elif r.state == RobotState.PARKED:
                parked = "—"
                if r.parked_at:
                    u, v, f = r.parked_at
                    parked = f"{u}|{v}|{f:.2f}"
                txt(f"  {r.id}  PARKED", rc, self._font_sm)
                txt(f"    {parked}", DIM_TEXT, self._font_sm)

            else:
                txt(f"  {r.id}  IDLE @ {r.current_node}", rc, self._font_sm)

            txt(f"    goal: {r.goal_node or '—'}", DIM_TEXT, self._font_sm)
            sep(4)

        boxes = (info or {}).get("boxes") or []
        if boxes:
            sep(12)
            hline()
            delivered = sum(1 for b in boxes if b.get("status") == "DONE")
            txt(f"BOXES  {delivered}/{len(boxes)}", HEADER_COLOR, self._font_lg)
            sep(2)
            for box in boxes:
                status   = box.get("status", "")
                if status == "DONE":
                    continue
                pipeline = box.get("pipeline", "BLUE")
                bc       = BOX_PIPELINE_COLORS.get(pipeline, TEXT_COLOR)
                bid      = box.get("box_id", "?")
                nxt      = box.get("next_waypoint") or "?"
                if status == "WAITING":
                    node = box.get("current_node") or "?"
                    txt(f"  b{bid} {pipeline[:3]} WAIT @{node}", bc, self._font_sm)
                elif status == "IN_TRANSIT":
                    carrier = box.get("carried_by") or "?"
                    txt(f"  b{bid} {pipeline[:3]} TRNS {carrier} →{nxt}", bc, self._font_sm)

        hint_y = self.window_height - 72

        for hint in ["SPACE  pause / retoma", "+/-    sub-steps", "Q      sair"]:
            self._screen.blit(
                self._font_sm.render(hint, True, DIM_TEXT),
                (px, hint_y),
            )
            hint_y += 16

    def _compute_positions(self, world: World) -> dict[str, _PixelPos]:
        result = {}

        for r in world.all_robots():
            pos = self._robot_screen_pos(r)
            if pos is not None:
                result[r.id] = (float(pos[0]), float(pos[1]))

        return result

    @staticmethod
    def _lerp(
        prev: dict[str, _PixelPos],
        curr: dict[str, _PixelPos],
        t: float,
    ) -> dict[str, _PixelPos]:
        result = {}

        for rid, c in curr.items():
            p = prev.get(rid, c)
            result[rid] = (
                p[0] + (c[0] - p[0]) * t,
                p[1] + (c[1] - p[1]) * t,
            )

        return result

    @staticmethod
    def _lerp_angle(a: float, b: float, t: float) -> float:
        """Interpola entre dois ângulos pelo caminho mais curto (sem wrap-around)."""
        diff = (b - a + math.pi) % (2 * math.pi) - math.pi
        return a + diff * t

    def _robot_angle(self, r: Robot) -> float:
        if r.state == RobotState.MOVING and r.from_node and r.to_node:
            fx, fy = self._to_screen(r.from_node)
            tx, ty = self._to_screen(r.to_node)
            dx, dy = tx - fx, ty - fy
            target = math.atan2(dy, dx) if (dx != 0 or dy != 0) else -math.pi / 2

            # Durante o delay de viragem (wait_ticks > 0), interpola do ângulo
            # de chegada para o ângulo de partida para animar a rotação do robot.
            if r.wait_ticks > 0 and r.turn_ticks_total > 0 and r.came_from:
                cfx, cfy = self._to_screen(r.came_from)
                sdx, sdy = fx - cfx, fy - cfy
                source = math.atan2(sdy, sdx) if (sdx != 0 or sdy != 0) else target
                t = 1.0 - r.wait_ticks / r.turn_ticks_total
                return self._lerp_angle(source, target, t)

            return target

        if r.came_from and r.current_node:
            fx, fy = self._to_screen(r.came_from)
            tx, ty = self._to_screen(r.current_node)
            dx, dy = tx - fx, ty - fy
            if dx != 0 or dy != 0:
                return math.atan2(dy, dx)

        if r.parked_at:
            u, v, _ = r.parked_at
            fx, fy = self._to_screen(u)
            tx, ty = self._to_screen(v)
            dx, dy = tx - fx, ty - fy
            if dx != 0 or dy != 0:
                return math.atan2(dy, dx)

        return -math.pi / 2

    def _build_transform(self, padding: int) -> dict:
        coords = [self.graph.node_position(n) for n in self.graph.graph.nodes]

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        gx_min, gx_max = min(xs), max(xs)
        gy_min, gy_max = min(ys), max(ys)

        avail_w = self._graph_w - 2 * padding
        avail_h = self.window_height - 2 * padding

        scale = min(
            avail_w / max(gx_max - gx_min, 1),
            avail_h / max(gy_max - gy_min, 1),
        )

        ox = padding + (avail_w - (gx_max - gx_min) * scale) / 2
        oy = padding + (avail_h - (gy_max - gy_min) * scale) / 2

        return {
            "scale": scale,
            "gx_min": gx_min,
            "gy_min": gy_min,
            "ox": ox,
            "oy": oy,
        }

    def _to_screen(self, node: str) -> tuple[int, int]:
        gx, gy = self.graph.node_position(node)
        t = self._transform

        return (
            int(t["ox"] + (gx - t["gx_min"]) * t["scale"]),
            int(t["oy"] + (gy - t["gy_min"]) * t["scale"]),
        )

    def _parking_point_to_screen(self, u: str, v: str, fraction: float) -> tuple[int, int]:
        x, y = self.graph.parking_point_position(u, v, fraction)
        t = self._transform

        return (
            int(t["ox"] + (x - t["gx_min"]) * t["scale"]),
            int(t["oy"] + (y - t["gy_min"]) * t["scale"]),
        )

    def _robot_screen_pos(self, r: Robot) -> Optional[tuple[int, int]]:
        if r.state == RobotState.MOVING and r.from_node and r.to_node:
            ax, ay = self._to_screen(r.from_node)
            bx, by = self._to_screen(r.to_node)

            return (
                int(ax + (bx - ax) * r.progress),
                int(ay + (by - ay) * r.progress),
            )

        if r.parked_at is not None:
            u, v, fraction = r.parked_at
            return self._parking_point_to_screen(u, v, fraction)

        if r.current_node:
            return self._to_screen(r.current_node)

        return None