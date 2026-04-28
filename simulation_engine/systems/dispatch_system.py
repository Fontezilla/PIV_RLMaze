import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from simulation_engine.core.world_state import WorldState
from simulation_engine.core.entities import RobotState
from simulation_engine.core.graph import FactoryGraph

# ---------------------------------------------------------------------------
# Constantes das regras físicas
# ---------------------------------------------------------------------------

# Regra 2 — Zona de curva
# A zona NÃO é 30% da aresta atual.
# É 30% da menor aresta relevante do mapa, para ter tamanho fixo.
CURVE_ZONE_FRACTION = 0.30
CURVE_SPEED_RATIO = 0.70

# Regra 3 — Manobra de viragem
TURN_TICKS = 2
TURN_THRESHOLD_RAD = 0.10  # ~5.7°


@dataclass
class DispatchResult:
    accepted: bool
    dispatched: bool = False
    speed_applied: bool = False
    reason: str = "ok"


class DispatchSystem:
    """
    Aplica ações do agente.

    Recebe:
        {robot_id: (next_node_or_None, speed_cmd)}

    speed_cmd:
        "acc" | "dec" | "hold"

    Contrato:
        - Robot IDLE + next_node → tenta iniciar movimento e aplica speed_cmd
          no mesmo tick se o dispatch for aceite.
        - Robot MOVING           → apenas aplica speed_cmd.

    Regras físicas aplicadas:
        1. Docking edges (adj. a entry/exit/process)
              - entrar:  speed >= 0 (sempre de frente)
              - sair:    dispatch cria estado de recuo (docking_exit=True);
                         speed <= 0 enquanto em modo de saída
        2. Zona de curva
              - tamanho fixo = 30% da menor aresta relevante do mapa
              - speed_max na zona = vel_max × 0.70
        3. Manobra de viragem em junctions
              - se o ângulo entre a aresta de chegada e a de saída > threshold:
                robot espera TURN_TICKS antes de poder avançar
    """

    def __init__(
        self,
        graph: FactoryGraph,
        vel_max: float = 1.0,
        acc_step: Optional[float] = None,
    ):
        self.graph = graph
        self.vel_max = vel_max
        self.acc_step = acc_step if acc_step is not None else vel_max * 0.4
        self.curve_zone_length = self._compute_curve_zone_length()

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def apply(
        self,
        world: WorldState,
        actions: Dict[str, Tuple[Optional[str], str]],
    ) -> Dict[str, DispatchResult]:
        occupied_edges: set[tuple[str, str]] = set()
        results: Dict[str, DispatchResult] = {}

        for robot_id, action in actions.items():
            if robot_id not in world.robots:
                continue

            robot = world.robots[robot_id]

            if isinstance(action, tuple):
                next_node, speed_cmd = action
            else:
                next_node, speed_cmd = action, "acc"

            speed_cmd = self._normalize_speed_cmd(speed_cmd)

            # ----------------------------------------------------------
            # DISPATCH — só para robots IDLE com destino indicado
            # ----------------------------------------------------------
            if robot.state == RobotState.IDLE and next_node is not None:
                result = self._try_dispatch(
                    robot=robot,
                    next_node=next_node,
                    speed_cmd=speed_cmd,
                    occupied_edges=occupied_edges,
                    world=world,
                )
                results[robot_id] = result
                continue

            # ----------------------------------------------------------
            # SPEED COMMAND — só para robots MOVING
            # ----------------------------------------------------------
            if robot.state == RobotState.MOVING:
                self._apply_speed_cmd(robot, speed_cmd)
                results[robot_id] = DispatchResult(
                    accepted=True,
                    dispatched=False,
                    speed_applied=True,
                    reason="speed_control",
                )
                continue

            # ----------------------------------------------------------
            # Sem efeito
            # ----------------------------------------------------------
            results[robot_id] = DispatchResult(
                accepted=True,
                dispatched=False,
                speed_applied=False,
                reason="no_op",
            )

        return results

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _try_dispatch(
        self,
        robot,
        next_node: str,
        speed_cmd: str,
        occupied_edges: set[tuple[str, str]],
        world: WorldState,
    ) -> DispatchResult:
        if robot.current_node is None:
            return DispatchResult(False, reason="no_current_node")

        if robot.yield_ticks > 0:
            return DispatchResult(False, reason="yield_cooldown")

        if robot.turn_cooldown > 0:
            return DispatchResult(False, reason="turn_cooldown")

        current_node = robot.current_node
        neighbors = self.graph.neighbors(current_node)

        if next_node not in neighbors:
            return DispatchResult(False, reason="non_neighbor")

        # evitar U-turn imediato se houver alternativas
        if robot.prev_node is not None and next_node == robot.prev_node and len(neighbors) > 1:
            return DispatchResult(False, reason="u_turn_blocked")

        # Regra 3 — viragem em junction
        if self.graph.is_junction(current_node) and self._needs_turn(robot.prev_node, current_node, next_node):
            robot.turn_cooldown = TURN_TICKS
            # Configura animação de rotação
            robot.rotation_angle_from = self._edge_angle(robot.prev_node, current_node)
            robot.rotation_angle_to   = self._edge_angle(current_node, next_node)
            robot.rotation_ticks_total = TURN_TICKS
            robot.is_turning = True
            return DispatchResult(False, reason="turn_required")

        # hard-block: tráfego oposto já em movimento na aresta destino
        if world.edge_has_opposing_traffic(current_node, next_node):
            return DispatchResult(False, reason="opposing_traffic")

        # docking edges são corredor único — bloquear se já existe robot na aresta
        # (qualquer direção, inclui docking exits que estão a sair)
        if self.graph.is_docking_edge(current_node, next_node):
            if world.robots_on_edge(current_node, next_node) or \
               world.robots_on_edge(next_node, current_node):
                return DispatchResult(False, reason="docking_edge_occupied")

        # docking exit: impedir se o nó de chegada (junction) já tem robot IDLE
        # sem este guard, o robot recua para cima de um robot parado no junction
        if self.graph.is_special(current_node) and world.robots_at_node(next_node):
            return DispatchResult(False, reason="docking_exit_dest_occupied")

        # impedir dois robots na mesma aresta no mesmo tick
        edge = (current_node, next_node)
        reverse = (next_node, current_node)
        if edge in occupied_edges or reverse in occupied_edges:
            return DispatchResult(False, reason="edge_reserved_this_tick")

        # impedir entrada se há robot no mesmo sentido demasiado perto do início
        # ou completamente parado na aresta (ambos causam colisão inevitável)
        same_dir = world.robots_on_edge(current_node, next_node)
        if same_dir:
            try:
                edge_len = self.graph.distance(current_node, next_node)
                min_safe = (robot.collision_radius * 2.0) / edge_len if edge_len > 0 else 1.0
                if any(r.progress < min_safe or r.speed <= 0.0 for r in same_dir):
                    return DispatchResult(False, reason="edge_entry_blocked")
            except Exception:
                pass

        occupied_edges.add(edge)
        occupied_edges.add(reverse)

        # --- Regra 1: saída de nó especial (docking exit) ---
        if self.graph.is_special(current_node):
            # robot sai de nó especial → recua ao longo da aresta
            # from_node=next_node(junction), to_node=current_node(special)
            # progress começa em 1.0 (robot está no to_node) e desce até 0
            robot.from_node = next_node
            robot.to_node = current_node
            robot.current_node = None
            robot.progress = 1.0
            robot.speed = 0.0
            robot.state = RobotState.MOVING
            robot.docking_exit = True

        else:
            # despacho normal
            robot.from_node = current_node
            robot.to_node = next_node
            robot.current_node = None
            robot.progress = 0.0
            robot.speed = 0.0
            robot.state = RobotState.MOVING
            robot.docking_exit = False

        # aplicar comando de velocidade no mesmo tick do dispatch
        self._apply_speed_cmd(robot, speed_cmd)

        # robot saiu — reset do contador de starvation
        robot.idle_stuck_ticks = 0

        return DispatchResult(
            accepted=True,
            dispatched=True,
            speed_applied=True,
            reason="dispatch_ok",
        )

    # ------------------------------------------------------------------
    # Speed command
    # ------------------------------------------------------------------

    def _normalize_speed_cmd(self, speed_cmd: str) -> str:
        if speed_cmd not in {"acc", "dec", "hold"}:
            return "hold"
        return speed_cmd

    def _apply_speed_cmd(self, robot, speed_cmd: str) -> None:
        """
        Aplica incremento/decremento de velocidade e faz cumprir as
        restrições físicas das Regras 1 e 2.
        """

        # Para docking_exit, "acc" significa acelerar o recuo
        if robot.docking_exit:
            if speed_cmd == "acc":
                speed_cmd = "dec"
            elif speed_cmd == "dec":
                speed_cmd = "acc"

        if speed_cmd == "acc":
            new_speed = robot.speed + self.acc_step
        elif speed_cmd == "dec":
            new_speed = robot.speed - self.acc_step
        else:
            new_speed = robot.speed

        # Regra 1 — restrição de direção em docking edges
        if robot.from_node and robot.to_node and self.graph.is_docking_edge(robot.from_node, robot.to_node):
            if robot.docking_exit:
                new_speed = min(new_speed, 0.0)
            else:
                new_speed = max(new_speed, 0.0)

        # Regra 2 — limite de velocidade na zona de curva
        curve_limit = self._curve_zone_limit(robot)
        if curve_limit is not None:
            new_speed = max(min(new_speed, curve_limit), -curve_limit)

        # Clamp global
        new_speed = min(new_speed, self.vel_max)
        # Robots normais (não docking_exit) não podem reverter na aresta —
        # o DQN oscilava: dec→speed<0→reverter, acc→positivo, loop infinito.
        # O escape de deadlock é feito deterministicamente via blocked_ticks.
        if not robot.docking_exit:
            new_speed = max(new_speed, 0.0)
        else:
            new_speed = max(new_speed, -self.vel_max * 0.5)
        robot.speed = new_speed

    # ------------------------------------------------------------------
    # Regra 2 — zona de curva
    # ------------------------------------------------------------------

    def _compute_curve_zone_length(self) -> float:
        """
        Calcula o tamanho fixo da zona de curva em world units.

        Usa 30% da menor aresta relevante onde a regra pode aplicar,
        isto é, arestas não-docking com pelo menos um junction.
        """
        candidate_lengths = []

        for u, v in self.graph.graph.edges():
            if self.graph.is_docking_edge(u, v):
                continue

            if not (self.graph.is_junction(u) or self.graph.is_junction(v)):
                continue

            try:
                dist = self.graph.distance(u, v)
            except Exception:
                continue

            if dist > 0:
                candidate_lengths.append(dist)

        if not candidate_lengths:
            return 0.0

        return min(candidate_lengths) * CURVE_ZONE_FRACTION

    def _curve_zone_limit(self, robot) -> Optional[float]:
        """
        Devolve o limite de speed (positivo) se o robot está na zona de curva,
        ou None caso contrário.

        A zona é uma distância fixa em world units, não uma percentagem da
        aresta atual.
        """
        fn = robot.from_node
        tn = robot.to_node
        if fn is None or tn is None:
            return None

        if self.graph.is_docking_edge(fn, tn):
            return None

        try:
            edge_length = self.graph.distance(fn, tn)
        except Exception:
            return None

        if edge_length <= 0 or self.curve_zone_length <= 0:
            return None

        limit = self.vel_max * CURVE_SPEED_RATIO
        progress = max(0.0, min(1.0, robot.progress))

        dist_from_start = progress * edge_length
        dist_to_end = (1.0 - progress) * edge_length

        # zona fixa ao sair de um junction
        # — ignorada se o robot vem em linha reta (prev_node disponível e sem viragem)
        if self.graph.is_junction(fn) and dist_from_start < self.curve_zone_length:
            prev = robot.prev_node
            if prev is None or self._needs_turn(prev, fn, tn):
                return limit
            # linha reta confirmada — sem curve zone na saída

        # zona fixa ao entrar num junction
        # — ignorada se o robot vai continuar em linha reta (buffer disponível e sem viragem)
        if self.graph.is_junction(tn) and dist_to_end < self.curve_zone_length:
            buf = robot.buffered_next_node
            if buf is None or self._needs_turn(fn, tn, buf):
                return limit
            # linha reta confirmada — sem curve zone na entrada

        return None

    # ------------------------------------------------------------------
    # Regra 3 — deteção de viragem
    # ------------------------------------------------------------------

    def _edge_angle(self, from_node: Optional[str], to_node: str) -> float:
        """Ângulo (radianos) da aresta from_node → to_node no espaço do mapa."""
        if from_node is None:
            return 0.0
        fx, fy = self.graph.node_position(from_node)
        tx, ty = self.graph.node_position(to_node)
        return math.atan2(ty - fy, tx - fx)

    def _needs_turn(self, prev_node: Optional[str], current: str, next_node: str) -> bool:
        """
        Retorna True se a direção da aresta de saída (current→next_node)
        difere significativamente da aresta de chegada (prev_node→current).
        """
        if prev_node is None:
            return False

        p1 = self.graph.node_position(prev_node)
        p2 = self.graph.node_position(current)
        p3 = self.graph.node_position(next_node)

        dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
        dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]

        a1 = math.atan2(dy1, dx1)
        a2 = math.atan2(dy2, dx2)

        diff = abs(a2 - a1) % (2 * math.pi)
        if diff > math.pi:
            diff = 2 * math.pi - diff

        return diff > TURN_THRESHOLD_RAD