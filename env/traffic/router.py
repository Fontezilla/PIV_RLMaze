from __future__ import annotations

from env.core.entities import Robot, RobotState
from env.core.graph import FactoryGraph
from env.traffic.astar import plan
from env.traffic.locks import NodeLock, ParkingLock, SegmentLock
from env.traffic.prediction import (
    LOOKAHEAD_STEPS,
    build_future_edge_usage,
    find_safe_parking_candidate,
    parked_robot_should_leave,
    parking_candidate_to_target,
)

DEADLOCK_THRESHOLD = 12
DEADLOCK_LOOKAHEAD_TICKS = 4

# Máximo de ticks que o waiting inteligente pode segurar um robot.
# Se este limite for atingido, o robot aceita qualquer alternativa disponível
# para evitar livelocks circulares (A espera B, B espera A).
SMART_WAIT_MAX_TICKS = 20

MAX_DETOUR_FACTOR = 3.0
IMMEDIATE_REVERSE_PENALTY = 3000.0

# Espera máxima antes de aceitar parking como optimização,
# mesmo sem deadlock declarado.
WAIT_LIMIT_ENTRY_EXIT_PROCESS = 22
WAIT_LIMIT_JUNCTION = 24
WAIT_LIMIT_NORMAL_NODE = 32

FUTURE_EDGE_CONGESTION_PENALTY = 1600.0
PARKED_EDGE_CONGESTION_PENALTY = 2500.0


class Router:
    """
    Controlo de tráfego reactivo:
      - A* no grafo dinâmico
      - Locks de segmento e nó
      - Penalização anti-loop
      - Waiting inteligente com limite de tempo (evita livelocks)
      - Evasão para nós especiais
      - Parking goal-aligned em subnós geométricos
    """

    def __init__(self, graph: FactoryGraph) -> None:
        self.graph = graph
        self.segment_lock = SegmentLock()
        self.node_lock = NodeLock()
        self.parking_lock = ParkingLock()
        self._plans: dict[str, list[str]] = {}

    def reset(self) -> None:
        self.segment_lock = SegmentLock()
        self.node_lock = NodeLock()
        self.parking_lock = ParkingLock()
        self._plans.clear()

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def register(self, robot: Robot) -> None:
        if robot.current_node:
            self.node_lock.try_acquire(robot.current_node, robot.id)
            robot.push_visited(robot.current_node)

    def decide(self, robot: Robot, robots: list[Robot]) -> tuple[str, str | None] | None:
        if robot.goal_node is None:
            return None

        if robot.is_parked():
            target = self._unpark_target(robot, robots)
            if target is not None:
                return ("unpark", target)
            return None

        if robot.current_node is None:
            return None

        if robot.current_node == robot.goal_node:
            return None

        # --- 1º plano: tenta manter a rota actual ---
        if not self._has_valid_plan(robot):
            self._replan(robot, robots)

        next_node = self._next_in_plan(robot)
        if next_node and self._try_move(robot, next_node):
            return ("move", next_node)

        # --- Waiting inteligente ---
        # Espera apenas se o bloqueio aparenta ser curto e se esperar for
        # mais barato do que desviar.
        if robot.wait_ticks_in_junction < SMART_WAIT_MAX_TICKS:
            ideal_next = self._ideal_next_node(robot)
            if (
                ideal_next is not None
                and not self.node_lock.is_free_for(ideal_next, robot.id)
            ):
                ticks_to_free = self._estimate_ticks_to_free(ideal_next, robots)
                if ticks_to_free is not None:
                    alt_plan = self._compute_alt_plan(robot, robots)
                    alt_cost = self._plan_cost_ticks(alt_plan) if alt_plan else 9999
                    if ticks_to_free < alt_cost:
                        return None

        # --- 2º plano: recalcula com bloqueios e previsão ---
        self._replan(robot, robots)
        next_node = self._next_in_plan(robot)
        if next_node and self._try_move(robot, next_node):
            return ("move", next_node)

        # --- Evasão para nós especiais ---
        evasion = self._find_evasion(robot, robots)
        evasion_is_dead_end = (
            evasion is not None and self.graph.graph.degree(evasion) == 1
        )

        # Evasão para nós não-dead-end: move directamente.
        if evasion is not None and not evasion_is_dead_end:
            if self._try_move(robot, evasion):
                return ("move", evasion)

        blocker_coming = self._blocker_arriving_soon(robot, robots)
        is_deadlocked = self._is_deadlock(robot)

        # --- Parking optimizado ---
        # Parking é sempre tentado se:
        #   a) o robot atingiu o limite de espera (deadlock/wait_limit), OU
        #   b) a única evasão é um dead-end (parking é mais eficiente).
        # O bloqueador a chegar NÃO impede parking: parking liberta o nó
        # actual para o bloqueador passar — é exactamente o que queremos.
        # Só não parkamos se o bloqueador está MESMO a chegar E não estamos
        # em deadlock (permite que o robot passe naturalmente).
        if self._should_try_parking(robot, robots) or evasion_is_dead_end:
            if not blocker_coming or is_deadlocked:
                park = self._find_parking(robot, robots)
                if park:
                    return ("park", park)

        # Dead-end como último recurso (se parking não estiver disponível).
        if evasion_is_dead_end and self._try_move(robot, evasion):
            return ("move", evasion)

        return None

    def arrived(self, robot: Robot, node: str) -> None:
        if robot.came_from and robot.came_from != node:
            self.segment_lock.release(robot.came_from, node, robot.id)
            self.node_lock.release(robot.came_from, robot.id)
            robot.push_edge(robot.came_from, node)

        self.node_lock.try_acquire(node, robot.id)
        robot.push_visited(node)

    def parked(self, robot: Robot, u: str, v: str, fraction: float) -> None:
        robot.parked_at = (u, v, fraction)
        robot.parking_reserved_edge = (u, v)

        # O robot está num ponto geométrico da aresta, não num nó real.
        # Libertar o nó de origem evita que o parking continue a bloquear
        # o próprio nó que queria libertar.
        self.node_lock.release(u, robot.id)
        self.segment_lock.release(u, v, robot.id)
        self.parking_lock.try_acquire(u, v, fraction, robot.id)

        robot.current_node = None
        robot.from_node = u
        robot.to_node = v
        robot.progress = fraction
        robot.speed = 0.0
        robot.state = RobotState.PARKED
        robot.parked_ticks = 0

        robot.push_edge(u, v)

    def unparked(self, robot: Robot) -> None:
        if robot.parked_at:
            u, v, fraction = robot.parked_at
            self.parking_lock.release(u, v, fraction, robot.id)

        robot.parked_at = None
        robot.parking_reserved_edge = None
        robot.parked_ticks = 0

    def release_all(self, robot: Robot) -> None:
        if robot.current_node:
            self.node_lock.release(robot.current_node, robot.id)
        if robot.from_node and robot.to_node:
            self.segment_lock.release(robot.from_node, robot.to_node, robot.id)
        if robot.parked_at:
            u, v, fraction = robot.parked_at
            self.segment_lock.release(u, v, robot.id)
            self.parking_lock.release(u, v, fraction, robot.id)
        self.parking_lock.release_all_for(robot.id)

    def sync_moving_node_locks(self, robots: list[Robot]) -> None:
        for robot in robots:
            if robot.state == RobotState.MOVING and robot.from_node is not None:
                self.node_lock.release(robot.from_node, robot.id)

    # ------------------------------------------------------------------
    # Planeamento
    # ------------------------------------------------------------------

    def _has_valid_plan(self, robot: Robot) -> bool:
        robot_plan = self._plans.get(robot.id, [])
        if len(robot_plan) < 2:
            return False
        return robot_plan[0] == robot.current_node

    def _make_plan_path(
        self,
        robot  : Robot,
        robots : list[Robot] | None = None,
    ) -> list[str] | None:
        """
        Executa A* com bloqueios e penalizações. Devolve o caminho ou None.
        Partilhado por _replan (que guarda o resultado) e _compute_alt_plan
        (que apenas estima o custo sem guardar).

        Parked edges are NOT hard-blocked here — they're penalised via
        congested_edges so A* routes around them when alternatives exist.
        Hard-blocking parked junction edges causes cascade deadlocks when
        multiple corridors are simultaneously occupied.
        """
        if robot.current_node is None or robot.goal_node is None:
            return None

        blocked_nodes   = self.node_lock.blocked_nodes_for(robot.id)
        blocked_edges   = self.segment_lock.blocked_edges_for(robot.id)
        avoid_nodes     = robot.recent_node_penalties(robot.goal_node)
        avoid_edges     = robot.recent_edge_penalties()
        congested_edges = self._congested_edges_for(robot, robots)

        n_neighbors = len(list(self.graph.neighbors(robot.current_node)))
        rev_penalty = 0.0 if n_neighbors <= 1 else IMMEDIATE_REVERSE_PENALTY

        path = plan(
            graph                    = self.graph,
            src                      = robot.current_node,
            dst                      = robot.goal_node,
            came_from                = robot.came_from,
            blocked_nodes            = blocked_nodes,
            blocked_edges            = blocked_edges,
            avoid_nodes              = avoid_nodes,
            avoid_edges              = avoid_edges,
            congested_edges          = congested_edges,
            immediate_reverse_penalty= rev_penalty,
        )

        if not path or self._is_excessive_detour(robot, path):
            return None

        return path

    def _replan(self, robot: Robot, robots: list[Robot] | None = None) -> None:
        path = self._make_plan_path(robot, robots)
        self._plans[robot.id] = path if path else []

    def _compute_alt_plan(self, robot: Robot, robots: list[Robot] | None = None) -> list[str] | None:
        """Gera o plano alternativo sem guardar — usado para estimar custo antes de decidir esperar."""
        return self._make_plan_path(robot, robots)

    def _is_excessive_detour(self, robot: Robot, path: list[str]) -> bool:
        if robot.current_node is None or robot.goal_node is None:
            return False

        normal_distance = self.graph.shortest_distance(
            robot.current_node, robot.goal_node
        )

        if normal_distance == float("inf") or normal_distance <= 0:
            return False

        path_distance = self._path_distance(path)
        if path_distance <= normal_distance * MAX_DETOUR_FACTOR:
            return False

        # Path parece excessivo — mas só é realmente excessivo se existir um
        # caminho mais curto SEM os bloqueios actuais de nós.
        # Se o caminho directo passa por nós bloqueados (head-on deadlock, etc.),
        # este é o único caminho viável e deve ser aceite.
        blocked = self.node_lock.blocked_nodes_for(robot.id)
        unblocked_path = plan(
            graph=self.graph,
            src=robot.current_node,
            dst=robot.goal_node,
            came_from=robot.came_from,
        )
        if unblocked_path is None:
            return False  # sem caminho de todo — aceita o que tiver

        unblocked_dist = self._path_distance(unblocked_path)
        if unblocked_dist > normal_distance * MAX_DETOUR_FACTOR:
            return False  # mesmo sem bloqueios o caminho é longo — grafo sem alternativa

        # Verifica se o caminho sem bloqueios passa por nós actualmente bloqueados.
        # Se sim, o desvio é forçado pelos bloqueios, não pelos loops — aceitar.
        for node in unblocked_path[1:-1]:  # nós intermédios
            if node in blocked:
                return False  # caminho directo bloqueado → desvio necessário

        return True

    def _path_distance(self, path: list[str]) -> float:
        total = 0.0
        for u, v in zip(path, path[1:]):
            if not self.graph.has_edge(u, v):
                return float("inf")
            total += self.graph.edge_distance(u, v)
        return total

    def _next_in_plan(self, robot: Robot) -> str | None:
        robot_plan = self._plans.get(robot.id, [])
        if len(robot_plan) >= 2 and robot_plan[0] == robot.current_node:
            return robot_plan[1]
        return None

    def _advance_plan(self, robot: Robot) -> None:
        robot_plan = self._plans.get(robot.id, [])
        if robot_plan:
            self._plans[robot.id] = robot_plan[1:]

    # ------------------------------------------------------------------
    # Waiting inteligente
    # ------------------------------------------------------------------

    def _ideal_next_node(self, robot: Robot) -> str | None:
        """Próximo nó no caminho ideal sem bloqueios."""
        if robot.current_node is None or robot.goal_node is None:
            return None
        try:
            path = plan(
                graph=self.graph,
                src=robot.current_node,
                dst=robot.goal_node,
                came_from=robot.came_from,
            )
            if path and len(path) >= 2:
                return path[1]
        except Exception:
            pass
        return None

    def _estimate_ticks_to_free(
        self, node: str, robots: list[Robot]
    ) -> int | None:
        """
        Estima em quantos ticks o nó ficará livre. Lookahead de 2 passos.

        Caso 1 — robot em trânsito direto (to_node == node, MOVING):
            ticks = restante_da_aresta + saída_mínima_do_nó

        Caso 2 — robot já no nó, parado (IDLE/WAITING):
            ticks = saída_mínima_do_nó
            EXCEPÇÃO: se esse robot também está em WAITING há muitos ticks,
            provavelmente também está bloqueado — não contar como "vai libertar".

        Caso 3 — lookahead: robot a 1 passo (to_node vizinho de node):
            ticks = restante_até_to_node + tempo(to_node→node) + saída_de_node

        Retorna None se ninguém vai libertar o nó.
        """
        for other in robots:
            # Caso 1: em trânsito direto
            if other.to_node == node and other.state == RobotState.MOVING:
                ticks_arrival = self._ticks_remaining(other)
                ticks_exit = self._estimate_exit_ticks(node)
                return ticks_arrival + ticks_exit

            # Caso 2: já está no nó parado
            # Não contar se o bloqueante também está há muito tempo bloqueado
            # (sinal de livelock — o outro robot também está à espera de algo)
            if (
                other.current_node == node
                and other.state == RobotState.WAITING
                and other.wait_ticks_in_junction > SMART_WAIT_MAX_TICKS // 2
            ):
                # Bloqueante também está preso — não vai libertar em breve
                return None

            if (
                other.current_node == node
                and other.state in (RobotState.IDLE, RobotState.WAITING)
            ):
                return self._estimate_exit_ticks(node)

            # Caso 3: a 1 passo, usa topologia
            if (
                other.state == RobotState.MOVING
                and other.to_node is not None
                and other.to_node != node
                and self.graph.has_edge(other.to_node, node)
            ):
                ticks_to_intermediate = self._ticks_remaining(other)
                try:
                    inter_dist = self.graph.edge_distance(other.to_node, node)
                    from env.core.physics import MAX_SPEED
                    ticks_intermediate = max(int(inter_dist / max(MAX_SPEED, 1)), 1)
                except Exception:
                    ticks_intermediate = 15
                ticks_exit = self._estimate_exit_ticks(node)
                return ticks_to_intermediate + ticks_intermediate + ticks_exit

        return None

    def _ticks_remaining(self, robot: Robot) -> int:
        if robot.from_node is None or robot.to_node is None:
            return 1
        try:
            dist = self.graph.edge_distance(robot.from_node, robot.to_node)
            from env.core.physics import MAX_SPEED, SHORTEST_EDGE
            remaining = (dist / max(MAX_SPEED, 1)) * (1.0 - robot.progress)
            return max(int(remaining), 1)
        except Exception:
            from env.core.physics import MAX_SPEED, SHORTEST_EDGE
            return max(int(SHORTEST_EDGE / max(MAX_SPEED, 1)), 1)

    def _estimate_exit_ticks(self, node: str) -> int:
        try:
            from env.core.physics import MAX_SPEED
            neighbors = list(self.graph.neighbors(node))
            if not neighbors:
                return 1
            min_dist = min(self.graph.edge_distance(node, nb) for nb in neighbors)
            return max(int(min_dist / max(MAX_SPEED, 1)), 1)
        except Exception:
            from env.core.physics import MAX_SPEED, SHORTEST_EDGE
            return max(int(SHORTEST_EDGE / max(MAX_SPEED, 1)), 1)

    def _plan_cost_ticks(self, path: list[str]) -> int:
        try:
            from env.core.physics import MAX_SPEED
            dist = self._path_distance(path)
            if dist == float("inf"):
                return 9999
            return max(int(dist / max(MAX_SPEED, 1)), 1)
        except Exception:
            return 9999

    def _congested_edges_for(
        self,
        robot: Robot,
        robots: list[Robot] | None,
    ) -> dict[tuple[str, str], float]:
        """
        Cria penalizações suaves para arestas que provavelmente vão ser usadas.

        Não bloqueia a rota: apenas torna menos atractivos caminhos que cruzam
        trajectórias previstas ou arestas com robots estacionados.
        """
        if not robots:
            return {}

        congested: dict[tuple[str, str], float] = {}

        future_usage = build_future_edge_usage(
            graph=self.graph,
            robots=robots,
            exclude_robot_id=robot.id,
            steps=LOOKAHEAD_STEPS,
        )

        for edge, users in future_usage.items():
            congested[edge] = max(
                congested.get(edge, 0.0),
                FUTURE_EDGE_CONGESTION_PENALTY * max(len(users), 1),
            )

        for edge in self.parking_lock.occupied_segments():
            congested[edge] = max(
                congested.get(edge, 0.0),
                PARKED_EDGE_CONGESTION_PENALTY,
            )

        return congested

    # ------------------------------------------------------------------
    # Movimento
    # ------------------------------------------------------------------

    def _try_move(self, robot: Robot, next_node: str) -> bool:
        if robot.current_node is None:
            return False

        # Não atravessa arestas onde OUTRO robot está estacionado.
        # Usa blocked_edges_for (exclui o próprio robot) para não bloquear
        # o robot quando tenta sair do seu próprio parking spot.
        seg = self.graph.segment_id(robot.current_node, next_node)
        if seg in self.parking_lock.blocked_edges_for(robot.id):
            return False

        if not self.segment_lock.try_acquire(robot.current_node, next_node, robot.id):
            return False

        if not self.node_lock.try_acquire(next_node, robot.id):
            self.segment_lock.release(robot.current_node, next_node, robot.id)
            return False

        self._advance_plan(robot)
        return True

    # ------------------------------------------------------------------
    # Evasão
    # ------------------------------------------------------------------

    def _find_evasion(self, robot: Robot, robots: list[Robot]) -> str | None:
        if robot.current_node is None:
            return None

        goals = {r.goal_node for r in robots if r.id != robot.id}
        planned = self._nodes_in_other_plans(robot.id)

        for neighbor in self.graph.neighbors(robot.current_node):
            if not self.graph.is_evasion_node(neighbor):
                continue
            if neighbor in goals:
                continue
            if neighbor in planned:
                continue
            if not self.node_lock.is_free(neighbor):
                continue
            if not self.segment_lock.is_free_for(robot.current_node, neighbor, robot.id):
                continue
            if robot.came_from is not None and neighbor == robot.came_from:
                continue
            return neighbor

        return None

    def _nodes_in_other_plans(self, robot_id: str) -> set[str]:
        nodes = set()
        for rid, path in self._plans.items():
            if rid != robot_id:
                nodes.update(path)
        return nodes

    # ------------------------------------------------------------------
    # Decisão WAITING vs PARKING
    # ------------------------------------------------------------------

    def _should_try_parking(self, robot: Robot, robots: list[Robot]) -> bool:
        if robot.current_node is None:
            return False

        if self._is_deadlock(robot):
            return True

        # Se outro robot precisa deste nó, ainda assim damos alguns ticks
        # para o bloqueio resolver naturalmente. Isto evita parking prematuro
        # causado por previsões curtas ou por robots que afinal seguem outra direcção.
        if self._current_node_needed_by_other(robot, robots):
            return robot.wait_ticks_in_junction >= 4

        return robot.wait_ticks_in_junction >= self._wait_limit_for_node(robot.current_node)

    def _wait_limit_for_node(self, node: str) -> int:
        if self.graph.is_entry(node) or self.graph.is_exit(node) or self.graph.is_process_node(node):
            return WAIT_LIMIT_ENTRY_EXIT_PROCESS

        if self.graph.is_junction(node):
            return WAIT_LIMIT_JUNCTION

        return WAIT_LIMIT_NORMAL_NODE

    def _current_node_needed_by_other(self, robot: Robot, robots: list[Robot]) -> bool:
        """
        True se o nó actual do robot aparece como necessidade provável
        de outro robot nos próximos passos.
        """
        node = robot.current_node
        if node is None:
            return False

        for other in robots:
            if other.id == robot.id:
                continue

            if other.goal_node == node:
                return True

            if other.to_node == node and other.state == RobotState.MOVING:
                return True

            other_plan = self._plans.get(other.id, [])
            if node in other_plan[1:LOOKAHEAD_STEPS + 1]:
                return True

            if other.current_node is None or other.goal_node is None:
                continue

            try:
                predicted_path = plan(
                    graph=self.graph,
                    src=other.current_node,
                    dst=other.goal_node,
                    came_from=other.came_from,
                )
            except Exception:
                predicted_path = None

            if predicted_path and node in predicted_path[1:LOOKAHEAD_STEPS + 1]:
                return True

        return False

    # ------------------------------------------------------------------
    # Deadlock
    # ------------------------------------------------------------------

    def _is_deadlock(self, robot: Robot) -> bool:
        return robot.wait_ticks_in_junction >= DEADLOCK_THRESHOLD

    def _blocker_arriving_soon(self, robot: Robot, robots: list[Robot]) -> bool:
        if robot.current_node is None:
            return False

        # Robots at dead-end nodes have no alternative exit: never block them.
        if self.graph.graph.degree(robot.current_node) == 1:
            return False

        next_node = self._next_in_plan(robot)
        if next_node is None:
            return False

        for other in robots:
            if other.id == robot.id:
                continue

            # Another robot is heading TO next_node and about to arrive.
            if other.to_node == next_node and other.state == RobotState.MOVING:
                remaining = 1.0 - other.progress
                total_ticks = self._edge_ticks(other)
                lookahead_fraction = DEADLOCK_LOOKAHEAD_TICKS / max(total_ticks, 1)
                if remaining <= lookahead_fraction:
                    return True

            # NOTE: we intentionally do NOT check other.from_node == next_node.
            # A robot moving FROM next_node is vacating it (node lock already
            # released by sync_moving_node_locks), so it is not a blocker.

        return False

    def _edge_ticks(self, robot: Robot) -> int:
        if robot.from_node is None or robot.to_node is None:
            return 20
        try:
            dist = self.graph.edge_distance(robot.from_node, robot.to_node)
            from env.core.physics import MAX_SPEED
            return max(int(dist / max(MAX_SPEED, 1)), 1)
        except Exception:
            return 20

    # ------------------------------------------------------------------
    # Parking
    # ------------------------------------------------------------------

    def _find_parking(self, robot: Robot, robots: list[Robot]) -> str | None:
        """
        Escolhe parking através do módulo de previsão.

        A reserva fica feita aqui para evitar que outro robot escolha o mesmo
        segmento/ponto no mesmo tick.
        """
        candidate = find_safe_parking_candidate(
            graph=self.graph,
            robot=robot,
            robots=robots,
        )

        if candidate is None:
            return None

        u, v, fraction = parking_candidate_to_target(candidate)

        if not self.segment_lock.is_free_for(u, v, robot.id):
            return None

        if not self.parking_lock.try_acquire(u, v, fraction, robot.id):
            return None

        if not self.segment_lock.try_acquire(u, v, robot.id):
            self.parking_lock.release(u, v, fraction, robot.id)
            return None

        return f"{u}|{v}|{fraction}"

    def _unpark_target(self, robot: Robot, robots: list[Robot]) -> str | None:
        """
        Escolhe o melhor extremo da aresta de parking.

        Em vez de obrigar o robot a voltar ao nó de origem, compara os dois
        extremos da aresta e escolhe o que reduz mais o custo até ao objectivo.
        """
        if not robot.parked_at or not robot.goal_node:
            return None

        if not parked_robot_should_leave(self.graph, robot, robots):
            return None

        u, v, fraction = robot.parked_at
        target = self._best_unpark_endpoint(robot, u, v, fraction)

        if target is not None and self._reserve_unpark(robot, u, v, target):
            return target

        if target is None:
            return None

        fallback = v if target == u else u
        if self._endpoint_is_usable(robot, u, v, fallback):
            if self._reserve_unpark(robot, u, v, fallback):
                return fallback

        return None

    def _best_unpark_endpoint(
        self,
        robot: Robot,
        u: str,
        v: str,
        fraction: float,
    ) -> str | None:
        if robot.goal_node is None:
            return None

        edge_dist = self.graph.edge_distance(u, v)

        candidates: list[tuple[float, str]] = []

        for target, partial_cost, came_from in (
            (u, edge_dist * fraction, v),
            (v, edge_dist * (1.0 - fraction), u),
        ):
            if not self._endpoint_is_usable(robot, u, v, target):
                continue

            route_distance = self.graph.shortest_distance(target, robot.goal_node)
            if route_distance == float("inf"):
                continue

            penalty = 0.0

            # Evita sair para um dead-end, excepto se esse extremo for o objectivo.
            if self.graph.graph.degree(target) <= 1 and target != robot.goal_node:
                penalty += 5000.0

            if robot.came_from is not None and target == robot.came_from:
                penalty += 800.0

            # Pequena penalização para voltar à origem quando o outro extremo
            # também é viável. Isto reduz o padrão parking -> origem -> rota.
            if target == u:
                penalty += 250.0

            candidates.append((partial_cost + route_distance + penalty, target))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _endpoint_is_usable(
        self,
        robot: Robot,
        u: str,
        v: str,
        target: str,
    ) -> bool:
        if target not in {u, v}:
            return False

        if not self.segment_lock.is_free_for(u, v, robot.id):
            return False

        return self.node_lock.is_free_for(target, robot.id)

    def _reserve_unpark(
        self,
        robot: Robot,
        u: str,
        v: str,
        target: str,
    ) -> bool:
        if target == v:
            start, end = u, v
        else:
            start, end = v, u

        if not self.segment_lock.try_acquire(start, end, robot.id):
            return False

        if not self.node_lock.try_acquire(target, robot.id):
            self.segment_lock.release(start, end, robot.id)
            return False

        return True
