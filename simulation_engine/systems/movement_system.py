from simulation_engine.core.world_state import WorldState
from simulation_engine.core.entities import RobotState, Robot
from simulation_engine.core.graph import FactoryGraph

YIELD_AFTER_REVERSE = 1
BLOCKED_ESCAPE_TICKS = 30   # ticks parado mid-edge antes de forçar retorno


class MovementSystem:
    """
    Atualiza o movimento contínuo dos robôs.

    - Movimento é contínuo ao longo da aresta (progress ∈ [0, 1])
    - Velocidade controlada pelo agente via DispatchSystem
    - Speed < 0 apenas em docking_exit → robot recua ao longo da aresta
    - Chegada ao to_node   quando progress >= 1  → IDLE em to_node
    - Retorno ao from_node quando progress <= 0  → IDLE em from_node
    - Calcula posição contínua (world_x, world_y)
    - NÃO trata colisões
    """

    def __init__(self, graph: FactoryGraph):
        self.graph = graph

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, world: WorldState) -> None:
        """
        Atualiza movimento e posição dos robots.

        Nota:
        - Não faz reset de flags de colisão; isso pertence ao CollisionSystem.
        """
        newly_idled: set[str] = self._update_moving(world)
        self._update_idle(world, newly_idled)

    # ------------------------------------------------------------------
    # MOVING — movimento contínuo (forward e reverse)
    # ------------------------------------------------------------------

    def _update_moving(self, world: WorldState) -> set[str]:
        newly_idled: set[str] = set()

        for robot in world.robots.values():
            if robot.state != RobotState.MOVING:
                continue

            if robot.from_node is None or robot.to_node is None:
                fallback_node = robot.current_node or robot.from_node or robot.to_node
                self._force_idle_on_node(robot, fallback_node)
                newly_idled.add(robot.id)
                continue

            try:
                distance = self.graph.distance(robot.from_node, robot.to_node)
            except Exception:
                self._force_idle_on_node(robot, robot.from_node)
                newly_idled.add(robot.id)
                continue

            if distance <= 0:
                self._force_idle_on_node(robot, robot.from_node)
                newly_idled.add(robot.id)
                continue

            robot.progress += robot.speed / distance

            # ----------------------------------------------------------
            # CHEGADA AO to_node
            # ----------------------------------------------------------
            if robot.progress >= 1.0 and robot.speed >= 0:
                robot.progress = 1.0
                self._update_world_position(robot)

                arrived_node = robot.to_node
                previous_node = robot.from_node

                robot.current_node = arrived_node
                robot.prev_node = previous_node
                robot.from_node = None
                robot.to_node = None
                robot.progress = 0.0
                robot.speed = 0.0
                robot.state = RobotState.IDLE
                robot.docking_exit = False
                robot.blocked_ticks = 0

                newly_idled.add(robot.id)

            # ----------------------------------------------------------
            # RETORNO AO from_node (reversão — speed < 0)
            # ----------------------------------------------------------
            elif robot.progress <= 0.0 and robot.speed < 0:
                robot.progress = 0.0
                self._update_world_position(robot)

                # Docking exit: junction pode ter ficado ocupado depois do
                # dispatch. Esperar até estar livre para evitar freeze.
                if robot.docking_exit and world.robots_at_node(robot.from_node):
                    robot.speed = 0.0
                    robot.blocked_ticks += 1
                    continue

                origin_node = robot.from_node
                attempted_node = robot.to_node

                robot.current_node = origin_node
                robot.prev_node = attempted_node
                robot.from_node = None
                robot.to_node = None
                robot.progress = 0.0
                robot.speed = 0.0
                robot.state = RobotState.IDLE
                robot.docking_exit = False
                robot.blocked_ticks = 0

                # pausa antes de re-dispatch para separar robots
                robot.yield_ticks = YIELD_AFTER_REVERSE

                newly_idled.add(robot.id)

            # ----------------------------------------------------------
            # EM TRÂNSITO
            # ----------------------------------------------------------
            else:
                self._update_world_position(robot)
                # Contar ticks onde o robot está parado a meio da aresta
                if robot.speed <= 0.0 and 0.0 < robot.progress < 1.0:
                    robot.blocked_ticks += 1
                    # Escape determinístico: parado demasiado tempo → retorna ao from_node
                    if robot.blocked_ticks >= BLOCKED_ESCAPE_TICKS and not robot.docking_exit:
                        origin_node   = robot.from_node
                        attempted_node = robot.to_node
                        robot.current_node = origin_node
                        robot.prev_node    = attempted_node
                        robot.from_node    = None
                        robot.to_node      = None
                        robot.progress     = 0.0
                        robot.speed        = 0.0
                        robot.state        = RobotState.IDLE
                        robot.blocked_ticks = 0
                        robot.yield_ticks  = YIELD_AFTER_REVERSE
                        newly_idled.add(robot.id)
                else:
                    robot.blocked_ticks = 0

        return newly_idled

    # ------------------------------------------------------------------
    # IDLE — mantém posição no nó
    # ------------------------------------------------------------------

    def _update_idle(self, world: WorldState, newly_idled: set[str]) -> None:
        for robot in world.robots.values():
            if robot.state != RobotState.IDLE:
                continue

            if robot.current_node is None:
                continue

            x, y = self.graph.node_position(robot.current_node)
            robot.world_x = x
            robot.world_y = y

            # Não consumir cooldowns no mesmo tick em que o robot acabou
            # de passar para IDLE.
            if robot.id in newly_idled:
                continue

            if robot.yield_ticks > 0:
                robot.yield_ticks -= 1

            # Contar ticks em que o robot está pronto a mover mas bloqueado
            if robot.yield_ticks == 0 and robot.turn_cooldown == 0:
                robot.idle_stuck_ticks += 1

            # contagem decrescente da manobra de viragem
            if robot.turn_cooldown > 0:
                robot.turn_cooldown -= 1
                if robot.turn_cooldown == 0:
                    # Manobra concluída — limpa prev_node para que o próximo
                    # dispatch não reatrive _needs_turn para o mesmo nó de origem
                    # (o ciclo infinito: _needs_turn True → cooldown → _needs_turn True → …)
                    robot.prev_node = None
                    robot.is_turning = False

    # ------------------------------------------------------------------
    # POSIÇÃO CONTÍNUA
    # ------------------------------------------------------------------

    def _update_world_position(self, robot: Robot) -> None:
        if robot.from_node is None or robot.to_node is None:
            return

        x1, y1 = self.graph.node_position(robot.from_node)
        x2, y2 = self.graph.node_position(robot.to_node)

        t = max(0.0, min(1.0, robot.progress))
        robot.world_x = x1 + (x2 - x1) * t
        robot.world_y = y1 + (y2 - y1) * t

    # ------------------------------------------------------------------
    # FAIL-SAFE
    # ------------------------------------------------------------------

    def _force_idle_on_node(self, robot: Robot, node_id: str | None) -> None:
        """
        Recupera de um estado MOVING inválido para evitar robots presos.

        Se existir um nó de referência, o robot é colocado nesse nó.
        """
        robot.state = RobotState.IDLE
        robot.speed = 0.0
        robot.progress = 0.0
        robot.from_node = None
        robot.to_node = None
        robot.docking_exit = False
        robot.current_node = node_id
        robot.yield_ticks = 0
        robot.turn_cooldown = 0
        robot.is_turning = False
        robot.blocked_ticks = 0
        robot.idle_stuck_ticks = 0

        if node_id is not None:
            x, y = self.graph.node_position(node_id)
            robot.world_x = x
            robot.world_y = y