from __future__ import annotations

from env.core.entities import Robot, RobotState


class World:
    """Estado global da simulação — contém todos os robots."""

    def __init__(self) -> None:
        """Inicializa o mundo vazio."""
        self._robots: dict[str, Robot] = {}
        self.tick: int = 0

    def reset(self) -> None:
        """Reinicia o mundo — remove todos os robots e faz reset ao tick."""
        self._robots.clear()
        self.tick = 0

    def step_tick(self) -> None:
        """Incrementa o contador de ticks."""
        self.tick += 1

    def add_robot(self, robot: Robot) -> None:
        """Adiciona ou substitui um robot no mundo."""
        self._robots[robot.id] = robot

    def remove_robot(self, robot_id: str) -> None:
        """Remove um robot do mundo."""
        if robot_id in self._robots:
            del self._robots[robot_id]

    def has_robot(self, robot_id: str) -> bool:
        """Indica se existe um robot com o id indicado."""
        return robot_id in self._robots

    def get_robot(self, robot_id: str) -> Robot:
        """Retorna o robot pelo id."""
        return self._robots[robot_id]

    def all_robots(self) -> list[Robot]:
        """Retorna todos os robots."""
        return list(self._robots.values())

    def other_robots(self, robot_id: str) -> list[Robot]:
        """Retorna todos os robots excepto o indicado."""
        return [
            robot
            for robot in self._robots.values()
            if robot.id != robot_id
        ]

    def robots_by_state(self, state: RobotState) -> list[Robot]:
        """Retorna todos os robots num determinado estado."""
        return [
            robot
            for robot in self._robots.values()
            if robot.state == state
        ]

    def idle_robots(self) -> list[Robot]:
        """Retorna robots em IDLE."""
        return self.robots_by_state(RobotState.IDLE)

    def moving_robots(self) -> list[Robot]:
        """Retorna robots em MOVING."""
        return self.robots_by_state(RobotState.MOVING)

    def waiting_robots(self) -> list[Robot]:
        """Retorna robots em WAITING."""
        return self.robots_by_state(RobotState.WAITING)

    def parked_robots(self) -> list[Robot]:
        """Retorna robots em PARKED."""
        return self.robots_by_state(RobotState.PARKED)

    def robot_count(self) -> int:
        """Retorna o número de robots no mundo."""
        return len(self._robots)

    def moving_count(self) -> int:
        """Retorna o número de robots em movimento."""
        return len(self.moving_robots())

    def waiting_count(self) -> int:
        """Retorna o número de robots à espera."""
        return len(self.waiting_robots())

    def parked_count(self) -> int:
        """Retorna o número de robots estacionados."""
        return len(self.parked_robots())

    def robots_on_node(self, node_id: str) -> list[Robot]:
        """
        Retorna robots que estão actualmente num nó real.

        Só conta robots cujo current_node é igual ao nó indicado.
        Robots a meio de uma aresta não entram aqui.
        """
        return [
            robot
            for robot in self._robots.values()
            if robot.current_node == node_id
        ]

    def robots_on_edge(self, u: str, v: str, *, directed: bool = False) -> list[Robot]:
        """
        Retorna robots que estão numa aresta.

        Se directed=True:
            apenas conta robots em u -> v.

        Se directed=False:
            conta robots em u -> v e v -> u.
        """
        robots: list[Robot] = []

        for robot in self._robots.values():
            if robot.from_node is None or robot.to_node is None:
                continue

            if directed:
                if robot.from_node == u and robot.to_node == v:
                    robots.append(robot)
            else:
                same_direction = robot.from_node == u and robot.to_node == v
                opposite_direction = robot.from_node == v and robot.to_node == u

                if same_direction or opposite_direction:
                    robots.append(robot)

        return robots

    def parked_on_edge(self, u: str, v: str) -> list[Robot]:
        """
        Retorna robots estacionados numa determinada aresta.

        A comparação é não-direccional.
        """
        robots: list[Robot] = []

        edge = {u, v}

        for robot in self.parked_robots():
            if robot.parked_at is None:
                continue

            park_u, park_v, _ = robot.parked_at

            if {park_u, park_v} == edge:
                robots.append(robot)

        return robots

    def occupied_nodes(self) -> set[str]:
        """
        Retorna nós ocupados por robots parados em nós reais.

        Não inclui robots estacionados em parking points.
        """
        nodes: set[str] = set()

        for robot in self._robots.values():
            if robot.current_node is not None:
                nodes.add(robot.current_node)

        return nodes

    def occupied_edges(self) -> set[tuple[str, str]]:
        """
        Retorna arestas ocupadas por robots em movimento ou estacionados.

        As arestas são guardadas em formato canónico:
            (min(u, v), max(u, v))
        """
        edges: set[tuple[str, str]] = set()

        for robot in self._robots.values():
            if robot.from_node is not None and robot.to_node is not None:
                u = min(robot.from_node, robot.to_node)
                v = max(robot.from_node, robot.to_node)
                edges.add((u, v))

            if robot.parked_at is not None:
                park_u, park_v, _ = robot.parked_at
                u = min(park_u, park_v)
                v = max(park_u, park_v)
                edges.add((u, v))

        return edges

    def has_parked_robot_on_edge(self, u: str, v: str) -> bool:
        """Indica se existe algum robot estacionado na aresta u-v."""
        return len(self.parked_on_edge(u, v)) > 0

    def has_moving_robot_on_edge(
        self,
        u: str,
        v: str,
        *,
        directed: bool = False,
    ) -> bool:
        """Indica se existe algum robot em movimento na aresta indicada."""
        return len(self.robots_on_edge(u, v, directed=directed)) > 0

    def snapshot(self) -> dict:
        """
        Gera um snapshot simples do estado actual.

        Útil para debug, logs, visualização ou treino futuro de RL.
        """
        return {
            "tick": self.tick,
            "robot_count": self.robot_count(),
            "moving_count": self.moving_count(),
            "waiting_count": self.waiting_count(),
            "parked_count": self.parked_count(),
            "robots": [
                {
                    "id": robot.id,
                    "state": robot.state.name,
                    "current_node": robot.current_node,
                    "from_node": robot.from_node,
                    "to_node": robot.to_node,
                    "progress": robot.progress,
                    "goal_node": robot.goal_node,
                    "came_from": robot.came_from,
                    "wait_ticks": robot.wait_ticks,
                    "wait_ticks_in_junction": robot.wait_ticks_in_junction,
                    "parked_at": robot.parked_at,
                    "carrying_box": robot.carrying_box,
                    "world_x": robot.world_x,
                    "world_y": robot.world_y,
                }
                for robot in self._robots.values()
            ],
        }

    def __repr__(self) -> str:
        return (
            f"World("
            f"tick={self.tick}, "
            f"robots={self.robot_count()}, "
            f"moving={self.moving_count()}, "
            f"waiting={self.waiting_count()}, "
            f"parked={self.parked_count()})"
        )