"""Estado global da simulação — colecção de robots."""

from __future__ import annotations

from env.core.entities import Robot, RobotState


class World:
    """Mundo da simulação: tick global + colecção de robots."""

    def __init__(self) -> None:
        self._robots: dict[str, Robot] = {}
        self.tick: int = 0

    def reset(self) -> None:
        """Limpa robots e reinicia o tick."""
        self._robots.clear()
        self.tick = 0

    def step_tick(self) -> None:
        """Incrementa o tick global."""
        self.tick += 1

    def add_robot(self, robot: Robot) -> None:
        """Adiciona ou substitui um robot."""
        self._robots[robot.id] = robot

    def has_robot(self, robot_id: str) -> bool:
        return robot_id in self._robots

    def all_robots(self) -> list[Robot]:
        return list(self._robots.values())

    def get_robot(self, robot_id: str) -> Robot:
        return self._robots[robot_id]

    def robots_by_state(self, state: RobotState) -> list[Robot]:
        return [r for r in self._robots.values() if r.state == state]

    def moving_robots(self) -> list[Robot]:
        return self.robots_by_state(RobotState.MOVING)

    def waiting_robots(self) -> list[Robot]:
        return self.robots_by_state(RobotState.WAITING)

    def parked_robots(self) -> list[Robot]:
        return self.robots_by_state(RobotState.PARKED)

    def robot_count(self) -> int:
        return len(self._robots)

    def moving_count(self) -> int:
        return len(self.moving_robots())

    def waiting_count(self) -> int:
        return len(self.waiting_robots())

    def parked_count(self) -> int:
        return len(self.parked_robots())

    def robots_on_node(self, node_id: str) -> list[Robot]:
        """Robots cujo current_node é node_id (ignora os em aresta)."""
        return [r for r in self._robots.values() if r.current_node == node_id]

    def parked_on_edge(self, u: str, v: str) -> list[Robot]:
        """Robots estacionados na aresta {u, v} (não-direccional)."""
        edge = {u, v}
        result: list[Robot] = []
        for robot in self.parked_robots():
            if robot.parked_at is None:
                continue
            pu, pv, _ = robot.parked_at
            if {pu, pv} == edge:
                result.append(robot)
        return result

    def snapshot(self) -> dict:
        """Snapshot do estado actual para debug/info."""
        return {
            "tick": self.tick,
            "robot_count":   self.robot_count(),
            "moving_count":  self.moving_count(),
            "waiting_count": self.waiting_count(),
            "parked_count":  self.parked_count(),
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
            f"World(tick={self.tick}, "
            f"robots={self.robot_count()}, "
            f"moving={self.moving_count()}, "
            f"waiting={self.waiting_count()}, "
            f"parked={self.parked_count()})"
        )
