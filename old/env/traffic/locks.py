"""Locks de tráfego: segmentos, nós e parking points geométricos."""

from __future__ import annotations


class SegmentLock:
    """Locks de segmentos (arestas) — um robot em movimento por aresta."""

    def __init__(self) -> None:
        self._segments: dict[tuple[str, str], dict] = {}

    def _key(self, u: str, v: str) -> tuple[str, str]:
        return (min(u, v), max(u, v))

    def try_acquire(self, u: str, v: str, robot_id: str) -> bool:
        """True se conseguiu adquirir (ou já tinha) o segmento."""
        key = self._key(u, v)
        if key not in self._segments:
            self._segments[key] = {
                "direction": (u, v),
                "robots": {robot_id},
            }
            return True
        return robot_id in self._segments[key]["robots"]

    def release(self, u: str, v: str, robot_id: str) -> None:
        """Liberta o segmento para o robot."""
        key = self._key(u, v)
        if key not in self._segments:
            return
        entry = self._segments[key]
        entry["robots"].discard(robot_id)
        if not entry["robots"]:
            del self._segments[key]

    def release_all_for(self, robot_id: str) -> None:
        """Liberta todos os segmentos ocupados pelo robot."""
        to_delete: list[tuple[str, str]] = []
        for key, entry in self._segments.items():
            entry["robots"].discard(robot_id)
            if not entry["robots"]:
                to_delete.append(key)
        for key in to_delete:
            del self._segments[key]

    def is_free(self, u: str, v: str) -> bool:
        return self._key(u, v) not in self._segments

    def is_free_for(self, u: str, v: str, robot_id: str) -> bool:
        """True se o segmento está livre ou já pertence ao robot."""
        key = self._key(u, v)
        if key not in self._segments:
            return True
        return robot_id in self._segments[key]["robots"]

    def occupied_segments(self) -> set[tuple[str, str]]:
        return set(self._segments.keys())

    def blocked_edges_for(self, robot_id: str) -> set[tuple[str, str]]:
        """Segmentos ocupados por outros robots."""
        return {
            key for key, entry in self._segments.items()
            if robot_id not in entry["robots"]
        }

    def __repr__(self) -> str:
        return f"SegmentLock(segments_occupied={len(self._segments)})"


class NodeLock:
    """Locks de nós — um robot por nó."""

    def __init__(self) -> None:
        self._nodes: dict[str, str] = {}

    def try_acquire(self, node_id: str, robot_id: str) -> bool:
        """True se conseguiu ocupar (ou já era do robot)."""
        current = self._nodes.get(node_id)
        if current is None or current == robot_id:
            self._nodes[node_id] = robot_id
            return True
        return False

    def release(self, node_id: str, robot_id: str) -> None:
        """Liberta o nó se pertencer ao robot."""
        if self._nodes.get(node_id) == robot_id:
            del self._nodes[node_id]

    def release_all_for(self, robot_id: str) -> None:
        """Liberta todos os nós ocupados pelo robot."""
        to_delete = [n for n, rid in self._nodes.items() if rid == robot_id]
        for node in to_delete:
            del self._nodes[node]

    def is_free(self, node_id: str) -> bool:
        return node_id not in self._nodes

    def is_free_for(self, node_id: str, robot_id: str) -> bool:
        """True se livre ou já pertence ao robot."""
        current = self._nodes.get(node_id)
        return current is None or current == robot_id

    def blocked_nodes_for(self, robot_id: str) -> set[str]:
        """Nós ocupados por outros robots."""
        return {n for n, rid in self._nodes.items() if rid != robot_id}

    def __repr__(self) -> str:
        return f"NodeLock(nodes_occupied={len(self._nodes)})"


class ParkingLock:
    """Locks de parking points geométricos (u, v, fracção)."""

    def __init__(self) -> None:
        self._parking: dict[tuple[str, str, int], str] = {}

    def _segment_key(self, u: str, v: str) -> tuple[str, str]:
        return (min(u, v), max(u, v))

    def _fraction_key(self, fraction: float) -> int:
        return int(round(fraction * 1000))

    def _key(self, u: str, v: str, fraction: float) -> tuple[str, str, int]:
        a, b = self._segment_key(u, v)
        return a, b, self._fraction_key(fraction)

    def try_acquire(self, u: str, v: str, fraction: float, robot_id: str) -> bool:
        """True se conseguiu reservar (ou já pertencia ao robot)."""
        key = self._key(u, v, fraction)
        current = self._parking.get(key)
        if current is None or current == robot_id:
            self._parking[key] = robot_id
            return True
        return False

    def release(self, u: str, v: str, fraction: float, robot_id: str) -> None:
        """Liberta o parking point se pertencer ao robot."""
        key = self._key(u, v, fraction)
        if self._parking.get(key) == robot_id:
            del self._parking[key]

    def release_all_for(self, robot_id: str) -> None:
        """Liberta todos os parking points do robot."""
        to_delete = [k for k, rid in self._parking.items() if rid == robot_id]
        for key in to_delete:
            del self._parking[key]

    def occupied_segments(self) -> set[tuple[str, str]]:
        """Segmentos com pelo menos um parking point ocupado."""
        return {(u, v) for u, v, _ in self._parking.keys()}

    def blocked_edges_for(self, robot_id: str) -> set[tuple[str, str]]:
        """Segmentos com parking ocupado por outros robots."""
        return {
            (u, v) for (u, v, _), rid in self._parking.items()
            if rid != robot_id
        }

    def __repr__(self) -> str:
        return f"ParkingLock(points_occupied={len(self._parking)})"
