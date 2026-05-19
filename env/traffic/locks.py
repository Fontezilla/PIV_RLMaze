from __future__ import annotations


class SegmentLock:
    """
    Controla o acesso a segmentos do grafo.

    Regras:
    - cada segmento só pode estar ocupado por um robot em movimento de cada vez;
    - evita colisões frontais;
    - evita colisões por trás;
    - um robot pode renovar o lock do segmento que já ocupa;
    - robots estacionados são tratados separadamente em ParkingLock.
    """

    def __init__(self) -> None:
        # segment_id -> {"direction": (from, to), "robots": set[str]}
        self._segments: dict[tuple[str, str], dict] = {}

    def _key(self, u: str, v: str) -> tuple[str, str]:
        """Chave canónica do segmento, independente da direcção."""
        return (min(u, v), max(u, v))

    def try_acquire(self, u: str, v: str, robot_id: str) -> bool:
        """
        Tenta adquirir o segmento u->v para o robot.

        Retorna:
        - True se o segmento está livre;
        - True se o próprio robot já tinha o segmento;
        - False se outro robot ocupa o segmento.
        """
        key = self._key(u, v)
        direction = (u, v)

        if key not in self._segments:
            self._segments[key] = {
                "direction": direction,
                "robots": {robot_id},
            }
            return True

        entry = self._segments[key]

        if robot_id in entry["robots"]:
            return True

        return False

    def release(self, u: str, v: str, robot_id: str) -> None:
        """Liberta o segmento para o robot dado."""
        key = self._key(u, v)

        if key not in self._segments:
            return

        entry = self._segments[key]
        entry["robots"].discard(robot_id)

        if not entry["robots"]:
            del self._segments[key]

    def release_all_for(self, robot_id: str) -> None:
        """Liberta todos os segmentos ocupados por um robot."""
        to_delete: list[tuple[str, str]] = []

        for key, entry in self._segments.items():
            entry["robots"].discard(robot_id)

            if not entry["robots"]:
                to_delete.append(key)

        for key in to_delete:
            del self._segments[key]

    def is_free(self, u: str, v: str) -> bool:
        """True se o segmento está completamente livre."""
        return self._key(u, v) not in self._segments

    def is_free_for(self, u: str, v: str, robot_id: str) -> bool:
        """
        True se:
        - o segmento está livre;
        - ou o próprio robot já ocupa o segmento.

        False se outro robot ocupa o segmento.
        """
        key = self._key(u, v)

        if key not in self._segments:
            return True

        entry = self._segments[key]
        return robot_id in entry["robots"]

    def occupied_segments(self) -> set[tuple[str, str]]:
        """Retorna todos os segmentos actualmente ocupados."""
        return set(self._segments.keys())

    def blocked_edges_for(self, robot_id: str) -> set[tuple[str, str]]:
        """
        Retorna os segmentos bloqueados para este robot.

        Como o segmento é exclusivo, qualquer segmento ocupado por outro robot
        fica bloqueado.
        """
        blocked: set[tuple[str, str]] = set()

        for key, entry in self._segments.items():
            if robot_id not in entry["robots"]:
                blocked.add(key)

        return blocked

    def direction_of(self, u: str, v: str) -> tuple[str, str] | None:
        """Retorna a direcção actual do segmento ou None se livre."""
        key = self._key(u, v)
        entry = self._segments.get(key)
        return entry["direction"] if entry else None

    def robots_in(self, u: str, v: str) -> set[str]:
        """Retorna os robots que estão no segmento."""
        key = self._key(u, v)
        entry = self._segments.get(key)
        return set(entry["robots"]) if entry else set()

    def occupied_by_other(self, u: str, v: str, robot_id: str) -> bool:
        """True se o segmento está ocupado por outro robot."""
        key = self._key(u, v)
        entry = self._segments.get(key)

        if entry is None:
            return False

        return robot_id not in entry["robots"]

    def __repr__(self) -> str:
        occupied = len(self._segments)
        return f"SegmentLock(segments_occupied={occupied})"


class NodeLock:
    """
    Controla a ocupação de nós.

    Cada nó só pode ser ocupado por um robot de cada vez.
    """

    def __init__(self) -> None:
        # node_id -> robot_id
        self._nodes: dict[str, str] = {}

    def try_acquire(self, node_id: str, robot_id: str) -> bool:
        """
        Tenta ocupar o nó para o robot.

        Retorna:
        - True se conseguiu ocupar;
        - True se o nó já era do próprio robot;
        - False se está ocupado por outro robot.
        """
        current = self._nodes.get(node_id)

        if current is None or current == robot_id:
            self._nodes[node_id] = robot_id
            return True

        return False

    def release(self, node_id: str, robot_id: str) -> None:
        """Liberta o nó se estava ocupado por este robot."""
        if self._nodes.get(node_id) == robot_id:
            del self._nodes[node_id]

    def release_all_for(self, robot_id: str) -> None:
        """Liberta todos os nós ocupados por um robot."""
        to_delete = [
            node
            for node, rid in self._nodes.items()
            if rid == robot_id
        ]

        for node in to_delete:
            del self._nodes[node]

    def is_free(self, node_id: str) -> bool:
        """True se o nó está livre."""
        return node_id not in self._nodes

    def is_free_for(self, node_id: str, robot_id: str) -> bool:
        """True se o nó está livre ou já pertence a este robot."""
        current = self._nodes.get(node_id)
        return current is None or current == robot_id

    def occupied_by(self, node_id: str) -> str | None:
        """Retorna o robot_id que ocupa o nó, ou None se livre."""
        return self._nodes.get(node_id)

    def occupied_nodes(self) -> set[str]:
        """Retorna todos os nós ocupados."""
        return set(self._nodes.keys())

    def blocked_nodes_for(self, robot_id: str) -> set[str]:
        """
        Retorna os nós bloqueados para este robot,
        ou seja, nós ocupados por outros robots.
        """
        return {
            node
            for node, rid in self._nodes.items()
            if rid != robot_id
        }

    def __repr__(self) -> str:
        return f"NodeLock(nodes_occupied={len(self._nodes)})"


class ParkingLock:
    """
    Controla parking points geométricos em arestas.

    Um parking point é identificado por:
        (u, v, fraction)

    A chave da aresta é canónica, mas a fracção é mantida.
    Isto permite evitar dois robots estacionados no mesmo ponto.
    """

    def __init__(self) -> None:
        # (segment_u, segment_v, fraction_key) -> robot_id
        self._parking: dict[tuple[str, str, int], str] = {}

    def _segment_key(self, u: str, v: str) -> tuple[str, str]:
        """Chave canónica do segmento."""
        return (min(u, v), max(u, v))

    def _fraction_key(self, fraction: float) -> int:
        """
        Converte a fracção para chave estável.

        Exemplo:
            0.5   -> 500
            0.333 -> 333
            0.667 -> 667
        """
        return int(round(fraction * 1000))

    def _key(self, u: str, v: str, fraction: float) -> tuple[str, str, int]:
        a, b = self._segment_key(u, v)
        return a, b, self._fraction_key(fraction)

    def try_acquire(self, u: str, v: str, fraction: float, robot_id: str) -> bool:
        """
        Tenta reservar um parking point para um robot.

        Retorna:
        - True se o parking point está livre;
        - True se já pertence ao próprio robot;
        - False se está ocupado por outro robot.
        """
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
        """Liberta todos os parking points ocupados por um robot."""
        to_delete = [
            key
            for key, rid in self._parking.items()
            if rid == robot_id
        ]

        for key in to_delete:
            del self._parking[key]

    def is_free(self, u: str, v: str, fraction: float) -> bool:
        """True se o parking point está livre."""
        return self._key(u, v, fraction) not in self._parking

    def is_free_for(self, u: str, v: str, fraction: float, robot_id: str) -> bool:
        """True se o parking point está livre ou já pertence ao robot."""
        key = self._key(u, v, fraction)
        current = self._parking.get(key)

        return current is None or current == robot_id

    def occupied_by(self, u: str, v: str, fraction: float) -> str | None:
        """Retorna o robot_id que ocupa o parking point, ou None se livre."""
        return self._parking.get(self._key(u, v, fraction))

    def occupied_points(self) -> dict[tuple[str, str, int], str]:
        """Retorna uma cópia dos parking points ocupados."""
        return dict(self._parking)

    def occupied_segments(self) -> set[tuple[str, str]]:
        """
        Retorna os segmentos que têm pelo menos um robot estacionado.

        Útil para o planner penalizar arestas com robots estacionados.
        """
        return {
            (u, v)
            for u, v, _fraction in self._parking.keys()
        }

    def blocked_edges_for(self, robot_id: str) -> set[tuple[str, str]]:
        """
        Retorna segmentos com parking ocupado por outros robots.

        Isto não deve ser usado sempre como bloqueio absoluto.
        Para fluidez, muitas vezes é melhor passar isto ao planner como
        congested_edges em vez de blocked_edges.
        """
        blocked: set[tuple[str, str]] = set()

        for (u, v, _fraction), rid in self._parking.items():
            if rid != robot_id:
                blocked.add((u, v))

        return blocked

    def __repr__(self) -> str:
        return f"ParkingLock(points_occupied={len(self._parking)})"