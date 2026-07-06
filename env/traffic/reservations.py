"""Tabela de reservas espaço-tempo para o Cooperative A*.

Cada robot planeado reserva os nós e arestas que ocupa, com o intervalo de
tempo (ticks) correspondente. Robots planeados a seguir consultam esta
tabela para evitar esses intervalos — é isto que substitui os locks
reactivos e o replanning do router antigo.

Suporta também PRIORIDADE dinâmica: uma reserva de outro robot só bloqueia
se a prioridade do dono for >= à de quem pergunta (`priorities` + `
requester_priority`); menor prioridade é ignorada (o dono é "atropelado" —
ver `conflicts_below_priority`, usado por quem comete uma reserva
prioritária para descobrir quem precisa de replanear)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    """Intervalo de tempo [start, end] (ticks)."""
    start: float
    end: float

    def overlaps(self, other: "Interval", buffer: float = 0.0) -> bool:
        """True se este intervalo se sobrepõe a `other`, com margem `buffer`."""
        return self.start < other.end + buffer and other.start < self.end + buffer


class ReservationTable:
    """Reservas de nós e arestas por intervalo de tempo, partilhadas entre
    robots — a base da coordenação sem colisões do Space-Time A*."""

    def __init__(self) -> None:
        """Cria uma tabela de reservas vazia."""
        self._node_reservations: dict[str, list[tuple[Interval, str]]] = {}
        self._edge_reservations: dict[tuple[str, str], list[tuple[Interval, str]]] = {}

    def _edge_key(self, u: str, v: str) -> tuple[str, str]:
        """Chave canónica (não-direccional) de uma aresta."""
        return (u, v) if u <= v else (v, u)

    def _blocks(
        self,
        owner: str,
        robot_id: str,
        priorities: dict[str, float] | None,
        requester_priority: float,
        reserved_start: float | None = None,
        now: float | None = None,
    ) -> bool:
        """True se a reserva de `owner` deve contar como bloqueio para
        `robot_id`: sempre, a não ser que `priorities` esteja definido e
        `owner` tenha prioridade estritamente menor que `requester_priority`
        (nesse caso é ignorada — `owner` será "atropelado").

        Uma reserva que ainda NÃO COMEÇOU (`reserved_start > now`) nunca é
        atropelável, seja qual for a prioridade — representa algo que o
        dono ainda vai fazer no futuro (ex.: uma pausa breve mais à frente
        no seu trajecto actual), e nesse instante ele pode estar a meio de
        OUTRA aresta, sem forma segura de o interromper agora. Só reservas
        já em curso (`reserved_start <= now` — o dono está mesmo ali,
        agora) podem ser atropeladas."""
        if owner == robot_id:
            return False
        if priorities is None:
            return True
        if now is not None and reserved_start is not None and reserved_start > now + 1e-9:
            return True
        return priorities.get(owner, float("inf")) >= requester_priority

    def safe_intervals(
        self,
        node: str,
        robot_id: str,
        buffer: float = 0.0,
        priorities: dict[str, float] | None = None,
        requester_priority: float = float("inf"),
        now: float | None = None,
    ) -> list[tuple[float, float]]:
        """Intervalos [lo, hi] em que `node` está livre (para presença
        pontual), a base do SIPP. Uma reserva bloqueante [s, e] bloqueia
        (s-buffer, e+buffer); os intervalos seguros são o complemento sobre
        [0, ∞). Sempre pelo menos um (o último estende-se a ∞)."""
        blocked = sorted(
            (r.start - buffer, r.end + buffer)
            for r, owner in self._node_reservations.get(node, [])
            if self._blocks(owner, robot_id, priorities, requester_priority, r.start, now)
        )
        safe: list[tuple[float, float]] = []
        cursor = 0.0
        for lo, hi in blocked:
            if lo > cursor:
                safe.append((cursor, lo))
            cursor = max(cursor, hi)
        safe.append((cursor, float("inf")))
        return safe

    def is_node_free(
        self,
        node: str,
        interval: Interval,
        robot_id: str,
        buffer: float = 0.0,
        priorities: dict[str, float] | None = None,
        requester_priority: float = float("inf"),
        now: float | None = None,
    ) -> bool:
        """True se `node` está livre (de reservas bloqueantes) durante `interval`."""
        for reserved, owner in self._node_reservations.get(node, []):
            if self._blocks(owner, robot_id, priorities, requester_priority, reserved.start, now) and interval.overlaps(reserved, buffer):
                return False
        return True

    def is_edge_free(
        self,
        u: str,
        v: str,
        interval: Interval,
        robot_id: str,
        buffer: float = 0.0,
        priorities: dict[str, float] | None = None,
        requester_priority: float = float("inf"),
    ) -> bool:
        """True se a aresta u-v está livre (de reservas bloqueantes) durante `interval`."""
        key = self._edge_key(u, v)
        for reserved, owner in self._edge_reservations.get(key, []):
            if self._blocks(owner, robot_id, priorities, requester_priority) and interval.overlaps(reserved, buffer):
                return False
        return True

    def earliest_free_edge_start(
        self,
        u: str,
        v: str,
        duration: float,
        earliest_start: float,
        robot_id: str,
        buffer: float = 0.0,
        max_iterations: int = 50,
        priorities: dict[str, float] | None = None,
        requester_priority: float = float("inf"),
    ) -> float | None:
        """Menor instante de partida (>= earliest_start) tal que ocupar a
        aresta u-v durante `duration` não colide com reservas bloqueantes.

        Ao contrário de um nó (onde chegar mais cedo e ficar mais tempo
        parado nunca resolve um conflito com uma reserva futura — a
        janela cresce mas o início não muda), uma aresta pode
        simplesmente ser atravessada mais tarde: atrasar a partida desloca
        o intervalo inteiro no tempo, o que resolve genuinamente
        conflitos transitórios (ex.: outro robot só está a passar).
        """
        key = self._edge_key(u, v)
        reservations = self._edge_reservations.get(key, [])
        start = earliest_start

        for _ in range(max_iterations):
            candidate = Interval(start, start + duration)
            conflict_end = None
            for reserved, owner in reservations:
                if not self._blocks(owner, robot_id, priorities, requester_priority):
                    continue
                if candidate.overlaps(reserved, buffer):
                    if conflict_end is None or reserved.end > conflict_end:
                        conflict_end = reserved.end
            if conflict_end is None:
                return start
            if conflict_end == float("inf"):
                return None
            start = conflict_end + buffer + 1e-9

        return None

    def earliest_free_node_instant(
        self,
        node: str,
        earliest_t: float,
        robot_id: str,
        buffer: float = 0.0,
        max_iterations: int = 50,
        priorities: dict[str, float] | None = None,
        requester_priority: float = float("inf"),
        now: float | None = None,
    ) -> float | None:
        """Menor instante (>= earliest_t) em que `node` está livre (de
        reservas bloqueantes).

        Ao contrário de esperar mais tempo *parado* num nó já ocupado
        (isso nunca resolve nada — ver `earliest_free_edge_start`), isto
        responde a uma pergunta diferente: "posso sequer estar neste nó
        neste instante?". Se não, empurra o instante para o fim da
        reserva conflituosa e repete — como só verifica um ponto (não um
        intervalo com início fixo), isto SIM se resolve deslocando-o.
        """
        entries = self._node_reservations.get(node, [])
        t = earliest_t

        for _ in range(max_iterations):
            point = Interval(t, t)
            conflict_end = None
            for reserved, owner in entries:
                if not self._blocks(owner, robot_id, priorities, requester_priority, reserved.start, now):
                    continue
                if point.overlaps(reserved, buffer):
                    if conflict_end is None or reserved.end > conflict_end:
                        conflict_end = reserved.end
            if conflict_end is None:
                return t
            if conflict_end == float("inf"):
                return None
            t = conflict_end + buffer + 1e-9

        return None

    def conflicts_below_priority(
        self,
        node_claims: list[tuple[str, Interval]],
        edge_claims: list[tuple[str, str, Interval]],
        robot_id: str,
        priorities: dict[str, float],
        requester_priority: float,
        buffer: float = 0.0,
        now: float | None = None,
    ) -> set[str]:
        """Robot_ids com reservas de NÓ que se sobrepõem às reservas em
        `node_claims` e têm prioridade ESTRITAMENTE menor que
        `requester_priority` E já começaram (`reserved.start <= now`) —
        os "atropelados" que `robot_id` vai precisar de mandar replanear.
        Reservas ainda no futuro nunca contam (ver `_blocks`); `edge_claims`
        nunca gera vítimas (arestas são sempre FCFS puro)."""
        victims: set[str] = set()
        for node, interval in node_claims:
            for reserved, owner in self._node_reservations.get(node, []):
                if owner == robot_id or owner == "__gate__":
                    continue
                if now is not None and reserved.start > now + 1e-9:
                    continue
                if priorities.get(owner, float("inf")) < requester_priority and interval.overlaps(reserved, buffer):
                    victims.add(owner)
        return victims

    def reserve_node(self, node: str, interval: Interval, robot_id: str) -> None:
        """Reserva `node` para `robot_id` durante `interval`."""
        self._node_reservations.setdefault(node, []).append((interval, robot_id))

    def reserve_edge(self, u: str, v: str, interval: Interval, robot_id: str) -> None:
        """Reserva a aresta u-v para `robot_id` durante `interval`."""
        key = self._edge_key(u, v)
        self._edge_reservations.setdefault(key, []).append((interval, robot_id))

    def clear_robot(self, robot_id: str) -> None:
        """Remove todas as reservas de um robot (para replanear)."""
        for node, entries in self._node_reservations.items():
            self._node_reservations[node] = [e for e in entries if e[1] != robot_id]
        for edge, entries in self._edge_reservations.items():
            self._edge_reservations[edge] = [e for e in entries if e[1] != robot_id]

    def prune_before(self, t: float) -> None:
        """Descarta reservas já totalmente no passado (fim < t) — evita as
        tabelas crescerem sem limite num cenário de replaneamento contínuo."""
        for node in list(self._node_reservations):
            kept = [e for e in self._node_reservations[node] if e[0].end >= t]
            if kept:
                self._node_reservations[node] = kept
            else:
                del self._node_reservations[node]
        for edge in list(self._edge_reservations):
            kept = [e for e in self._edge_reservations[edge] if e[0].end >= t]
            if kept:
                self._edge_reservations[edge] = kept
            else:
                del self._edge_reservations[edge]

    def reset(self) -> None:
        """Apaga todas as reservas."""
        self._node_reservations.clear()
        self._edge_reservations.clear()
