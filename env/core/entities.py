"""
env/core/entities.py
~~~~~~~~~~~~~~~~~~~~
Definição das entidades do ambiente de fábrica.

Robot  — AGV autónomo
Box    — caixa com pipeline e waypoints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


MAX_HISTORY_LEN  = 10
MAX_PARKED_TICKS = 80


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RobotState(Enum):
    """Estado do robot."""
    IDLE    = auto()
    MOVING  = auto()
    WAITING = auto()
    PARKED  = auto()


class BoxStatus(Enum):
    """Estado de uma caixa no ambiente."""
    WAITING    = auto()   # no nó, à espera de ser apanhada
    IN_TRANSIT = auto()   # a ser transportada por um robot
    DONE       = auto()   # entregue no exit node


class PipelineType(Enum):
    """Pipeline atribuída a uma caixa (define os waypoints intermédios)."""
    BLUE  = auto()   # entry → exit (directo)
    GREEN = auto()   # entry → processA → exit
    RED   = auto()   # entry → processA → processB → exit


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------

@dataclass
class Box:
    """
    Representa uma caixa no ambiente.

    Campos
    ------
    box_id       : identificador único
    pipeline     : tipo de pipeline (BLUE / GREEN / RED)
    waypoints    : sequência ordenada de nós que a caixa deve visitar,
                   incluindo o nó inicial (entry) e o final (exit)
    waypoint_idx : índice do próximo waypoint a atingir (começa em 1,
                   pois waypoints[0] é o nó de spawn onde a caixa já está)
    current_node : nó onde a caixa se encontra (None quando em trânsito)
    status       : WAITING | IN_TRANSIT | DONE
    carried_by   : id do robot que carrega a caixa, ou None
    """

    box_id       : int
    pipeline     : PipelineType
    waypoints    : list[str]
    current_node : str
    waypoint_idx : int        = 1
    status       : BoxStatus  = BoxStatus.WAITING
    carried_by   : str | None = None

    # ------------------------------------------------------------------
    # Propriedades de conveniência
    # ------------------------------------------------------------------

    @property
    def next_waypoint(self) -> str | None:
        """Próximo nó que a caixa deve atingir, ou None se entregue."""
        if self.waypoint_idx < len(self.waypoints):
            return self.waypoints[self.waypoint_idx]
        return None

    @property
    def is_waiting(self) -> bool:
        return self.status == BoxStatus.WAITING

    @property
    def is_in_transit(self) -> bool:
        return self.status == BoxStatus.IN_TRANSIT

    @property
    def is_done(self) -> bool:
        return self.status == BoxStatus.DONE

    @property
    def is_available(self) -> bool:
        """Pode ser apanhada: WAITING num nó e sem robot atribuído."""
        return self.status == BoxStatus.WAITING and self.carried_by is None

    # ------------------------------------------------------------------
    # Mutações
    # ------------------------------------------------------------------

    def pick_up(self, robot_id: str) -> None:
        """Regista que o robot apanhou a caixa."""
        self.carried_by   = robot_id
        self.status       = BoxStatus.IN_TRANSIT
        self.current_node = None

    def advance_waypoint(self, node: str) -> bool:
        """
        Regista chegada ao próximo waypoint.

        Devolve True se a caixa foi entregue (último waypoint = exit node).
        """
        self.waypoint_idx += 1
        self.current_node  = node

        if self.next_waypoint is None:
            self.status     = BoxStatus.DONE
            self.carried_by = None
            return True

        # Waypoint intermédio — fica a aguardar no nó actual
        self.status     = BoxStatus.WAITING
        self.carried_by = None
        return False

    def __repr__(self) -> str:
        nxt = self.next_waypoint or "—"
        return (
            f"Box(id={self.box_id}, "
            f"pipeline={self.pipeline.name}, "
            f"status={self.status.name}, "
            f"node={self.current_node}, "
            f"next_wp={nxt}, "
            f"carried_by={self.carried_by})"
        )


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

@dataclass
class Robot:
    """
    AGV autónomo no ambiente da fábrica.

    O histórico anti-loop (last_visited, last_edges) e as penalizações
    associadas são responsabilidade do Router, não desta entidade.
    """

    id    : str
    state : RobotState = RobotState.IDLE

    # Posição no grafo
    current_node : str | None = None
    from_node    : str | None = None
    to_node      : str | None = None
    progress     : float      = 0.0

    # Velocidade
    speed        : float = 0.0
    target_speed : float = 0.0

    # Navegação
    goal_node  : str | None = None
    came_from  : str | None = None

    # Contadores de espera
    wait_ticks             : int = 0
    wait_ticks_in_junction : int = 0
    turn_ticks_total       : int = 0

    # Posição no mundo (para rendering)
    world_x : float = 0.0
    world_y : float = 0.0

    # Histórico anti-loop — gerido pelo Router
    last_visited : list[str]              = field(default_factory=list)
    last_edges   : list[tuple[str, str]]  = field(default_factory=list)

    # Parking temporário: (u, v, fraction) ou None
    parked_at               : tuple[str, str, float] | None = None
    parked_ticks            : int       = 0
    park_reason             : str | None = None
    parking_reserved_edge   : tuple[str, str] | None = None

    # Caixa transportada (box_id ou None)
    carrying_box    : int | None = None

    # Box_id explicitamente assignado pelo agente (None após pick ou sem assignment de caixa)
    assigned_box_id : int | None = None

    # ------------------------------------------------------------------
    # Predicados de estado
    # ------------------------------------------------------------------

    def is_idle(self) -> bool:
        return self.state == RobotState.IDLE

    def is_moving(self) -> bool:
        return self.state == RobotState.MOVING

    def is_waiting(self) -> bool:
        return self.state == RobotState.WAITING

    def is_parked(self) -> bool:
        return self.state == RobotState.PARKED

    def is_free(self) -> bool:
        """Livre para receber novo assignment: IDLE ou WAITING sem caixa."""
        return (
            self.state in (RobotState.IDLE, RobotState.WAITING)
            and self.carrying_box is None
        )

    def reached_goal(self) -> bool:
        return (
            self.state == RobotState.IDLE
            and self.current_node is not None
            and self.current_node == self.goal_node
        )

    # ------------------------------------------------------------------
    # Histórico anti-loop  (interface mantida para compatibilidade com Router)
    # ------------------------------------------------------------------

    def push_visited(self, node: str, max_len: int = MAX_HISTORY_LEN) -> None:
        if not node:
            return
        if not self.last_visited or self.last_visited[-1] != node:
            self.last_visited.append(node)
        if len(self.last_visited) > max_len:
            self.last_visited = self.last_visited[-max_len:]

    def push_edge(
        self,
        from_node : str,
        to_node   : str,
        max_len   : int = MAX_HISTORY_LEN,
    ) -> None:
        if not from_node or not to_node:
            return
        edge = (from_node, to_node)
        if not self.last_edges or self.last_edges[-1] != edge:
            self.last_edges.append(edge)
        if len(self.last_edges) > max_len:
            self.last_edges = self.last_edges[-max_len:]

    def recent_node_penalties(
        self,
        goal_node: str | None = None,
    ) -> dict[str, float]:
        penalties: dict[str, float] = {}
        counts: dict[str, int] = {}
        for n in self.last_visited:
            counts[n] = counts.get(n, 0) + 1

        for idx, node in enumerate(self.last_visited):
            if node == goal_node:
                continue
            recency        = idx + 1
            penalty        = recency * 450.0
            repeated_count = counts[node]
            if repeated_count > 1:
                penalty += repeated_count * 900.0
            penalties[node] = max(penalties.get(node, 0.0), penalty)

        return penalties

    def recent_edge_penalties(self) -> dict[tuple[str, str], float]:
        penalties  : dict[tuple[str, str], float] = {}
        edge_counts: dict[tuple[str, str], int]   = {}
        for e in self.last_edges:
            edge_counts[e] = edge_counts.get(e, 0) + 1

        for idx, (u, v) in enumerate(self.last_edges):
            recency = idx + 1

            same_pen    = recency * 800.0
            reverse_pen = recency * 2200.0

            repeated_same    = edge_counts.get((u, v), 0)
            repeated_reverse = edge_counts.get((v, u), 0)

            if repeated_same > 1:
                same_pen += repeated_same * 1200.0
            if repeated_reverse > 0:
                reverse_pen += repeated_reverse * 1800.0

            penalties[(u, v)] = max(penalties.get((u, v), 0.0), same_pen)
            penalties[(v, u)] = max(penalties.get((v, u), 0.0), reverse_pen)

        return penalties

    # ------------------------------------------------------------------
    # Mutações de estado
    # ------------------------------------------------------------------

    def enter_parking(
        self,
        u        : str,
        v        : str,
        fraction : float,
        reason   : str | None = None,
    ) -> None:
        self.state                  = RobotState.PARKED
        self.from_node              = u
        self.to_node                = v
        self.progress               = fraction
        self.speed                  = 0.0
        self.current_node           = None
        self.parked_at              = (u, v, fraction)
        self.parking_reserved_edge  = (u, v)
        self.parked_ticks           = 0
        self.park_reason            = reason

    def leave_parking(self) -> None:
        self.parked_at              = None
        self.parking_reserved_edge  = None
        self.parked_ticks           = 0
        self.park_reason            = None
        self.speed                  = 0.0
        self.progress               = 0.0
        self.state                  = RobotState.IDLE

    def tick_waiting(self) -> None:
        self.wait_ticks             += 1
        self.wait_ticks_in_junction += 1

    def reset_waiting(self) -> None:
        self.wait_ticks             = 0
        self.wait_ticks_in_junction = 0

    def tick_parked(self) -> None:
        self.parked_ticks += 1

    def parking_expired(self, max_ticks: int = MAX_PARKED_TICKS) -> bool:
        return self.state == RobotState.PARKED and self.parked_ticks >= max_ticks

    def clear_motion(self) -> None:
        self.from_node = None
        self.to_node   = None
        self.progress  = 0.0
        self.speed     = 0.0

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parked   = ""
        carrying = ""

        if self.parked_at is not None:
            u, v, frac = self.parked_at
            parked = f", parked_at={u}->{v}@{frac:.2f}"

        if self.carrying_box is not None:
            carrying = f", box={self.carrying_box}"

        return (
            f"Robot({self.id}, "
            f"state={self.state.name}, "
            f"node={self.current_node}, "
            f"goal={self.goal_node}"
            f"{parked}"
            f"{carrying})"
        )