"""Entidades do ambiente de fábrica: Robot (AGV) e Box (caixa com pipeline)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


MAX_HISTORY_LEN = 10


class RobotState(Enum):
    """Estado do robot."""
    IDLE    = auto()
    MOVING  = auto()
    WAITING = auto()
    PARKED  = auto()


class BoxStatus(Enum):
    """Estado de uma caixa."""
    WAITING    = auto()
    IN_TRANSIT = auto()
    DONE       = auto()


class PipelineType(Enum):
    """Pipeline da caixa (define os waypoints intermédios)."""
    BLUE  = auto()
    GREEN = auto()
    RED   = auto()


@dataclass
class Box:
    """Caixa com pipeline dinâmica.

    A box mantém `pipeline_remaining` (fases por fazer) e `pipeline_constraints`
    (opções de cada fase). O `next_waypoint` é decidido pelo agent no momento
    do assignment — não há waypoints fixos pré-computados.

    Cada fase nas constraints é uma de duas formas:
      - lista de strings (singleton)  → e.g. exits: agent escolhe um node.
      - lista de pares [entry, exit]  → e.g. processos: agent escolhe um par
        (passa o `entry`), e o drop em `entry` faz auto-advance para o `exit`.
    """

    box_id               : int
    pipeline             : PipelineType
    current_node         : str | None
    pipeline_remaining   : list[str]                = field(default_factory=list)
    pipeline_constraints : dict[str, list]          = field(default_factory=dict)
    pipeline_total_steps : int                      = 0
    next_waypoint        : str | None               = None
    status               : BoxStatus                = BoxStatus.WAITING
    carried_by           : str | None               = None

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
        """WAITING e sem robot atribuído (precisa decisão se next_waypoint=None)."""
        return self.status == BoxStatus.WAITING and self.carried_by is None

    @property
    def steps_done(self) -> int:
        """Quantas fases da pipeline já completou."""
        return self.pipeline_total_steps - len(self.pipeline_remaining)

    def target_options(self) -> list[str]:
        """Nodes válidos como next_waypoint para o próximo step da pipeline.

        Para steps singleton, devolve as próprias opções.
        Para steps com pares (entry, exit), devolve só os entries (o exit
        do par é alcançado via auto-advance no drop).
        """
        if not self.pipeline_remaining:
            return []
        step = self.pipeline_remaining[0]
        options = self.pipeline_constraints.get(step, [])
        out: list[str] = []
        for opt in options:
            if isinstance(opt, list):
                if opt:
                    out.append(opt[0])
            else:
                out.append(opt)
        return out

    def pick_up(self, robot_id: str) -> None:
        """Regista pickup pelo robot."""
        self.carried_by   = robot_id
        self.status       = BoxStatus.IN_TRANSIT
        self.current_node = None

    def apply_drop(self, node: str) -> bool:
        """Aplica drop em `node`; avança a pipeline. Devolve True se DONE.

        Se o step actual é um par (process_entry, process_exit) e `node` é o
        entry do par, faz auto-advance: current_node passa a ser o exit.
        Caso contrário (singleton), current_node = node.
        Em ambos os casos: pop step de pipeline_remaining, reset next_waypoint,
        status volta a WAITING (ou DONE se não há mais steps).
        """
        if not self.pipeline_remaining:
            # já estava no fim — não devia chegar aqui, mas seguro
            self.current_node = node
            self.status       = BoxStatus.DONE
            self.carried_by   = None
            self.next_waypoint = None
            return True

        step    = self.pipeline_remaining[0]
        options = self.pipeline_constraints.get(step, [])

        landed: str = node
        for opt in options:
            if isinstance(opt, list) and opt and opt[0] == node:
                # par (entry, exit) — auto-advance para o exit
                if len(opt) > 1:
                    landed = opt[1]
                break

        self.pipeline_remaining.pop(0)
        self.current_node   = landed
        self.next_waypoint  = None
        self.carried_by     = None

        if not self.pipeline_remaining:
            self.status = BoxStatus.DONE
            return True

        self.status = BoxStatus.WAITING
        return False

    def __repr__(self) -> str:
        nxt = self.next_waypoint or "—"
        rem = "/".join(self.pipeline_remaining) or "DONE"
        return (
            f"Box(id={self.box_id}, "
            f"pipeline={self.pipeline.name}, "
            f"status={self.status.name}, "
            f"node={self.current_node}, "
            f"next_wp={nxt}, "
            f"remaining={rem}, "
            f"carried_by={self.carried_by})"
        )


@dataclass
class Robot:
    """AGV autónomo no ambiente da fábrica."""

    id    : str
    state : RobotState = RobotState.IDLE

    current_node : str | None = None
    from_node    : str | None = None
    to_node      : str | None = None
    progress     : float      = 0.0

    speed        : float = 0.0
    target_speed : float = 0.0

    goal_node  : str | None = None
    came_from  : str | None = None

    wait_ticks             : int = 0
    wait_ticks_in_junction : int = 0
    turn_ticks_total       : int = 0

    world_x : float = 0.0
    world_y : float = 0.0

    last_visited : list[str]              = field(default_factory=list)
    last_edges   : list[tuple[str, str]]  = field(default_factory=list)

    parked_at    : tuple[str, str, float] | None = None

    carrying_box    : int | None = None
    assigned_box_id : int | None = None

    # True quando o agent escolheu idle e ainda não houve novidade no mundo.
    # Evita re-decisão em loop quando o agent não tem candidatos úteis.
    idle_acknowledged : bool = False

    # Prioridade dinâmica (calculada por tick no env). Robots com caixa
    # têm priority > 0; quanto mais perto do delivery final, maior.
    # Usado pelo router para decidir quem cede em conflitos.
    priority : float = 0.0

    def is_idle(self) -> bool:
        return self.state == RobotState.IDLE

    def is_waiting(self) -> bool:
        return self.state == RobotState.WAITING

    def is_parked(self) -> bool:
        return self.state == RobotState.PARKED

    def is_free(self) -> bool:
        """Livre para novo assignment: IDLE/WAITING/PARKED, sem caixa, sem reserva."""
        return (
            self.state in (RobotState.IDLE, RobotState.WAITING, RobotState.PARKED)
            and self.carrying_box is None
            and self.assigned_box_id is None
        )

    def reached_goal(self) -> bool:
        return (
            self.state == RobotState.IDLE
            and self.current_node is not None
            and self.current_node == self.goal_node
        )

    def push_visited(self, node: str, max_len: int = MAX_HISTORY_LEN) -> None:
        """Empurra nó visitado no histórico anti-loop."""
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
        """Empurra aresta percorrida no histórico anti-loop."""
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
        """Penalizações por nó com base no histórico recente."""
        penalties: dict[str, float] = {}
        counts: dict[str, int] = {}
        for n in self.last_visited:
            counts[n] = counts.get(n, 0) + 1

        for idx, node in enumerate(self.last_visited):
            if node == goal_node:
                continue
            recency = idx + 1
            penalty = recency * 450.0
            if counts[node] > 1:
                penalty += counts[node] * 900.0
            penalties[node] = max(penalties.get(node, 0.0), penalty)

        return penalties

    def recent_edge_penalties(self) -> dict[tuple[str, str], float]:
        """Penalizações por aresta com base no histórico recente."""
        penalties: dict[tuple[str, str], float] = {}
        edge_counts: dict[tuple[str, str], int] = {}
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
