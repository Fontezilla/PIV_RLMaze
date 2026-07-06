"""Entidades de dados do ambiente. Por agora só a `Box` (a caixa com a sua
pipeline). O estado do robot vive no wrapper do env, conduzido pelos
schedules do Space-Time A*, por isso não há aqui uma classe Robot pesada
como no projecto antigo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class BoxStatus(Enum):
    """Estado de uma caixa: à espera num nó, em transporte, a ser
    processada (par entry->exit, com delay), ou entregue."""
    WAITING    = auto()
    IN_TRANSIT = auto()
    PROCESSING = auto()
    DONE       = auto()


class PipelineType(Enum):
    """Tipo de pipeline: BLUE (entry→exit), GREEN (+processA),
    RED (+processA+processB)."""
    BLUE  = auto()
    GREEN = auto()
    RED   = auto()


@dataclass
class Box:
    """Caixa com pipeline dinâmica.

    A caixa nasce só com o entry fixo; o destino de cada fase seguinte é
    decidido pelo agente no momento do assignment (`next_waypoint`). Cada
    fase em `pipeline_constraints` é uma de duas formas:
      - lista de strings (ex.: exits) → o agente escolhe um nó.
      - lista de pares [entry, exit] (processos) → o agente escolhe o par
        (passa o `entry`); o drop no `entry` faz auto-advance para o `exit`.
    """

    box_id               : int
    pipeline             : PipelineType
    current_node         : str | None
    pipeline_remaining   : list[str]       = field(default_factory=list)
    pipeline_constraints : dict[str, list] = field(default_factory=dict)
    pipeline_total_steps : int             = 0
    next_waypoint        : str | None      = None
    status               : BoxStatus       = BoxStatus.WAITING
    carried_by           : str | None      = None
    process_ready_at     : float | None    = None
    process_landed       : str | None      = None
    process_entry        : str | None      = None

    @property
    def is_waiting(self) -> bool:
        """True se está pousada à espera de robot."""
        return self.status == BoxStatus.WAITING

    @property
    def is_in_transit(self) -> bool:
        """True se está a ser transportada."""
        return self.status == BoxStatus.IN_TRANSIT

    @property
    def is_done(self) -> bool:
        """True se a pipeline está completa (entregue)."""
        return self.status == BoxStatus.DONE

    @property
    def is_available(self) -> bool:
        """True se está WAITING e sem robot atribuído (assignável)."""
        return self.status == BoxStatus.WAITING and self.carried_by is None

    @property
    def steps_done(self) -> int:
        """Nº de fases da pipeline já concluídas."""
        return self.pipeline_total_steps - len(self.pipeline_remaining)

    def target_options(self) -> list[str]:
        """Nós válidos como next_waypoint para o próximo step da pipeline.

        Steps singleton devolvem as próprias opções. Steps com pares
        (entry, exit) devolvem só os entries (o exit é alcançado por
        auto-advance no drop)."""
        if not self.pipeline_remaining:
            return []
        options = self.pipeline_constraints.get(self.pipeline_remaining[0], [])
        out: list[str] = []
        for opt in options:
            if isinstance(opt, list):
                if opt:
                    out.append(opt[0])
            else:
                out.append(opt)
        return out

    def pick_up(self, robot_id: str) -> None:
        """Marca a caixa como transportada por `robot_id`."""
        self.carried_by   = robot_id
        self.status       = BoxStatus.IN_TRANSIT
        self.current_node = None

    def peek_landing(self, node: str) -> str:
        """Onde a caixa aterraria ao fazer drop em `node`, sem aplicar
        nenhuma mudança de estado — só consulta `pipeline_constraints` (a
        fase actual da pipeline). Devolve `node` se não for entry de um
        par (entry, exit)."""
        if not self.pipeline_remaining:
            return node
        options = self.pipeline_constraints.get(self.pipeline_remaining[0], [])
        for opt in options:
            if isinstance(opt, list) and opt and opt[0] == node and len(opt) > 1:
                return opt[1]
        return node

    def apply_drop(
        self, node: str, now: float, process_delay: float = 0.0, machine_free: bool = True,
    ) -> bool:
        """Aplica drop em `node`, avança a pipeline e devolve True se ficou
        DONE. Se o step actual é um par (entry, exit) e `node` é o entry,
        entra em PROCESSING (visível no entry, `current_node=node`). Se a
        máquina estiver livre (`machine_free`), arranca logo o temporizador
        (`process_ready_at`); senão fica em fila (`process_ready_at=None`)
        até `BoxManager._promote_queued_at` a arrancar, quando o exit for
        libertado. A recompensa de waypoint já é dada aqui, não se atrasa."""
        if not self.pipeline_remaining:
            self.current_node  = node
            self.status        = BoxStatus.DONE
            self.carried_by    = None
            self.next_waypoint = None
            return True

        landed = self.peek_landing(node)
        self.pipeline_remaining.pop(0)
        self.next_waypoint = None
        self.carried_by    = None

        if landed != node and process_delay > 0.0:
            self.status           = BoxStatus.PROCESSING
            self.current_node     = node
            self.process_landed   = landed
            self.process_entry    = node
            self.process_ready_at = now + process_delay if machine_free else None
            return False

        self.current_node = landed
        if not self.pipeline_remaining:
            self.status = BoxStatus.DONE
            return True
        self.status = BoxStatus.WAITING
        return False

    def finish_processing(self) -> None:
        """Termina o processamento: a caixa aparece no exit (`process_landed`),
        WAITING (ou DONE se essa era a última fase da pipeline)."""
        self.current_node     = self.process_landed
        self.process_landed   = None
        self.process_entry    = None
        self.process_ready_at = None
        self.status = BoxStatus.DONE if not self.pipeline_remaining else BoxStatus.WAITING

    def __repr__(self) -> str:
        """Resumo compacto da caixa (id, pipeline, estado, posição, destino)."""
        rem = "/".join(self.pipeline_remaining) or "DONE"
        return (f"Box(id={self.box_id}, {self.pipeline.name}, {self.status.name}, "
                f"node={self.current_node}, next={self.next_waypoint or '—'}, rem={rem})")
