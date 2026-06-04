"""Gestão do ciclo de vida das caixas (criar, spawn, pick, drop, auto-advance).

As caixas nascem só com pipeline + entry; o agent decide o próximo destino
(next_waypoint) a cada drop intermédio. Pares process_entry/exit fazem
auto-advance dentro do `apply_drop` da Box.
"""

from __future__ import annotations

import random
import yaml
from pathlib import Path
from typing import Optional

from env.core.entities import Box, BoxStatus, PipelineType, Robot


REWARD_WAYPOINT = 1.0
REWARD_DELIVERY = 10.0
REWARD_PICK     = 0.3


_PIPELINE_MAP: dict[str, PipelineType] = {
    "blue":  PipelineType.BLUE,
    "green": PipelineType.GREEN,
    "red":   PipelineType.RED,
}


class BoxManager:
    """Gere ciclo de vida das caixas: spawn lazy, pick/drop, target options."""

    def __init__(
        self,
        pipeline_path : str | Path,
        n_boxes       : int = 8,
        seed          : Optional[int] = None,
    ) -> None:
        self._pipeline_path = Path(pipeline_path)
        self._n_boxes       = n_boxes
        self._rng           = random.Random(seed)
        self._pipeline_cfg  = self._load_pipeline_cfg()

        self._boxes       : dict[int, Box] = {}
        self._spawn_queue : list[Box]      = []
        self._next_id     : int            = 0

    def reset(self, seed: Optional[int] = None) -> None:
        """Cria todas as caixas e activa as que cabem nas entries livres."""
        if seed is not None:
            self._rng = random.Random(seed)

        self._boxes.clear()
        self._spawn_queue.clear()
        self._next_id = 0

        all_entries: list[str] = []
        for cfg in self._pipeline_cfg.values():
            entries = [
                o for o in cfg.get("constraints", {}).get("entry", [])
                if isinstance(o, str)
            ]
            if entries:
                all_entries = entries
                break

        if all_entries and len(all_entries) >= self._n_boxes:
            entry_pool = self._rng.sample(all_entries, self._n_boxes)
        elif all_entries:
            entry_pool = [all_entries[i % len(all_entries)] for i in range(self._n_boxes)]
            self._rng.shuffle(entry_pool)
        else:
            entry_pool = [None] * self._n_boxes

        for forced_entry in entry_pool:
            box = self._create_pending_box(forced_entry=forced_entry)
            self._spawn_queue.append(box)

        self.tick_spawn()

    def tick_spawn(self) -> list[int]:
        """Activa caixas da fila quando o entry está livre. Devolve novos ids."""
        newly_placed: list[int] = []
        still_waiting: list[Box] = []

        for box in self._spawn_queue:
            entry = box.current_node
            if entry is not None and self.is_node_drop_available(entry):
                self._boxes[box.box_id] = box
                newly_placed.append(box.box_id)
            else:
                still_waiting.append(box)

        self._spawn_queue = still_waiting
        return newly_placed

    def is_node_drop_available(self, node: str) -> bool:
        """True se o nó pode receber uma caixa (regra: máx. 1 por nó)."""
        for box in self._boxes.values():
            if box.is_waiting and box.current_node == node:
                return False
            if box.is_in_transit and box.next_waypoint == node:
                return False
        return True

    def on_pick(self, robot: Robot, box_id: int) -> float:
        """Regista pickup; devolve REWARD_PICK ou 0.0 se falhou."""
        box = self._boxes.get(box_id)
        if box is None or not box.is_available:
            return 0.0
        if box.current_node != robot.current_node:
            return 0.0
        # Sem next_waypoint definido o agent ainda não escolheu destino → não pick.
        if box.next_waypoint is None:
            return 0.0

        box.pick_up(robot.id)
        robot.carrying_box = box_id
        return REWARD_PICK

    def on_drop(self, robot: Robot, node: str) -> tuple[float, bool, int | None]:
        """Tenta drop no nó; devolve (reward, delivered, box_id)."""
        if robot.carrying_box is None:
            return 0.0, False, None

        box = self._boxes.get(robot.carrying_box)
        if box is None:
            robot.carrying_box = None
            return 0.0, False, None

        if box.next_waypoint != node:
            return 0.0, False, None

        box_id    = box.box_id
        delivered = box.apply_drop(node)
        robot.carrying_box = None

        reward = REWARD_DELIVERY if delivered else REWARD_WAYPOINT
        return reward, delivered, box_id

    def get_box(self, box_id: int) -> Box | None:
        return self._boxes.get(box_id)

    def boxes(self) -> list[Box]:
        return list(self._boxes.values())

    def active_boxes(self) -> list[Box]:
        """Caixas activas ainda não entregues."""
        return [b for b in self._boxes.values() if not b.is_done]

    def available_boxes(self) -> list[Box]:
        """Caixas WAITING (assignable pelo agente).

        Inclui boxes com next_waypoint ainda por decidir — é trabalho do agent
        decidir o destino. Filtra boxes sem opções (DONE).
        """
        return [
            b for b in self._boxes.values()
            if b.is_available and b.target_options()
        ]

    def available_boxes_at(self, node: str) -> list[Box]:
        """Caixas WAITING num nó (para features do agente)."""
        return [
            b for b in self._boxes.values()
            if b.is_available and b.current_node == node
        ]

    def delivered_count(self) -> int:
        return sum(1 for b in self._boxes.values() if b.is_done)

    def queue_count(self) -> int:
        """Número de caixas ainda em fila de espera."""
        return len(self._spawn_queue)

    def all_delivered(self) -> bool:
        """True quando todas entregues E spawn_queue vazia."""
        return (
            not self._spawn_queue
            and all(b.is_done for b in self._boxes.values())
        )

    def snapshot(self) -> list[dict]:
        """Estado das caixas activas para observação/debug."""
        return [
            {
                "box_id":            b.box_id,
                "pipeline":          b.pipeline.name,
                "status":            b.status.name,
                "current_node":      b.current_node,
                "next_waypoint":     b.next_waypoint,
                "carried_by":        b.carried_by,
                "target_options":    b.target_options(),
                "steps_done":        b.steps_done,
                "steps_total":       b.pipeline_total_steps,
                "pipeline_remaining": list(b.pipeline_remaining),
            }
            for b in self._boxes.values()
        ]

    def _create_pending_box(self, forced_entry: str | None = None) -> Box:
        """Cria caixa com pipeline; só fixa o entry, restantes fases são decisão do agent."""
        pipeline_name = self._rng.choice(list(self._pipeline_cfg.keys()))
        pipeline_type = _PIPELINE_MAP[pipeline_name]
        cfg           = self._pipeline_cfg[pipeline_name]

        constraints: dict[str, list] = cfg.get("constraints", {})
        sequence    : list[str]      = list(cfg.get("sequence", []))

        # Escolhe o entry (forçado ou aleatório das opções).
        entry_opts = [o for o in constraints.get("entry", []) if isinstance(o, str)]
        entry_node = (
            forced_entry if forced_entry is not None and forced_entry in entry_opts
            else (self._rng.choice(entry_opts) if entry_opts else None)
        )

        # pipeline_remaining = todos os steps menos "entry" (que já está fixo).
        pipeline_remaining = [s for s in sequence if s != "entry"]

        box = Box(
            box_id               = self._next_id,
            pipeline             = pipeline_type,
            current_node         = entry_node,
            pipeline_remaining   = pipeline_remaining,
            pipeline_constraints = constraints,
            pipeline_total_steps = len(pipeline_remaining),
            status               = BoxStatus.WAITING,
        )
        self._next_id += 1
        return box

    def _load_pipeline_cfg(self) -> dict:
        with open(self._pipeline_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        pipelines = data.get("pipelines", {})
        return {
            name: cfg
            for name, cfg in pipelines.items()
            if name in _PIPELINE_MAP
        }
