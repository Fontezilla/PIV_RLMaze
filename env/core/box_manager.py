"""Gestão do ciclo de vida das caixas: criar, spawn nos entries, pick/drop
com auto-advance dos pares process_entry/exit, recompensas.

As caixas nascem só com pipeline + entry; o agente decide o próximo destino
(`next_waypoint`) a cada drop intermédio. Portado do projecto antigo — é
lógica pura de dados, independente da física/tráfego.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import yaml

from env.core.entities import Box, BoxStatus, PipelineType


REWARD_WAYPOINT = 1.0
REWARD_DELIVERY = 10.0
REWARD_PICK     = 0.3

# Retornos decrescentes por exit: a n-ésima caixa entregue NO MESMO exit
# (contando a partir de 0) vale REWARD_DELIVERY * DECAY**n. Assim a 1ª caixa
# num exit vale 10, a 2ª no mesmo exit vale 10*DECAY, etc. — incentiva o
# agente a espalhar as entregas por exits diferentes em vez de as concentrar.
# DECAY=1.0 desliga o efeito (volta a reward fixo); mais baixo = mais pressão
# para distribuir. O contador é por episódio (reset em `reset`).
REWARD_DELIVERY_DECAY = 0.6

# Tempo de "máquina" simulado num par process entry->exit — só efeito
# visual (dá tempo a uma animação no render 3D); a recompensa de waypoint
# já é dada no drop, não se atrasa. 1 tick = 0.05s (ver env/render/renderer.py).
PROCESS_DELAY_TICKS = 20.0

_PIPELINE_MAP: dict[str, PipelineType] = {
    "blue":  PipelineType.BLUE,
    "green": PipelineType.GREEN,
    "red":   PipelineType.RED,
}

_LETTER_TO_PIPELINE: dict[str, str] = {"B": "blue", "R": "red", "G": "green"}


def parse_box_layout(layout: str) -> list[tuple[int, str]]:
    """Traduz um layout de caixas por slot de entry numa lista ordenada de
    (índice_do_entry, nome_da_pipeline).

    Formato: grupos separados por espaços, um por entry começando no entryA
    (slot 0 = entryA, 1 = entryB, ...). Cada grupo é a sequência de cores
    que nasce nesse entry, por ordem. Letras: B=blue, R=red, G=green;
    '-'/'.'/'_' marcam um slot vazio.

    Ex.: "BB RG GG B" → entryA lança 2 azuis, entryB 1 vermelha depois 1
    verde, entryC 2 verdes, entryD 1 azul (7 caixas no total).
    """
    out: list[tuple[int, str]] = []
    for slot_idx, group in enumerate(layout.split()):
        for ch in group:
            if ch in "-._":
                continue
            key = _LETTER_TO_PIPELINE.get(ch.upper())
            if key is None:
                raise ValueError(
                    f"Letra de cor desconhecida em box_layout: {ch!r} (usa B/R/G)"
                )
            out.append((slot_idx, key))
    return out


class BoxManager:
    """Ciclo de vida das caixas: spawn lazy nos entries, pick/drop, opções."""

    def __init__(
        self,
        pipeline_path: str | Path,
        n_boxes: int = 8,
        seed: Optional[int] = None,
        layout: str | None = None,
    ) -> None:
        """Carrega a config de pipelines e prepara o gerador de caixas.

        Se `layout` for dado (ex. "BB RG GG B"), as caixas são criadas
        deterministicamente (entry + cor por slot, ver `parse_box_layout`) e
        `n_boxes` passa a ser o total do layout, ignorando o argumento."""
        self._pipeline_path = Path(pipeline_path)
        self._rng           = random.Random(seed)
        self._pipeline_cfg  = self._load_pipeline_cfg()

        self._layout : list[tuple[int, str]] | None = (
            parse_box_layout(layout) if layout else None
        )
        self._n_boxes = len(self._layout) if self._layout is not None else n_boxes

        self._boxes       : dict[int, Box] = {}
        self._spawn_queue : list[Box]      = []
        self._next_id     : int            = 0
        self._exit_deliveries : dict[str, int] = {}

    @property
    def n_boxes(self) -> int:
        """Número total de caixas do episódio (derivado do layout se houver)."""
        return self._n_boxes

    def reset(self, seed: Optional[int] = None, n_boxes: Optional[int] = None) -> None:
        """Cria todas as caixas e activa as que cabem nos entries livres.

        `n_boxes` (só no modo aleatório, sem layout) permite variar o número
        de caixas por episódio — ignorado se houver `layout`."""
        if seed is not None:
            self._rng = random.Random(seed)
        if n_boxes is not None and self._layout is None:
            self._n_boxes = n_boxes
        self._boxes.clear()
        self._spawn_queue.clear()
        self._exit_deliveries.clear()
        self._next_id = 0

        all_entries: list[str] = []
        for cfg in self._pipeline_cfg.values():
            entries = [o for o in cfg.get("constraints", {}).get("entry", []) if isinstance(o, str)]
            if entries:
                all_entries = entries
                break

        if self._layout is not None:
            # Layout explícito: entry por slot (0=entryA...) e cor fixa por
            # caixa, na ordem dada. Agrupados por entry → a 1ª de cada entry
            # nasce logo, as seguintes ficam em fila até o entry libertar.
            for slot_idx, pipeline_name in self._layout:
                entry = all_entries[slot_idx] if slot_idx < len(all_entries) else None
                self._spawn_queue.append(
                    self._create_pending_box(entry, forced_pipeline=pipeline_name)
                )
            self.tick_spawn()
            return

        if all_entries and len(all_entries) >= self._n_boxes:
            entry_pool = self._rng.sample(all_entries, self._n_boxes)
        elif all_entries:
            entry_pool = [all_entries[i % len(all_entries)] for i in range(self._n_boxes)]
            self._rng.shuffle(entry_pool)
        else:
            entry_pool = [None] * self._n_boxes

        for forced_entry in entry_pool:
            self._spawn_queue.append(self._create_pending_box(forced_entry))
        self.tick_spawn()

    def tick_spawn(self) -> list[int]:
        """Activa caixas da fila quando o entry está livre. Devolve novos ids."""
        newly_placed: list[int] = []
        still_waiting: list[Box] = []
        for box in self._spawn_queue:
            if box.current_node is not None and self.is_node_drop_available(box.current_node):
                self._boxes[box.box_id] = box
                newly_placed.append(box.box_id)
            else:
                still_waiting.append(box)
        self._spawn_queue = still_waiting
        return newly_placed

    def is_node_drop_available(self, node: str) -> bool:
        """True se o nó pode receber uma caixa (máx. 1 por nó). Um entry de
        máquina já ocupado (caixa em fila ou a processar) também bloqueia —
        só uma caixa de cada vez por máquina do lado do entry."""
        for box in self._boxes.values():
            if box.is_waiting and box.current_node == node:
                return False
            if box.is_in_transit and box.next_waypoint == node:
                return False
            if box.status == BoxStatus.PROCESSING and box.process_entry == node:
                return False
        return True

    def on_pick(self, robot_id: str, box_id: int, robot_node: str, now: float) -> float:
        """Regista pickup; devolve REWARD_PICK ou 0.0 se falhou. Se este nó
        era o exit de uma máquina, liberta-a — promove quem estiver em fila
        no entry correspondente a arrancar já o processamento.

        NOTA: não usa `is_available` aqui — `carried_by` já foi atribuído
        no momento do assignment (`_commit_journey`, bem antes do robot
        chegar fisicamente à caixa), não neste pickup físico. Confirma-se
        antes que é mesmo ESTE robot que a está a levantar."""
        box = self._boxes.get(box_id)
        if box is None or box.status != BoxStatus.WAITING or box.carried_by != robot_id:
            return 0.0
        if box.current_node != robot_node:
            return 0.0
        if box.next_waypoint is None:
            return 0.0
        box.pick_up(robot_id)
        self._promote_queued_at(robot_node, now)
        return REWARD_PICK

    def on_drop(self, box_id: int, node: str, now: float) -> tuple[float, bool]:
        """Tenta drop da caixa `box_id` no nó; devolve (reward, delivered).
        Se `node` é o entry de uma máquina cujo exit ainda está ocupado
        (caixa anterior por levantar), a caixa fica em fila (visível no
        entry, sem processar) até `_promote_queued_at` a arrancar."""
        box = self._boxes.get(box_id)
        if box is None or box.next_waypoint != node:
            return 0.0, False
        landed = box.peek_landing(node)
        machine_free = landed == node or not self._is_node_occupied(landed)
        delivered = box.apply_drop(node, now, process_delay=PROCESS_DELAY_TICKS,
                                    machine_free=machine_free)
        if not delivered:
            return REWARD_WAYPOINT, False
        # Retornos decrescentes: escala pela nº de caixas já entregues NESTE
        # exit (antes desta). Ver REWARD_DELIVERY_DECAY.
        n = self._exit_deliveries.get(node, 0)
        self._exit_deliveries[node] = n + 1
        return REWARD_DELIVERY * (REWARD_DELIVERY_DECAY ** n), True

    def _is_node_occupied(self, node: str) -> bool:
        """True se alguma caixa está WAITING (pousada, por levantar) nesse nó."""
        return any(b.is_waiting and b.current_node == node for b in self._boxes.values())

    def _promote_queued_at(self, freed_node: str, now: float) -> None:
        """Depois de uma caixa ser levantada de `freed_node` (o exit de uma
        máquina), arranca o processamento de quem estivesse em fila no
        entry correspondente (só pode haver uma, `is_node_drop_available`
        bloqueia um segundo drop no mesmo entry enquanto ocupado)."""
        for box in self._boxes.values():
            if (box.status == BoxStatus.PROCESSING and box.process_ready_at is None
                    and box.process_landed == freed_node):
                box.process_ready_at = now + PROCESS_DELAY_TICKS
                break

    def next_process_ready(self) -> float | None:
        """Menor `process_ready_at` entre as caixas em processamento (só as
        já a contar, não as em fila), ou None se nenhuma estiver activa."""
        times = [b.process_ready_at for b in self._boxes.values()
                 if b.status == BoxStatus.PROCESSING and b.process_ready_at is not None]
        return min(times) if times else None

    def tick_processing(self, now: float) -> None:
        """Liberta (WAITING/DONE no exit) as caixas cujo processamento
        activo já terminou até `now` (ignora as ainda em fila)."""
        for box in self._boxes.values():
            if (box.status == BoxStatus.PROCESSING and box.process_ready_at is not None
                    and now >= box.process_ready_at - 1e-9):
                box.finish_processing()

    def get_box(self, box_id: int) -> Box | None:
        """Devolve a caixa por id (ou None)."""
        return self._boxes.get(box_id)

    def boxes(self) -> list[Box]:
        """Todas as caixas activas (já spawned)."""
        return list(self._boxes.values())

    def active_boxes(self) -> list[Box]:
        """Caixas ainda não entregues."""
        return [b for b in self._boxes.values() if not b.is_done]

    def available_boxes(self) -> list[Box]:
        """Caixas WAITING assignáveis (com opções de destino por decidir)."""
        return [b for b in self._boxes.values() if b.is_available and b.target_options()]

    def delivered_count(self) -> int:
        """Número de caixas já entregues."""
        return sum(1 for b in self._boxes.values() if b.is_done)

    def queue_count(self) -> int:
        """Número de caixas ainda por spawnar (fila)."""
        return len(self._spawn_queue)

    def all_delivered(self) -> bool:
        """True se todas as caixas foram entregues e a fila está vazia."""
        return not self._spawn_queue and all(b.is_done for b in self._boxes.values())

    def snapshot(self) -> list[dict]:
        """Estado serializável de todas as caixas (para observação/render)."""
        return [
            {
                "box_id":             b.box_id,
                "pipeline":           b.pipeline.name,
                "status":             b.status.name,
                "current_node":       b.current_node,
                "next_waypoint":      b.next_waypoint,
                "carried_by":         b.carried_by,
                "target_options":     b.target_options(),
                "steps_done":         b.steps_done,
                "steps_total":        b.pipeline_total_steps,
                "pipeline_remaining": list(b.pipeline_remaining),
                "process_landed":     b.process_landed,
                "process_entry":      b.process_entry,
                "process_ready_at":   b.process_ready_at,
            }
            for b in self._boxes.values()
        ]

    def _create_pending_box(
        self, forced_entry: str | None = None, forced_pipeline: str | None = None,
    ) -> Box:
        """Cria uma caixa nova, pronta a spawnar. Pipeline forçada
        (`forced_pipeline`) ou aleatória; entry fixo (forçado) ou aleatório."""
        pipeline_name = (
            forced_pipeline if forced_pipeline is not None
            else self._rng.choice(list(self._pipeline_cfg.keys()))
        )
        cfg           = self._pipeline_cfg[pipeline_name]
        constraints   = cfg.get("constraints", {})
        sequence      = list(cfg.get("sequence", []))

        entry_opts = [o for o in constraints.get("entry", []) if isinstance(o, str)]
        entry_node = (
            forced_entry if forced_entry is not None and forced_entry in entry_opts
            else (self._rng.choice(entry_opts) if entry_opts else None)
        )
        pipeline_remaining = [s for s in sequence if s != "entry"]

        box = Box(
            box_id               = self._next_id,
            pipeline             = _PIPELINE_MAP[pipeline_name],
            current_node         = entry_node,
            pipeline_remaining   = pipeline_remaining,
            pipeline_constraints = constraints,
            pipeline_total_steps = len(pipeline_remaining),
            status               = BoxStatus.WAITING,
        )
        self._next_id += 1
        return box

    def _load_pipeline_cfg(self) -> dict:
        """Lê do YAML as pipelines conhecidas (blue/green/red)."""
        with open(self._pipeline_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return {
            name: cfg for name, cfg in data.get("pipelines", {}).items()
            if name in _PIPELINE_MAP
        }
