"""
env/core/box_manager.py
~~~~~~~~~~~~~~~~~~~~~~~
Gestão do ciclo de vida das caixas no ambiente de fábrica.

Responsabilidades
-----------------
  - Criar caixas no reset() com pipelines e waypoints aleatórios
    gerados a partir do box_pipeline.yaml
  - on_pick(robot, box_id)   — robot apanha caixa (chamado pelo env após
                               decisão do agente RL)
  - on_drop(robot, node)     — drop automático quando robot chega ao
                               next_waypoint da caixa que transporta;
                               devolve (reward, delivered, box_id)
  - available_boxes()        — caixas disponíveis para assignment
                               (filtra por capacidade do nó destino)
  - tick_spawn()             — tenta colocar caixas da fila de espera
                               quando o nó de entrada fica livre
  - is_node_drop_available() — verifica capacidade (máx. 1 caixa por nó)

Regra de capacidade
-------------------
  Cada nó especial (entry, exit, processA_entry, processA_exit,
  processB_entry, processB_exit) suporta no máximo 1 caixa.
  Uma caixa ocupa um nó enquanto estiver WAITING.
  Um nó também está "reservado" quando uma caixa IN_TRANSIT se dirige
  para ele (next_waypoint) — evita que dois robots levem duas caixas
  para o mesmo destino.

Spawn lazy
----------
  As caixas são criadas no reset() mas não colocadas imediatamente.
  São mantidas numa fila (_spawn_queue). Em cada tick, tick_spawn()
  tenta colocar as caixas da fila nos nós de entrada livres.
  Isto garante que nunca ficam 2 caixas no mesmo nó entry ao início.
"""

from __future__ import annotations

import random
import yaml
from pathlib import Path
from typing import Optional

from env.core.entities import Box, BoxStatus, PipelineType, Robot


# ---------------------------------------------------------------------------
# Rewards por evento de caixa
# ---------------------------------------------------------------------------

REWARD_WAYPOINT = 1.0     # waypoint intermédio concluído
REWARD_DELIVERY = 10.0    # entrega completa (exit node)
REWARD_PICK     = 0.3     # pick bem-sucedido


# ---------------------------------------------------------------------------
# Mapeamento nome → enum
# ---------------------------------------------------------------------------

_PIPELINE_MAP: dict[str, PipelineType] = {
    "blue":  PipelineType.BLUE,
    "green": PipelineType.GREEN,
    "red":   PipelineType.RED,
}


# ---------------------------------------------------------------------------
# BoxManager
# ---------------------------------------------------------------------------

class BoxManager:
    """
    Gere o ciclo de vida das caixas.

    Não toma decisões de alocação — essa responsabilidade é do agente RL.
    Expõe o estado das caixas e processa eventos de pick/drop.

    Parâmetros
    ----------
    pipeline_path : caminho para box_pipeline.yaml
    n_boxes       : número total de caixas no episódio
    seed          : seed para reprodutibilidade (opcional)
    """

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

        # Caixas activas (já colocadas no ambiente)
        self._boxes   : dict[int, Box]  = {}
        # Caixas em fila de espera (ainda não colocadas)
        self._spawn_queue : list[Box]   = []
        self._next_id : int = 0

    # ------------------------------------------------------------------
    # Ciclo de vida do episódio
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Cria todas as caixas e tenta colocar as que cabem nos nós de entrada.

        As restantes ficam em fila de espera (_spawn_queue) e são
        colocadas à medida que os nós ficam livres via tick_spawn().
        """
        if seed is not None:
            self._rng = random.Random(seed)

        self._boxes.clear()
        self._spawn_queue.clear()
        self._next_id = 0

        # Cria todas as caixas com pipelines e waypoints já definidos
        for _ in range(self._n_boxes):
            box = self._create_pending_box()
            self._spawn_queue.append(box)

        # Coloca as que couberem imediatamente
        self.tick_spawn()

    def tick_spawn(self) -> list[int]:
        """
        Tenta colocar caixas da fila de espera nos nós de entrada livres.

        Uma caixa só é activada se o seu nó de entrada (waypoints[0])
        não tiver outra caixa WAITING nem uma caixa IN_TRANSIT a caminho.

        Devolve lista de box_ids recém-colocados (pode ser vazia).
        """
        newly_placed: list[int] = []
        still_waiting: list[Box] = []

        for box in self._spawn_queue:
            entry = box.waypoints[0]
            if self.is_node_drop_available(entry):
                self._boxes[box.box_id] = box
                newly_placed.append(box.box_id)
            else:
                still_waiting.append(box)

        self._spawn_queue = still_waiting
        return newly_placed

    # ------------------------------------------------------------------
    # Capacidade de nós
    # ------------------------------------------------------------------

    def is_node_drop_available(self, node: str) -> bool:
        """
        True se o nó pode receber uma nova caixa (regra: máx. 1 por nó).

        Um nó está ocupado se:
          - já tem uma caixa WAITING, OU
          - uma caixa IN_TRANSIT tem esse nó como next_waypoint
            (reservado por um robot que já está a caminho).
        """
        for box in self._boxes.values():
            if box.is_waiting and box.current_node == node:
                return False
            if box.is_in_transit and box.next_waypoint == node:
                return False
        return True

    # ------------------------------------------------------------------
    # Eventos de pick / drop  (chamados pelo FactoryEnv)
    # ------------------------------------------------------------------

    def on_pick(self, robot: Robot, box_id: int) -> float:
        """
        Regista que o robot apanhou a caixa.

        Devolve reward de pick, ou 0.0 se o pick falhou.
        """
        box = self._boxes.get(box_id)
        if box is None:
            return 0.0
        if not box.is_available:
            return 0.0
        if box.current_node != robot.current_node:
            return 0.0

        box.pick_up(robot.id)
        robot.carrying_box = box_id
        return REWARD_PICK

    def on_drop(self, robot: Robot, node: str) -> tuple[float, bool, int | None]:
        """
        Tenta fazer drop da caixa que o robot transporta.

        Só faz drop se o nó for o next_waypoint da caixa.

        Devolve (reward, delivered, box_id).
          reward    : reward do evento (waypoint ou entrega)
          delivered : True se foi entrega final
          box_id    : id da caixa envolvida, ou None se não houve drop
        """
        if robot.carrying_box is None:
            return 0.0, False, None

        box = self._boxes.get(robot.carrying_box)
        if box is None:
            robot.carrying_box = None
            return 0.0, False, None

        if box.next_waypoint != node:
            return 0.0, False, None

        box_id    = box.box_id
        delivered = box.advance_waypoint(node)
        robot.carrying_box = None

        reward = REWARD_DELIVERY if delivered else REWARD_WAYPOINT
        return reward, delivered, box_id

    # ------------------------------------------------------------------
    # Consultas de estado  (usadas pelo agente RL e pelo FactoryEnv)
    # ------------------------------------------------------------------

    def get_box(self, box_id: int) -> Box | None:
        """Devolve a caixa activa pelo id, ou None."""
        return self._boxes.get(box_id)

    def boxes(self) -> list[Box]:
        """Todas as caixas activas (qualquer estado)."""
        return list(self._boxes.values())

    def active_boxes(self) -> list[Box]:
        """Caixas activas ainda não entregues."""
        return [b for b in self._boxes.values() if not b.is_done]

    def available_boxes(self) -> list[Box]:
        """
        Caixas disponíveis para assignment pelo agente RL.

        Uma caixa está disponível se:
          - status == WAITING e não está a ser transportada
          - tem um next_waypoint definido
          - o next_waypoint tem capacidade livre (nenhuma outra caixa lá
            ou a caminho) — regra de 1 caixa por nó
        """
        return [
            b for b in self._boxes.values()
            if b.is_available
            and b.next_waypoint is not None
            and self.is_node_drop_available(b.next_waypoint)
        ]

    def available_boxes_at(self, node: str) -> list[Box]:
        """
        Caixas WAITING num nó específico.

        Usado apenas para features do GNN (n_boxes_waiting),
        sem verificar capacidade do destino.
        """
        return [
            b for b in self._boxes.values()
            if b.is_available and b.current_node == node
        ]

    def box_carried_by(self, robot_id: str) -> Box | None:
        """Devolve a caixa transportada por um robot, ou None."""
        for box in self._boxes.values():
            if box.carried_by == robot_id:
                return box
        return None

    def delivered_count(self) -> int:
        return sum(1 for b in self._boxes.values() if b.is_done)

    def queue_count(self) -> int:
        """Número de caixas ainda em fila de espera (não colocadas)."""
        return len(self._spawn_queue)

    def all_delivered(self) -> bool:
        """
        True quando todas as caixas foram entregues E a fila de espera
        está vazia (não há mais caixas a spawnar).
        """
        return (
            not self._spawn_queue
            and all(b.is_done for b in self._boxes.values())
        )

    def snapshot(self) -> list[dict]:
        """Estado completo das caixas activas — para observação e debug."""
        return [
            {
                "box_id":        b.box_id,
                "pipeline":      b.pipeline.name,
                "status":        b.status.name,
                "current_node":  b.current_node,
                "next_waypoint": b.next_waypoint,
                "carried_by":    b.carried_by,
                "waypoint_idx":  b.waypoint_idx,
                "n_waypoints":   len(b.waypoints),
                "waypoints":     b.waypoints,
            }
            for b in self._boxes.values()
        ]

    # ------------------------------------------------------------------
    # Internos — criação de caixas
    # ------------------------------------------------------------------

    def _create_pending_box(self) -> Box:
        """
        Cria uma caixa com pipeline e waypoints já determinados,
        mas sem a colocar no ambiente (fica em fila de espera).
        """
        pipeline_name = self._rng.choice(list(self._pipeline_cfg.keys()))
        pipeline_type = _PIPELINE_MAP[pipeline_name]
        cfg           = self._pipeline_cfg[pipeline_name]
        waypoints     = self._build_waypoints(cfg)

        box = Box(
            box_id       = self._next_id,
            pipeline     = pipeline_type,
            waypoints    = waypoints,
            current_node = waypoints[0],
            status       = BoxStatus.WAITING,
        )
        self._next_id += 1
        return box

    def _build_waypoints(self, cfg: dict) -> list[str]:
        """
        Constrói a lista de waypoints para uma pipeline.

        cfg["sequence"]    → ordem dos passos (ex: ["entry","processA","exit"])
        cfg["constraints"] → passo → lista de opções

        Cada opção pode ser:
          - str  → nó único  (ex: "entryA")
          - list → par de nós para uma estação de processo
                   (ex: ["processA1_entry","processA1_exit"])
                   Os dois nós são adicionados como waypoints consecutivos.
        """
        waypoints: list[str] = []
        constraints: dict[str, list] = cfg.get("constraints", {})

        for step in cfg.get("sequence", []):
            options = constraints.get(step, [])
            if not options:
                continue
            choice = self._rng.choice(options)
            if isinstance(choice, list):
                # Par (entry + exit) de uma estação de processo
                waypoints.extend(choice)
            else:
                waypoints.append(choice)

        return waypoints

    # ------------------------------------------------------------------
    # Internos — carregamento do yaml
    # ------------------------------------------------------------------

    def _load_pipeline_cfg(self) -> dict:
        with open(self._pipeline_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        pipelines = data.get("pipelines", {})
        return {
            name: cfg
            for name, cfg in pipelines.items()
            if name in _PIPELINE_MAP
        }
