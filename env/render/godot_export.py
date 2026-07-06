"""Exporta um episódio gravado (`record=True`) para JSON, consumível por um
script de replay em Godot — alternativa ao `renderer.py` (pygame) para
visualização 3D, sem o Godot precisar de correr nada de Python.

Desenho: o Python nunca manda coordenadas — só nomes de nós (strings). O
lado Godot resolve cada nome (ex. "N", "processA1_entry") para uma posição
3D via um `Marker3D` com esse nome, colocado no mapa já construído. Isto
desacopla completamente as duas partes: o Python não sabe nada do espaço
3D, e o Godot só lê este ficheiro.

Estrutura exportada:
  robots: robot_id -> lista de paragens (nó, chegada, partida, espera,
          velocidade) — a "espera" (`wait`) é a janela onde uma animação
          de rotação/idle pode encaixar sem atropelar o movimento seguinte.
  box_log: log bruto (t, snapshot de todas as caixas) — para posição/
           estado de qualquer caixa em qualquer instante.
  events: pickup/dropoff/processed já derivados (comparando snapshots
          consecutivos) — poupa o Godot de ter de inferir transições.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from env.physics.engine import SchedulePlayer


def _export_robot_schedule(player: SchedulePlayer) -> list[dict]:
    """Serializa o horário de um robot: uma entrada por nó, com a janela de
    espera (`wait` = depart - arrival) já calculada."""
    return [
        {
            "node": e.node,
            "arrival": e.arrival,
            "depart": e.depart,
            "wait": e.depart - e.arrival,
            "speed": e.speed,
        }
        for e in player.schedule
    ]


def _derive_events(box_log: list[tuple[float, list[dict]]]) -> list[dict]:
    """Deriva eventos discretos (pickup/dropoff/queued/processed) comparando
    cada caixa com o seu snapshot anterior. Uma máquina ocupada (exit ainda
    por levantar) faz a caixa nova ficar `queued` no entry — visível, sem
    processar — até o exit libertar; só aí `process_start` arranca o
    temporizador (útil para distinguir "à espera" de "máquina a trabalhar"
    no render 3D)."""
    events: list[dict] = []
    prev: dict[int, dict] = {}

    for t, snapshot in box_log:
        for box in snapshot:
            bid = box["box_id"]
            before = prev.get(bid)
            if before is not None:
                picked_up = before["carried_by"] is None and box["carried_by"] is not None
                dropped   = before["carried_by"] is not None and box["carried_by"] is None
                finished_processing = (
                    before["status"] == "PROCESSING" and box["status"] != "PROCESSING"
                )
                started_processing = (
                    before["status"] == "PROCESSING" and box["status"] == "PROCESSING"
                    and before.get("process_ready_at") is None
                    and box.get("process_ready_at") is not None
                )

                if picked_up:
                    events.append({
                        "t": t, "type": "pickup",
                        "robot_id": box["carried_by"], "box_id": bid,
                        "node": before.get("current_node"),
                    })
                if dropped:
                    node = (
                        box.get("process_entry")
                        if box["status"] == "PROCESSING"
                        else box.get("current_node")
                    )
                    events.append({
                        "t": t, "type": "dropoff",
                        "robot_id": before["carried_by"], "box_id": bid,
                        "node": node,
                        "queued": box["status"] == "PROCESSING" and box.get("process_ready_at") is None,
                    })
                if started_processing:
                    events.append({
                        "t": t, "type": "process_start",
                        "box_id": bid,
                        "node": box.get("process_entry"),
                    })
                if finished_processing:
                    events.append({
                        "t": t, "type": "processed",
                        "box_id": bid,
                        "from_node": before.get("process_entry"),
                        "to_node": box.get("current_node"),
                    })
            prev[bid] = box

    return sorted(events, key=lambda e: e["t"])


def export_episode(env: Any, path: str | Path) -> None:
    """Exporta o episódio gravado em `env` (precisa `record=True`) para
    `path` em JSON."""
    if not env.record:
        raise ValueError("env tem de ter record=True para exportar (recria com record=True).")

    data = {
        "duration": env.clock,
        "robots": {
            robot_id: _export_robot_schedule(player)
            for robot_id, player in env.render_players.items()
        },
        "box_log": [[t, snapshot] for t, snapshot in env.box_log],
        "events": _derive_events(env.box_log),
    }

    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
