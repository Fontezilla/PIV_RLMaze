"""Pretty-print das decisões do agent — para debug em modo eval."""

from __future__ import annotations

from typing import Any

from agent.actor_critic import RobotDecision


NEG_INF = float("-inf")


def explain_step(
    state          : dict[str, Any],
    decisions      : list[RobotDecision],
    action_indices : list[int],
    reward         : float | None = None,
) -> str:
    """Devolve string multi-linha com a explicação detalhada deste step.

    Mostra, por robot pendente:
      - estado (current_node, goal, carrying, assigned)
      - candidatos ordenados por score, marcando o escolhido e os mascarados
        (já-assignados a outro robot neste step, idle-mask)
    """
    tick = state.get("tick", "?")
    pending_ids = state.get("pending_robot_ids", [])
    robots_by_id = {r["id"]: r for r in state.get("robots", [])}
    boxes_by_id  = {b["box_id"]: b for b in state.get("boxes",  [])}
    n_avail = len(state.get("available_box_ids", []))

    lines: list[str] = []
    header = f"=== tick={tick}  pending={len(pending_ids)}  avail_boxes={n_avail}"
    if reward is not None:
        header += f"  step_reward={reward:+.3f}"
    header += " " + "=" * max(0, 80 - len(header))
    lines.append(header)

    if not decisions:
        lines.append("  (sem decisões — agent devolveu vazio)")
        return "\n".join(lines)

    # Replica o estado do masking sequencial à medida que percorremos as decisões
    assigned_boxes : set[int] = set()
    assigned_nodes : set[str] = set()

    for i, decision in enumerate(decisions):
        if i >= len(action_indices):
            continue

        # Recupera robot_id via decision.robot_idx (índice no tensor de robots).
        # NÃO usa pending_ids[i] por posição — pending_robot_indices pode ser
        # um subconjunto filtrado de pending_ids, quebrando a correspondência 1:1.
        robots_in_state = state.get("robots", [])
        robot_id = (
            robots_in_state[decision.robot_idx]["id"]
            if decision.robot_idx < len(robots_in_state)
            else f"<idx={decision.robot_idx}>"
        )
        r = robots_by_id.get(robot_id, {})
        action_idx = action_indices[i]
        chosen_key = (
            decision.candidates[action_idx]
            if action_idx < len(decision.candidates) else None
        )

        # Cabecalho do robot
        lines.append(
            f"> {robot_id} @ {r.get('current_node') or '<transit>'}  "
            f"goal={r.get('goal_node') or '-'}  "
            f"carry={r.get('carrying_box') if r.get('carrying_box') is not None else '-'}  "
            f"asn={r.get('assigned_box_id') if r.get('assigned_box_id') is not None else '-'}"
        )

        # ── Determina máscaras aplicadas (replica _mask_logits) ────────
        logits  = decision.logits.detach().cpu().tolist()
        masked  : list[str | None] = [None] * len(logits)

        for j, key in enumerate(decision.candidates):
            ctype = key[0]
            if ctype == "box":
                _, box_id, target_node = key
                if int(box_id) in assigned_boxes:
                    masked[j] = "box assigned this step"
                elif target_node is not None and str(target_node) in assigned_nodes:
                    masked[j] = f"target {target_node} taken this step"

        # Idle-mask: se há pelo menos 1 candidato real não-mascarado
        has_real = any(
            decision.candidates[j][0] != "idle" and masked[j] is None
            for j in range(len(decision.candidates))
        )
        if has_real:
            for j, key in enumerate(decision.candidates):
                if key[0] == "idle":
                    masked[j] = "idle-mask (ha trabalho)"

        # ── Ordena candidatos por score (descending) ───────────────────
        order = sorted(range(len(logits)), key=lambda j: -logits[j])

        for rank, j in enumerate(order):
            key = decision.candidates[j]
            score = logits[j]
            label = _format_candidate(key, boxes_by_id)

            mark = ""
            if chosen_key is not None and chosen_key == key:
                mark = " <-- CHOSEN"
            elif masked[j] is not None:
                mark = f"  [masked: {masked[j]}]"

            lines.append(f"    {score:+7.3f}  {label}{mark}")

        # ── Actualiza o masking para o próximo robot ───────────────────
        if chosen_key is not None:
            ctype = chosen_key[0]
            if ctype == "box":
                _, box_id, target_node = chosen_key
                assigned_boxes.add(int(box_id))
                if target_node is not None:
                    assigned_nodes.add(str(target_node))

    return "\n".join(lines)


def _format_candidate(
    key   : tuple,
    boxes : dict[int, dict],
) -> str:
    """Linha-resumo de um candidato (box / node / idle)."""
    ctype = key[0]
    if ctype == "box":
        _, box_id, target_node = key
        b = boxes.get(int(box_id), {})
        pipeline = (b.get("pipeline") or "?")[:5]
        loc      = b.get("current_node") or "<transit>"
        return f"[box ] b{box_id:<3} {pipeline:<5} @ {loc:<18} -> {target_node}"
    return "[idle]"
