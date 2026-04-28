"""
action_builder.py — Filtragem de ações válidas para o MotionAgent.

Ação: (next_node_or_None, speed_cmd)
  - IDLE   → (neighbor, "acc")  ou  (None, "hold")
  - MOVING → (None, "dec") | (None, "hold") | (None, "acc")

Filtros aplicados em robots IDLE (cada filtro só é aplicado se deixar
pelo menos uma opção; caso contrário o filtro é ignorado):

  1. Anti-tráfego-oposto   — exclui vizinhos com tráfego oposto confirmado
  2. Anti-congestionamento — prefere vizinhos não-especiais sem robots IDLE
                             (evita pile-up em junctions; nós especiais
                              são sempre incluídos como destinos válidos)
                             BYPASSADO quando idle_stuck_ticks > STARVATION_THRESHOLD
                             ou quando o robot está num leaf node.
  3. Docking edges         — evita entrar numa docking edge já ocupada
  4. Leaf node ocupado     — cede passagem se o leaf vizinho tem robot IDLE;
                             BYPASSADO por starvation para evitar livelock.

Nota: o anti-U-turn foi removido intencionalmente. O robot deve poder
recuar para prev_node como estratégia de desvio; o compass reward
(negativo ao afastar do goal) desincentiva U-turns desnecessários.
"""

from typing import List, Optional, Tuple

from simulation_engine.interface.observation_builder import MotionObs
from simulation_engine.core.entities import RobotState

# Após este nº de ticks parado sem conseguir despachar, os filtros 2 e 4
# são ignorados para que o robot possa sair de um deadlock total.
STARVATION_THRESHOLD = 40

Action = Tuple[Optional[str], str]


def get_valid_actions(obs: MotionObs) -> List[Action]:
    """
    Devolve lista de ações válidas para o estado atual do robot.

    Para robots MOVING:
        apenas comandos de velocidade.

    Para robots IDLE:
        vizinhos filtrados pelas regras acima + hold.

    Nota:
        Este builder deve manter-se alinhado com o DispatchSystem.
        Só deve mascarar ações que o dispatch também rejeitaria.
    """

    # ------------------------------------------------------------------
    # MOVING — apenas controlar velocidade
    # ------------------------------------------------------------------
    if obs.state == RobotState.MOVING:
        return [(None, "dec"), (None, "hold"), (None, "acc")]

    # ------------------------------------------------------------------
    # IDLE — sem capacidade de dispatch
    # ------------------------------------------------------------------
    if not obs.can_dispatch():
        return [(None, "hold")]

    candidates = list(obs.neighbors)

    # Sem vizinhos → só hold
    if not candidates:
        return [(None, "hold")]

    # ------------------------------------------------------------------
    # Regra 1 — sem tráfego oposto
    # ------------------------------------------------------------------
    no_opposing = [n for n in candidates if not obs.has_opposing(n)]
    if no_opposing:
        candidates = no_opposing

    # ------------------------------------------------------------------
    # Regra 2 — anti-congestionamento
    # Prefere nós vizinhos sem robots IDLE (junctions vazios).
    # Nós especiais (entry/exit/process) são sempre permitidos — podem
    # ser o destino final e apenas cabem 1 robot de cada vez (docking).
    # Bypassed se o robot está há demasiado tempo parado (starvation).
    # Também bypassed se o robot está num leaf node (só 1 vizinho) — não
    # tem alternativa, tem de sair por aí independentemente de congestionamento.
    # ------------------------------------------------------------------
    at_leaf = len(obs.neighbors) <= 1
    if not at_leaf and obs.idle_stuck_ticks <= STARVATION_THRESHOLD:
        not_congested = [
            n for n in candidates
            if obs.neighbor_special.get(n, False)   # destino especial: sempre ok
            or obs.neighbor_idle.get(n, 0) == 0     # junction/corredor: só se livre
        ]
        if not_congested:
            candidates = not_congested

    # Fallback defensivo
    if not candidates:
        candidates = list(obs.neighbors)

    # ------------------------------------------------------------------
    # Regra 3 — evitar docking edges ocupadas
    # ------------------------------------------------------------------
    no_docking_block = [
        n for n in candidates
        if not (
            obs.neighbor_special.get(n, False) and
            (obs.edge_robots.get(n, 0) > 0 or obs.edge_opposing.get(n, False))
        )
    ]

    if no_docking_block:
        candidates = no_docking_block

    # ------------------------------------------------------------------
    # Regra 4 — leaf node ocupado: cede passagem
    # Um leaf node tem apenas uma aresta. Entrar enquanto ocupado arrisca
    # bloquear a saída do ocupante. Espera que ele saia primeiro.
    # Bypass por starvation: se o robot está há demasiado tempo parado
    # (o ocupante devia ter saído entretanto via Rule 2 at_leaf skip),
    # força a entrada para evitar livelock.
    # ------------------------------------------------------------------
    if obs.idle_stuck_ticks <= STARVATION_THRESHOLD:
        no_occupied_leaf = [
            n for n in candidates
            if not (obs.neighbor_is_leaf.get(n, False) and obs.neighbor_idle.get(n, 0) > 0)
        ]
        if no_occupied_leaf:
            candidates = no_occupied_leaf

    return [(n, "acc") for n in candidates] + [(None, "hold")]