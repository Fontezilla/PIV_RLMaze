"""
state_encoder.py — Codificação de estado para Q_route e Q_velocity.

Q_route    → (cur_idx, dst_idx, prev_idx, opp_ahead, occ_ahead,
              min_adjacent_bin, has_box, n_alt_bin, dist_to_goal_bin)

Q_velocity → (progress_bin, lead_gap_bin, speed_bin, converging_bin,
              approaching_dest, dist_ahead_bin, collided_bit, dest_occupied_bit)

─── Route state ──────────────────────────────────────────────────────────────
cur_idx / dst_idx / prev_idx
    Identidade da posição, goal e histório (anti-U-turn).

opp_ahead (0/1)
    O vizinho greedy tem tráfego oposto? Sinal para desvio de rota.

occ_ahead (0/1)
    O vizinho greedy tem robots IDLE? Congestionamento na direção pretendida.

min_adjacent_bin (0-3)
    Proximidade do robot mais avançado em arestas adjacentes.
    0=nenhum, 1=<5u, 2=<15u, 3=≥15u.

has_box (0/1)
    O robot transporta uma caixa? Carga urgente pode justificar prioridade.

n_alt_bin (0-2)
    Rotas livres disponíveis (sem tráfego oposto, excluindo U-turn).
    0=nenhuma, 1=uma, 2=duas+.
    Com opp_ahead=1 e n_alt_bin≥1 → aprender a ceder.
    Com opp_ahead=1 e n_alt_bin=0 → negociar/esperar.

dist_to_goal_bin (0-4)
    Distância Dijkstra ao goal em unidades mundo, binned.
    0=no goal, 1=muito perto, 2=perto, 3=médio, 4=longe.
    Sinal de urgência: robot perto do goal deve ter prioridade;
    robot longe pode fazer detour sem grande custo.

─── Velocity state ───────────────────────────────────────────────────────────
progress_bin (0-9)
    Posição na aresta atual [0,1] → 10 bins.

lead_gap_bin (0-9)
    Gap normalizado [0,1] ao robot mais próximo à frente na mesma aresta.
    Baixo → risco de colisão traseira → travar.

speed_bin (0-9)
    Velocidade atual normalizada.

converging_bin (0-3)
    Robots a dirigir-se para o mesmo to_node. Congestionamento no destino.

approaching_dest (0/1)
    1 se estiver a ≤ APPROACH_DEST_DISTANCE unidades do nó goal.

dist_ahead_bin (0-3)
    Distância absoluta (mundo) ao robot mais próximo à frente na mesma aresta.
    Complemento ao lead_gap_bin (que é normalizado pelo progress).
    0=nenhum, 1=<5u, 2=<15u, 3=≥15u.

collided_bit (0/1)
    1 se houve sobreposição física no tick anterior.
    Permite ao agente reagir ao estado de colisão ativa.

dest_occupied_bit (0/1)
    1 se o nó destino (to_node) já tem um robot IDLE.
    Sinal para travar antes de chegar — evita acumulação em nós lotados.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from simulation_engine.core.graph import FactoryGraph
from simulation_engine.interface.observation_builder import MotionObs

# Sentinels
_NO_PREV      = -1
_UNKNOWN_NODE = -2

# Bins para quantidades contínuas
_PROGRESS_BINS = 10
_GAP_BINS      = 10
_SPEED_BINS    = 10
_CONVERGE_BINS = 4    # 0, 1, 2, 3+

APPROACH_DEST_DISTANCE = 30.0

# Thresholds para dist_to_goal_bin (unidades mundo)
# Ajustar conforme o tamanho do mapa
_GOAL_DIST_THRESHOLDS = (0.0, 80.0, 220.0, 450.0)   # 4 limites → 5 bins (0–4)


def _node_index(graph: FactoryGraph) -> Dict[str, int]:
    if not hasattr(graph, "_node_idx_cache"):
        graph._node_idx_cache = {
            n: i for i, n in enumerate(sorted(graph.graph.nodes()))
        }
    return graph._node_idx_cache


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _dist_to_goal_bin(dist: float) -> int:
    """
    Converte distância Dijkstra em bin de urgência.
    0 = no goal / muito perto  …  4 = longe.
    """
    for i, threshold in enumerate(_GOAL_DIST_THRESHOLDS):
        if dist <= threshold:
            return i
    return len(_GOAL_DIST_THRESHOLDS)   # bin mais alto


def _greedy_neighbor(obs: MotionObs, goal: str, graph: FactoryGraph) -> Optional[str]:
    """Vizinho mais próximo do goal por Dijkstra, excluindo U-turn."""
    candidates = [
        n for n in obs.neighbors
        if not (obs.prev_node is not None and n == obs.prev_node and len(obs.neighbors) > 1)
    ]
    if not candidates:
        candidates = list(obs.neighbors)
    if not candidates:
        return None

    best: Optional[str] = None
    best_dist = float("inf")
    for nb in candidates:
        try:
            d = graph.shortest_path_length(nb, goal)
            if d < best_dist:
                best_dist = d
                best = nb
        except Exception:
            pass
    return best


def _count_free_alternatives(obs: MotionObs) -> int:
    """
    Vizinhos sem tráfego oposto, excluindo U-turn.
    Indica ao Q_route se tem onde ceder quando o caminho greedy está bloqueado.
    """
    candidates = list(obs.neighbors)
    if not candidates:
        return 0
    if obs.prev_node is not None and len(candidates) > 1:
        candidates = [n for n in candidates if n != obs.prev_node]
    return sum(1 for n in candidates if not obs.has_opposing(n))


# ---------------------------------------------------------------------------
# ROUTE STATE
# ---------------------------------------------------------------------------

def encode_route_state(
    obs: MotionObs,
    goal: str,
    graph: FactoryGraph,
) -> Tuple[int, int, int, int, int, int, int, int, int]:
    """
    Estado de routing — chamado quando IDLE a decidir o próximo nó.

    Returns
    -------
    (cur_idx, dst_idx, prev_idx,
     opp_ahead, occ_ahead,
     min_adjacent_bin, has_box, n_alt_bin,
     dist_to_goal_bin)
    """
    idx = _node_index(graph)

    cur_idx  = idx.get(obs.current_node, _UNKNOWN_NODE)
    dst_idx  = idx.get(goal, _UNKNOWN_NODE)
    prev_idx = idx.get(obs.prev_node, _NO_PREV) if obs.prev_node is not None else _NO_PREV

    best_nb   = _greedy_neighbor(obs, goal, graph)
    opp_ahead = int(best_nb is not None and obs.has_opposing(best_nb))
    occ_ahead = int(best_nb is not None and obs.neighbor_idle.get(best_nb, 0) > 0)

    has_box          = 1 if obs.carried_box is not None else 0
    n_alt_bin        = min(2, _count_free_alternatives(obs))
    min_adjacent_bin = obs.min_adjacent_bin

    try:
        raw_dist = graph.shortest_path_length(obs.current_node, goal) if obs.current_node else 0.0
    except Exception:
        raw_dist = 0.0
    dist_to_goal_bin = _dist_to_goal_bin(raw_dist)

    return (cur_idx, dst_idx, prev_idx,
            opp_ahead, occ_ahead,
            min_adjacent_bin, has_box, n_alt_bin,
            dist_to_goal_bin)


# ---------------------------------------------------------------------------
# VELOCITY STATE
# ---------------------------------------------------------------------------

def encode_velocity_state(
    obs: MotionObs,
    goal: str,
    vel_max: float,
    graph: FactoryGraph,
) -> Tuple[int, int, int, int, int, int, int, int]:
    """
    Estado de velocidade — chamado a cada tick enquanto MOVING.

    Returns
    -------
    (progress_bin, lead_gap_bin, speed_bin, converging_bin,
     approaching_dest, dist_ahead_bin, collided_bit, dest_occupied_bit)
    """
    progress = _clip01(obs.progress)
    lead_gap = _clip01(obs.lead_gap)

    progress_bin   = min(_PROGRESS_BINS - 1, int(progress * _PROGRESS_BINS))
    lead_gap_bin   = min(_GAP_BINS      - 1, int(lead_gap * _GAP_BINS))
    speed_norm     = _clip01(abs(obs.speed) / vel_max) if vel_max > 0 else 0.0
    speed_bin      = min(_SPEED_BINS    - 1, int(speed_norm * _SPEED_BINS))
    converging_bin = min(_CONVERGE_BINS - 1, max(0, int(obs.converging_robots)))

    approaching_dest = 0
    if obs.to_node == goal and obs.from_node is not None:
        try:
            edge_len = graph.distance(obs.from_node, obs.to_node)
            remaining = (1.0 - progress) * edge_len
            approaching_dest = int(remaining <= APPROACH_DEST_DISTANCE)
        except Exception:
            approaching_dest = int(progress > 0.70)

    dist_ahead_bin    = obs.dist_ahead_bin        # 0=nenhum, 1=<5u, 2=<15u, 3=≥15u
    collided_bit      = int(obs.collided)          # sobreposição física no tick anterior
    dest_occupied_bit = int(obs.to_node_occupied)  # nó destino já tem robot IDLE

    return (progress_bin, lead_gap_bin, speed_bin, converging_bin,
            approaching_dest, dist_ahead_bin, collided_bit, dest_occupied_bit)
