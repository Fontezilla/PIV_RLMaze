"""
obs_encoder.py — Vetores de observação contínuos para os DQNs de routing e velocidade.

─── Route observation  (ROUTE_OBS_DIM = 117) ─────────────────────────────────

Self (7):
  progress               — progresso na aresta atual [0,1]
  speed_norm             — velocidade normalizada [0,1]
  dist_to_goal_norm      — distância Dijkstra ao goal, normalizada
  has_box                — 0/1, transporta caixa
  idle_stuck_norm        — ticks sem mover, normalizado (indica robot preso)
  robots_targeting_ref   — robots a ir para ref_node, normalizado
                           (alto = recuar para cá é arriscado)
  lead_has_box           — 0/1, robot à frente na mesma aresta tem caixa

Por vizinho — agregado (6 slots × 8 = 48):
  valid             — 0/1, este slot tem um vizinho real
  is_greedy         — 0/1, este vizinho é o próximo hop greedy para o goal
  n_robots_norm     — robots na aresta ref→vizinho, normalizado
  has_opposing      — 0/1, tráfego oposto nessa aresta
  n_idle_norm       — robots IDLE no vizinho, normalizado
  traffic_cost_norm — custo do melhor k-path via este vizinho, pesado por tráfego
  is_special        — 0/1, nó especial (entry/exit/process)
  incoming_norm     — robots a aproximar-se de nb por outras direções (2-hop)

Por vizinho — robot individual (6 slots × 10 = 60):
  fwd_valid         — 0/1, existe robot em frente na aresta ref→nb
  fwd_progress      — progress do robot fwd mais próximo de ref [0,1]
  fwd_speed_norm    — velocidade do robot fwd [0,1]
  opp_valid         — 0/1, existe robot oposto na aresta nb→ref
  opp_progress      — progress do robot oposto mais próximo de ref [0,1]
                       (alto = quase a chegar a ref → perigo imediato)
  opp_speed_norm    — velocidade do robot oposto [0,1]
  opp_has_box       — 0/1, robot oposto transporta caixa (prioridade máxima)
  opp_next_conflicts— 0/1, buffered_next do oposto == nb (conflito persiste após cruzamento)
  inc_valid         — 0/1, existe robot a aproximar-se de nb por 2ª direção
  inc_progress      — progress do robot incoming mais próximo de nb [0,1]
                       (alto = quase a chegar a nb → nb vai ficar congestionado)

Contexto (2):
  collided          — 0/1, sobreposição física no tick anterior
  converging_norm   — robots a dirigir-se ao mesmo to_node, normalizado

─── Velocity observation  (VEL_OBS_DIM = 16) ────────────────────────────────

  progress          — posição na aresta [0,1]
  speed_norm        — velocidade normalizada [0,1]
  lead_gap          — gap normalizado ao robot mais próximo à frente [0,1]
  converging_norm   — robots a convergir para o mesmo nó destino
  approaching_dest  — 0/1, distância restante ao próximo nó ≤ 40 unidades
  dist_ahead_norm   — distância ao robot mais próximo à frente (bins/3)
  collided          — 0/1, colisão no tick anterior
  to_node_occupied  — 0/1, nó destino tem robot IDLE
  docking_exit      — 0/1, em manobra de docking reverso

  lead_progress     — progress exato do robot à frente (1.0 = nenhum)
  lead_speed_norm   — velocidade do robot à frente [0,1]
  rear_progress     — progress do robot atrás (0.0 = nenhum)
  rear_speed_norm   — velocidade do robot atrás [0,1]

  is_final_edge     — 0/1, to_node == goal (estamos na última aresta para o goal)
  dist_to_goal_norm — distância Dijkstra ao goal normalizada [0,1]
                      permite ao DQN aprender "perto do goal → comportamento diferente"
  lead_has_box      — 0/1, robot à frente na mesma aresta tem caixa (dar espaço)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from simulation_engine.core.graph import FactoryGraph
from simulation_engine.interface.observation_builder import MotionObs

# Número máximo de vizinhos possíveis no mapa da fábrica
MAX_NEIGHBORS = 6

# Dimensões dos vetores
ROUTE_OBS_DIM = 7 + MAX_NEIGHBORS * 8 + MAX_NEIGHBORS * 10 + 2  # = 117
VEL_OBS_DIM   = 16

# Constantes de normalização
_MAP_DIAG       = 1800.0   # diagonal máxima aprox. do mapa
_VEL_MAX        = 30.0     # vel_max global
_TRAFFIC_LAMBDA = 3.0      # penalização de tráfego — 1 robot → 4× mais caro
# MAP_DIAG * (1 + LAMBDA) cobre um caminho completo com 1 robot/aresta
_TRAFFIC_COST_NORM = _MAP_DIAG * (1.0 + _TRAFFIC_LAMBDA)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _traffic_cost_for_hop(
    first_hop: str,
    ref_node:  str,
    goal:      str,
    graph:     FactoryGraph,
    obs:       MotionObs,
    world=None,
) -> float:
    """
    Custo do melhor k-path via first_hop, pesado pelo tráfego actual.
    Usa world para contagem real de robots por aresta (se disponível),
    caso contrário usa obs.edge_robots para o primeiro hop e ignora os restantes.
    """
    k_entries = graph.k_shortest_paths.get(ref_node, {}).get(goal, [])
    best = float("inf")
    for path, _ in k_entries:
        if len(path) < 2 or path[1] != first_hop:
            continue
        score = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            dist = graph.distance(u, v)
            if world is not None:
                n = len(world.robots_on_edge(u, v))
            else:
                n = obs.edge_robots.get(v, 0) if i == 0 else 0
            score += dist * (1.0 + _TRAFFIC_LAMBDA * n)
        if score < best:
            best = score
    if best == float("inf"):
        # fallback: distância estática + primeiro hop
        try:
            d = graph.shortest_path_length(first_hop, goal)
        except Exception:
            d = _MAP_DIAG
        n = obs.edge_robots.get(first_hop, 0) if world is None else (
            len(world.robots_on_edge(ref_node, first_hop)) if world else 0)
        first_dist = graph.distance(ref_node, first_hop) if ref_node else 0.0
        best = first_dist * (1.0 + _TRAFFIC_LAMBDA * n) + d
    return best


def _greedy_next(obs: MotionObs, goal: str, graph: FactoryGraph, world=None) -> Optional[str]:
    """
    Escolhe o próximo hop com base nos k-shortest paths pré-computados.
    Se world fornecido: pondera tráfego em TODAS as arestas do path.
    Sem world: penaliza apenas o primeiro hop via obs.edge_robots.
    """
    ref_node = obs.current_node or obs.from_node
    if ref_node is None:
        return None

    candidates = set(
        n for n in obs.neighbors
        if not (obs.prev_node is not None and n == obs.prev_node and len(obs.neighbors) > 1)
    )
    if not candidates:
        candidates = set(obs.neighbors)
    if not candidates:
        return None

    k_entries = graph.k_shortest_paths.get(ref_node, {}).get(goal, [])

    best_hop, best_score = None, float("inf")
    for path, _ in k_entries:
        if len(path) < 2:
            continue
        first_hop = path[1]
        if first_hop not in candidates:
            continue

        score = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            dist = graph.distance(u, v)
            if world is not None:
                n = len(world.robots_on_edge(u, v))
            else:
                n = obs.edge_robots.get(v, 0) if i == 0 else 0
            score += dist * (1.0 + _TRAFFIC_LAMBDA * n)

        if score < best_score:
            best_score, best_hop = score, first_hop

    # fallback: se nenhum k-path cobre os candidatos
    if best_hop is None:
        for nb in candidates:
            try:
                d = graph.shortest_path_length(nb, goal)
                n = obs.edge_robots.get(nb, 0)
                score = d * (1.0 + _TRAFFIC_LAMBDA * n)
                if score < best_score:
                    best_score, best_hop = score, nb
            except Exception:
                pass

    return best_hop


# ---------------------------------------------------------------------------
# Route observation
# ---------------------------------------------------------------------------

def build_route_obs(
    obs:   MotionObs,
    goal:  str,
    graph: FactoryGraph,
    world=None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Constrói o vetor de observação para o DQN de routing.

    Returns
    -------
    obs_vec   — float32 (ROUTE_OBS_DIM,)
    mask      — bool (MAX_NEIGHBORS,)  True = slot com vizinho válido
    neighbors — lista dos nomes dos vizinhos reais (len ≤ MAX_NEIGHBORS)
    """
    # ---- self features ----
    try:
        dist_to_goal = graph.shortest_path_length(obs.current_node, goal) \
                       if obs.current_node else 0.0
    except Exception:
        dist_to_goal = 0.0

    self_feats = np.array([
        float(obs.progress),
        min(abs(obs.speed) / _VEL_MAX, 1.0),
        min(dist_to_goal / _MAP_DIAG, 1.0),
        float(obs.carried_box is not None),
        min(obs.idle_stuck_ticks / 60.0, 1.0),
        min(obs.robots_targeting_ref / 4.0, 1.0),
        float(obs.lead_robot_has_box),               # robot à frente tem caixa → cede
    ], dtype=np.float32)

    # ---- greedy next hop ----
    greedy = _greedy_next(obs, goal, graph, world)

    # ---- per-neighbor aggregate features ----
    valid_neighbors = list(obs.neighbors)[:MAX_NEIGHBORS]
    agg_feats  = np.zeros((MAX_NEIGHBORS, 8),  dtype=np.float32)
    indiv_feats = np.zeros((MAX_NEIGHBORS, 10), dtype=np.float32)
    mask       = np.zeros(MAX_NEIGHBORS, dtype=bool)

    ref_node = obs.current_node or obs.from_node

    for i, nb in enumerate(valid_neighbors):
        mask[i] = True

        traffic_cost = _traffic_cost_for_hop(nb, ref_node, goal, graph, obs, world) \
                       if ref_node is not None else _MAP_DIAG
        traffic_cost_norm = min(traffic_cost / _TRAFFIC_COST_NORM, 1.0)

        agg_feats[i] = [
            1.0,                                                         # valid
            float(nb == greedy),                                         # is_greedy
            min(obs.edge_robots.get(nb, 0)   / 4.0, 1.0),               # n_robots_norm
            float(obs.edge_opposing.get(nb, False)),                     # has_opposing
            min(obs.neighbor_idle.get(nb, 0) / 2.0, 1.0),               # n_idle_norm
            traffic_cost_norm,                                           # traffic_cost_norm
            float(obs.neighbor_special.get(nb, False)),                  # is_special
            min(obs.adj_incoming_count.get(nb, 0) / 4.0, 1.0),          # incoming_norm (2-hop)
        ]

        # Individual robot visibility: nearest fwd + nearest opp + nearest incoming (2-hop)
        fwd = obs.adj_fwd_nearest.get(nb)
        opp = obs.adj_opp_nearest.get(nb)
        inc = obs.adj_incoming_nearest.get(nb)

        fwd_valid    = float(fwd is not None)
        fwd_progress = fwd[0] if fwd is not None else 0.0
        fwd_speed    = min(abs(fwd[1]) / _VEL_MAX, 1.0) if fwd is not None else 0.0

        opp_valid    = float(opp is not None)
        opp_progress = opp[0] if opp is not None else 0.0
        opp_speed    = min(abs(opp[1]) / _VEL_MAX, 1.0) if opp is not None else 0.0
        opp_has_box  = float(obs.adj_opp_has_box.get(nb, False)) if opp is not None else 0.0
        # conflito persiste se o oposto planeia voltar para nb após chegar a ref
        opp_next     = obs.adj_opp_buffered_next.get(nb) if opp is not None else None
        opp_next_conflicts = float(opp_next == nb) if opp is not None else 0.0

        # inc_progress alto = robot quase a chegar a nb por outra direção
        inc_valid    = float(inc is not None)
        inc_progress = inc[0] if inc is not None else 0.0

        indiv_feats[i] = [fwd_valid, fwd_progress, fwd_speed,
                          opp_valid, opp_progress, opp_speed, opp_has_box, opp_next_conflicts,
                          inc_valid, inc_progress]

    # ---- context features ----
    context_feats = np.array([
        float(obs.collided),
        min(obs.converging_robots / 4.0, 1.0),
    ], dtype=np.float32)

    obs_vec = np.concatenate([
        self_feats,
        agg_feats.flatten(),
        indiv_feats.flatten(),
        context_feats,
    ])
    return obs_vec, mask, valid_neighbors


# ---------------------------------------------------------------------------
# Velocity observation
# ---------------------------------------------------------------------------

def build_velocity_obs(
    obs:   MotionObs,
    goal:  str,
    graph: FactoryGraph,
) -> np.ndarray:
    """
    Constrói o vetor de observação para o DQN de velocidade.

    Returns
    -------
    float32 (VEL_OBS_DIM,)
    """
    approaching_dest = 0.0
    if obs.from_node is not None and obs.to_node is not None:
        try:
            edge_len  = graph.distance(obs.from_node, obs.to_node)
            remaining = (1.0 - float(obs.progress)) * edge_len
            approaching_dest = float(remaining <= 40.0)
        except Exception:
            approaching_dest = float(obs.progress > 0.80)

    # Distância ao goal (para is_final_edge e dist_to_goal_norm)
    try:
        ref = obs.from_node if obs.from_node is not None else obs.current_node
        dist_to_goal = graph.shortest_path_length(ref, goal) if ref else 0.0
    except Exception:
        dist_to_goal = 0.0

    return np.array([
        float(obs.progress),
        min(abs(obs.speed) / _VEL_MAX, 1.0),
        float(obs.lead_gap),
        min(obs.converging_robots / 4.0, 1.0),
        approaching_dest,
        obs.dist_ahead_bin / 3.0,
        float(obs.collided),
        float(obs.to_node_occupied),
        float(obs.docking_exit),
        # 4 — robot à frente e atrás na mesma aresta
        obs.lead_robot_progress,
        min(abs(obs.lead_robot_speed) / _VEL_MAX, 1.0),
        obs.rear_robot_progress,
        min(abs(obs.rear_robot_speed) / _VEL_MAX, 1.0),
        # 2 — contexto do goal
        float(obs.to_node is not None and obs.to_node == goal),  # is_final_edge
        min(dist_to_goal / _MAP_DIAG, 1.0),                     # dist_to_goal_norm
        float(obs.lead_robot_has_box),                           # lead tem caixa → dar espaço
    ], dtype=np.float32)
