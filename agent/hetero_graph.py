"""
agents/gnn_ppo/hetero_graph.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Constrói o grafo heterogéneo (PyTorch Geometric) a partir do estado
bruto devolvido por FactoryEnv._build_state().

Tipos de nós
------------
  "robot"    — um nó por robot activo
  "box"      — um nó por caixa activa (não DONE)
  "mapnode"  — um nó por nó do mapa da fábrica

Tipos de arestas
----------------
  ("robot",   "at",        "mapnode")  — robot está neste nó (ou a caminho)
  ("box",     "at",        "mapnode")  — caixa está neste nó
  ("robot",   "carries",   "box")      — robot transporta esta caixa
  ("box",     "next_wp",   "mapnode")  — próximo waypoint da caixa
  ("mapnode", "connected", "mapnode")  — conectividade do mapa (bidireccional)

Features por tipo de nó
------------------------
  robot   (dim=9):
    [0:4]  state one-hot (IDLE, MOVING, WAITING, PARKED)
    [4]    progress na aresta [0, 1]
    [5]    wait_norm = wait_ticks_in_junction / DEADLOCK_THRESHOLD [0, 1]
    [6]    has_box (bool)
    [7]    x normalizado [0, 1]
    [8]    y normalizado [0, 1]

  box     (dim=9):
    [0:3]  status one-hot (WAITING, IN_TRANSIT, DONE)
    [3:6]  pipeline one-hot (BLUE, GREEN, RED)
    [6]    waypoint_progress = (idx-1) / (n_waypoints-1) [0, 1]
    [7]    x do current_node normalizado (0 se IN_TRANSIT)
    [8]    y do current_node normalizado (0 se IN_TRANSIT)

  mapnode (dim=9):
    [0:7]  type one-hot (junction, entry, exit, processA_entry,
                          processA_exit, processB_entry, processB_exit)
    [7]    n_robots_here normalizado
    [8]    n_boxes_waiting normalizado

Dimensões de features de arestas
----------------------------------
  connected: [0] distância normalizada [0, 1]
  outras:    sem features (edge_attr=None)

Índices de acção
----------------
  O grafo expõe dois índices auxiliares usados pelo actor:
    pending_robot_indices  : list[int]  — índices em robot_nodes
    available_box_indices  : list[int]  — índices em box_nodes

  O actor itera sobre pending_robot_indices (sequencialmente) e, para
  cada robot, calcula scores sobre available_box_indices + preposition
  nodes (mapnode_indices para nós de pre-posição).
"""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import HeteroData

from env.traffic.router import DEADLOCK_THRESHOLD


# ---------------------------------------------------------------------------
# Constantes de normalização
# ---------------------------------------------------------------------------

# Coordenadas máximas do mapa (lidas do map_factory_4.yaml)
_MAP_X_MAX = 1160.0
_MAP_Y_MAX = 1385.0

# Número máximo de robots / caixas para normalizar contagens
_MAX_ROBOTS_PER_NODE = 4
_MAX_BOXES_PER_NODE  = 4

# Tipos de nó do mapa — ordem fixa para one-hot
_NODE_TYPES = [
    "junction",
    "entry",
    "exit",
    "processA_entry",
    "processA_exit",
    "processB_entry",
    "processB_exit",
]
_NODE_TYPE_IDX: dict[str, int] = {t: i for i, t in enumerate(_NODE_TYPES)}

# Pipelines — ordem fixa para one-hot
_PIPELINE_TYPES = ["BLUE", "GREEN", "RED"]
_PIPELINE_IDX: dict[str, int] = {p: i for i, p in enumerate(_PIPELINE_TYPES)}

# Estados do robot — ordem fixa para one-hot
_ROBOT_STATES = ["IDLE", "MOVING", "WAITING", "PARKED"]
_ROBOT_STATE_IDX: dict[str, int] = {s: i for i, s in enumerate(_ROBOT_STATES)}

# Estados da caixa — ordem fixa para one-hot
_BOX_STATUSES = ["WAITING", "IN_TRANSIT", "DONE"]
_BOX_STATUS_IDX: dict[str, int] = {s: i for i, s in enumerate(_BOX_STATUSES)}

# Nós de pre-posição (tipos que o agente pode usar como target)
# Inclui processA_exit e processB_exit porque são agora waypoints reais
# onde caixas ficam WAITING após processamento.
_PREPOSITION_TYPES = frozenset({
    "entry",
    "exit",
    "processA_entry",
    "processA_exit",
    "processB_entry",
    "processB_exit",
})

# Dimensões das features
DIM_ROBOT   = 9
DIM_BOX     = 9
DIM_MAPNODE = 9
DIM_EDGE    = 1   # só arestas "connected" têm features


# ---------------------------------------------------------------------------
# Builder principal
# ---------------------------------------------------------------------------

def build_hetero_graph(state: dict[str, Any]) -> HeteroData:
    """
    Constrói um HeteroData PyG a partir do estado bruto do FactoryEnv.

    Parâmetros
    ----------
    state : dict devolvido por FactoryEnv._build_state()

    Devolve
    -------
    HeteroData com:
      - x para cada tipo de nó
      - edge_index para cada tipo de aresta
      - edge_attr para arestas "connected"
      - atributos auxiliares:
          .pending_robot_indices  : list[int]
          .available_box_indices  : list[int]
          .preposition_node_indices : list[int]
          .robot_id_to_idx        : dict[str, int]
          .box_id_to_idx          : dict[int, int]
          .mapnode_id_to_idx      : dict[str, int]
          .idx_to_robot_id        : list[str]
          .idx_to_box_id          : list[int]
          .idx_to_mapnode_id      : list[str]
    """
    data = HeteroData()

    robots_raw    = state["robots"]           # list[dict]
    boxes_raw     = state["boxes"]            # list[dict]
    graph_nodes   = state["graph_nodes"]      # dict[node_id, dict]
    # Manter a ordem original do env (lista) para que o masking sequencial
    # no actor seja consistente entre o forward de sampling e o de update PPO.
    pending_ids   = state["pending_robot_ids"]    # list[str], ordem estável
    available_ids = state["available_box_ids"]    # list[int], ordem estável

    # Filtra apenas caixas activas (não DONE)
    active_boxes = [b for b in boxes_raw if b["status"] != "DONE"]

    # ------------------------------------------------------------------
    # Índices
    # ------------------------------------------------------------------

    robot_ids   : list[str] = [r["id"] for r in robots_raw]
    box_ids     : list[int] = [b["box_id"] for b in active_boxes]
    mapnode_ids : list[str] = sorted(graph_nodes.keys())

    robot_id_to_idx   : dict[str, int] = {rid: i for i, rid in enumerate(robot_ids)}
    box_id_to_idx     : dict[int, int] = {bid: i for i, bid in enumerate(box_ids)}
    mapnode_id_to_idx : dict[str, int] = {nid: i for i, nid in enumerate(mapnode_ids)}

    n_robots   = len(robot_ids)
    n_boxes    = len(box_ids)
    n_mapnodes = len(mapnode_ids)

    # ------------------------------------------------------------------
    # Features dos nós
    # ------------------------------------------------------------------

    # --- robot ---
    robot_x = torch.zeros(n_robots, DIM_ROBOT, dtype=torch.float32)
    for i, r in enumerate(robots_raw):
        state_idx = _ROBOT_STATE_IDX.get(r["state"], 0)
        robot_x[i, state_idx] = 1.0                                      # [0:4] one-hot state
        robot_x[i, 4] = float(r.get("progress", 0.0))                   # [4] progress
        wait = r.get("wait_ticks_in_junction", 0)
        robot_x[i, 5] = min(wait / max(DEADLOCK_THRESHOLD, 1), 1.0)     # [5] wait_norm
        robot_x[i, 6] = 1.0 if r.get("carrying_box") is not None else 0.0  # [6] has_box
        robot_x[i, 7] = _norm_x(r.get("world_x", 0.0))                  # [7] x norm
        robot_x[i, 8] = _norm_y(r.get("world_y", 0.0))                  # [8] y norm

    data["robot"].x = robot_x

    # --- box ---
    box_x = torch.zeros(n_boxes, DIM_BOX, dtype=torch.float32)
    for i, b in enumerate(active_boxes):
        status_idx = _BOX_STATUS_IDX.get(b["status"], 0)
        box_x[i, status_idx] = 1.0                                        # [0:3] one-hot status

        pipeline_idx = _PIPELINE_IDX.get(b["pipeline"], 0)
        box_x[i, 3 + pipeline_idx] = 1.0                                  # [3:6] one-hot pipeline

        n_wp = b.get("n_waypoints", 1)
        wp_idx = b.get("waypoint_idx", 1)
        box_x[i, 6] = (wp_idx - 1) / max(n_wp - 1, 1)                   # [6] waypoint_progress

        cur_node = b.get("current_node")
        if cur_node and cur_node in graph_nodes:
            node_info = graph_nodes[cur_node]
            box_x[i, 7] = _norm_x(node_info["x"])                        # [7] x norm
            box_x[i, 8] = _norm_y(node_info["y"])                        # [8] y norm

    data["box"].x = box_x

    # --- mapnode ---
    mapnode_x = torch.zeros(n_mapnodes, DIM_MAPNODE, dtype=torch.float32)
    for i, nid in enumerate(mapnode_ids):
        info = graph_nodes[nid]
        type_idx = _NODE_TYPE_IDX.get(info["type"], 0)
        mapnode_x[i, type_idx] = 1.0                                      # [0:7] one-hot type
        n_r = info.get("n_robots_here", 0)
        n_b = info.get("n_boxes_waiting", 0)
        mapnode_x[i, 7] = min(n_r / _MAX_ROBOTS_PER_NODE, 1.0)           # [7] robots norm
        mapnode_x[i, 8] = min(n_b / _MAX_BOXES_PER_NODE, 1.0)            # [8] boxes norm

    data["mapnode"].x = mapnode_x

    # ------------------------------------------------------------------
    # Arestas
    # ------------------------------------------------------------------

    # --- (robot, at, mapnode) ---
    # robot está num nó actual ou está a mover-se para um nó
    r_at_src, r_at_dst = [], []
    for i, r in enumerate(robots_raw):
        cur  = r.get("current_node")
        to_n = r.get("to_node")
        target = cur if cur else to_n
        if target and target in mapnode_id_to_idx:
            r_at_src.append(i)
            r_at_dst.append(mapnode_id_to_idx[target])

    data["robot", "at", "mapnode"].edge_index = _make_edge_index(
        r_at_src, r_at_dst, n_robots, n_mapnodes
    )

    # --- (box, at, mapnode) ---
    # caixa está num nó (só se WAITING — IN_TRANSIT não tem nó)
    b_at_src, b_at_dst = [], []
    for i, b in enumerate(active_boxes):
        cur = b.get("current_node")
        if cur and cur in mapnode_id_to_idx:
            b_at_src.append(i)
            b_at_dst.append(mapnode_id_to_idx[cur])

    data["box", "at", "mapnode"].edge_index = _make_edge_index(
        b_at_src, b_at_dst, n_boxes, n_mapnodes
    )

    # --- (robot, carries, box) ---
    r_carries_src, r_carries_dst = [], []
    for i, r in enumerate(robots_raw):
        carried = r.get("carrying_box")
        if carried is not None and carried in box_id_to_idx:
            r_carries_src.append(i)
            r_carries_dst.append(box_id_to_idx[carried])

    data["robot", "carries", "box"].edge_index = _make_edge_index(
        r_carries_src, r_carries_dst, n_robots, n_boxes
    )

    # --- (box, next_wp, mapnode) ---
    b_wp_src, b_wp_dst = [], []
    for i, b in enumerate(active_boxes):
        nwp = b.get("next_waypoint")
        if nwp and nwp in mapnode_id_to_idx:
            b_wp_src.append(i)
            b_wp_dst.append(mapnode_id_to_idx[nwp])

    data["box", "next_wp", "mapnode"].edge_index = _make_edge_index(
        b_wp_src, b_wp_dst, n_boxes, n_mapnodes
    )

    # --- (mapnode, connected, mapnode) ---
    # Arestas do mapa — bidireccional com distância normalizada
    map_src, map_dst, map_attr = _build_map_edges(
        state, mapnode_id_to_idx, n_mapnodes
    )
    data["mapnode", "connected", "mapnode"].edge_index = _make_edge_index(
        map_src, map_dst, n_mapnodes, n_mapnodes
    )
    if map_attr:
        data["mapnode", "connected", "mapnode"].edge_attr = torch.tensor(
            map_attr, dtype=torch.float32
        ).unsqueeze(1)

    # ------------------------------------------------------------------
    # Metadados auxiliares para o actor
    # ------------------------------------------------------------------

    data.pending_robot_indices = [
        robot_id_to_idx[rid] for rid in pending_ids
        if rid in robot_id_to_idx
    ]
    # Deduplica mantendo a ordem (available_ids pode ter duplicados em edge cases)
    seen_box: set[int] = set()
    data.available_box_indices = []
    for bid in available_ids:
        if bid in box_id_to_idx and bid not in seen_box:
            data.available_box_indices.append(box_id_to_idx[bid])
            seen_box.add(bid)
    data.preposition_node_indices = [
        mapnode_id_to_idx[nid]
        for nid in mapnode_ids
        if graph_nodes[nid]["type"] in _PREPOSITION_TYPES
    ]

    # Mapeamentos inversos (para descodificar acções)
    data.robot_id_to_idx   = robot_id_to_idx
    data.box_id_to_idx     = box_id_to_idx
    data.mapnode_id_to_idx = mapnode_id_to_idx
    data.idx_to_robot_id   = robot_ids
    data.idx_to_box_id     = box_ids
    data.idx_to_mapnode_id = mapnode_ids

    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_x(x: float) -> float:
    return float(x) / _MAP_X_MAX


def _norm_y(y: float) -> float:
    return float(y) / _MAP_Y_MAX


def _make_edge_index(
    src     : list[int],
    dst     : list[int],
    n_src   : int,
    n_dst   : int,
) -> torch.Tensor:
    """Cria edge_index [2, E]. Devolve tensor vazio se não houver arestas."""
    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def _build_map_edges(
    state             : dict,
    mapnode_id_to_idx : dict[str, int],
    n_mapnodes        : int,
) -> tuple[list[int], list[int], list[float]]:
    """
    Constrói arestas do mapa a partir das arestas do grafo.

    As arestas são bidirecccionais (u→v e v→u).
    A distância é normalizada pelo comprimento máximo de aresta do mapa.
    """
    # O estado não inclui as arestas directamente — precisamos de as inferir
    # a partir dos graph_nodes. Mas o FactoryEnv não expõe as arestas no state.
    # Usamos o graph_edges se disponível, senão inferimos da conectividade.
    graph_edges = state.get("graph_edges", [])

    if not graph_edges:
        # Fallback: sem arestas (o GNN ainda funciona com as outras arestas)
        return [], [], []

    # Normalização pelo máximo
    distances = [e["distance"] for e in graph_edges if "distance" in e]
    max_dist  = max(distances) if distances else 1.0

    src, dst, attr = [], [], []
    for edge in graph_edges:
        u = edge.get("from")
        v = edge.get("to")
        d = edge.get("distance", 1.0)

        if u not in mapnode_id_to_idx or v not in mapnode_id_to_idx:
            continue

        ui = mapnode_id_to_idx[u]
        vi = mapnode_id_to_idx[v]
        d_norm = d / max_dist

        # Bidireccional
        src.extend([ui, vi])
        dst.extend([vi, ui])
        attr.extend([d_norm, d_norm])

    return src, dst, attr


# ---------------------------------------------------------------------------
# Dimensões exportadas (usadas pelo actor_critic.py para inicializar as redes)
# ---------------------------------------------------------------------------

NODE_DIMS = {
    "robot":   DIM_ROBOT,
    "box":     DIM_BOX,
    "mapnode": DIM_MAPNODE,
}

EDGE_TYPES = [
    ("robot",   "at",        "mapnode"),
    ("box",     "at",        "mapnode"),
    ("robot",   "carries",   "box"),
    ("box",     "next_wp",   "mapnode"),
    ("mapnode", "connected", "mapnode"),
]