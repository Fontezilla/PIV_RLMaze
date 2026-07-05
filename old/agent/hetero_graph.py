"""Constrói o HeteroData (PyG) a partir do estado bruto do FactoryEnv."""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import HeteroData

from old.env.traffic.router import DEADLOCK_THRESHOLD


# ---- Normalização espacial (factory map bounds) -------------------------
_MAP_X_MAX = 1160.0
_MAP_Y_MAX = 1385.0

MAX_ROBOTS_PER_NODE = 4   # normalização de n_robots_here nas features de nó
_MAX_BOXES_PER_NODE = 4

# ---- Vocabulários one-hot ------------------------------------------------
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

_PIPELINE_TYPES = ["BLUE", "GREEN", "RED"]
_PIPELINE_IDX: dict[str, int] = {p: i for i, p in enumerate(_PIPELINE_TYPES)}

_ROBOT_STATES = ["IDLE", "MOVING", "WAITING", "PARKED"]
_ROBOT_STATE_IDX: dict[str, int] = {s: i for i, s in enumerate(_ROBOT_STATES)}

_BOX_STATUSES = ["WAITING", "IN_TRANSIT", "DONE"]
_BOX_STATUS_IDX: dict[str, int] = {s: i for i, s in enumerate(_BOX_STATUSES)}

DIM_ROBOT   = 9
DIM_BOX     = 9
DIM_MAPNODE = 14   # 7 type + n_robots + n_boxes + betweenness + degree + 3 dists


# Normalizadores de distância — diagonal aproximada do mapa (1160 x 1385 ≈ 1810).
_DIST_MAX  = 2000.0
_DEGREE_MAX = 4.0


# ---- Edge types ---------------------------------------------------------
# Directos: aquilo que o estado descreve naturalmente.
# Os reverses são gerados em _add_reverse_edges() para garantir message
# passing em ambos sentidos no HeteroConv.
DIRECT_EDGE_TYPES: list[tuple[str, str, str]] = [
    ("robot", "at",          "mapnode"),
    ("box",   "at",          "mapnode"),
    ("robot", "carries",     "box"),
    ("robot", "goal",        "mapnode"),
    ("box",   "next_wp",     "mapnode"),
    # Future path: próximos K nodes que o robot vai visitar. Permite ao
    # agent ver as rotas planeadas dos outros para coordenar espacialmente.
    ("robot", "future_path", "mapnode"),
]

# Edge homogéneo (já bidirectional na construção das edges do mapa).
MAP_EDGE_TYPE: tuple[str, str, str] = ("mapnode", "connected", "mapnode")

# Lista completa, exposta a quem constrói HeteroConvs.
EDGE_TYPES: list[tuple[str, str, str]] = [
    *DIRECT_EDGE_TYPES,
    *[(dst, f"rev_{rel}", src) for src, rel, dst in DIRECT_EDGE_TYPES],
    MAP_EDGE_TYPE,
]


# ---- Construção do grafo ------------------------------------------------
def build_hetero_graph(state: dict[str, Any]) -> HeteroData:
    """Constrói HeteroData PyG a partir do state dict do FactoryEnv."""
    data = HeteroData()

    robots_raw        = state["robots"]
    boxes_raw         = state["boxes"]
    graph_nodes       = state["graph_nodes"]
    pending_ids       = state["pending_robot_ids"]
    available_targets = state.get("available_box_targets", [])

    active_boxes = [b for b in boxes_raw if b["status"] != "DONE"]

    robot_ids   : list[str] = [r["id"] for r in robots_raw]
    box_ids     : list[int] = [b["box_id"] for b in active_boxes]
    mapnode_ids : list[str] = sorted(graph_nodes.keys())

    robot_id_to_idx   : dict[str, int] = {rid: i for i, rid in enumerate(robot_ids)}
    box_id_to_idx     : dict[int, int] = {bid: i for i, bid in enumerate(box_ids)}
    mapnode_id_to_idx : dict[str, int] = {nid: i for i, nid in enumerate(mapnode_ids)}

    # ── Node features ────────────────────────────────────────────────────
    data["robot"].x   = _build_robot_features(robots_raw)
    data["box"].x     = _build_box_features(active_boxes, graph_nodes)
    data["mapnode"].x = _build_mapnode_features(mapnode_ids, graph_nodes)

    # ── Direct edges ─────────────────────────────────────────────────────
    data["robot", "at", "mapnode"].edge_index = _make_edge_index(
        *_robot_at_pairs(robots_raw, mapnode_id_to_idx)
    )
    data["box", "at", "mapnode"].edge_index = _make_edge_index(
        *_box_at_pairs(active_boxes, mapnode_id_to_idx)
    )
    data["robot", "carries", "box"].edge_index = _make_edge_index(
        *_robot_carries_pairs(robots_raw, box_id_to_idx)
    )
    data["robot", "goal", "mapnode"].edge_index = _make_edge_index(
        *_robot_goal_pairs(robots_raw, mapnode_id_to_idx)
    )
    data["box", "next_wp", "mapnode"].edge_index = _make_edge_index(
        *_box_next_wp_pairs(active_boxes, mapnode_id_to_idx)
    )
    data["robot", "future_path", "mapnode"].edge_index = _make_edge_index(
        *_robot_future_path_pairs(
            state.get("future_paths", {}),
            robot_id_to_idx,
            mapnode_id_to_idx,
        )
    )

    # ── Reverse edges (uniformes, gerados a partir dos directos) ─────────
    _add_reverse_edges(data)

    # ── Map topology (bidirectional já no edge_index) ────────────────────
    map_src, map_dst = _build_map_edges(state, mapnode_id_to_idx)
    data[MAP_EDGE_TYPE].edge_index = _make_edge_index(map_src, map_dst)

    # ── Metadados auxiliares para o actor ────────────────────────────────
    data.pending_robot_indices = [
        robot_id_to_idx[rid] for rid in pending_ids
        if rid in robot_id_to_idx
    ]
    # Pre-agrega info sobre ocupação por mapnode_idx, usado pelo AssignmentHead
    # para computar features explícitas dos candidatos (n_robots_at_target etc).
    n_at_node    : dict[int, int]      = {}
    is_someone_goal : set[int]         = set()
    future_paths = state.get("future_paths", {})
    is_in_future : set[int]            = set()

    for r in robots_raw:
        cur = r.get("current_node")
        if cur and cur in mapnode_id_to_idx:
            idx = mapnode_id_to_idx[cur]
            n_at_node[idx] = n_at_node.get(idx, 0) + 1
        goal = r.get("goal_node")
        if goal and goal in mapnode_id_to_idx:
            is_someone_goal.add(mapnode_id_to_idx[goal])

    for _rid, path in future_paths.items():
        for node in path:
            if node in mapnode_id_to_idx:
                is_in_future.add(mapnode_id_to_idx[node])

    data.node_n_robots      = n_at_node       # mapnode_idx -> count
    data.node_is_goal       = is_someone_goal # set of mapnode_idx
    data.node_in_future     = is_in_future    # set of mapnode_idx
    # Candidatos (box_idx, target_mapnode_idx). Cada box pode gerar múltiplos
    # candidatos se tiver várias opções de target ainda por escolher.
    seen_target: set[tuple[int, str]] = set()
    data.available_box_targets = []
    for bid, target_node in available_targets:
        key = (bid, target_node)
        if key in seen_target:
            continue
        if bid not in box_id_to_idx or target_node not in mapnode_id_to_idx:
            continue
        seen_target.add(key)
        data.available_box_targets.append(
            (box_id_to_idx[bid], mapnode_id_to_idx[target_node])
        )

    data.idx_to_robot_id   = robot_ids
    data.idx_to_box_id     = box_ids
    data.idx_to_mapnode_id = mapnode_ids

    return data


# ---- Features -----------------------------------------------------------
def _build_robot_features(robots_raw: list[dict]) -> torch.Tensor:
    """Tensor [n_robots, DIM_ROBOT]."""
    n = len(robots_raw)
    x = torch.zeros(n, DIM_ROBOT, dtype=torch.float32)
    for i, r in enumerate(robots_raw):
        state_idx = _ROBOT_STATE_IDX.get(r.get("state", "IDLE"), 0)
        x[i, state_idx] = 1.0
        x[i, 4] = float(r.get("progress", 0.0))
        wait = r.get("wait_ticks_in_junction", 0)
        x[i, 5] = min(wait / max(DEADLOCK_THRESHOLD, 1), 1.0)
        x[i, 6] = 1.0 if r.get("carrying_box") is not None else 0.0
        x[i, 7] = _norm_x(r.get("world_x", 0.0))
        x[i, 8] = _norm_y(r.get("world_y", 0.0))
    return x


def _build_box_features(
    active_boxes : list[dict],
    graph_nodes  : dict[str, dict],
) -> torch.Tensor:
    """Tensor [n_boxes, DIM_BOX]."""
    n = len(active_boxes)
    x = torch.zeros(n, DIM_BOX, dtype=torch.float32)
    for i, b in enumerate(active_boxes):
        status_idx = _BOX_STATUS_IDX.get(b.get("status", "WAITING"), 0)
        x[i, status_idx] = 1.0
        pipeline_idx = _PIPELINE_IDX.get(b.get("pipeline", "BLUE"), 0)
        x[i, 3 + pipeline_idx] = 1.0
        # Progresso = fases já feitas / fases totais da pipeline.
        steps_total = max(b.get("steps_total", 1), 1)
        x[i, 6] = b.get("steps_done", 0) / steps_total
        cur_node = b.get("current_node")
        if cur_node and cur_node in graph_nodes:
            info = graph_nodes[cur_node]
            x[i, 7] = _norm_x(info["x"])
            x[i, 8] = _norm_y(info["y"])
    return x


def _build_mapnode_features(
    mapnode_ids : list[str],
    graph_nodes : dict[str, dict],
) -> torch.Tensor:
    """Tensor [n_mapnodes, DIM_MAPNODE]."""
    n = len(mapnode_ids)
    x = torch.zeros(n, DIM_MAPNODE, dtype=torch.float32)
    for i, nid in enumerate(mapnode_ids):
        info = graph_nodes[nid]
        type_idx = _NODE_TYPE_IDX.get(info["type"], 0)
        x[i, type_idx] = 1.0
        # Dinâmicas
        x[i, 7]  = min(info.get("n_robots_here",   0) / MAX_ROBOTS_PER_NODE, 1.0)
        x[i, 8]  = min(info.get("n_boxes_waiting", 0) / _MAX_BOXES_PER_NODE,  1.0)
        # Estruturais (pre-computadas)
        x[i, 9]  = float(info.get("betweenness", 0.0))               # [0, 1]
        x[i, 10] = min(info.get("degree", 1) / _DEGREE_MAX, 1.0)
        x[i, 11] = _norm_dist(info.get("dist_to_exit",     0.0))
        x[i, 12] = _norm_dist(info.get("dist_to_processA", 0.0))
        x[i, 13] = _norm_dist(info.get("dist_to_processB", 0.0))
    return x


def _norm_dist(d: float) -> float:
    """Normaliza distância para [0, 1]; inf vira 1."""
    if d == float("inf"):
        return 1.0
    return min(float(d) / _DIST_MAX, 1.0)


# ---- Edge index builders ------------------------------------------------
def _robot_at_pairs(
    robots_raw       : list[dict],
    mapnode_id_to_idx: dict[str, int],
) -> tuple[list[int], list[int]]:
    """Onde o robot está (current_node ou to_node se em trânsito)."""
    src, dst = [], []
    for i, r in enumerate(robots_raw):
        cur    = r.get("current_node")
        to_n   = r.get("to_node")
        target = cur if cur else to_n
        if target and target in mapnode_id_to_idx:
            src.append(i)
            dst.append(mapnode_id_to_idx[target])
    return src, dst


def _robot_goal_pairs(
    robots_raw       : list[dict],
    mapnode_id_to_idx: dict[str, int],
) -> tuple[list[int], list[int]]:
    """Para onde o robot quer ir (goal_node), se definido."""
    src, dst = [], []
    for i, r in enumerate(robots_raw):
        goal = r.get("goal_node")
        if goal and goal in mapnode_id_to_idx:
            src.append(i)
            dst.append(mapnode_id_to_idx[goal])
    return src, dst


def _box_at_pairs(
    active_boxes     : list[dict],
    mapnode_id_to_idx: dict[str, int],
) -> tuple[list[int], list[int]]:
    """Onde a box está pousada (não tem edge se IN_TRANSIT)."""
    src, dst = [], []
    for i, b in enumerate(active_boxes):
        cur = b.get("current_node")
        if cur and cur in mapnode_id_to_idx:
            src.append(i)
            dst.append(mapnode_id_to_idx[cur])
    return src, dst


def _box_next_wp_pairs(
    active_boxes     : list[dict],
    mapnode_id_to_idx: dict[str, int],
) -> tuple[list[int], list[int]]:
    """Próximo waypoint para o qual a box quer ser entregue."""
    src, dst = [], []
    for i, b in enumerate(active_boxes):
        nwp = b.get("next_waypoint")
        if nwp and nwp in mapnode_id_to_idx:
            src.append(i)
            dst.append(mapnode_id_to_idx[nwp])
    return src, dst


def _robot_future_path_pairs(
    future_paths     : dict[str, list[str]],
    robot_id_to_idx  : dict[str, int],
    mapnode_id_to_idx: dict[str, int],
) -> tuple[list[int], list[int]]:
    """Edges (robot, future_path, mapnode) — 1 por (robot, node planeado)."""
    src, dst = [], []
    for robot_id, path in future_paths.items():
        if robot_id not in robot_id_to_idx:
            continue
        r_idx = robot_id_to_idx[robot_id]
        for node in path:
            if node in mapnode_id_to_idx:
                src.append(r_idx)
                dst.append(mapnode_id_to_idx[node])
    return src, dst


def _robot_carries_pairs(
    robots_raw    : list[dict],
    box_id_to_idx : dict[int, int],
) -> tuple[list[int], list[int]]:
    """Robot a transportar uma box (raro: só durante o transporte)."""
    src, dst = [], []
    for i, r in enumerate(robots_raw):
        carried = r.get("carrying_box")
        if carried is not None and carried in box_id_to_idx:
            src.append(i)
            dst.append(box_id_to_idx[carried])
    return src, dst


def _build_map_edges(
    state             : dict,
    mapnode_id_to_idx : dict[str, int],
) -> tuple[list[int], list[int]]:
    """Edges do grafo do mapa (bidireccionais; distance descartada para já)."""
    src, dst = [], []
    for edge in state.get("graph_edges", []):
        u = edge.get("from")
        v = edge.get("to")
        if u not in mapnode_id_to_idx or v not in mapnode_id_to_idx:
            continue
        ui = mapnode_id_to_idx[u]
        vi = mapnode_id_to_idx[v]
        src.extend([ui, vi])
        dst.extend([vi, ui])
    return src, dst


def _add_reverse_edges(data: HeteroData) -> None:
    """Para cada edge direccional, cria o reverso uniforme ('rev_<rel>')."""
    for src, rel, dst in DIRECT_EDGE_TYPES:
        ei = data[src, rel, dst].edge_index
        data[dst, f"rev_{rel}", src].edge_index = ei.flip(0)


# ---- Utilitários --------------------------------------------------------
def _norm_x(x: float) -> float:
    return float(x) / _MAP_X_MAX


def _norm_y(y: float) -> float:
    return float(y) / _MAP_Y_MAX


def _make_edge_index(src: list[int], dst: list[int]) -> torch.Tensor:
    """Cria edge_index [2, E]; tensor [2, 0] se não houver arestas."""
    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)
