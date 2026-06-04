"""Rede neuronal do agente RL: HeteroGNN + AssignmentHead (actor) + GlobalValueHead (critic)."""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from agent.hetero_graph import (
    DIM_BOX,
    DIM_MAPNODE,
    DIM_ROBOT,
    EDGE_TYPES,
    MAX_ROBOTS_PER_NODE,
)


CandidateKey = tuple                  # ("box", box_id, target_node) | ("idle", -1, None)
NEG_INF      = float("-inf")
Mode         = Literal["sample", "argmax", "log_prob"]


class RobotDecision(NamedTuple):
    """Resultado do actor para um robot pendente.

    Cada candidate é tuplo (tipo, id, target_node_ou_None).
    """
    robot_idx  : int
    logits     : Tensor
    candidates : list[CandidateKey]


# ─────────────────────────────────────────────────────────────────────────
#  Encoder
# ─────────────────────────────────────────────────────────────────────────
class HeteroGNN(nn.Module):
    """GNN heterogéneo com SAGEConv por tipo de aresta + residual + LayerNorm."""

    def __init__(
        self,
        hidden_dim : int = 128,
        n_layers   : int = 2,
        dropout    : float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout    = dropout

        self.input_proj = nn.ModuleDict({
            "robot":   nn.Linear(DIM_ROBOT,   hidden_dim),
            "box":     nn.Linear(DIM_BOX,     hidden_dim),
            "mapnode": nn.Linear(DIM_MAPNODE, hidden_dim),
        })

        self.convs = nn.ModuleList([
            HeteroConv(
                {etype: SAGEConv(hidden_dim, hidden_dim) for etype in EDGE_TYPES},
                aggr="sum",
            )
            for _ in range(n_layers)
        ])

        self.norms = nn.ModuleList([
            nn.ModuleDict({
                "robot":   nn.LayerNorm(hidden_dim),
                "box":     nn.LayerNorm(hidden_dim),
                "mapnode": nn.LayerNorm(hidden_dim),
            })
            for _ in range(n_layers)
        ])

    def forward(self, data: HeteroData) -> dict[str, Tensor]:
        """Devolve {ntype: tensor [n, hidden_dim]} após message passing."""
        device = next(self.parameters()).device

        # Embedding inicial — todos os 3 tipos têm tensor (vazio se n=0).
        x_dict: dict[str, Tensor] = {}
        for ntype in ("robot", "box", "mapnode"):
            x = data[ntype].x if ntype in data.node_types else None
            if x is None or x.shape[0] == 0:
                x_dict[ntype] = torch.zeros(0, self.hidden_dim, device=device)
            else:
                x_dict[ntype] = F.relu(self.input_proj[ntype](x))

        # Apenas passa edges com pelo menos 1 aresta e cujos endpoints existam.
        edge_index_dict = {
            etype: data[etype].edge_index
            for etype in EDGE_TYPES
            if data[etype].edge_index.shape[1] > 0
            and x_dict[etype[0]].shape[0] > 0
            and x_dict[etype[2]].shape[0] > 0
        }

        for i, conv in enumerate(self.convs):
            if not edge_index_dict:
                break
            out_dict = conv(x_dict, edge_index_dict)
            for ntype, h_new in out_dict.items():
                if x_dict[ntype].shape[0] == 0:
                    continue
                h = h_new + x_dict[ntype]                # residual
                h = self.norms[i][ntype](h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                x_dict[ntype] = h

        return x_dict


# ─────────────────────────────────────────────────────────────────────────
#  Actor
# ─────────────────────────────────────────────────────────────────────────
CANDIDATE_EXTRA_DIM = 3   # [n_robots_at_target_norm, is_someone_goal, in_future_path]


class AssignmentHead(nn.Module):
    """Scores additive-attention sobre candidatos (box ∪ idle).

    Cada candidato tem score = v(tanh(W_r·h_robot + W_c·candidate_emb +
    W_e·extra_features)), onde extra_features dá info explícita sobre
    ocupação do target (n_robots lá, é goal de alguém, está no future_path).
    """

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.W_robot     = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_candidate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_extra     = nn.Linear(CANDIDATE_EXTRA_DIM, hidden_dim, bias=False)
        self.v           = nn.Linear(hidden_dim, 1, bias=False)
        self.idle_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.01)

    def forward(
        self,
        h_dict : dict[str, Tensor],
        data   : HeteroData,
    ) -> list[RobotDecision]:
        """Decisões do actor para cada robot pendente. Sem masking sequencial aqui —
        o masking é aplicado em apply_sequential() para garantir consistência entre
        sample / argmax / log_prob."""
        h_robot   = h_dict["robot"]
        h_box     = h_dict["box"]
        h_mapnode = h_dict["mapnode"]

        pending_indices   = getattr(data, "pending_robot_indices", []) or []
        avail_box_targets = getattr(data, "available_box_targets", []) or []
        idx_to_box_id     = getattr(data, "idx_to_box_id",         []) or []
        idx_to_mapnode_id = getattr(data, "idx_to_mapnode_id",     []) or []
        node_n_robots     = getattr(data, "node_n_robots",         {}) or {}
        node_is_goal      = getattr(data, "node_is_goal",          set())
        node_in_future    = getattr(data, "node_in_future",        set())

        device = h_robot.device if h_robot.shape[0] > 0 else next(self.parameters()).device

        def target_extras(target_idx: int) -> Tensor:
            n   = node_n_robots.get(target_idx, 0) / MAX_ROBOTS_PER_NODE
            ig  = 1.0 if target_idx in node_is_goal   else 0.0
            ifp = 1.0 if target_idx in node_in_future else 0.0
            return torch.tensor([n, ig, ifp], dtype=torch.float32, device=device)

        # Pré-projecta candidatos uma única vez (partilhado entre robots).
        cand_embeddings : list[Tensor]       = []
        cand_keys       : list[CandidateKey] = []

        # Candidatos box: cada (box, target) é 1 entry. Embedding = box + target + extras.
        for box_idx, target_idx in avail_box_targets:
            if not (0 <= box_idx < h_box.shape[0]):
                continue
            if not (0 <= target_idx < h_mapnode.shape[0]):
                continue
            if box_idx >= len(idx_to_box_id) or target_idx >= len(idx_to_mapnode_id):
                continue
            emb = (
                self.W_candidate(h_box[box_idx] + h_mapnode[target_idx])
                + self.W_extra(target_extras(target_idx))
            )
            cand_embeddings.append(emb)
            cand_keys.append((
                "box",
                idx_to_box_id[box_idx],
                idx_to_mapnode_id[target_idx],
            ))

        # Idle sempre presente; sem features extras (não tem target).
        zero_extras = torch.zeros(CANDIDATE_EXTRA_DIM, device=device)
        cand_embeddings.append(
            self.W_candidate(self.idle_embedding) + self.W_extra(zero_extras)
        )
        cand_keys.append(("idle", -1, None))

        h_cands = torch.stack(cand_embeddings, dim=0)    # [n_cands, hidden_dim]

        decisions: list[RobotDecision] = []
        for robot_idx in pending_indices:
            if robot_idx >= h_robot.shape[0]:
                continue
            h_r    = self.W_robot(h_robot[robot_idx]).unsqueeze(0)   # [1, hidden_dim]
            scores = self.v(torch.tanh(h_r + h_cands)).squeeze(-1)   # [n_cands]
            decisions.append(RobotDecision(
                robot_idx  = robot_idx,
                logits     = scores,
                candidates = list(cand_keys),
            ))

        return decisions


# ─────────────────────────────────────────────────────────────────────────
#  Critic
# ─────────────────────────────────────────────────────────────────────────
class GlobalValueHead(nn.Module):
    """V(s) via max-pool por tipo + MLP. Max preserva sinal de eventos raros."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_dict: dict[str, Tensor]) -> Tensor:
        """Estima V(s) (escalar)."""
        device = next(self.parameters()).device
        pools: list[Tensor] = []
        for ntype in ("robot", "box", "mapnode"):
            h = h_dict.get(ntype)
            if h is None or h.shape[0] == 0:
                pools.append(torch.zeros(self.mlp[0].in_features // 3, device=device))
            else:
                pools.append(h.max(dim=0).values)
        h_global = torch.cat(pools, dim=0)
        return self.mlp(h_global).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────
#  Composição
# ─────────────────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    """GNN + AssignmentHead (actor) + GlobalValueHead (critic); encode partilhado."""

    def __init__(
        self,
        hidden_dim : int   = 128,
        n_layers   : int   = 2,
        dropout    : float = 0.1,
    ) -> None:
        super().__init__()
        self.gnn    = HeteroGNN(hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)
        self.actor  = AssignmentHead(hidden_dim=hidden_dim)
        self.critic = GlobalValueHead(hidden_dim=hidden_dim)

    def forward(
        self,
        data: HeteroData,
    ) -> tuple[list[RobotDecision], Tensor]:
        """Encode uma vez e devolve (decisions, value)."""
        h_dict    = self.gnn(data)
        decisions = self.actor(h_dict, data)
        value     = self.critic(h_dict)
        return decisions, value


# ─────────────────────────────────────────────────────────────────────────
#  Masking sequencial + sampling/log_prob — UMA implementação para tudo
# ─────────────────────────────────────────────────────────────────────────
def _mask_logits(
    logits         : Tensor,
    candidates     : list[CandidateKey],
    assigned_boxes : set[int],
    assigned_nodes : set[str],
) -> Tensor:
    """Aplica masking sequencial + idle-mask.

    - Box já-escolhida: mesma box_id ou mesmo target_node consumido.
    - Idle-mask: idle só é válido se não houver candidatos box disponíveis.
    """
    masked = logits.clone()
    for j, key in enumerate(candidates):
        ctype = key[0]
        if ctype == "box":
            _, box_id, target_node = key
            if int(box_id) in assigned_boxes:
                masked[j] = NEG_INF
            elif target_node is not None and str(target_node) in assigned_nodes:
                masked[j] = NEG_INF

    has_real = any(
        candidates[j][0] != "idle" and masked[j].item() != NEG_INF
        for j in range(len(candidates))
    )
    if has_real:
        for j, key in enumerate(candidates):
            if key[0] == "idle":
                masked[j] = NEG_INF

    return masked


def apply_sequential(
    decisions      : list[RobotDecision],
    mode           : Mode,
    action_indices : list[int] | None = None,
) -> tuple[list[int], Tensor]:
    """Loop sequencial sobre decisões com masking consistente.

    mode='sample'   → amostra estocástico; devolve (chosen_indices, log_probs).
    mode='argmax'   → argmax determinístico; devolve (chosen_indices, log_probs).
    mode='log_prob' → recomputa log_probs para `action_indices` dado (sem alterar a acção).
    """
    if mode == "log_prob" and action_indices is None:
        raise ValueError("mode='log_prob' requires action_indices.")

    assigned_boxes : set[int] = set()
    assigned_nodes : set[str] = set()

    chosen    : list[int]    = []
    log_probs : list[Tensor] = []

    for i, decision in enumerate(decisions):
        masked = _mask_logits(
            decision.logits,
            decision.candidates,
            assigned_boxes,
            assigned_nodes,
        )
        dist = torch.distributions.Categorical(logits=masked)

        if mode == "sample":
            action = int(dist.sample().item())
        elif mode == "argmax":
            action = int(masked.argmax().item())
        else:  # log_prob
            action = int(action_indices[i])

        log_probs.append(dist.log_prob(torch.tensor(action, device=masked.device)))
        chosen.append(action)

        if action < len(decision.candidates):
            key = decision.candidates[action]
            ctype = key[0]
            if ctype == "box":
                _, box_id, target_node = key
                assigned_boxes.add(int(box_id))
                if target_node is not None:
                    assigned_nodes.add(str(target_node))

    if not log_probs:
        device = decisions[0].logits.device if decisions else torch.device("cpu")
        return chosen, torch.zeros(0, device=device)
    return chosen, torch.stack(log_probs)


def compute_entropy(decisions: list[RobotDecision]) -> Tensor:
    """Entropia média sobre os robots pendentes, com idle-mask coerente."""
    if not decisions:
        device = next(iter(decisions), RobotDecision(0, torch.zeros(1), [])).logits.device
        return torch.zeros(1, device=device)

    entropies: list[Tensor] = []
    for decision in decisions:
        masked = _mask_logits(
            decision.logits,
            decision.candidates,
            set(), set(),
        )
        dist = torch.distributions.Categorical(logits=masked)
        entropies.append(dist.entropy())
    return torch.stack(entropies).mean()


def decode_actions(
    decisions      : list[RobotDecision],
    action_indices : list[int],
    data           : HeteroData,
) -> dict[str, tuple[int, str] | None]:
    """Converte índices em assignments {robot_id: (box_id, target_node) | None}."""
    idx_to_robot_id = getattr(data, "idx_to_robot_id", [])
    assignments: dict[str, tuple[int, str] | str | None] = {}

    for decision, action_idx in zip(decisions, action_indices):
        if decision.robot_idx >= len(idx_to_robot_id):
            continue
        robot_id = idx_to_robot_id[decision.robot_idx]

        if action_idx >= len(decision.candidates):
            assignments[robot_id] = None
            continue

        key = decision.candidates[action_idx]
        ctype = key[0]
        if ctype == "box":
            _, box_id, target_node = key
            assignments[robot_id] = (int(box_id), str(target_node))
        else:
            assignments[robot_id] = None

    return assignments
