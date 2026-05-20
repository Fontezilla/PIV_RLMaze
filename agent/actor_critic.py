"""
agents/gnn_ppo/actor_critic.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rede neuronal para o agente RL de gestão de fábrica.

Arquitectura
------------

  HeteroData (PyG)
       ↓
  HeteroGNN  (2-3 camadas de HeteroConv com SAGEConv)
       ↓
  Embeddings por tipo de nó:
    h_robot   [n_robots,   hidden_dim]
    h_box     [n_boxes,    hidden_dim]
    h_mapnode [n_mapnodes, hidden_dim]
       ↓
  ┌─────────────────────────────────┐
  │  Actor (AssignmentHead)         │
  │                                 │
  │  Para cada robot pendente r_i:  │
  │    candidatos = caixas          │
  │                 disponíveis     │
  │               + nós de          │
  │                 pre-posição     │
  │                                 │
  │    score_ij = v · tanh(         │
  │      W_r · h_robot_i            │
  │      + W_c · h_candidate_j      │
  │    )                            │
  │    → softmax → π(a | s, r_i)   │
  └─────────────────────────────────┘
       ↓
  ┌─────────────────────────────────┐
  │  Critic (GlobalPooling + MLP)   │
  │                                 │
  │    h_global = mean_pool(        │
  │      h_robot ∥ h_box ∥ h_map   │
  │    )                            │
  │    V(s) = MLP(h_global)         │
  └─────────────────────────────────┘

Acção
-----
  Para cada robot pendente (processados sequencialmente):
    acção ∈ {box_0, box_1, ..., box_M, node_0, node_1, ..., node_K}

  O espaço de acção varia com o número de candidatos disponíveis.
  O masking é feito internamente: caixas já assignadas neste step
  são removidas dos candidatos dos robots seguintes.

Forward pass
------------
  actor_forward(data) →
    list[ (robot_idx, logits, candidate_keys) ]
    onde candidate_keys = lista de ("box", box_id) ou ("node", node_id)
    com a mesma ordem que logits.

  critic_forward(data) →
    V(s) : tensor escalar
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool

from agent.hetero_graph import (
    DIM_BOX,
    DIM_MAPNODE,
    DIM_ROBOT,
    EDGE_TYPES,
)


# ---------------------------------------------------------------------------
# Tipos auxiliares
# ---------------------------------------------------------------------------

# Chave de candidato: ("box", box_id) ou ("node", node_id)
CandidateKey = tuple[str, int | str]


class RobotDecision(NamedTuple):
    """Resultado do actor para um robot pendente."""
    robot_idx    : int           # índice em h_robot
    logits       : Tensor        # [n_candidates] — não normalizado
    candidates   : list[CandidateKey]  # mesma ordem que logits


# ---------------------------------------------------------------------------
# HeteroGNN
# ---------------------------------------------------------------------------

class HeteroGNN(nn.Module):
    """
    GNN heterogéneo com SAGEConv por tipo de aresta.

    Todas as features de entrada são projectadas para hidden_dim antes
    das camadas de message passing — permite usar a mesma dimensão
    independentemente das features originais.

    Parâmetros
    ----------
    hidden_dim : dimensão dos embeddings intermédios e de saída
    n_layers   : número de camadas de message passing (recomendado: 2-3)
    dropout    : dropout aplicado entre camadas
    """

    def __init__(
        self,
        hidden_dim : int = 128,
        n_layers   : int = 2,
        dropout    : float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dropout    = dropout

        # Projecções de entrada — cada tipo de nó tem a sua
        self.input_proj = nn.ModuleDict({
            "robot":   nn.Linear(DIM_ROBOT,   hidden_dim),
            "box":     nn.Linear(DIM_BOX,     hidden_dim),
            "mapnode": nn.Linear(DIM_MAPNODE, hidden_dim),
        })

        # Camadas de message passing
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            conv = HeteroConv(
                {
                    ("robot",   "at",        "mapnode"): SAGEConv(hidden_dim, hidden_dim),
                    ("box",     "at",        "mapnode"): SAGEConv(hidden_dim, hidden_dim),
                    ("robot",   "carries",   "box"):     SAGEConv(hidden_dim, hidden_dim),
                    ("mapnode", "has_robot", "robot"):   SAGEConv(hidden_dim, hidden_dim),
                    ("box",     "carried_by","robot"):   SAGEConv(hidden_dim, hidden_dim),
                    ("box",     "next_wp",   "mapnode"): SAGEConv(hidden_dim, hidden_dim),
                    ("mapnode", "connected", "mapnode"): SAGEConv(hidden_dim, hidden_dim),
                },
                aggr="sum",
            )
            self.convs.append(conv)

        # Layer norm por tipo de nó (estabiliza o treino)
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                "robot":   nn.LayerNorm(hidden_dim),
                "box":     nn.LayerNorm(hidden_dim),
                "mapnode": nn.LayerNorm(hidden_dim),
            })
            for _ in range(n_layers)
        ])

    def forward(self, data: HeteroData) -> dict[str, Tensor]:
        """
        Devolve embeddings por tipo de nó após message passing.

        Devolve
        -------
        dict com chaves "robot", "box", "mapnode"
        cada tensor com shape [n_nodes, hidden_dim]
        """
        # Projecção inicial
        x_dict: dict[str, Tensor] = {
            ntype: F.relu(self.input_proj[ntype](data[ntype].x))
            for ntype in ("robot", "box", "mapnode")
            if data[ntype].x is not None and data[ntype].x.shape[0] > 0
        }

        # Garante que todos os tipos existem (mesmo que vazios)
        for ntype in ("robot", "box", "mapnode"):
            if ntype not in x_dict:
                x_dict[ntype] = torch.zeros(
                    0, self.hidden_dim,
                    device=next(self.parameters()).device,
                )

        # Message passing
        edge_index_dict = {
            etype: data[etype].edge_index
            for etype in EDGE_TYPES
            if hasattr(data[etype[0], etype[1], etype[2]], "edge_index")
        }

        for i, conv in enumerate(self.convs):
            # Filtra edge_types que têm arestas reais
            valid_edge_index = {
                etype: ei
                for etype, ei in edge_index_dict.items()
                if ei.shape[1] > 0
                and etype[0] in x_dict
                and etype[2] in x_dict
            }

            if valid_edge_index:
                out_dict = conv(x_dict, valid_edge_index)
                # Residual + norm + dropout
                for ntype in out_dict:
                    if ntype in x_dict and x_dict[ntype].shape[0] > 0:
                        h = out_dict[ntype] + x_dict[ntype]
                        h = self.norms[i][ntype](h)
                        h = F.dropout(h, p=self.dropout, training=self.training)
                        x_dict[ntype] = h

        return x_dict


# ---------------------------------------------------------------------------
# Actor — AssignmentHead
# ---------------------------------------------------------------------------

class AssignmentHead(nn.Module):
    """
    Cabeça de actor para decisão sequencial de assignments.

    Para cada robot pendente, calcula scores sobre:
      - caixas disponíveis (não yet assignadas neste step)
      - nós de pre-posição

    Score: v · tanh(W_r · h_robot + W_c · h_candidate)
    (additive attention — Bahdanau style)

    Parâmetros
    ----------
    hidden_dim  : dimensão dos embeddings do GNN
    """

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()

        self.W_robot     = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_candidate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v           = nn.Linear(hidden_dim, 1, bias=False)

        # Embedding aprendido para "idle" (robot não faz nada)
        # Permite ao agente optar por não assignar um robot
        self.idle_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.01)

    def forward(
        self,
        h_dict       : dict[str, Tensor],
        data         : HeteroData,
        assigned_boxes: set[int] | None = None,
    ) -> list[RobotDecision]:
        """
        Calcula decisões para todos os robots pendentes.

        Parâmetros
        ----------
        h_dict         : embeddings por tipo de nó (output do GNN)
        data           : HeteroData com metadados de acção
        assigned_boxes : box_ids já assignados neste step (masking sequencial)

        Devolve
        -------
        list[RobotDecision] — uma por robot pendente, na ordem de
        data.pending_robot_indices
        """
        if assigned_boxes is None:
            assigned_boxes = set()

        h_robot   = h_dict.get("robot")
        h_box     = h_dict.get("box")
        h_mapnode = h_dict.get("mapnode")

        decisions: list[RobotDecision] = []

        pending_indices     = getattr(data, "pending_robot_indices", [])
        available_box_idxs  = getattr(data, "available_box_indices", [])
        preposition_idxs    = getattr(data, "preposition_node_indices", [])
        idx_to_box_id       = getattr(data, "idx_to_box_id", [])
        idx_to_mapnode_id   = getattr(data, "idx_to_mapnode_id", [])

        for robot_idx in pending_indices:
            if h_robot is None or robot_idx >= h_robot.shape[0]:
                continue

            h_r = self.W_robot(h_robot[robot_idx])  # [hidden_dim]

            candidate_embeddings : list[Tensor]      = []
            candidate_keys       : list[CandidateKey] = []

            # --- Caixas disponíveis (não assignadas ainda) ---
            for box_idx in available_box_idxs:
                if box_idx >= len(idx_to_box_id):
                    continue
                box_id = idx_to_box_id[box_idx]
                if box_id in assigned_boxes:
                    continue  # masking sequencial
                if h_box is not None and box_idx < h_box.shape[0]:
                    candidate_embeddings.append(self.W_candidate(h_box[box_idx]))
                    candidate_keys.append(("box", box_id))

            # --- Nós de pre-posição ---
            for node_idx in preposition_idxs:
                if node_idx >= len(idx_to_mapnode_id):
                    continue
                node_id = idx_to_mapnode_id[node_idx]
                if h_mapnode is not None and node_idx < h_mapnode.shape[0]:
                    candidate_embeddings.append(self.W_candidate(h_mapnode[node_idx]))
                    candidate_keys.append(("node", node_id))

            # --- Opção idle ---
            # Sempre presente como safety net (caso todas as outras opções
            # sejam consumidas por robots anteriores na mesma decisão).
            # O masking de idle quando há trabalho útil é aplicado fora,
            # nas funções de sampling/argmax.
            candidate_embeddings.append(self.W_candidate(self.idle_embedding))
            candidate_keys.append(("idle", -1))

            if not candidate_embeddings:
                continue

            # Stack e score
            h_candidates = torch.stack(candidate_embeddings, dim=0)  # [n_cand, hidden_dim]
            h_r_expanded = h_r.unsqueeze(0).expand_as(h_candidates)  # [n_cand, hidden_dim]

            combined = torch.tanh(h_r_expanded + h_candidates)       # [n_cand, hidden_dim]
            logits   = self.v(combined).squeeze(-1)                   # [n_cand]

            decisions.append(RobotDecision(
                robot_idx  = robot_idx,
                logits     = logits,
                candidates = candidate_keys,
            ))

        return decisions


# ---------------------------------------------------------------------------
# Critic — GlobalValueHead
# ---------------------------------------------------------------------------

class GlobalValueHead(nn.Module):
    """
    Cabeça de critic — estima V(s) a partir do estado global.

    Usa mean pooling sobre todos os embeddings de nós (robot + box + mapnode)
    seguido de um MLP.

    Parâmetros
    ----------
    hidden_dim : dimensão dos embeddings do GNN
    """

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()

        # Projecção para dimensão comum antes do pooling
        self.proj_robot   = nn.Linear(hidden_dim, hidden_dim)
        self.proj_box     = nn.Linear(hidden_dim, hidden_dim)
        self.proj_mapnode = nn.Linear(hidden_dim, hidden_dim)

        # MLP sobre o embedding global
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_dict: dict[str, Tensor]) -> Tensor:
        """
        Estima V(s).

        Devolve
        -------
        Tensor de shape [] (escalar)
        """
        pools: list[Tensor] = []

        for ntype, proj in (
            ("robot",   self.proj_robot),
            ("box",     self.proj_box),
            ("mapnode", self.proj_mapnode),
        ):
            h = h_dict.get(ntype)
            if h is not None and h.shape[0] > 0:
                pools.append(proj(h).mean(dim=0))  # [hidden_dim]

        if not pools:
            device = next(self.parameters()).device
            return torch.zeros(1, device=device)

        # Média sobre os três tipos de nó
        h_global = torch.stack(pools, dim=0).mean(dim=0)  # [hidden_dim]
        return self.mlp(h_global).squeeze(-1)              # escalar


# ---------------------------------------------------------------------------
# ActorCritic — módulo principal
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Módulo principal do agente RL.

    Combina GNN + AssignmentHead (actor) + GlobalValueHead (critic).

    Parâmetros
    ----------
    hidden_dim : dimensão dos embeddings (recomendado: 128 ou 256)
    n_layers   : número de camadas GNN (recomendado: 2 ou 3)
    dropout    : dropout entre camadas GNN
    """

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

    def encode(self, data: HeteroData) -> dict[str, Tensor]:
        """Corre o GNN e devolve embeddings. Útil para reutilizar em actor + critic."""
        return self.gnn(data)

    def actor_forward(
        self,
        data           : HeteroData,
        assigned_boxes : set[int] | None = None,
    ) -> list[RobotDecision]:
        """
        Decisões do actor para todos os robots pendentes.

        Parâmetros
        ----------
        data           : HeteroData com o estado actual
        assigned_boxes : caixas já assignadas neste step (masking sequencial)

        Devolve
        -------
        list[RobotDecision]
        """
        h_dict = self.encode(data)
        return self.actor(h_dict, data, assigned_boxes)

    def critic_forward(self, data: HeteroData) -> Tensor:
        """
        Estimativa do valor do estado V(s).

        Devolve
        -------
        Tensor escalar
        """
        h_dict = self.encode(data)
        return self.critic(h_dict)

    def forward(
        self,
        data           : HeteroData,
        assigned_boxes : set[int] | None = None,
    ) -> tuple[list[RobotDecision], Tensor]:
        """
        Forward pass completo — actor + critic com GNN partilhado.

        Reutiliza os embeddings do GNN para ambas as cabeças.

        Devolve
        -------
        (decisions, value)
          decisions : list[RobotDecision]
          value     : Tensor escalar V(s)
        """
        h_dict    = self.encode(data)
        decisions = self.actor(h_dict, data, assigned_boxes)
        value     = self.critic(h_dict)
        return decisions, value


# ---------------------------------------------------------------------------
# Helpers de sampling e log_prob  (usados pelo PPO)
# ---------------------------------------------------------------------------

def sample_actions(
    decisions: list[RobotDecision],
) -> tuple[list[tuple[int, int]], Tensor]:
    """
    Amostra uma acção para cada robot pendente (sem masking sequencial).

    Devolve
    -------
    actions   : list[(robot_idx, candidate_idx)] — índice do candidato escolhido
    log_probs : Tensor [n_pending] — log probabilidade de cada acção
    """
    actions   : list[tuple[int, int]] = []
    log_probs : list[Tensor]          = []

    for decision in decisions:
        dist     = torch.distributions.Categorical(logits=decision.logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)

        actions.append((decision.robot_idx, int(action.item())))
        log_probs.append(log_prob)

    if not log_probs:
        return actions, torch.zeros(0)

    return actions, torch.stack(log_probs)


def _apply_idle_mask(
    logits     : Tensor,
    candidates : list[CandidateKey],
) -> Tensor:
    """
    Mascara a opção idle (-inf) se existir pelo menos um candidato não-idle
    com logit válido (não -inf). Idle só permanece disponível quando o robot
    genuinamente não tem trabalho — evita robots a bloquearem o mapa em IDLE
    com caixas por entregar.
    """
    has_non_idle_valid = False
    for j, (ctype, _) in enumerate(candidates):
        if ctype != "idle" and logits[j].item() != float("-inf"):
            has_non_idle_valid = True
            break

    if not has_non_idle_valid:
        return logits

    for j, (ctype, _) in enumerate(candidates):
        if ctype == "idle":
            logits[j] = float("-inf")
    return logits


def sample_actions_sequential(
    decisions: list[RobotDecision],
) -> tuple[list[tuple[int, int]], Tensor]:
    """
    Amostra acções sequencialmente, maskando caixas e nós já assignados.

    Cada robot vê apenas as caixas/nós que nenhum robot anterior escolheu
    neste step. Garante que dois robots nunca concorrem pelo mesmo alvo.

    Devolve
    -------
    actions   : list[(robot_idx, candidate_idx)]
    log_probs : Tensor [n_pending]
    """
    assigned_boxes: set[int] = set()
    assigned_nodes: set[str] = set()
    actions   : list[tuple[int, int]] = []
    log_probs : list[Tensor]          = []

    for decision in decisions:
        # Máscara: põe -inf nos alvos já assignados
        logits = decision.logits.clone()
        for j, (ctype, cid) in enumerate(decision.candidates):
            if ctype == "box" and int(cid) in assigned_boxes:
                logits[j] = float("-inf")
            elif ctype == "node" and str(cid) in assigned_nodes:
                logits[j] = float("-inf")

        logits = _apply_idle_mask(logits, decision.candidates)

        dist     = torch.distributions.Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)

        actions.append((decision.robot_idx, int(action.item())))
        log_probs.append(log_prob)

        # Regista o alvo escolhido para masking dos robots seguintes
        chosen = int(action.item())
        if chosen < len(decision.candidates):
            ctype, cid = decision.candidates[chosen]
            if ctype == "box":
                assigned_boxes.add(int(cid))
            elif ctype == "node":
                assigned_nodes.add(str(cid))

    if not log_probs:
        return actions, torch.zeros(0)

    return actions, torch.stack(log_probs)


def compute_log_probs(
    decisions    : list[RobotDecision],
    action_indices: list[int],
) -> Tensor:
    """
    Calcula log_prob de acções já tomadas (para PPO ratio, sem masking sequencial).

    Parâmetros
    ----------
    decisions      : list[RobotDecision] do forward pass actual
    action_indices : índices das acções originais (mesma ordem que decisions)

    Devolve
    -------
    Tensor [n_pending]
    """
    log_probs: list[Tensor] = []

    for decision, action_idx in zip(decisions, action_indices):
        dist     = torch.distributions.Categorical(logits=decision.logits)
        log_prob = dist.log_prob(
            torch.tensor(action_idx, device=decision.logits.device)
        )
        log_probs.append(log_prob)

    if not log_probs:
        return torch.zeros(0)

    return torch.stack(log_probs)


def compute_log_probs_sequential(
    decisions       : list[RobotDecision],
    action_indices  : list[int],
    candidate_lists : list[list[CandidateKey]],
) -> Tensor:
    """
    Recalcula log_probs com o mesmo masking sequencial usado em sample_actions_sequential.

    Usa candidate_lists (candidatos originais do sampling) para reconstruir
    o estado de assigned_boxes passo a passo, garantindo que old_log_probs
    e new_log_probs são computados na mesma distribuição mascarada.

    Parâmetros
    ----------
    decisions        : list[RobotDecision] do forward pass actual (pesos actualizados)
    action_indices   : índices das acções originais (mesma ordem que decisions)
    candidate_lists  : candidatos tal como estavam durante o sampling original

    Devolve
    -------
    Tensor [n_pending]
    """
    assigned_boxes: set[int] = set()
    assigned_nodes: set[str] = set()
    log_probs: list[Tensor] = []

    for decision, action_idx, orig_candidates in zip(
        decisions, action_indices, candidate_lists
    ):
        logits = decision.logits.clone()
        for j, (ctype, cid) in enumerate(decision.candidates):
            if ctype == "box" and int(cid) in assigned_boxes:
                logits[j] = float("-inf")
            elif ctype == "node" and str(cid) in assigned_nodes:
                logits[j] = float("-inf")

        logits = _apply_idle_mask(logits, decision.candidates)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs.append(
            dist.log_prob(torch.tensor(action_idx, device=logits.device))
        )

        # Replica o estado do masking usando os candidatos originais
        if action_idx < len(orig_candidates):
            ctype, cid = orig_candidates[action_idx]
            if ctype == "box":
                assigned_boxes.add(int(cid))
            elif ctype == "node":
                assigned_nodes.add(str(cid))

    if not log_probs:
        return torch.zeros(0)

    return torch.stack(log_probs)


def compute_entropy(decisions: list[RobotDecision]) -> Tensor:
    """
    Calcula a entropia média da política (para o bonus de exploração do PPO).

    Devolve
    -------
    Tensor escalar
    """
    entropies: list[Tensor] = []

    for decision in decisions:
        probs   = F.softmax(decision.logits, dim=-1)
        dist    = torch.distributions.Categorical(probs)
        entropies.append(dist.entropy())

    if not entropies:
        return torch.zeros(1)

    return torch.stack(entropies).mean()


def decode_actions(
    decisions    : list[RobotDecision],
    action_indices: list[int],
    data         : HeteroData,
) -> dict[str, int | str | None]:
    """
    Converte índices de acção em assignments {robot_id: target}.

    target = box_id (int)   se a acção é apanhar uma caixa
    target = node_id (str)  se a acção é ir para um nó de pre-posição
    target = None           se a acção é idle

    Parâmetros
    ----------
    decisions      : list[RobotDecision]
    action_indices : índice do candidato escolhido por cada robot
    data           : HeteroData com mapeamentos de índices

    Devolve
    -------
    dict {robot_id: target}
    """
    idx_to_robot_id = getattr(data, "idx_to_robot_id", [])
    assignments: dict[str, int | str | None] = {}

    for decision, action_idx in zip(decisions, action_indices):
        robot_idx = decision.robot_idx
        if robot_idx >= len(idx_to_robot_id):
            continue

        robot_id = idx_to_robot_id[robot_idx]

        if action_idx >= len(decision.candidates):
            assignments[robot_id] = None
            continue

        cand_type, cand_id = decision.candidates[action_idx]

        if cand_type == "box":
            assignments[robot_id] = int(cand_id)
        elif cand_type == "node":
            assignments[robot_id] = str(cand_id)
        else:
            # idle
            assignments[robot_id] = None

    return assignments