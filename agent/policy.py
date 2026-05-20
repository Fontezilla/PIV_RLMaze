"""
agents/gnn_ppo/policy.py
~~~~~~~~~~~~~~~~~~~~~~~~
Camada de política — liga o ActorCritic ao FactoryEnv.

Responsabilidades
-----------------
  - Recebe state dict do FactoryEnv
  - Constrói HeteroGraph (via hetero_graph.py)
  - Corre o ActorCritic (GNN + Attention)
  - Sampling sequencial com masking de caixas já assignadas
  - Devolve assignments {robot_id: target} para o FactoryEnv
  - Guarda trajectória (transitions) para o PPO update

Trajectória
-----------
  Cada "transition" corresponde a um step do FactoryEnv
  (que pode cobrir múltiplos ticks internos).

  Campos guardados por transition:
    graph        : HeteroData — estado no momento da decisão
    action_indices : list[int] — índice do candidato escolhido por robot
    log_probs    : Tensor [n_pending] — log π(a|s)
    value        : Tensor escalar — V(s)
    reward       : float — reward acumulado até ao próximo evento
    done         : bool — terminated ou truncated

Uso típico
----------
  policy = Policy(hidden_dim=128, n_layers=2)

  state, info = env.reset()
  done = False

  while not done:
      assignments, transition = policy.act(state)
      state, reward, terminated, truncated, info = env.step(assignments)
      done = terminated or truncated
      policy.record(transition, reward=reward, done=done)

  policy.update(optimizer, gamma=0.99, lam=0.95, clip_eps=0.2)
  policy.clear()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData

from agent.actor_critic import (
    ActorCritic,
    CandidateKey,
    RobotDecision,
    compute_entropy,
    compute_log_probs_sequential,
    decode_actions,
    sample_actions_sequential,
)
from agent.hetero_graph import build_hetero_graph


# ---------------------------------------------------------------------------
# Transition — um passo da trajectória
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """
    Dados de um step guardados para o PPO update.

    pending_robot_indices e action_indices têm o mesmo comprimento
    e a mesma ordem — o i-ésimo robot tomou a i-ésima acção.
    """
    graph                 : HeteroData      # estado no momento da decisão
    pending_robot_indices : list[int]       # robots que decidiram neste step
    action_indices        : list[int]       # índice do candidato escolhido
    candidate_lists       : list[list[CandidateKey]]  # candidatos por robot
    log_probs             : Tensor          # [n_pending]
    value                 : Tensor          # escalar V(s)
    reward                : float = 0.0    # preenchido depois pelo record()
    done                  : bool  = False  # preenchido depois pelo record()


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class Policy:
    """
    Política GNN + Attention + PPO para gestão de fábrica.

    Parâmetros
    ----------
    hidden_dim : dimensão dos embeddings GNN
    n_layers   : camadas de message passing
    dropout    : dropout entre camadas GNN
    device     : "cpu" ou "cuda"
    """

    def __init__(
        self,
        hidden_dim : int   = 128,
        n_layers   : int   = 2,
        dropout    : float = 0.1,
        device     : str   = "cpu",
    ) -> None:
        self.device = torch.device(device)

        self.net = ActorCritic(
            hidden_dim = hidden_dim,
            n_layers   = n_layers,
            dropout    = dropout,
        ).to(self.device)

        # Buffer de trajectória do episódio actual
        self._trajectory: list[Transition] = []

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def act(
        self,
        state      : dict,
        deterministic: bool = False,
    ) -> tuple[dict[str, int | str | None], Transition]:
        """
        Decide assignments para os robots pendentes.

        Parâmetros
        ----------
        state         : dict devolvido por FactoryEnv._build_state()
        deterministic : se True, escolhe argmax em vez de amostrar
                        (útil para avaliação)

        Devolve
        -------
        assignments : {robot_id: target} para passar ao FactoryEnv.step()
        transition  : Transition com os dados para o PPO (reward e done
                      ainda não preenchidos — usar record() depois)
        """
        self.net.eval()

        with torch.no_grad():
            graph = build_hetero_graph(state)
            graph = graph.to(self.device)

            decisions, value = self.net(graph)

            if not decisions:
                # Nenhum robot pendente — devolve assignments vazios
                empty = Transition(
                    graph                 = graph,
                    pending_robot_indices = [],
                    action_indices        = [],
                    candidate_lists       = [],
                    log_probs             = torch.zeros(0, device=self.device),
                    value                 = value,
                )
                return {}, empty

            if deterministic:
                action_indices, log_probs = _argmax_actions(decisions)
            else:
                action_tuples, log_probs = sample_actions_sequential(decisions)
                action_indices = [a[1] for a in action_tuples]
                log_probs      = log_probs.to(self.device)

            assignments = decode_actions(
                decisions,
                action_indices,
                graph,
            )

        transition = Transition(
            graph                 = graph,
            pending_robot_indices = [d.robot_idx for d in decisions],
            action_indices        = action_indices,
            candidate_lists       = [d.candidates for d in decisions],
            log_probs             = log_probs,
            value                 = value,
        )

        return assignments, transition

    def record(
        self,
        transition : Transition,
        reward     : float,
        done       : bool,
    ) -> None:
        """
        Preenche reward e done na transition e guarda na trajectória.

        Deve ser chamado após env.step() devolver o reward.
        """
        transition.reward = reward
        transition.done   = done
        self._trajectory.append(transition)

    def clear(self) -> None:
        """Limpa o buffer de trajectória."""
        self._trajectory.clear()

    @property
    def trajectory(self) -> list[Transition]:
        return list(self._trajectory)

    # ------------------------------------------------------------------
    # PPO Update
    # ------------------------------------------------------------------

    def update(
        self,
        optimizer  : torch.optim.Optimizer,
        gamma      : float = 0.99,
        lam        : float = 0.95,
        clip_eps   : float = 0.2,
        value_coef : float = 0.5,
        entropy_coef: float = 0.01,
        n_epochs   : int   = 4,
        normalize_advantages: bool = True,
        trajectory : list | None = None,
    ) -> dict[str, float]:
        """
        Actualiza os pesos da rede com PPO.

        Parâmetros
        ----------
        optimizer   : optimizador (ex: Adam)
        gamma       : factor de desconto
        lam         : parâmetro GAE (lambda)
        clip_eps    : clipping do ratio PPO
        value_coef  : peso da loss do critic
        entropy_coef: peso do bonus de entropia
        n_epochs    : épocas de update por trajectória
        normalize_advantages : normaliza vantagens (recomendado)

        Devolve
        -------
        dict com métricas de treino:
          policy_loss, value_loss, entropy, total_loss, approx_kl
        """
        traj = trajectory if trajectory is not None else self._trajectory
        if not traj:
            return {}

        self.net.train()

        # ------------------------------------------------------------------
        # 1. Calcula retornos e vantagens (GAE)
        # ------------------------------------------------------------------
        returns, advantages = self._compute_gae(gamma, lam, traj)

        advantages_t = torch.stack(advantages)
        if normalize_advantages and advantages_t.std() > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        advantages = list(advantages_t.unbind(0))

        # ------------------------------------------------------------------
        # 2. Recolhe dados antigos (old_log_probs, values)
        # ------------------------------------------------------------------
        old_log_probs_list = [t.log_probs for t in traj]
        old_values_list    = [t.value     for t in traj]

        # ------------------------------------------------------------------
        # 3. Épocas PPO
        # ------------------------------------------------------------------
        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss":  [],
            "entropy":     [],
            "approx_kl":   [],
        }

        for _ in range(n_epochs):
            for i, transition in enumerate(traj):
                if not transition.action_indices:
                    continue

                # Forward pass actual
                graph = transition.graph.to(self.device)
                decisions, value = self.net(graph)

                if not decisions:
                    continue

                # Log probs actuais (com o mesmo masking sequencial do sampling)
                new_log_probs = compute_log_probs_sequential(
                    decisions,
                    transition.action_indices,
                    transition.candidate_lists,
                )

                # Garante que old_log_probs tem o mesmo shape
                old_log_probs = old_log_probs_list[i].to(self.device)
                if old_log_probs.shape != new_log_probs.shape:
                    continue

                # Vantagem para este step (média se múltiplos robots)
                adv = advantages[i].to(self.device)

                # --- Policy loss (PPO clip) ---
                ratio       = torch.exp(new_log_probs - old_log_probs)
                surr1       = ratio * adv
                surr2       = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Value loss ---
                ret         = returns[i].to(self.device)
                value_loss  = F.mse_loss(value, ret)

                # --- Entropy bonus ---
                entropy     = compute_entropy(decisions)

                # --- Total loss ---
                total_loss = (
                    policy_loss
                    + value_coef  * value_loss
                    - entropy_coef * entropy
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                optimizer.step()

                # Métricas
                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean()

                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["approx_kl"].append(approx_kl.item())

        return {k: sum(v) / max(len(v), 1) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def _compute_gae(
        self,
        gamma      : float,
        lam        : float,
        trajectory : list | None = None,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """
        Calcula retornos e vantagens GAE para toda a trajectória.

        Devolve
        -------
        returns    : list[Tensor escalar] — retorno descontado por step
        advantages : list[Tensor escalar] — vantagem GAE por step
        """
        traj       = trajectory if trajectory is not None else self._trajectory
        n          = len(traj)
        returns    : list[Tensor] = [torch.zeros(1)] * n
        advantages : list[Tensor] = [torch.zeros(1)] * n

        gae = 0.0
        next_value = 0.0

        for i in reversed(range(n)):
            t          = traj[i]
            reward     = t.reward
            value      = t.value.detach().item()
            done       = t.done

            # TD error
            delta = reward + gamma * next_value * (1 - float(done)) - value

            # GAE
            gae   = delta + gamma * lam * (1 - float(done)) * gae

            returns[i]    = torch.tensor(value + gae, dtype=torch.float32)
            advantages[i] = torch.tensor(gae,         dtype=torch.float32)

            next_value = value

        return returns, advantages

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Guarda os pesos da rede."""
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        """Carrega os pesos da rede."""
        self.net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )

    def parameters(self):
        return self.net.parameters()


# ---------------------------------------------------------------------------
# Helper — acções determinísticas (argmax)
# ---------------------------------------------------------------------------

def _argmax_actions(
    decisions: list[RobotDecision],
) -> tuple[list[int], Tensor]:
    """
    Escolhe deterministicamente a acção de maior score para cada robot,
    com masking sequencial de caixas já assignadas.

    Devolve
    -------
    action_indices : list[int]
    log_probs      : Tensor [n_pending]
    """
    assigned_boxes: set[int]   = set()
    assigned_nodes: set[str]   = set()
    action_indices: list[int]  = []
    log_probs     : list[Tensor] = []

    for decision in decisions:
        logits = decision.logits.clone()
        for j, (ctype, cid) in enumerate(decision.candidates):
            if ctype == "box" and int(cid) in assigned_boxes:
                logits[j] = float("-inf")
            elif ctype == "node" and str(cid) in assigned_nodes:
                logits[j] = float("-inf")

        # Mascara idle se houver alternativas válidas (não bloqueadas).
        has_non_idle_valid = any(
            decision.candidates[j][0] != "idle" and logits[j].item() != float("-inf")
            for j in range(len(decision.candidates))
        )
        if has_non_idle_valid:
            for j, (ctype, _) in enumerate(decision.candidates):
                if ctype == "idle":
                    logits[j] = float("-inf")

        dist     = torch.distributions.Categorical(logits=logits)
        action   = int(logits.argmax().item())
        log_prob = dist.log_prob(torch.tensor(action, device=logits.device))

        action_indices.append(action)
        log_probs.append(log_prob)

        if action < len(decision.candidates):
            ctype, cid = decision.candidates[action]
            if ctype == "box":
                assigned_boxes.add(int(cid))
            elif ctype == "node":
                assigned_nodes.add(str(cid))

    if not log_probs:
        return action_indices, torch.zeros(0)

    return action_indices, torch.stack(log_probs)