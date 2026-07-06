"""Política GNN + PPO standard — interface com o FactoryEnv.

Portado do agente antigo quase sem alterações (PPO/GAE são lógica pura,
agnóstica ao env). A única mudança real: `Transition` ganhou o campo
`order` (a ordem do leilão usada por `apply_sequential` — ver
`agent/actor_critic.py`), que tem de ser guardada em `act_verbose` e
reenviada em `update()` ao recomputar log-probs, para o masking sequencial
do replay corresponder exactamente ao que foi amostrado."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData

from agent.actor_critic import (
    ActorCritic,
    RobotDecision,
    apply_sequential,
    compute_entropy,
    decode_actions,
)
from agent.hetero_graph import build_hetero_graph


@dataclass
class Transition:
    """Dados de um step para o PPO update.

    `terminated`: estado terminal real (e.g. all_delivered). V(s') = 0.
    `truncated` : limite externo (e.g. tick_limit). Bootstrap V(s_T).
    Mutuamente prioritários: se terminated, ignora truncated.
    `order`: ordem do leilão usada em `apply_sequential` ao amostrar esta
    transição — tem de ser reenviada tal-e-qual no update de PPO.
    """
    graph          : HeteroData
    action_indices : list[int]
    log_probs      : Tensor
    value          : Tensor
    order          : list[int] = field(default_factory=list)
    reward         : float = 0.0
    terminated     : bool  = False
    truncated      : bool  = False

    @property
    def done(self) -> bool:
        """Episódio acabou (terminated ou truncated). Para slicing de trajectórias."""
        return self.terminated or self.truncated


class Policy:
    """Política GNN + Attention + PPO."""

    def __init__(
        self,
        hidden_dim: int   = 128,
        n_layers: int   = 2,
        dropout: float = 0.1,
        device: str   = "cpu",
    ) -> None:
        """Cria a rede (`ActorCritic`) no device dado e a trajectória vazia."""
        self.device = torch.device(device)
        self.net    = ActorCritic(
            hidden_dim = hidden_dim,
            n_layers   = n_layers,
            dropout    = dropout,
        ).to(self.device)
        self._trajectory: list[Transition] = []

    def act(
        self,
        state: dict,
        deterministic: bool = False,
    ) -> tuple[dict[str, tuple[int, str] | None], Transition]:
        """Decide assignments e devolve (assignments, transition)."""
        assignments, transition, _ = self.act_verbose(state, deterministic)
        return assignments, transition

    def act_verbose(
        self,
        state: dict,
        deterministic: bool = False,
    ) -> tuple[dict[str, tuple[int, str] | None], Transition, list[RobotDecision]]:
        """Como `act()`, mas devolve também as decisions (logits, candidates)
        para debug/explicação."""
        self.net.eval()
        with torch.no_grad():
            graph = build_hetero_graph(state).to(self.device)
            decisions, value = self.net(graph)

            if not decisions:
                empty = Transition(
                    graph          = graph,
                    action_indices = [],
                    log_probs      = torch.zeros(0, device=self.device),
                    value          = value,
                    order          = [],
                )
                return {}, empty, []

            mode = "argmax" if deterministic else "sample"
            action_indices, log_probs, order = apply_sequential(decisions, mode=mode)
            assignments = decode_actions(decisions, action_indices, graph)

        transition = Transition(
            graph          = graph,
            action_indices = action_indices,
            log_probs      = log_probs,
            value          = value,
            order          = order,
        )
        return assignments, transition, decisions

    def record(
        self,
        transition: Transition,
        reward: float,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        """Anota reward/terminated/truncated e guarda na trajectória."""
        transition.reward     = reward
        transition.terminated = terminated
        transition.truncated  = truncated
        self._trajectory.append(transition)

    def clear(self) -> None:
        """Esvazia a trajectória acumulada."""
        self._trajectory.clear()

    @property
    def trajectory(self) -> list[Transition]:
        """Cópia da trajectória acumulada até agora."""
        return list(self._trajectory)

    def update(
        self,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_epochs: int   = 4,
        normalize_advantages: bool  = True,
        max_grad_norm: float = 0.5,
        trajectory: list | None = None,
    ) -> dict[str, float]:
        """PPO standard com batch agregado (1 backward por epoch)."""
        traj = trajectory if trajectory is not None else self._trajectory
        if not traj:
            return {}

        self.net.train()

        returns, advantages = self._compute_gae(gamma, lam, traj)

        if normalize_advantages and len(advantages) > 1:
            adv_t = torch.stack(advantages)
            if adv_t.std() > 1e-8:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            advantages = list(adv_t.unbind(0))

        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss":  [],
            "entropy":     [],
            "approx_kl":   [],
        }

        for _ in range(n_epochs):
            optimizer.zero_grad()

            policy_losses: list[Tensor] = []
            value_losses : list[Tensor] = []
            entropies    : list[Tensor] = []
            kl_running   : list[float]  = []

            for i, t in enumerate(traj):
                if not t.action_indices:
                    continue

                graph            = t.graph.to(self.device)
                decisions, value = self.net(graph)

                if not decisions or len(decisions) != len(t.action_indices):
                    continue

                _, new_log_probs, _ = apply_sequential(
                    decisions,
                    mode           = "log_prob",
                    action_indices = t.action_indices,
                    order          = t.order,
                )
                old_log_probs = t.log_probs.to(self.device)
                if old_log_probs.shape != new_log_probs.shape:
                    continue

                adv = advantages[i].to(self.device)
                ret = returns[i].to(self.device)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv

                policy_losses.append(-torch.min(surr1, surr2).mean())
                value_losses.append(F.mse_loss(value, ret))
                entropies.append(compute_entropy(decisions))

                with torch.no_grad():
                    kl_running.append((old_log_probs - new_log_probs).mean().item())

            if not policy_losses:
                continue

            policy_loss = torch.stack(policy_losses).mean()
            value_loss  = torch.stack(value_losses).mean()
            entropy     = torch.stack(entropies).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(entropy.item())
            metrics["approx_kl"].append(sum(kl_running) / max(len(kl_running), 1))

        return {k: sum(v) / max(len(v), 1) for k, v in metrics.items()}

    def _compute_gae(
        self,
        gamma: float,
        lam: float,
        traj: list[Transition],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """GAE com fronteiras por episódio (done ou truncated). Bootstrap em truncated."""
        n = len(traj)
        returns   : list[Tensor] = [torch.zeros(1) for _ in range(n)]
        advantages: list[Tensor] = [torch.zeros(1) for _ in range(n)]

        boundaries: list[tuple[int, int]] = []
        start = 0
        for i, t in enumerate(traj):
            if t.done:
                boundaries.append((start, i))
                start = i + 1
        if start < n:
            boundaries.append((start, n - 1))

        for s, e in boundaries:
            last_t = traj[e]

            if last_t.terminated:
                next_value = 0.0
            else:
                self.net.eval()
                with torch.no_grad():
                    _, v = self.net(last_t.graph.to(self.device))
                    next_value = float(v.item())
                self.net.train()

            gae = 0.0
            for i in range(e, s - 1, -1):
                t     = traj[i]
                value = float(t.value.detach().item())

                delta = t.reward + gamma * next_value - value
                gae   = delta + gamma * lam * gae

                returns[i]    = torch.tensor(value + gae, dtype=torch.float32)
                advantages[i] = torch.tensor(gae,         dtype=torch.float32)

                next_value = value

        return returns, advantages

    def save(self, path: str) -> None:
        """Grava os pesos da rede em `path`."""
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        """Carrega pesos da rede a partir de `path`."""
        self.net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )

    def parameters(self):
        """Parâmetros da rede (para o optimizer)."""
        return self.net.parameters()
