"""
dqn_network.py — Rede DQN Dueling + Double, replay buffer e agente de treino.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self._buf = deque(maxlen=capacity)

    def push(
        self,
        obs:       np.ndarray,
        action:    int,
        reward:    float,
        next_obs:  np.ndarray,
        next_mask: np.ndarray,
        done:      bool,
    ) -> None:
        self._buf.append((obs, action, reward, next_obs, next_mask, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        obs, actions, rewards, next_obs, next_masks, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs),        dtype=torch.float32),
            torch.tensor(actions,              dtype=torch.long),
            torch.tensor(rewards,              dtype=torch.float32),
            torch.tensor(np.array(next_obs),   dtype=torch.float32),
            torch.tensor(np.array(next_masks), dtype=torch.bool),
            torch.tensor(dones,                dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Dueling DQN
# ---------------------------------------------------------------------------

class DuelingDQN(nn.Module):
    def __init__(
        self,
        obs_dim:   int,
        n_actions: int,
        hidden:    Tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        self.shared = nn.Sequential(*layers)

        stream_dim = max(64, in_dim // 2)
        self.val_stream = nn.Sequential(
            nn.Linear(in_dim, stream_dim), nn.ReLU(),
            nn.Linear(stream_dim, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(in_dim, stream_dim), nn.ReLU(),
            nn.Linear(stream_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.shared(x)
        val  = self.val_stream(feat)                          # (B, 1)
        adv  = self.adv_stream(feat)                          # (B, A)
        return val + adv - adv.mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# DQN Agent (Double DQN + target network)
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Agente DQN genérico (Double DQN + Dueling) para espaços de ação discretos.

    Parâmetros
    ----------
    obs_dim           — dimensão do vetor de observação
    n_actions         — número total de ações (incluindo as mascaradas)
    hidden            — tamanhos das camadas partilhadas
    lr                — learning rate Adam
    gamma             — fator de desconto
    buffer_capacity   — capacidade máxima do replay buffer
    batch_size        — amostras por passo de treino
    target_update_freq— passos entre cópias online→target
    train_freq        — passos entre chamadas a _train_step
    replay_start      — mínimo de experiências antes de treinar
    device            — "cpu" ou "cuda"
    """

    def __init__(
        self,
        obs_dim:            int,
        n_actions:          int,
        hidden:             Tuple[int, ...] = (256, 128),
        lr:                 float = 1e-3,
        gamma:              float = 0.95,
        buffer_capacity:    int   = 50_000,
        batch_size:         int   = 128,
        target_update_freq: int   = 500,
        train_freq:         int   = 4,
        replay_start:       int   = 500,
        device:             str   = "cpu",
    ) -> None:
        self.n_actions          = n_actions
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.train_freq         = train_freq
        self.replay_start       = replay_start
        self.device             = torch.device(device)

        self.online = DuelingDQN(obs_dim, n_actions, hidden).to(self.device)
        self.target = DuelingDQN(obs_dim, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)

        self._step_count = 0

    # ------------------------------------------------------------------

    def select_action(
        self,
        obs:     np.ndarray,
        mask:    np.ndarray,   # bool, True = ação válida
        epsilon: float,
    ) -> int:
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return 0

        if random.random() < epsilon:
            return int(random.choice(valid))

        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(t).squeeze(0).cpu().numpy()

        q_masked = np.full(self.n_actions, -np.inf, dtype=np.float32)
        q_masked[mask] = q[mask]
        return int(np.argmax(q_masked))

    def push(
        self,
        obs:       np.ndarray,
        action:    int,
        reward:    float,
        next_obs:  np.ndarray,
        next_mask: np.ndarray,
        done:      bool,
    ) -> None:
        self.buffer.push(obs, action, reward, next_obs, next_mask, done)
        self._step_count += 1

        if (len(self.buffer) >= self.replay_start
                and len(self.buffer) >= self.batch_size
                and self._step_count % self.train_freq == 0):
            self._train_step()

        if self._step_count % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

    def _train_step(self) -> None:
        obs, actions, rewards, next_obs, next_masks, dones = \
            self.buffer.sample(self.batch_size)

        obs        = obs.to(self.device)
        actions    = actions.to(self.device)
        rewards    = rewards.to(self.device)
        next_obs   = next_obs.to(self.device)
        next_masks = next_masks.to(self.device)
        dones      = dones.to(self.device)

        q_vals = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online seleciona a ação, target avalia o valor
            q_next_online = self.online(next_obs)
            q_next_online[~next_masks] = -1e9
            best_actions = q_next_online.argmax(dim=1)
            q_next_target = self.target(next_obs).gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)
            target_vals = rewards + self.gamma * q_next_target * (1.0 - dones)

        loss = F.smooth_l1_loss(q_vals, target_vals)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "online":      self.online.state_dict(),
            "target":      self.target.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "step_count":  self._step_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step_count = ckpt.get("step_count", 0)
