"""
motion_agent — DQN para navegação de robots.

Exporta:
    DQNController    — controller principal (dqn_route + dqn_vel)
    MotionController — controller tabular legado (compatibilidade)
    RewardWeights    — pesos do reward (ajustáveis)
"""

from agents.motion_agent.dqn_controller import DQNController
from agents.motion_agent.motion_controller import MotionController
from agents.motion_agent.reward import RewardWeights

__all__ = ["DQNController", "MotionController", "RewardWeights"]
