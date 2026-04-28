"""
reward.py — Cálculo de reward para o MotionController / DQNController.

R_route    = α·Δdijkstra_norm - γ·P_colisão - γ·P_congestionamento + arrival_bonus
R_velocity = β·(vel/vel_max)  - γ·P_colisão - δ·P_tick - γ·P_blocked

Onde:
    Δdijkstra_norm = (dist_before - dist_after) / MAP_REF_DIST

    Normalização por MAP_REF_DIST reduz a variância do sinal de compass:
    sem ela, hops de 50 unidades dão R≈20 e hops de 700 dão R≈280,
    tornando o Q-value instável e difícil de aprender.

    arrival_bonus faz com que o terminal state seja consistentemente
    valorizado, independentemente do comprimento da última aresta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from simulation_engine.core.events import Events

# ---------------------------------------------------------------------------
# Pesos e magnitudes
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    alpha:         float = 0.4    # peso do progresso de distância (compass)
    beta:          float = 0.2    # peso do throughput (velocidade normalizada)
    gamma:         float = 0.25   # peso das penalizações (colisão, bloqueio, congestionamento)
    delta:         float = 0.1    # peso da penalização por tick
    arrival_bonus: float = 50.0   # bónus fixo ao chegar ao goal (terminal state)


# Distância de referência para normalizar o sinal compass.
# Valor aproximado da aresta média no mapa da fábrica (~200 u).
# Mantém os reward por hop no intervalo [-1, +1] aprox.
MAP_REF_DIST: float = 200.0

COLLISION_PENALTY:     float = 9.0   # penalização por sobreposição física real
TICK_PENALTY:          float = 0.02
BLOCKED_PENALTY:       float = 1.0   # MOVING mas speed=0
TAILGATE_PENALTY:      float = 0.8   # lead_gap < TAILGATE_THRESHOLD
TAILGATE_THRESHOLD:    float = 0.25  # lead_gap [0,1] abaixo do qual há risco de colisão traseira
NODE_OCCUPIED_PENALTY: float = 1.0   # custo por escolher nó já ocupado

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_collisions(robot_id: str, events: Events) -> int:
    """Número de colisões envolvendo robot_id neste tick."""
    return sum(
        1 for c in events.collisions
        if c.robot_a == robot_id or c.robot_b == robot_id
    )

# ---------------------------------------------------------------------------
# Route reward — calculado na chegada ao próximo nó
# ---------------------------------------------------------------------------

def route_reward(
    dist_before: float,
    dist_after: float,
    n_collisions: int,
    weights: RewardWeights,
    node_occupied: bool = False,
) -> float:
    """
    Reward de routing.

    Parameters
    ----------
    dist_before:
        Distância Dijkstra do nó anterior ao goal.
    dist_after:
        Distância Dijkstra do nó atual ao goal.
    n_collisions:
        Número de colisões acumuladas durante a travessia da aresta.
    node_occupied:
        True se o nó de destino já tinha robot IDLE no momento da escolha.
    """
    dist_before  = max(0.0, float(dist_before))
    dist_after   = max(0.0, float(dist_after))
    n_collisions = max(0, int(n_collisions))

    # Compass normalizado — reduz variância entre hops curtos e longos
    delta_d_norm = (dist_before - dist_after) / MAP_REF_DIST
    r_compass    = weights.alpha * delta_d_norm

    r_collision  = -weights.gamma * COLLISION_PENALTY * n_collisions
    r_congestion = -weights.gamma * NODE_OCCUPIED_PENALTY if node_occupied else 0.0

    # Bónus de chegada: sinal consistente para o terminal state,
    # independente do comprimento da última aresta
    r_arrival = weights.arrival_bonus if dist_after < 1e-6 else 0.0

    return r_compass + r_collision + r_congestion + r_arrival

# ---------------------------------------------------------------------------
# Velocity reward — calculado por tick enquanto MOVING
# ---------------------------------------------------------------------------

def velocity_reward(
    speed: float,
    vel_max: float,
    n_collisions: int,
    weights: RewardWeights,
    is_blocked: bool = False,
    lead_gap: float = 1.0,
) -> float:
    """
    Reward de velocidade.

    Parameters
    ----------
    speed:
        Velocidade atual do robot após aplicar o comando (>= 0).
    vel_max:
        Velocidade máxima usada para normalização.
    n_collisions:
        Número de colisões (sobreposições físicas) envolvendo o robot neste tick.
    is_blocked:
        True se o robot continua MOVING mas com speed == 0.
    lead_gap:
        Gap normalizado [0,1] ao robot da frente na mesma aresta.
        1.0 significa ninguém à frente.
    """
    speed = max(0.0, float(speed))
    vel_max = max(0.0, float(vel_max))
    n_collisions = max(0, int(n_collisions))
    lead_gap = max(0.0, min(1.0, float(lead_gap)))

    speed_ratio = _clip01((speed / vel_max) if vel_max > 0.0 else 0.0)

    r_vel      = weights.beta  * speed_ratio
    r_col      = -weights.gamma * COLLISION_PENALTY * n_collisions
    r_tick     = -weights.delta * TICK_PENALTY
    r_blocked  = -weights.gamma * BLOCKED_PENALTY  if is_blocked else 0.0
    r_tailgate = -weights.gamma * TAILGATE_PENALTY if lead_gap < TAILGATE_THRESHOLD else 0.0

    return r_vel + r_col + r_tick + r_blocked + r_tailgate


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))