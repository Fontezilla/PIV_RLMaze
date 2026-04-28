from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RobotState(Enum):
    IDLE = "idle"
    MOVING = "moving"


class BoxState(Enum):
    AT_NODE = "at_node"
    IN_TRANSPORT = "in_transport"
    IN_PROCESS = "in_process"
    DELIVERED = "delivered"


@dataclass
class Robot:
    id: str

    state: RobotState = RobotState.IDLE

    # posição lógica (só válido quando IDLE)
    current_node: Optional[str] = None

    # movimento (aresta atual)
    from_node: Optional[str] = None
    to_node: Optional[str] = None
    progress: float = 0.0  # [0, 1]

    # posição contínua no mundo (IMPORTANTE para colisões)
    world_x: float = 0.0
    world_y: float = 0.0

    # velocidade (controlo 1D)
    speed: float = 0.0

    # colisões
    collision_radius: float = 20.0
    collided: bool = False
    collision_ticks: int = 0

    # carga
    carried_box: Optional[str] = None
    reserved_box: Optional[str] = None

    # controlo
    prev_node: Optional[str] = None
    yield_ticks: int = 0

    # Rule 3 — manobra de viragem (ticks restantes antes de poder mover)
    turn_cooldown: int = 0

    # Animação de rotação (sincronizada com turn_cooldown)
    is_turning: bool = False
    rotation_angle_from: float = 0.0   # ângulo inicial (radianos)
    rotation_angle_to: float = 0.0     # ângulo final (radianos)
    rotation_ticks_total: int = 0      # = TURN_TICKS no início da manobra

    # Rule 1 — sinaliza que o robot está a sair de um nó especial de recuo
    # (from_node=junction, to_node=special, progress desce de 1.0 → 0.0)
    docking_exit: bool = False

    # Node buffer
    buffered_next_node: Optional[str] = None

    # Anti-deadlock counters
    # blocked_ticks     — ticks consecutivos MOVING com speed ≤ 0 (e progress ∈ ]0,1[)
    # idle_stuck_ticks  — ticks consecutivos IDLE pronto a despachar mas sem conseguir
    blocked_ticks: int = 0
    idle_stuck_ticks: int = 0


@dataclass
class Box:
    id: str

    pipeline_type: str  # "blue", "green", "red"

    state: BoxState = BoxState.AT_NODE

    current_node: Optional[str] = None
    carried_by: Optional[str] = None

    # pipeline
    next_step_index: int = 0
    destination_node: Optional[str] = None


@dataclass
class Process:
    node_id: str
    busy: bool = False