"""
observation_builder.py — Observação local do MotionAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from simulation_engine.core.entities import RobotState
from simulation_engine.core.world_state import WorldState
from simulation_engine.core.graph import FactoryGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bin_distance(dist: float | None) -> int:
    if dist is None:
        return 0
    if dist < 5:
        return 1
    elif dist < 15:
        return 2
    else:
        return 3


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class MotionObs:
    robot_id: str
    state:    RobotState

    # Posição
    current_node:  Optional[str] = None
    prev_node:     Optional[str] = None
    from_node:     Optional[str] = None
    to_node:       Optional[str] = None
    progress:      float         = 0.0
    speed:         float         = 0.0
    docking_exit:  bool          = False

    # Cooldowns
    yield_ticks:   int = 0
    turn_cooldown: int = 0

    # Tráfego local — agregados por aresta/nó
    neighbors:        List[str]        = field(default_factory=list)
    edge_robots:      Dict[str, int]   = field(default_factory=dict)
    edge_opposing:    Dict[str, bool]  = field(default_factory=dict)
    neighbor_idle:    Dict[str, int]   = field(default_factory=dict)
    neighbor_special: Dict[str, bool]  = field(default_factory=dict)
    neighbor_is_leaf: Dict[str, bool]  = field(default_factory=dict)

    dist_ahead_bin:   int = 0
    min_adjacent_bin: int = 0

    # Q_velocity
    lead_gap:          float = 1.0
    converging_robots: int   = 0
    to_node_occupied:  bool  = False

    # Carga
    carried_box: Optional[str] = None

    # Anti-starvation
    idle_stuck_ticks: int = 0

    # Colisão
    collided: bool = False

    # ------------------------------------------------------------------
    # Visibilidade individual de robots vizinhos
    # ------------------------------------------------------------------

    # Aresta própria — robot à frente e atrás (quando MOVING)
    # progress 1.0 / 0.0 = sentinela "nenhum robot nesta direção"
    lead_robot_progress: float = 1.0
    lead_robot_speed:    float = 0.0
    lead_robot_has_box:  bool  = False
    rear_robot_progress: float = 0.0
    rear_robot_speed:    float = 0.0

    # Arestas adjacentes — robot mais próximo de ref por direção
    # chave = nó vizinho nb
    # valor = (progress, speed) do robot mais próximo, ou None se vazio
    #
    # adj_fwd_nearest[nb]: robot mais próximo de ref na direção ref→nb
    #                       (progress baixo = acabou de entrar, perto de nós)
    # adj_opp_nearest[nb]: robot mais próximo de ref na direção nb→ref
    #                       (progress alto = quase a chegar a ref)
    adj_fwd_nearest: Dict[str, Optional[Tuple[float, float]]] = field(default_factory=dict)
    adj_opp_nearest: Dict[str, Optional[Tuple[float, float]]] = field(default_factory=dict)

    # Intenção e carga do robot oposto mais próximo por aresta
    # adj_opp_has_box[nb]:       robot oposto tem caixa?
    # adj_opp_buffered_next[nb]: próximo hop planeado após chegar a ref
    #                             (buffered_next_node do oposto)
    adj_opp_has_box:       Dict[str, bool]          = field(default_factory=dict)
    adj_opp_buffered_next: Dict[str, Optional[str]] = field(default_factory=dict)

    # Visibilidade 2-hop: robots a aproximar-se de nb por direções que não ref
    # robots_targeting_ref: quantos robots têm to_node == ref_node (o nó de origem/atual)
    #   → informa o DQN que recuar para cá vai criar conflito
    # adj_incoming_count[nb]:   robots com to_node==nb e from_node!=ref (outras direções)
    # adj_incoming_nearest[nb]: o robot mais próximo de nb vindo dessas direções
    #                            (progress alto = quase a chegar a nb)
    robots_targeting_ref:  int = 0
    adj_incoming_count:    Dict[str, int]                       = field(default_factory=dict)
    adj_incoming_nearest:  Dict[str, Optional[Tuple[float, float]]] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def can_dispatch(self) -> bool:
        return (
            self.state == RobotState.IDLE
            and self.current_node is not None
            and self.yield_ticks == 0
            and self.turn_cooldown == 0
        )

    def is_edge_free(self, neighbor: str) -> bool:
        return self.edge_robots.get(neighbor, 0) == 0

    def has_opposing(self, neighbor: str) -> bool:
        return self.edge_opposing.get(neighbor, False)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_motion_obs(
    world: WorldState,
    graph: FactoryGraph,
    robot_id: str,
) -> MotionObs:

    robot = world.robots[robot_id]

    ref_node = (
        robot.current_node
        if robot.state == RobotState.IDLE
        else robot.from_node
    )

    neighbors:        List[str]       = []
    edge_robots:      Dict[str, int]  = {}
    edge_opposing:    Dict[str, bool] = {}
    neighbor_idle:    Dict[str, int]  = {}
    neighbor_special: Dict[str, bool] = {}
    neighbor_is_leaf: Dict[str, bool] = {}

    if ref_node is not None:
        neighbors = graph.neighbors(ref_node)

        for nb in neighbors:
            edge_robots[nb]      = len(world.robots_on_edge(ref_node, nb))
            edge_opposing[nb]    = world.edge_has_opposing_traffic(ref_node, nb)
            neighbor_idle[nb]    = len(world.robots_at_node(nb))
            neighbor_special[nb] = graph.is_special(nb)
            neighbor_is_leaf[nb] = (len(graph.neighbors(nb)) == 1)

    dist_ahead = None

    if robot.from_node and robot.to_node:
        x1, y1 = graph.node_position(robot.from_node)
        x2, y2 = graph.node_position(robot.to_node)
        edge_len = math.hypot(x2 - x1, y2 - y1)

        for other in world.robots.values():
            if other.id == robot.id:
                continue

            if other.from_node == robot.from_node and other.to_node == robot.to_node:
                if other.progress > robot.progress:
                    d = (other.progress - robot.progress) * edge_len
                    if dist_ahead is None or d < dist_ahead:
                        dist_ahead = d

    dist_ahead_bin = _bin_distance(dist_ahead)
    
    min_adj_dist = None

    if ref_node is not None:
        for n in neighbors:
            for other in world.robots.values():
                if other.id == robot.id:
                    continue

                if other.from_node == ref_node and other.to_node == n:
                    x1, y1 = graph.node_position(ref_node)
                    x2, y2 = graph.node_position(n)
                    edge_len = math.hypot(x2 - x1, y2 - y1)

                    d = other.progress * edge_len

                    if min_adj_dist is None or d < min_adj_dist:
                        min_adj_dist = d

    min_adjacent_bin = _bin_distance(min_adj_dist)

    # ------------------------------------------------------------------
    # Q_velocity + aresta própria
    # ------------------------------------------------------------------
    lead_gap             = 1.0
    converging_robots    = 0
    to_node_occupied     = False
    lead_robot_progress  = 1.0
    lead_robot_speed     = 0.0
    lead_robot_has_box   = False
    rear_robot_progress  = 0.0
    rear_robot_speed     = 0.0

    if robot.state == RobotState.MOVING and robot.from_node and robot.to_node:
        for r in world.robots.values():
            if r.id == robot_id:
                continue
            if r.from_node == robot.from_node and r.to_node == robot.to_node:
                gap = r.progress - robot.progress
                if 0.0 < gap < lead_gap:
                    lead_gap = gap
                # robot mais próximo à frente (maior progress acima do ego)
                if r.progress > robot.progress and r.progress < lead_robot_progress:
                    lead_robot_progress = r.progress
                    lead_robot_speed    = r.speed
                    lead_robot_has_box  = r.carried_box is not None
                # robot mais próximo atrás (menor progress abaixo do ego)
                elif r.progress < robot.progress and r.progress > rear_robot_progress:
                    rear_robot_progress = r.progress
                    rear_robot_speed    = r.speed

        converging_robots = sum(
            1 for r in world.robots.values()
            if r.id != robot_id and r.to_node == robot.to_node and not r.docking_exit
        )

        to_node_occupied = len(world.robots_at_node(robot.to_node)) > 0

    # ------------------------------------------------------------------
    # Arestas adjacentes — robot mais próximo por direção (fwd e opp)
    # ------------------------------------------------------------------
    adj_fwd_nearest:       Dict[str, Optional[Tuple[float, float]]] = {}
    adj_opp_nearest:       Dict[str, Optional[Tuple[float, float]]] = {}
    adj_opp_has_box:       Dict[str, bool]          = {}
    adj_opp_buffered_next: Dict[str, Optional[str]] = {}

    if ref_node is not None:
        for nb in neighbors:
            # Frente: ref_node → nb  (robot com progress mais baixo = mais perto de ref)
            best_fwd: Optional[Tuple[float, float]] = None
            for r in world.robots.values():
                if r.id == robot_id:
                    continue
                if (r.state == RobotState.MOVING
                        and r.from_node == ref_node and r.to_node == nb):
                    if best_fwd is None or r.progress < best_fwd[0]:
                        best_fwd = (r.progress, r.speed)
            adj_fwd_nearest[nb] = best_fwd

            # Oposto: nb → ref_node  (robot com progress mais alto = mais perto de ref)
            best_opp: Optional[Tuple[float, float]] = None
            best_opp_has_box       = False
            best_opp_buffered_next: Optional[str] = None
            for r in world.robots.values():
                if r.id == robot_id:
                    continue
                if (r.state == RobotState.MOVING
                        and r.from_node == nb and r.to_node == ref_node):
                    if best_opp is None or r.progress > best_opp[0]:
                        best_opp               = (r.progress, r.speed)
                        best_opp_has_box       = r.carried_box is not None
                        best_opp_buffered_next = r.buffered_next_node
            adj_opp_nearest[nb]       = best_opp
            adj_opp_has_box[nb]       = best_opp_has_box
            adj_opp_buffered_next[nb] = best_opp_buffered_next

    # ------------------------------------------------------------------
    # 2-hop visibility
    # ------------------------------------------------------------------
    robots_targeting_ref = 0
    adj_incoming_count:   Dict[str, int]                       = {}
    adj_incoming_nearest: Dict[str, Optional[Tuple[float, float]]] = {}

    if ref_node is not None:
        robots_targeting_ref = sum(
            1 for r in world.robots.values()
            if r.id != robot_id
            and r.state == RobotState.MOVING
            and r.to_node == ref_node
        )

        for nb in neighbors:
            incoming = [
                r for r in world.robots.values()
                if r.id != robot_id
                and r.state == RobotState.MOVING
                and r.to_node == nb
                and r.from_node != ref_node  # excluir os que saem de ref (já em adj_fwd)
            ]
            adj_incoming_count[nb] = len(incoming)
            # o mais próximo de nb = maior progress
            best_inc = max(incoming, key=lambda r: r.progress) if incoming else None
            adj_incoming_nearest[nb] = (best_inc.progress, best_inc.speed) if best_inc else None

    # ------------------------------------------------------------------
    return MotionObs(
        robot_id          = robot_id,
        state             = robot.state,
        current_node      = robot.current_node,
        prev_node         = robot.prev_node,
        from_node         = robot.from_node,
        to_node           = robot.to_node,
        progress          = robot.progress,
        speed             = robot.speed,
        docking_exit      = robot.docking_exit,
        yield_ticks       = robot.yield_ticks,
        turn_cooldown     = robot.turn_cooldown,
        neighbors         = neighbors,
        edge_robots       = edge_robots,
        edge_opposing     = edge_opposing,
        neighbor_idle     = neighbor_idle,
        neighbor_special  = neighbor_special,
        neighbor_is_leaf  = neighbor_is_leaf,

        dist_ahead_bin    = dist_ahead_bin,
        min_adjacent_bin  = min_adjacent_bin,

        lead_gap              = lead_gap,
        converging_robots     = converging_robots,
        to_node_occupied      = to_node_occupied,
        carried_box           = robot.carried_box,
        idle_stuck_ticks      = robot.idle_stuck_ticks,
        collided              = robot.collided,

        lead_robot_progress   = lead_robot_progress,
        lead_robot_speed      = lead_robot_speed,
        lead_robot_has_box    = lead_robot_has_box,
        rear_robot_progress   = rear_robot_progress,
        rear_robot_speed      = rear_robot_speed,
        adj_fwd_nearest       = adj_fwd_nearest,
        adj_opp_nearest       = adj_opp_nearest,
        adj_opp_has_box       = adj_opp_has_box,
        adj_opp_buffered_next = adj_opp_buffered_next,

        robots_targeting_ref  = robots_targeting_ref,
        adj_incoming_count    = adj_incoming_count,
        adj_incoming_nearest  = adj_incoming_nearest,
    )