import yaml
from typing import Dict, Optional, Tuple

from simulation_engine.core.world_state import WorldState
from simulation_engine.core.graph import FactoryGraph
from simulation_engine.core.events import Events
from simulation_engine.core.entities import Robot, Box, Process, RobotState

from simulation_engine.systems.dispatch_system import DispatchSystem
from simulation_engine.systems.movement_system import MovementSystem
from simulation_engine.systems.collision_system import CollisionSystem
from simulation_engine.systems.interaction_system import InteractionSystem
from simulation_engine.interface.robot_interface import RobotInterface


_AUTO_ABORT_TICKS = 20   # ticks consecutivos parado a meio → forçar reversão


class SimulationEngine:
    """
    Loop principal do ambiente.

    Ordem de execução por tick:
        1. Dispatch         — aplica ações dos agentes
        2. Collision        — trava aproximações e resolve sobreposição física
        2b. Auto-abort      — força reversão de robots bloqueados há demasiado tempo
        3. Movement         — avança progresso com speeds já validados
        4. Interaction      — pickup, processo, delivery
        5. Buffered dispatch — tenta consumir buffered_next_node se ainda for válido
    """

    def __init__(
        self,
        graph_path: str,
        pipeline_config_path: str,
        vel_max: float = 1.0,
    ):
        self.graph = FactoryGraph(graph_path)

        with open(pipeline_config_path, "r") as f:
            self.pipeline_config = yaml.safe_load(f)

        self.vel_max = vel_max
        self.world = WorldState()

        self.dispatch_system = DispatchSystem(
            self.graph,
            vel_max=vel_max,
            acc_step=vel_max * 0.15,
        )
        self.movement_system = MovementSystem(self.graph)
        self.collision_system = CollisionSystem(self.graph)
        self.interaction_system = InteractionSystem(self.graph, self.pipeline_config)

        # Interface read-only para os agentes
        self.interface = RobotInterface(self.world, self.graph)

    # ------------------------------------------------------------------
    # SETUP HELPERS
    # ------------------------------------------------------------------

    def add_robot(self, robot_id: str, node: str) -> Robot:
        """Cria um robot e coloca-o num nó (estado IDLE)."""
        robot = Robot(id=robot_id, current_node=node, state=RobotState.IDLE)
        robot.world_x, robot.world_y = self.graph.node_position(node)
        self.world.add_robot(robot)
        return robot

    def spawn_box(self, box_id: str, pipeline_type: str, node: str) -> Box:
        """Cria uma box e coloca-a num nó (estado AT_NODE, step 1 — após entry)."""
        box = Box(
            id=box_id,
            pipeline_type=pipeline_type,
            current_node=node,
            next_step_index=1,
        )
        self.world.add_box(box)
        return box

    def add_process(self, node_id: str) -> Process:
        """Regista um nó de processo no world state."""
        process = Process(node_id=node_id)
        self.world.add_process(process)
        return process

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self):
        """
        Limpa o estado do mundo para um novo episódio.
        Após reset, usa add_robot / spawn_box / add_process para repovoar.
        """
        self.world.reset()
        return self.world

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------

    def step(self, actions: Dict) -> Tuple[Events, Dict]: 
        events = Events() 
        # 1. aplicar ações dos agentes 
        dispatch_results = self.dispatch_system.apply(self.world, actions) 
        
        # 2. colisões — trava aproximações e resolve sobreposição ANTES de mover
        self.collision_system.update(self.world, events)

        # colisão invalida qualquer plano curto buffered
        self._invalidate_buffers_on_collision()

        # 2b. auto-abort: forçar reversão de robots bloqueados há demasiado tempo
        self._auto_abort_blocked()

        # 3. movimento físico — avança progresso com speeds já validados pela colisão
        self.movement_system.update(self.world)

        # 4. interações com boxes e processos
        self.interaction_system.update(self.world, events)
        
        # 5. tentar consumir buffered_next_node
        buffered_results = self._consume_buffered_dispatch()
        
        # juntar resultados buffered aos dispatch_results
        dispatch_results.update(buffered_results)
        
        # 6. avançar tempo
        self.world.tick += 1
        
        return events, dispatch_results

    # ------------------------------------------------------------------
    # Anti-deadlock
    # ------------------------------------------------------------------

    def _auto_abort_blocked(self) -> None:
        """
        Se um robot MOVING estiver bloqueado (speed ≤ 0) durante mais de
        _AUTO_ABORT_TICKS ticks consecutivos E ainda estiver na primeira
        metade da aresta (progress < 0.5), força reversão.

        Não actua em docking_exit — esses têm velocidade negativa por design.
        """
        for robot in self.world.robots.values():
            if robot.state != RobotState.MOVING:
                continue
            if robot.docking_exit:
                continue
            if robot.blocked_ticks < _AUTO_ABORT_TICKS:
                continue
            if robot.progress >= 0.5:
                continue

            # Forçar recuo: speed negativa que o MovementSystem vai processar
            robot.speed = -self.dispatch_system.acc_step
            robot.buffered_next_node = None
            robot.blocked_ticks = 0

    # ------------------------------------------------------------------
    # Buffered lookahead
    # ------------------------------------------------------------------

    def _invalidate_buffers_on_collision(self) -> None:
        """
        Colisão torna o plano curto obsoleto.
        """
        for robot in self.world.robots.values():
            if robot.collided:
                robot.buffered_next_node = None

    def _consume_buffered_dispatch(self) -> Dict:
        """
        Tenta auto-despachar robots IDLE com buffered_next_node ainda válido.

        Regras de invalidação (limpa buffer):
        - robot não está IDLE ou não tem nó atual
        - yield_ticks > 0  (robot reverteu — buffer aponta para nó errado)
        - next_node já não é vizinho do nó atual
        - tráfego oposto na aresta destino
        - docking edge destino já ocupada

        Regras de espera (preserva buffer, tenta no próximo tick):
        - turn_cooldown > 0  (curva em curso; após a curva o buffer ainda é válido
                               porque prev_node foi limpo e _needs_turn → False)
        - junction destino ocupada  (esperar que o nó fique livre)
        """
        buffered_actions: Dict[str, Tuple[Optional[str], str]] = {}

        for robot in self.world.robots.values():
            if robot.state != RobotState.IDLE:
                continue

            if robot.current_node is None:
                robot.buffered_next_node = None
                continue

            if robot.buffered_next_node is None:
                continue

            # Reversão → buffer inválido (robot voltou atrás, contexto mudou)
            if robot.yield_ticks > 0:
                robot.buffered_next_node = None
                continue

            # Curva em curso → preservar buffer; será consumido quando turn_cooldown=0
            if robot.turn_cooldown > 0:
                continue

            next_node = robot.buffered_next_node

            if next_node not in self.graph.neighbors(robot.current_node):
                robot.buffered_next_node = None
                continue

            if self._has_opposing_traffic(robot.current_node, next_node):
                robot.buffered_next_node = None
                continue

            # Docking edge ocupada → buffer inválido
            if self.graph.is_docking_edge(robot.current_node, next_node):
                if (self.world.robots_on_edge(robot.current_node, next_node) or
                        self.world.robots_on_edge(next_node, robot.current_node)):
                    robot.buffered_next_node = None
                    continue

            # Junction destino ocupada → esperar (alinha com action_builder Rule 3)
            if (not self.graph.is_special(next_node) and
                    self.world.robots_at_node(next_node)):
                continue  # preservar buffer, retry no próximo tick

            buffered_actions[robot.id] = (next_node, "acc")

        if not buffered_actions:
            return {}

        results = self.dispatch_system.apply(self.world, buffered_actions)

        for robot_id, result in results.items():
            robot = self.world.robots.get(robot_id)
            if robot is None:
                continue

            if result.accepted and result.dispatched:
                # Speed carry-over: o dispatch zera a speed e aplica acc_step (→ 4.5).
                # Sobrepõe com o máximo permitido pelas restrições físicas, evitando
                # a rampa de aceleração completa em passagens sem paragem real.
                curve_lim = self.dispatch_system._curve_zone_limit(robot)
                robot.speed = curve_lim if curve_lim is not None else self.dispatch_system.vel_max
                robot.buffered_next_node = None
            else:
                robot.buffered_next_node = None

        return results

    def _has_opposing_traffic(self, from_node: str, to_node: str) -> bool:
        """
        Verifica se existe tráfego físico em sentido contrário na aresta.
        Delega para world_state para garantir consistência com dispatch.
        """
        return self.world.edge_has_opposing_traffic(from_node, to_node)