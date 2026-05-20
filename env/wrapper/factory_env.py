"""
env/wrapper/factory_env.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Orquestrador do ambiente de fábrica v0.7.

Arquitectura
------------
  env/core/   ←  World, Robot, Box, BoxManager, physics
  env/traffic/ ←  Router
       ↑
  FactoryEnv  (este ficheiro)

Responsabilidades
-----------------
  - Correr a simulação tick a tick
  - Detectar eventos que requerem decisão do agente RL:
      * robot ficou livre (chegou ao goal sem caixa, ou pre-posição vazia)
      * nova caixa disponível → re-decide robots IDLE/WAITING sem caixa
  - Executar assignments decididos pelo agente: {robot_id: target}
      target = box_id (int)  →  robot vai buscar a caixa
      target = node_id (str) →  robot vai para nó de pre-posição
  - Acumular reward entre eventos e devolvê-lo no step()
  - Expor estado bruto para o agente construir o HeteroGraph

Interface com o agente RL
-------------------------
  O agente RL recebe um dict com o estado do mundo e decide assignments.
  O FactoryEnv não constrói observações — isso é responsabilidade do agente.

  Loop típico de treino:

    state, info = env.reset()
    done = False
    while not done:
        assignments = agent.decide(state)        # {robot_id: target}
        state, reward, terminated, truncated, info = env.step(assignments)
        done = terminated or truncated

Terminated / Truncated
-----------------------
  terminated : True quando todas as caixas foram entregues
  truncated  : True quando world.tick >= tick_limit
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from env.core.entities import Robot, RobotState
from env.core.graph import FactoryGraph
from env.core.physics import MAX_SPEED, tick as physics_tick, turn_delay
from env.core.world import World
from env.core.box_manager import BoxManager, REWARD_PICK
from env.traffic.router import Router, DEADLOCK_THRESHOLD


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SPAWN_NODE    = "N"
SPAWN_STAGGER = 10
TICK_LIMIT    = 5_000

REWARD_PER_TICK  = -0.001   # custo de tempo por robot activo por tick
REWARD_DEADLOCK  = -0.5     # penalização por deadlock detectado
REWARD_WAITING   = -0.02    # penalização por tick em WAITING

# Nós candidatos a pre-posição — onde caixas vão aparecer ou chegar
# Inclui processA_exit e processB_exit porque são agora waypoints reais
# onde caixas ficam WAITING após processamento.
_PREPOSITION_TYPES = frozenset({
    "entry",
    "exit",
    "processA_entry",
    "processA_exit",
    "processB_entry",
    "processB_exit",
})

DEFAULT_MAP      = Path(".configs/map_factory.yaml")
DEFAULT_CACHE    = Path(".configs/graph_cache.pkl")
DEFAULT_PIPELINE = Path(".configs/box_pipeline.yaml")


# ---------------------------------------------------------------------------
# Helpers de despacho  (idênticos à versão anterior)
# ---------------------------------------------------------------------------

def _dispatch_move(robot: Robot, next_node: str, graph: FactoryGraph) -> None:
    delay = (
        turn_delay(graph, robot.came_from, robot.current_node, next_node)
        if robot.current_node else 0
    )
    robot.from_node              = robot.current_node
    robot.to_node                = next_node
    robot.current_node           = None
    robot.progress               = 0.0
    robot.wait_ticks             = delay
    robot.turn_ticks_total       = delay
    robot.speed                  = 0.0 if delay > 0 else robot.speed
    robot.state                  = RobotState.MOVING
    robot.target_speed           = MAX_SPEED
    robot.wait_ticks_in_junction = 0
    robot.parked_at              = None


def _dispatch_park(robot: Robot, park_spec: str, router: Router) -> None:
    u, v, frac_str = park_spec.split("|")
    fraction = float(frac_str)
    robot.from_node              = u
    robot.to_node                = v
    robot.current_node           = None
    robot.progress               = 0.0
    robot.wait_ticks             = 0
    robot.speed                  = 0.0
    robot.state                  = RobotState.MOVING
    robot.target_speed           = MAX_SPEED
    robot.wait_ticks_in_junction = 0
    robot.parked_at              = (u, v, fraction)
    router.node_lock.release(u, robot.id)


def _dispatch_unpark(
    robot       : Robot,
    target_node : str | None,
    router      : Router,
) -> None:
    if robot.parked_at is None:
        return
    u, v, frac = robot.parked_at
    if target_node == v:
        if not router.node_lock.try_acquire(v, robot.id):
            return
        robot.from_node    = u
        robot.to_node      = v
        robot.current_node = None
        robot.came_from    = u
        robot.progress     = frac
    else:
        if not router.node_lock.try_acquire(u, robot.id):
            return
        robot.from_node    = v
        robot.to_node      = u
        robot.current_node = None
        robot.came_from    = v
        robot.progress     = 1.0 - frac
    robot.wait_ticks             = 0
    robot.state                  = RobotState.MOVING
    robot.speed                  = 0.0
    robot.target_speed           = MAX_SPEED
    robot.wait_ticks_in_junction = 0
    router.unparked(robot)


# ---------------------------------------------------------------------------
# FactoryEnv
# ---------------------------------------------------------------------------

class FactoryEnv:
    """
    Orquestrador do ambiente de fábrica.

    Não é um gym.Env directamente — o wrapper gym fica na camada do agente RL
    para suportar espaços de observação/acção variáveis (HeteroGraph).

    Parâmetros
    ----------
    n_robots      : número de robots activos em simultâneo
    n_boxes       : número de caixas criadas no reset()
    tick_limit    : ticks máximos antes de truncation
    map_path      : caminho para map_factory.yaml
    cache_path    : caminho para graph_cache.pkl
    pipeline_path : caminho para box_pipeline.yaml
    render_mode   : "human" para pygame, None para headless
    seed          : seed para reprodutibilidade
    """

    def __init__(
        self,
        n_robots      : int           = 4,
        n_boxes       : int           = 8,
        tick_limit    : int           = TICK_LIMIT,
        map_path      : str | Path    = DEFAULT_MAP,
        cache_path    : str | Path    = DEFAULT_CACHE,
        pipeline_path : str | Path    = DEFAULT_PIPELINE,
        render_mode   : Optional[str] = None,
        seed          : Optional[int] = None,
    ) -> None:
        self.n_robots    = n_robots
        self.n_boxes     = n_boxes
        self.tick_limit  = tick_limit
        self.render_mode = render_mode
        self._seed       = seed
        self._rng        = random.Random(seed)

        self.graph = FactoryGraph(str(map_path), str(cache_path))

        # Nós de pre-posição expostos ao agente
        self.preposition_nodes: list[str] = sorted(
            n for n in self.graph.all_nodes()
            if self.graph.node_type(n) in _PREPOSITION_TYPES
        )

        self._box_manager = BoxManager(
            pipeline_path = pipeline_path,
            n_boxes       = n_boxes,
            seed          = seed,
        )

        # Estado da simulação (populado no reset)
        self.world        : World  | None = None
        self.router       : Router | None = None
        self._spawn_ticks : list[int]     = []
        self._robots_list : list[Robot]   = []

        # Robots que aguardam decisão do agente neste momento
        self._pending_robots: list[Robot] = []

        # Renderer lazy
        self._renderer = None

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def reset(
        self,
        seed    : Optional[int]  = None,
        options : Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """
        Reinicia o episódio.

        Devolve (state, info) onde state é o estado bruto do mundo
        para o agente construir o HeteroGraph.
        """
        if seed is not None:
            self._rng = random.Random(seed)

        self.world  = World()
        self.router = Router(self.graph)

        self._box_manager.reset(seed=seed)

        self._spawn_ticks   = [i * SPAWN_STAGGER for i in range(self.n_robots)]
        self._robots_list   = [Robot(id=f"robot_{i}") for i in range(self.n_robots)]
        self._pending_robots = []

        # Corre ticks até haver robots spawnados e prontos para decisão
        self._run_until_decision_needed(accumulate_reward=False)

        state = self._build_state()
        info  = self._build_info()
        return state, info

    def step(
        self,
        assignments: dict[str, int | str],
    ) -> tuple[dict, float, bool, bool, dict]:
        """
        Executa os assignments decididos pelo agente e avança a simulação
        até ao próximo evento de decisão.

        Parâmetros
        ----------
        assignments : {robot_id: target}
            target = box_id (int)   → robot vai buscar a caixa
            target = node_id (str)  → robot vai para nó de pre-posição

        Devolve
        -------
        state      : estado bruto do mundo
        reward     : reward acumulado desde o último step
        terminated : todas as caixas entregues
        truncated  : tick_limit atingido
        info       : snapshot para debug
        """
        assert self.world  is not None, "Chama reset() antes de step()."
        assert self.router is not None

        # 1. Executa assignments sequencialmente
        #    (a ordem é a de self._pending_robots, decidida pelo env)
        reward = self._apply_assignments(assignments)

        # 2. Limpa lista de pendentes
        self._pending_robots.clear()

        # 3. Corre ticks até próximo evento
        reward += self._run_until_decision_needed(accumulate_reward=True)

        # 4. Terminação
        terminated = self._box_manager.all_delivered()
        truncated  = self.world.tick >= self.tick_limit

        state = self._build_state()
        info  = self._build_info(reward=reward)

        if self.render_mode == "human":
            self._do_render()

        return state, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Estado exposto ao agente
    # ------------------------------------------------------------------

    def pending_robots(self) -> list[Robot]:
        """Robots que aguardam assignment neste momento."""
        return list(self._pending_robots)

    def available_boxes(self):
        """Caixas disponíveis para assignment."""
        return self._box_manager.available_boxes()

    # ------------------------------------------------------------------
    # Internos — loop de simulação
    # ------------------------------------------------------------------

    def _run_until_decision_needed(
        self,
        accumulate_reward: bool = True,
    ) -> float:
        """
        Corre ticks até haver pelo menos um robot a precisar de decisão,
        ou até terminação.

        Devolve reward acumulado durante os ticks.
        """
        reward = 0.0

        while True:
            # Condição de saída: há robots pendentes
            if self._pending_robots:
                break

            # Condição de saída: episódio terminado
            if self._box_manager.all_delivered():
                break
            if self.world.tick >= self.tick_limit:
                break

            tick_reward = self._tick()
            if accumulate_reward:
                reward += tick_reward

            if self.render_mode == "human":
                self._do_render()

        return reward

    def _tick(self) -> float:
        """Avança um tick da simulação. Devolve reward do tick."""
        assert self.world  is not None
        assert self.router is not None

        self.world.step_tick()
        reward = 0.0

        # 1. Spawn de robots
        self._maybe_spawn_robots()

        # 1b. Spawn de caixas (lazy — só coloca se o nó de entrada estiver livre)
        newly_spawned = self._box_manager.tick_spawn()

        # 2. Reset WAITING → IDLE
        for robot in self.world.all_robots():
            if robot.state == RobotState.WAITING:
                robot.state = RobotState.IDLE

        # 3. Física + pick/drop
        newly_available_boxes: list[int] = []  # box_ids que ficaram disponíveis
        any_pick_happened = False

        for robot in self.world.all_robots():
            arrived = physics_tick(self.graph, robot)

            if not arrived:
                continue

            if robot.state == RobotState.PARKED:
                if robot.parked_at is not None:
                    u, v, frac = robot.parked_at
                    self.router.parked(robot, u, v, frac)
                continue

            if robot.current_node is None:
                continue

            self.router.arrived(robot, robot.current_node)
            robot.wait_ticks_in_junction = 0

            # --- Drop ---
            drop_reward, delivered, box_id = self._box_manager.on_drop(
                robot, robot.current_node
            )
            reward += drop_reward

            if box_id is not None:
                # Caixa acabou de ficar disponível num waypoint intermédio
                # ou foi entregue — notifica para re-planning
                if not delivered:
                    newly_available_boxes.append(box_id)
                # Robot ficou livre após drop
                self._mark_robot_pending(robot)
                continue

            # --- Pick automático se robot chegou ao nó da sua caixa assignada ---
            if robot.carrying_box is None and robot.current_node is not None:
                pick_reward = self._try_auto_pick(robot)
                reward += pick_reward
                if pick_reward > 0:
                    # Pick libertou o nó fonte — pode desbloquear outras caixas
                    any_pick_happened = True

            # --- Robot chegou ao goal sem caixa (pre-posição ou sem assignment) ---
            if robot.reached_goal() and robot.carrying_box is None:
                self.router.release_all(robot)
                robot.last_visited.clear()
                robot.last_edges.clear()
                self.router.register(robot)
                self._mark_robot_pending(robot)

        # 3b. Poll de auto-advance: caixas paradas em processX_entry
        #     onde o exit emparelhado entretanto ficou livre (e.g. outro
        #     robot apanhou a caixa que estava no exit) avançam agora.
        newly_advanced_boxes = self._box_manager.tick_process_advance()
        for adv_id in newly_advanced_boxes:
            if adv_id not in newly_available_boxes:
                newly_available_boxes.append(adv_id)

        # 4. Liberta nós de origem
        self.router.sync_moving_node_locks(self.world.all_robots())

        # 5. Decisão de movimento (Router)
        robots = self.world.all_robots()
        for robot in robots:
            if robot.state == RobotState.MOVING:
                continue
            if robot.is_idle() and robot.current_node == robot.goal_node:
                continue

            decision = self.router.decide(robot, robots)
            if decision is None:
                robot.wait_ticks_in_junction += 1
                if robot.is_idle():
                    robot.state = RobotState.WAITING
                continue

            action, payload = decision
            if action == "move" and payload is not None:
                _dispatch_move(robot, payload, self.graph)
            elif action == "park" and payload is not None:
                _dispatch_park(robot, payload, self.router)
            elif action == "unpark":
                _dispatch_unpark(robot, payload, self.router)

        # 6. Reward de tempo e bloqueio
        active = self.world.all_robots()
        reward += REWARD_PER_TICK * len(active)
        for robot in active:
            if robot.state == RobotState.WAITING:
                reward += REWARD_WAITING
            if robot.wait_ticks_in_junction >= DEADLOCK_THRESHOLD:
                reward += REWARD_DEADLOCK

        # 7. Re-planning quando há novos eventos:
        #    - caixa dropada num waypoint intermédio (fica disponível)
        #    - pick aconteceu (libertou o nó fonte — pode desbloquear outras caixas)
        #    - nova caixa spawnada (nova opção de assignment)
        if newly_available_boxes or any_pick_happened or newly_spawned:
            self._replan_idle_robots()

        return reward

    def _try_auto_pick(self, robot: Robot) -> float:
        """
        Pick automático quando o robot chega ao nó de uma caixa.

        Prioridade:
          1. Caixa explicitamente assignada pelo agente (assigned_box_id):
             se está disponível neste nó, apanha-a.
             Se o robot tem assignment mas a caixa não está aqui, não faz
             pickup oportunista — está em trânsito para outro destino.
          2. Sem assignment explícito (pré-posição ou robot livre): apanha
             qualquer caixa disponível neste nó que não esteja reservada
             a outro robot e cujo próximo waypoint esteja livre.
        """
        if robot.current_node is None:
            return 0.0

        # --- 1. Assignment explícito ---
        if robot.assigned_box_id is not None:
            box = self._box_manager.get_box(robot.assigned_box_id)
            if (
                box is not None
                and box.is_available
                and box.current_node == robot.current_node
            ):
                reward = self._box_manager.on_pick(robot, robot.assigned_box_id)
                if reward > 0:
                    robot.assigned_box_id = None
                    self._sync_goal_to_waypoint(robot)
                    return reward
            # Tem assignment mas não está no nó certo — deixa o robot continuar
            # em direcção à sua caixa-alvo sem apanhar outras pelo caminho.
            return 0.0

        # --- 2. Pickup oportunista ---
        if robot.carrying_box is not None:
            return 0.0

        all_robots = self.world.all_robots()
        assigned_to_others = {
            r.assigned_box_id
            for r in all_robots
            if r.id != robot.id and r.assigned_box_id is not None
        }

        for box in self._box_manager.available_boxes_at(robot.current_node):
            if box.box_id in assigned_to_others:
                continue
            next_wp = box.next_waypoint
            if next_wp is None:
                continue
            if not self._box_manager.is_node_drop_available(next_wp):
                continue

            reward = self._box_manager.on_pick(robot, box.box_id)
            if reward > 0:
                self._sync_goal_to_waypoint(robot)
                return reward

        return 0.0

    def _sync_goal_to_waypoint(self, robot: Robot) -> None:
        """Força goal_node para o próximo waypoint da caixa transportada."""
        if robot.carrying_box is None:
            return
        box = self._box_manager.get_box(robot.carrying_box)
        if box is None or box.next_waypoint is None:
            return
        target = box.next_waypoint
        if robot.goal_node != target:
            self.router.release_all(robot)
            robot.goal_node = target
            robot.last_visited.clear()
            robot.last_edges.clear()
            self.router.register(robot)

    def _mark_robot_pending(self, robot: Robot) -> None:
        """Marca robot como a precisar de decisão do agente."""
        if robot not in self._pending_robots:
            self._pending_robots.append(robot)

    def _replan_idle_robots(self) -> None:
        """
        Quando uma nova caixa fica disponível, marca todos os robots
        IDLE/WAITING sem caixa como pendentes para re-decisão.
        """
        for robot in self.world.all_robots():
            if robot.is_free() and robot not in self._pending_robots:
                self._pending_robots.append(robot)

    # ------------------------------------------------------------------
    # Internos — assignments
    # ------------------------------------------------------------------

    def _apply_assignments(self, assignments: dict[str, int | str]) -> float:
        """
        Executa os assignments do agente.

        Para cada robot pendente, se houver assignment:
          - int   → robot vai buscar a caixa (goal = current_node da caixa)
          - str   → robot vai para nó de pre-posição
          - None  → robot fica IDLE (sem assignment)

        Garante:
          - nenhuma caixa é assignada duas vezes no mesmo step
          - nenhum destino (next_waypoint) é reservado por dois robots
            no mesmo step (regra de capacidade 1 caixa por nó)

        Devolve reward imediato (0 — reward chega no drop/pick).
        """
        reward = 0.0

        # Conjunto de caixas já assignadas neste step
        assigned_boxes: set[int] = set()
        # Conjunto de next_waypoints reservados neste step
        assigned_next_waypoints: set[str] = set()
        # Conjunto de nós de pre-posição já assignados neste step
        assigned_goals: set[str] = set()

        for robot in self._pending_robots:
            if not self.world.has_robot(robot.id):
                continue

            # Limpa o assignment anterior antes de processar o novo
            robot.assigned_box_id = None

            target = assignments.get(robot.id)

            if target is None:
                # Sem assignment — robot fica IDLE no nó actual
                robot.state = RobotState.IDLE
                continue

            if isinstance(target, int):
                # Assignment para caixa
                box = self._box_manager.get_box(target)
                if box is None or not box.is_available:
                    robot.state = RobotState.IDLE
                    continue
                if target in assigned_boxes:
                    # Caixa já assignada a outro robot neste step
                    robot.state = RobotState.IDLE
                    continue

                # Verificação de capacidade: o destino da caixa deve estar livre
                next_wp = box.next_waypoint
                if next_wp is not None:
                    if next_wp in assigned_next_waypoints:
                        # Outro robot já vai para este nó neste step
                        robot.state = RobotState.IDLE
                        continue
                    if not self._box_manager.is_node_drop_available(next_wp):
                        # Já há uma caixa lá ou a caminho
                        robot.state = RobotState.IDLE
                        continue
                    assigned_next_waypoints.add(next_wp)

                assigned_boxes.add(target)
                robot.assigned_box_id = target
                goal = box.current_node
                self._set_robot_goal(robot, goal)

            elif isinstance(target, str):
                # Assignment para nó de pre-posição (sem caixa específica)
                if target in assigned_goals:
                    robot.state = RobotState.IDLE
                    continue
                assigned_goals.add(target)
                self._set_robot_goal(robot, target)

        return reward

    def _set_robot_goal(self, robot: Robot, goal: str) -> None:
        """Define o goal do robot e regista no router."""
        self.router.release_all(robot)
        robot.goal_node = goal
        robot.last_visited.clear()
        robot.last_edges.clear()
        robot.state = RobotState.IDLE
        self.router.register(robot)

    # ------------------------------------------------------------------
    # Internos — spawn
    # ------------------------------------------------------------------

    def _maybe_spawn_robots(self) -> None:
        tick = self.world.tick
        for i, robot in enumerate(self._robots_list):
            if self.world.has_robot(robot.id):
                continue
            if tick < self._spawn_ticks[i]:
                continue

            robot.current_node           = SPAWN_NODE
            robot.state                  = RobotState.IDLE
            robot.came_from              = None
            robot.goal_node              = None
            robot.target_speed           = MAX_SPEED
            robot.speed                  = 0.0
            robot.wait_ticks             = 0
            robot.wait_ticks_in_junction = 0
            robot.parked_at              = None
            robot.carrying_box           = None

            x, y = self.graph.node_position(SPAWN_NODE)
            robot.world_x = x
            robot.world_y = y

            self.world.add_robot(robot)
            self.router.register(robot)

            # Robot recém-spawnado precisa de assignment imediato
            self._mark_robot_pending(robot)

    # ------------------------------------------------------------------
    # Internos — estado exposto ao agente
    # ------------------------------------------------------------------

    def _build_state(self) -> dict:
        """
        Estado bruto do mundo para o agente construir o HeteroGraph.

        Contém toda a informação necessária sem processamento.
        """
        all_robots = self.world.all_robots() if self.world else []

        # Robots pendentes vão re-decidir agora — as suas reservas anteriores
        # não devem mascarar as caixas/nós disponíveis (senão o agente não as
        # vê e o robot acaba sem assignment válido).
        pending_ids: set[str] = {r.id for r in self._pending_robots}

        # Caixas já assignadas (assigned_box_id set) mas ainda não apanhadas
        already_assigned_boxes = {
            r.assigned_box_id
            for r in all_robots
            if r.assigned_box_id is not None and r.id not in pending_ids
        }

        # Nós já usados como goal por outros robots
        already_targeted_nodes = {
            r.goal_node
            for r in all_robots
            if r.goal_node is not None and r.id not in pending_ids
        }

        # Pre-posições úteis: nós actuais de caixas disponíveis
        # + próximos waypoints de caixas em trânsito
        # → só faz sentido pré-posicionar onde as caixas estão ou vão parar
        useful_preposition: set[str] = set()
        for box in self._box_manager.active_boxes():
            if box.is_waiting and box.current_node:
                useful_preposition.add(box.current_node)
            if box.is_in_transit and box.next_waypoint:
                useful_preposition.add(box.next_waypoint)

        return {
            "tick":               self.world.tick if self.world else 0,
            "robots":             self.world.snapshot()["robots"] if self.world else [],
            "boxes":              self._box_manager.snapshot(),
            "pending_robot_ids":  [r.id for r in self._pending_robots],
            "available_box_ids":  [
                b.box_id for b in self._box_manager.available_boxes()
                if b.box_id not in already_assigned_boxes
            ],
            "preposition_nodes":  [
                n for n in self.preposition_nodes
                if n in useful_preposition and n not in already_targeted_nodes
            ],
            "graph_nodes":        self._graph_node_features(),
            "graph_edges":        self._graph_edge_features(),
        }

    def _graph_node_features(self) -> dict[str, dict]:
        """Features de cada nó do mapa para o HeteroGraph."""
        features = {}
        for node in self.graph.all_nodes():
            node_type = self.graph.node_type(node)
            x, y      = self.graph.node_position(node)
            features[node] = {
                "type": node_type,
                "x":    x,
                "y":    y,
                "n_robots_here":   len(self.world.robots_on_node(node)) if self.world else 0,
                "n_boxes_waiting": len(self._box_manager.available_boxes_at(node)),
            }
        return features

    def _graph_edge_features(self) -> list[dict]:
        """Arestas do mapa com distâncias — para o HeteroGraph."""
        edges = []
        for u, v, data in self.graph.graph.edges(data=True):
            edges.append({
                "from":     u,
                "to":       v,
                "distance": data.get("distance", 1.0),
            })
        return edges

    # ------------------------------------------------------------------
    # Internos — info
    # ------------------------------------------------------------------

    def _build_info(self, reward: float = 0.0) -> dict:
        if self.world is None:
            return {}
        return {
            **self.world.snapshot(),
            "boxes":              self._box_manager.snapshot(),
            "delivered":          self._box_manager.delivered_count(),
            "queued_boxes":       self._box_manager.queue_count(),
            "all_delivered":      self._box_manager.all_delivered(),
            "pending_robot_ids":  [r.id for r in self._pending_robots],
            "preposition_nodes":  self.preposition_nodes,
            "step_reward":        reward,
        }

    # ------------------------------------------------------------------
    # Render / close
    # ------------------------------------------------------------------

    def render(self) -> None:
        if self.render_mode == "human":
            self._do_render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _do_render(self) -> None:
        if self._renderer is None:
            from render.renderer import Renderer
            self._renderer = Renderer(self.graph)

        info = {
            "tick":   self.world.tick,
            "robots": self.world.robot_count(),
            "boxes":  self._box_manager.snapshot(),
        }
        result = self._renderer.render(self.world, info)
        if result == "quit":
            self.close()