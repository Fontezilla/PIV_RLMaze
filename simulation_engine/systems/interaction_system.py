from typing import Optional, Tuple, List

from simulation_engine.core.world_state import WorldState
from simulation_engine.core.entities import Robot, Box, BoxState
from simulation_engine.core.events import Events, PickupEvent, DeliveryEvent


class InteractionSystem:
    """
    Responsável por:

    - Pickup de caixas
    - Entrega (delivery)
    - Progressão na pipeline (processA, processB)

    Assume:
    - Robôs já estão em IDLE no nó (após collision_system)

    Usa pipeline_config para determinar o próximo passo de cada box,
    evitando dependência direta nos tipos de nó do grafo.
    """

    def __init__(self, graph, pipeline_config: dict):
        self.graph = graph
        self.pipeline_config = pipeline_config

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def update(self, world: WorldState, events: Events):
        # Reset process busy flags each tick so each process handles one robot per tick
        for process in world.processes.values():
            process.busy = False

        for robot in world.robots.values():

            if robot.current_node is None:
                continue

            if robot.carried_box is None:
                self._try_pickup(robot, world, events)
            else:
                self._try_process_or_deliver(robot, world, events)

    # ------------------------------------------------------------------
    # PICKUP
    # ------------------------------------------------------------------

    def _try_pickup(self, robot: Robot, world: WorldState, events: Events):
        boxes = world.boxes_at_node(robot.current_node)

        if not boxes:
            return

        box = boxes[0]

        robot.carried_box = box.id
        box.state = BoxState.IN_TRANSPORT
        box.carried_by = robot.id
        box.current_node = None
        self._update_destination_node(box)

        events.pickups.append(
            PickupEvent(
                robot_id=robot.id,
                box_id=box.id,
                node_id=robot.current_node,
            )
        )

    # ------------------------------------------------------------------
    # PROCESS / DELIVERY
    # ------------------------------------------------------------------

    def _try_process_or_deliver(self, robot: Robot, world: WorldState, events: Events):
        if robot.carried_box not in world.boxes:
            return
        box = world.get_box(robot.carried_box)
        node = robot.current_node

        result = self._next_step(box)
        if result is None:
            return

        step_name, valid_nodes = result

        if node not in valid_nodes:
            return

        # ------------------------------------------------------------------
        # DELIVERY
        # ------------------------------------------------------------------

        if step_name == "exit":
            box.state = BoxState.DELIVERED
            box.carried_by = None
            world.delivered_boxes.append(box.id)
            robot.carried_box = None

            events.deliveries.append(
                DeliveryEvent(
                    robot_id=robot.id,
                    box_id=box.id,
                    node_id=node,
                )
            )
            return

        # ------------------------------------------------------------------
        # PROCESS
        # ------------------------------------------------------------------

        process = world.processes.get(node)
        if process is None or process.busy:
            return

        # processamento instantâneo
        process.busy = True

        box.next_step_index += 1
        box.state = BoxState.AT_NODE
        box.carried_by = None
        box.current_node = node

        robot.carried_box = None
        self._update_destination_node(box)

    # ------------------------------------------------------------------
    # PIPELINE HELPER
    # ------------------------------------------------------------------

    def _update_destination_node(self, box: Box) -> None:
        result = self._next_step(box)
        if result is None:
            box.destination_node = None
        else:
            _, valid_nodes = result
            box.destination_node = valid_nodes[0] if valid_nodes else None

    def _next_step(self, box: Box) -> Optional[Tuple[str, List[str]]]:
        """
        Retorna (step_name, [valid_nodes]) para o próximo passo da box.
        Retorna None se a pipeline não existir ou a box já terminou.
        """
        config = self.pipeline_config.get("pipelines", {}).get(box.pipeline_type)
        if config is None:
            return None

        sequence = config["sequence"]
        constraints = config.get("constraints", {})

        idx = box.next_step_index
        if idx >= len(sequence):
            return None

        step_name = sequence[idx]
        valid_nodes = constraints.get(step_name, [])

        return step_name, valid_nodes
