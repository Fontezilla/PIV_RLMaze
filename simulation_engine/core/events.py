from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Event containers
# ---------------------------------------------------------------------------

@dataclass
class CollisionEvent:
    robot_a: str
    robot_b: str
    node_from: str
    node_to: str


@dataclass
class ArrivalEvent:
    robot_id: str
    node_id: str


@dataclass
class PickupEvent:
    robot_id: str
    box_id: str
    node_id: str


@dataclass
class DeliveryEvent:
    robot_id: str
    box_id: str
    node_id: str


# ---------------------------------------------------------------------------
# Events bundle (por tick)
# ---------------------------------------------------------------------------

@dataclass
class Events:
    collisions: List[CollisionEvent] = field(default_factory=list)
    arrivals: List[ArrivalEvent] = field(default_factory=list)
    pickups: List[PickupEvent] = field(default_factory=list)
    deliveries: List[DeliveryEvent] = field(default_factory=list)

    def clear(self):
        self.collisions.clear()
        self.arrivals.clear()
        self.pickups.clear()
        self.deliveries.clear()