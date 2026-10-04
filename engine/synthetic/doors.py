from dataclasses import dataclass, asdict
from typing import Literal


DoorOrientation = Literal["horizontal", "vertical"]


@dataclass(frozen=True)
class Door:
    id: str
    wall_id: str
    x: int
    y: int
    width: int
    orientation: DoorOrientation
    swing: str = "inward"

    def to_dict(self) -> dict:
        return asdict(self)


def generate_doors() -> list[Door]:
    """
    Generate deterministic door ground truth.

    x/y represent the center of the door opening.
    The opening belongs to a specific wall.
    """

    return [
        Door(
            id="door_001",
            wall_id="wall_001",
            x=350,
            y=100,
            width=90,
            orientation="horizontal",
            swing="inward",
        ),
        Door(
            id="door_002",
            wall_id="wall_002",
            x=1100,
            y=300,
            width=90,
            orientation="vertical",
            swing="inward",
        ),
    ]
