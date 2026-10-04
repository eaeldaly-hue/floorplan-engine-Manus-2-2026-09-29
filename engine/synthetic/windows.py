from dataclasses import dataclass, asdict
from typing import Literal


WindowOrientation = Literal["horizontal", "vertical"]


@dataclass(frozen=True)
class Window:
    id: str
    wall_id: str
    x: int
    y: int
    width: int
    orientation: WindowOrientation

    def to_dict(self) -> dict:
        return asdict(self)


def generate_windows() -> list[Window]:
    """
    Generate deterministic window ground truth.

    Each window is attached to a wall.
    x/y represent the center point of the opening.
    """

    return [
        Window(
            id="window_001",
            wall_id="wall_001",
            x=750,
            y=100,
            width=140,
            orientation="horizontal",
        ),
        Window(
            id="window_002",
            wall_id="wall_003",
            x=600,
            y=800,
            width=160,
            orientation="horizontal",
        ),
    ]
