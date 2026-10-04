from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Wall:
    id: str
    x1: int
    y1: int
    x2: int
    y2: int
    thickness: int = 20

    def to_dict(self) -> dict:
        return asdict(self)


def generate_walls(
    width: int = 1200,
    height: int = 900,
    margin: int = 100,
    wall_thickness: int = 20,
) -> list[Wall]:
    """
    Generate a simple rectangular floor-plan shell.

    Coordinates represent wall centerlines.
    This geometry is the Ground Truth for the synthetic image.
    """

    x1 = margin
    y1 = margin
    x2 = width - margin
    y2 = height - margin

    return [
        Wall(
            id="wall_001",
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y1,
            thickness=wall_thickness,
        ),
        Wall(
            id="wall_002",
            x1=x2,
            y1=y1,
            x2=x2,
            y2=y2,
            thickness=wall_thickness,
        ),
        Wall(
            id="wall_003",
            x1=x2,
            y1=y2,
            x2=x1,
            y2=y2,
            thickness=wall_thickness,
        ),
        Wall(
            id="wall_004",
            x1=x1,
            y1=y2,
            x2=x1,
            y2=y1,
            thickness=wall_thickness,
        ),
    ]