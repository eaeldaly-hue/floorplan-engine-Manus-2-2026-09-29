import json
import random
from pathlib import Path

from .walls import generate_walls
from .doors import generate_doors
from .windows import generate_windows
from .renderer import render_floorplan


DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 900


def generate_floorplan(
    output_dir: str | Path = "synthetic/generated",
    seed: int = 42,
    name: str = "floorplan_0001",
) -> dict:
    """
    Generate one synthetic floor plan and its exact Ground Truth.

    The seed is stored in the JSON so the generation is reproducible.
    """

    random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    walls = generate_walls(
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
    )

    doors = generate_doors()
    windows = generate_windows()

    image_path = output_dir / f"{name}.png"
    json_path = output_dir / f"{name}.json"

    render_floorplan(
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        walls=walls,
        doors=doors,
        windows=windows,
        output_path=image_path,
    )

    ground_truth = {
        "schema_version": "1.0",
        "generator_version": "0.1.0",
        "seed": seed,
        "image": {
            "filename": image_path.name,
            "width": DEFAULT_WIDTH,
            "height": DEFAULT_HEIGHT,
        },
        "walls": [wall.to_dict() for wall in walls],
        "doors": [door.to_dict() for door in doors],
        "windows": [window.to_dict() for window in windows],
    }

    json_path.write_text(
        json.dumps(ground_truth, indent=2),
        encoding="utf-8",
    )

    return ground_truth


if __name__ == "__main__":
    result = generate_floorplan()

    print("Synthetic floor plan generated")
    print(f"Image: synthetic/generated/{result['image']['filename']}")
    print("Ground Truth: synthetic/generated/floorplan_0001.json")
