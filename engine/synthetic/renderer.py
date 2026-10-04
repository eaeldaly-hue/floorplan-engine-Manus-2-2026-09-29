from pathlib import Path

import cv2
import numpy as np

from .walls import Wall
from .doors import Door
from .windows import Window


BACKGROUND = 255
WALL_COLOR = 0
SYMBOL_COLOR = 0


def _wall_unit_vector(wall: Wall) -> tuple[float, float]:
    dx = wall.x2 - wall.x1
    dy = wall.y2 - wall.y1
    length = (dx * dx + dy * dy) ** 0.5

    if length == 0:
        raise ValueError(f"Wall {wall.id} has zero length")

    return dx / length, dy / length


def _wall_normal(wall: Wall) -> tuple[float, float]:
    ux, uy = _wall_unit_vector(wall)
    return -uy, ux


def _opening_position(wall: Wall, x: int, y: int) -> float:
    ux, uy = _wall_unit_vector(wall)
    return (x - wall.x1) * ux + (y - wall.y1) * uy


def _draw_wall(
    image: np.ndarray,
    wall: Wall,
    openings: list[tuple[float, int]],
) -> None:

    ux, uy = _wall_unit_vector(wall)
    nx, ny = _wall_normal(wall)

    half_thickness = wall.thickness / 2

    wall_length = (
        (wall.x2 - wall.x1) ** 2
        + (wall.y2 - wall.y1) ** 2
    ) ** 0.5

    cuts = []

    for center, width in openings:
        start = max(0.0, center - width / 2)
        end = min(wall_length, center + width / 2)
        cuts.append((start, end))

    cuts.sort()

    for side in (-1, 1):

        ox = nx * half_thickness * side
        oy = ny * half_thickness * side

        cursor = 0.0

        for start, end in cuts:

            if start > cursor:

                p1 = (
                    round(wall.x1 + ux * cursor + ox),
                    round(wall.y1 + uy * cursor + oy),
                )

                p2 = (
                    round(wall.x1 + ux * start + ox),
                    round(wall.y1 + uy * start + oy),
                )

                cv2.line(
                    image,
                    p1,
                    p2,
                    WALL_COLOR,
                    2,
                    lineType=cv2.LINE_AA,
                )

            cursor = max(cursor, end)

        if cursor < wall_length:

            p1 = (
                round(wall.x1 + ux * cursor + ox),
                round(wall.y1 + uy * cursor + oy),
            )

            p2 = (
                round(wall.x2 + ox),
                round(wall.y2 + oy),
            )

            cv2.line(
                image,
                p1,
                p2,
                WALL_COLOR,
                2,
                lineType=cv2.LINE_AA,
            )


def _draw_door(
    image: np.ndarray,
    door: Door,
    wall: Wall,
) -> None:

    ux, uy = _wall_unit_vector(wall)
    nx, ny = _wall_normal(wall)

    half = door.width / 2

    # Opening endpoints.
    a = (
        door.x - ux * half,
        door.y - uy * half,
    )

    # Hinge.
    hinge = a

    # Door leaf opens inward.
    leaf_end = (
        hinge[0] + nx * door.width,
        hinge[1] + ny * door.width,
    )

    cv2.line(
        image,
        (round(hinge[0]), round(hinge[1])),
        (round(leaf_end[0]), round(leaf_end[1])),
        SYMBOL_COLOR,
        2,
        lineType=cv2.LINE_AA,
    )

    # 90 degree swing arc.
    points = []

    steps = 40

    for i in range(steps + 1):

        angle = (np.pi / 2) * i / steps

        px = (
            hinge[0]
            + door.width
            * (
                ux * np.cos(angle)
                + nx * np.sin(angle)
            )
        )

        py = (
            hinge[1]
            + door.width
            * (
                uy * np.cos(angle)
                + ny * np.sin(angle)
            )
        )

        points.append(
            [round(px), round(py)]
        )

    points = np.array(
        points,
        dtype=np.int32,
    )

    cv2.polylines(
        image,
        [points],
        False,
        SYMBOL_COLOR,
        2,
        lineType=cv2.LINE_AA,
    )


def _draw_window(
    image: np.ndarray,
    window: Window,
    wall: Wall,
) -> None:

    ux, uy = _wall_unit_vector(wall)
    nx, ny = _wall_normal(wall)

    half = window.width / 2

    offset = wall.thickness * 0.32

    a = (
        window.x - ux * half,
        window.y - uy * half,
    )

    b = (
        window.x + ux * half,
        window.y + uy * half,
    )

    for side in (-1, 1):

        p1 = (
            round(a[0] + nx * offset * side),
            round(a[1] + ny * offset * side),
        )

        p2 = (
            round(b[0] + nx * offset * side),
            round(b[1] + ny * offset * side),
        )

        cv2.line(
            image,
            p1,
            p2,
            SYMBOL_COLOR,
            2,
            lineType=cv2.LINE_AA,
        )


def render_floorplan(
    width: int,
    height: int,
    walls: list[Wall],
    doors: list[Door],
    windows: list[Window],
    output_path: str | Path,
) -> None:

    image = np.full(
        (height, width),
        BACKGROUND,
        dtype=np.uint8,
    )

    wall_by_id = {
        wall.id: wall
        for wall in walls
    }

    openings = {
        wall.id: []
        for wall in walls
    }

    for door in doors:

        wall = wall_by_id[door.wall_id]

        position = _opening_position(
            wall,
            door.x,
            door.y,
        )

        openings[wall.id].append(
            (position, door.width)
        )

    for window in windows:

        wall = wall_by_id[window.wall_id]

        position = _opening_position(
            wall,
            window.x,
            window.y,
        )

        openings[wall.id].append(
            (position, window.width)
        )

    # Draw walls.
    for wall in walls:

        _draw_wall(
            image,
            wall,
            openings[wall.id],
        )

    # Draw doors.
    for door in doors:

        _draw_door(
            image,
            door,
            wall_by_id[door.wall_id],
        )

    # Draw windows.
    for window in windows:

        _draw_window(
            image,
            window,
            wall_by_id[window.wall_id],
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise RuntimeError(
            f"Failed to write image: {output_path}"
        )