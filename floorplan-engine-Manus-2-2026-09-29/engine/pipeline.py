"""
Floor plan engine - entry point.

Pipeline: image -> wall mask -> wall segments (with real thickness) ->
room mask (doors/windows bridged) -> enclosed rooms.

This replaced an earlier version that ran two unfinished, disconnected
detection systems side by side (a fragile Hough-line pipeline in
`geometry/`, and a half-wired region-based system in `structural/` that
was never actually called from here). See MIGRATION_NOTES.md for what
changed and why.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from engine.walls.mask import WallMaskBuilder
from engine.walls.segments import WallSegmentExtractor
from engine.walls.room_mask import RoomMaskBuilder
from engine.rooms.detect import RoomDetector


IMAGE_PATH = "test_floorplan.png"
OUTPUT_DIR = Path("output")


def _draw_walls(image, segments):
    vis = image.copy()
    for s in segments:
        cv2.line(vis, s["start"], s["end"], (0, 0, 255), max(2, int(s["thickness"] / 6)))
    return vis


def _draw_rooms(image, rooms):
    vis = image.copy()
    for room in rooms:
        polygon = np.array(room["polygon"], dtype=np.int32)
        cv2.polylines(vis, [polygon], True, (0, 140, 255), 4)
        cv2.putText(
            vis, room["id"], room["center"],
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA,
        )
    return vis


def run(image_path=IMAGE_PATH, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    print(f"Loaded {image_path} ({image.shape[1]}x{image.shape[0]})")

    # 1. Wall mask: thick, solid strokes only (text/door-swings/window
    # ticks are thin and get filtered out by design, not by a fixed
    # pixel count - see WallMaskBuilder for why).
    wall_mask = WallMaskBuilder(image).build()
    cv2.imwrite(str(output_dir / "wall_mask.png"), wall_mask)

    # 2. Wall segments: row/column run-scanning, immune to the T/L
    # junction problem that breaks connected-components extraction.
    segments = WallSegmentExtractor(wall_mask).detect()
    print(f"Wall segments detected: {len(segments)}")

    cv2.imwrite(
        str(output_dir / "wall_segments.png"),
        _draw_walls(image, segments),
    )

    # 3. Room mask: same segments, but with door/window-sized gaps
    # bridged so a real room doesn't leak into its neighbour through an
    # open doorway. A separate, capped pass seals the true exterior
    # silhouette even where it's drawn with an unusually faint line.
    room_mask_builder = RoomMaskBuilder(segments, wall_mask.shape, source_image=image)
    room_mask = room_mask_builder.build()
    cv2.imwrite(str(output_dir / "room_mask.png"), room_mask)

    if room_mask_builder.last_unsealed_gap:
        print(
            f"WARNING: a {room_mask_builder.last_unsealed_gap}px gap in the "
            "drawing was too wide to safely auto-bridge (likely a faint "
            "line or an open-concept boundary). Any room that opens onto "
            "it will read as merged with the outside. See MIGRATION_NOTES.md."
        )

    # 4. Rooms: flood fill from outside the closed room mask.
    rooms = RoomDetector(room_mask).detect()
    print(f"Rooms detected: {len(rooms)}")
    for room in rooms:
        print(
            f"  {room['id']} | center={room['center']} | "
            f"area_px={room['area_pixels']}"
        )

    cv2.imwrite(
        str(output_dir / "detected_rooms.png"),
        _draw_rooms(image, rooms),
    )

    result = {
        "image": str(image_path),
        "wall_segments": segments,
        "rooms": rooms,
    }
    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nOutputs written to: {output_dir}/")
    return result


if __name__ == "__main__":
    run()
