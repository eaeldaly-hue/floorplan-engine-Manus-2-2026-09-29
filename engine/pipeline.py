"""
Floor plan engine - command-line geometry pipeline (no OCR).

Pipeline: image -> wall mask -> wall bands -> wall gaps (opening candidates)
-> sealed spaces -> topology -> door/window classification.

Uses the same engine.structure pass as the web analyzer, so the CLI and
the app agree on walls, spaces and openings.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from engine.opening_detection import classify_openings, draw_openings_overlay
from engine.structure import analyze_structure


IMAGE_PATH = "test_floorplan.png"
OUTPUT_DIR = Path("output")


def _draw_walls(image, bands):
    vis = image.copy()
    for band in bands:
        cv2.line(vis, tuple(map(int, band.p0)), tuple(map(int, band.p1)), (0, 0, 255), max(2, int(band.thickness / 6)))
    return vis


def _draw_rooms(image, rooms):
    vis = image.copy()
    for room in rooms:
        polygon = np.array(room["polygon"], dtype=np.int32)
        cv2.polylines(vis, [polygon], True, (0, 140, 255), 4)
        cv2.putText(vis, room["id"], room["center"], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    return vis


def run(image_path=IMAGE_PATH, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    print(f"Loaded {image_path} ({image.shape[1]}x{image.shape[0]})")

    structure = analyze_structure(image)
    cv2.imwrite(str(output_dir / "wall_mask.png"), structure.wall_mask)
    cv2.imwrite(str(output_dir / "room_mask.png"), structure.sealed_mask)
    print(f"Wall bands: {len(structure.bands)} (typical thickness {structure.wall_thickness:.0f}px)")
    cv2.imwrite(str(output_dir / "wall_segments.png"), _draw_walls(image, structure.bands))

    rooms = [{**space, "id": f"room_{k}"} for k, space in enumerate(structure.spaces, 1)]
    print(f"Rooms detected: {len(rooms)}")
    for room in rooms:
        print(f"  {room['id']} | center={room['center']} | area_px={room['area_pixels']}")
    cv2.imwrite(str(output_dir / "detected_rooms.png"), _draw_rooms(image, rooms))

    openings = classify_openings(image, structure)
    counts = {kind: sum(o["type"] == kind for o in openings) for kind in ("door", "window", "opening")}
    print(f"Openings: {len(openings)} ({counts['door']} doors, {counts['window']} windows, {counts['opening']} unknown)")
    (output_dir / "openings.png").write_bytes(draw_openings_overlay(image, openings))

    result = {
        "image": str(image_path),
        "wall_segments": [
            {"orientation": b.orientation, "start": [round(b.p0[0]), round(b.p0[1])], "end": [round(b.p1[0]), round(b.p1[1])],
             "length": round(float(np.hypot(b.p1[0] - b.p0[0], b.p1[1] - b.p0[1])), 1), "thickness": round(b.thickness, 1)}
            for b in structure.bands
        ],
        "rooms": rooms,
        "openings": openings,
        "room_adjacency": sorted([list(p) for p in structure.wall_adjacency]),
    }
    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nOutputs written to: {output_dir}/")
    return result


if __name__ == "__main__":
    run()
