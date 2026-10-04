from pathlib import Path
import json

import cv2

from engine.analyzer import FloorPlanAnalyzer
from engine.rooms.reconstruction_engine import (
    RoomReconstructionEngine,
    draw_reconstruction,
)
from engine.walls.segments import WallSegmentExtractor


IMAGE = Path("test_floorplan.png")
OUT_DIR = Path("output_v2")
OUT_DIR.mkdir(exist_ok=True)

captured_segments = []

_original_detect = WallSegmentExtractor.detect


def _capture_detect(self):
    segments = _original_detect(self)
    captured_segments.extend(segments)
    return segments


WallSegmentExtractor.detect = _capture_detect


image = cv2.imread(str(IMAGE))

if image is None:
    raise SystemExit(f"Could not read {IMAGE}")


analyzer = FloorPlanAnalyzer()
analyzer.analyze(
    image,
    source_name=IMAGE.name,
)


print()
print("=" * 60)
print("ROOM RECONSTRUCTION ENGINE v1")
print("=" * 60)

print(f"Image               : {image.shape[1]} x {image.shape[0]}")
print(f"Wall segments       : {len(captured_segments)}")


engine = RoomReconstructionEngine(
    snap_tolerance=10,
    min_room_area=7000,
    max_room_area_ratio=0.85,
    min_rectangularity=0.15,
)

result = engine.reconstruct(
    captured_segments,
    image.shape,
)


print(f"Wall groups         : {result.wall_groups}")
print(f"Virtual bridges     : {result.virtual_bridges}")
print(f"Graph nodes         : {result.graph_nodes}")
print(f"Graph edges         : {result.graph_edges}")
print(f"Closed faces        : {result.raw_faces}")
print(f"Valid room polygons : {result.valid_rooms}")
print()


for room in result.rooms:
    print(
        f"ROOM {room.id:02d} | "
        f"area={room.area:,.0f} | "
        f"bbox={room.bbox} | "
        f"rect={room.rectangularity:.2f} | "
        f"vertices={len(room.polygon)}"
    )


json_data = {
    "wall_segments": result.wall_segments,
    "wall_groups": result.wall_groups,
    "virtual_bridges": result.virtual_bridges,
    "graph_nodes": result.graph_nodes,
    "graph_edges": result.graph_edges,
    "raw_faces": result.raw_faces,
    "valid_rooms": result.valid_rooms,
    "rooms": [
        {
            "id": room.id,
            "node_ids": room.node_ids,
            "polygon": room.polygon,
            "area": room.area,
            "bbox": room.bbox,
            "rectangularity": room.rectangularity,
            "perimeter": room.perimeter,
        }
        for room in result.rooms
    ],
}


json_path = OUT_DIR / "room-reconstruction-engine-v1.json"

json_path.write_text(
    json.dumps(json_data, indent=2),
    encoding="utf-8",
)


image_path = OUT_DIR / "room-reconstruction-engine-v1.png"

draw_reconstruction(
    image,
    result,
    image_path,
)


print()
print(f"JSON output         : {json_path}")
print(f"Visualization       : {image_path}")
print("=" * 60)
