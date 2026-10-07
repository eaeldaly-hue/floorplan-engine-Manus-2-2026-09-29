import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.pipeline import run


ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "test_floorplan.png"
TRUTH = ROOT / "tests" / "fixtures" / "sample_plan_truth.json"


def _label_anchor_coverage(rooms, anchors):
    polygons = [
        np.asarray(room["polygon"], dtype=np.int32)
        for room in rooms
        if len(room["polygon"]) >= 3
    ]
    covered = 0
    for item in anchors:
        point = tuple(float(value) for value in item["label_center_px"])
        if any(cv2.pointPolygonTest(polygon, point, False) >= 0 for polygon in polygons):
            covered += 1
    return covered


def test_pipeline_returns_in_bounds_room_polygons(tmp_path):
    output_dir = tmp_path / "nested" / "analysis"
    result = run(str(IMAGE), output_dir)

    assert result["wall_segments"]
    assert result["rooms"]
    assert (output_dir / "result.json").is_file()
    assert (output_dir / "detected_rooms.png").is_file()

    height, width = cv2.imread(str(IMAGE)).shape[:2]
    for room in result["rooms"]:
        assert room["area_pixels"] > 0
        assert len(room["polygon"]) >= 3
        assert all(0 <= x < width and 0 <= y < height for x, y in room["polygon"])


def test_baseline_fixture_covers_the_full_source_image():
    data = json.loads(TRUTH.read_text(encoding="utf-8"))
    width, height = data["image_size_px"]
    assert cv2.imread(str(IMAGE)).shape[1::-1] == (width, height)
    assert all(
        0 <= x < width and 0 <= y < height
        for room in data["rooms"]
        for x, y in [room["label_center_px"]]
    )


def test_room_boundaries_cover_at_least_15_of_17_printed_room_labels(tmp_path):
    # Was a strict xfail at 7/17 before the Phase 3 structure engine (now 17/17).
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    result = run(str(IMAGE), tmp_path / "label-analysis")
    assert _label_anchor_coverage(result["rooms"], truth["rooms"]) >= 15
