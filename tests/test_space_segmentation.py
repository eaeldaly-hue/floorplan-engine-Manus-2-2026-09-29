from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.preprocessing.text_mask import TextMasker
from engine.rooms.space_segmentation import SpaceSegmenter
from engine.structural.wall_region_detector import WallRegionDetector


def test_sample_plan_yields_multiple_enclosed_spaces():
    image = cv2.imread(str(PROJECT_ROOT / "test_floorplan.png"))
    assert image is not None, "sample floor plan fixture must be readable"

    clean_image, _ = TextMasker(image).remove_text()
    wall_mask = WallRegionDetector(clean_image).detect()
    spaces = SpaceSegmenter(clean_image, wall_mask=wall_mask).detect_spaces()

    # The sample has an open-plan living/kitchen area and several enclosed
    # rooms. A range accommodates minor OpenCV-version contour differences.
    assert 8 <= len(spaces) <= 12
    assert all(len(space["polygon"]) >= 4 for space in spaces)
    assert all(space["area_pixels"] >= 10000 for space in spaces)

    height, width = image.shape[:2]
    for space in spaces:
        box = space["bbox"]
        assert box["x"] > 0
        assert box["y"] > 0
        assert box["x"] + box["width"] < width
        assert box["y"] + box["height"] < height
