from pathlib import Path

import cv2
import pytest

from engine.analyzer import FloorPlanAnalyzer, _parse_dimensions, _room_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "test_floorplan.png"


def test_dimension_parser_handles_common_units_and_rejects_partial_ocr():
    imperial = _parse_dimensions('13\'4" x 17\'0"')
    assert imperial is not None
    assert imperial["unit"] == "ft"
    assert imperial["area"] == pytest.approx(226.67, abs=0.02)

    inches_and_compact = _parse_dimensions('19" x 5\'4!')
    assert inches_and_compact is not None
    assert inches_and_compact["area"] == pytest.approx(8.44, abs=0.03)
    assert _parse_dimensions("4'2\" x 4'a\"") is None

    metric = _parse_dimensions("3.5 m x 4.0 m")
    assert metric is not None
    assert metric["unit"] == "m"
    assert metric["area"] == pytest.approx(14.0)

    arabic_metric = _parse_dimensions("٣٫٥ م × ٤ م")
    assert arabic_metric is not None
    assert arabic_metric["area"] == pytest.approx(14.0)
    assert _room_name("غرفة المعيشة") == "غرفة معيشة"
    assert _room_name("مطبخ") == "مطبخ"


def test_bundled_floor_plan_returns_room_records_and_in_image_boundaries():
    image = cv2.imread(str(SAMPLE_PATH), cv2.IMREAD_COLOR)
    assert image is not None
    result = FloorPlanAnalyzer().analyze(image, SAMPLE_PATH.name)

    assert result["room_count"] >= 14
    assert result["image"] == {"width": image.shape[1], "height": image.shape[0]}
    assert result["pixel_scale"] is not None
    assert result["pixel_scale"]["unit"] == "ft"
    assert len(result["overlay_png"]) > 50_000

    names = {room["name"].lower() for room in result["rooms"]}
    assert "kitchen" in names
    assert "garage" in names
    assert any("bedroom" in name for name in names)

    with_dimensions = [room for room in result["rooms"] if room["dimensions"]]
    with_boundaries = [room for room in result["rooms"] if room["boundary"]]
    with_area = [room for room in result["rooms"] if room["area"]["value"] is not None]
    assert len(with_dimensions) >= 12
    assert len(with_boundaries) >= result["room_count"] - 2
    assert len(with_area) >= result["room_count"] - 2

    for room in with_boundaries:
        boundary = room["boundary"]
        polygon = boundary["polygon"]
        assert len(polygon) >= 4
        for point in polygon:
            assert 0 <= point["x"] < image.shape[1]
            assert 0 <= point["y"] < image.shape[0]

    kitchen = next(room for room in result["rooms"] if room["name"] == "Kitchen")
    assert kitchen["area"]["unit"] == "ft²"
    assert kitchen["area"]["value"] == pytest.approx(226.67, abs=0.1)
