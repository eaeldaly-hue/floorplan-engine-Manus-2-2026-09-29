import json
from pathlib import Path

import pytest

from engine.analyzer import _parse_dimensions


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_plan_truth.json"


def test_sample_truth_has_all_17_printed_room_labels():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rooms = data["rooms"]
    assert len(rooms) == 17
    assert len({room["id"] for room in rooms}) == 17
    assert all(room["name"] and room["dimensions"] for room in rooms)
    assert all(len(room["label_center_px"]) == 2 for room in rooms)


def test_all_reference_dimensions_parse_to_positive_rectangular_estimates():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for room in data["rooms"]:
        result = _parse_dimensions(room["dimensions"])
        assert result is not None, room["id"]
        assert result["unit"] == "ft"
        assert result["width"] > 0
        assert result["height"] > 0
        assert result["area"] > 0


def test_feet_and_inches_convert_to_square_feet():
    result = _parse_dimensions('22\'11" x 18\'0"')
    assert result is not None
    assert result["width"] == pytest.approx(22 + 11 / 12, abs=0.001)
    assert result["height"] == pytest.approx(18)
    assert result["area"] == pytest.approx(412.5)


def test_unicode_multiplication_and_prime_marks_are_supported():
    result = _parse_dimensions("3′10″ × 4′0″")
    assert result is not None
    assert result["width"] == pytest.approx(46 / 12, abs=0.001)
    assert result["height"] == pytest.approx(4)


@pytest.mark.parametrize("text", ["", "13' x 17m", "13 x 17", "13'12\" x 8'0\""])
def test_invalid_or_unsupported_dimensions_are_rejected(text):
    assert _parse_dimensions(text) is None
