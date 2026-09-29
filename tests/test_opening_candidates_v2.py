import numpy as np
import pytest

from engine.geometry_v2.opening_candidates_v2 import OpeningCandidateGeneratorV2


def _image_for(mask):
    return np.zeros((*mask.shape, 3), dtype=np.uint8)


def _horizontal_gap_mask():
    mask = np.zeros((140, 220), dtype=np.uint8)
    mask[42:54, 10:210] = 255
    mask[42:54, 100:140] = 0
    return mask


def _vertical_gap_mask():
    mask = np.zeros((200, 160), dtype=np.uint8)
    mask[20:180, 60:72] = 255
    mask[90:125, 60:72] = 0
    return mask


def test_horizontal_candidate_uses_the_actual_wall_and_gap_coordinates():
    mask = _horizontal_gap_mask()
    candidates = OpeningCandidateGeneratorV2().detect(_image_for(mask), mask)

    horizontal = [item for item in candidates if item.orientation == "horizontal"]
    assert len(horizontal) == 1
    candidate = horizontal[0]
    assert candidate.start[0] == 100
    assert candidate.end[0] == 139
    assert 42 <= candidate.start[1] < 54
    assert candidate.start[1] == candidate.end[1]
    assert candidate.width == pytest.approx(40)
    assert candidate.wall_thickness == pytest.approx(12)


def test_vertical_candidate_uses_the_actual_wall_and_gap_coordinates():
    mask = _vertical_gap_mask()
    candidates = OpeningCandidateGeneratorV2().detect(_image_for(mask), mask)

    vertical = [item for item in candidates if item.orientation == "vertical"]
    assert len(vertical) == 1
    candidate = vertical[0]
    assert candidate.start[1] == 90
    assert candidate.end[1] == 124
    assert 60 <= candidate.start[0] < 72
    assert candidate.start[0] == candidate.end[0]
    assert candidate.width == pytest.approx(35)
    assert candidate.wall_thickness == pytest.approx(12)


def test_minimum_and_maximum_width_factors_are_applied():
    mask = _horizontal_gap_mask()
    image = _image_for(mask)
    assert OpeningCandidateGeneratorV2(min_width_factor=4).detect(image, mask) == []
    assert OpeningCandidateGeneratorV2(max_width_factor=3).detect(image, mask) == []


def test_uninterrupted_wall_does_not_produce_candidates():
    mask = np.zeros((120, 220), dtype=np.uint8)
    mask[40:52, 10:210] = 255
    assert OpeningCandidateGeneratorV2().detect(_image_for(mask), mask) == []


def test_invalid_width_factor_range_is_rejected():
    with pytest.raises(ValueError):
        OpeningCandidateGeneratorV2(min_width_factor=3, max_width_factor=2)
