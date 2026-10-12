"""Duplicate single-line openings (engine.structure._same_drawn_line): two Hough pieces of one
drawn line, tilted the opposite ways, are one opening; distinct nearby openings never merge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.structure import _same_drawn_line

T = 10.0                                         # wall thickness, px


def test_crossing_pieces_of_one_line_are_one_opening():
    # 5.jpeg: the two candidates reported for door D6 (580..626 at y ~312)
    assert _same_drawn_line((580.1, 315.7), (626.0, 311.9), (580.0, 308.0), (626.0, 314.0), T)
    # 11.png: two pieces of the same short line, crossing
    assert _same_drawn_line((204.0, 18.8), (223.0, 20.2), (204.0, 20.0), (223.0, 19.0), 6.0)


def test_parallel_outlines_of_one_opening_are_not_merged():
    # an opening's two face lines, 0.5 t apart: they never cross
    assert not _same_drawn_line((100, 200), (160, 200), (100, 205), (160, 205), T)


def test_adjacent_openings_on_one_wall_line_are_not_merged():
    # a door and the window next to it, end to end on the same wall line
    assert not _same_drawn_line((100, 200), (150, 200), (160, 200), (230, 200), T)
    # overlapping only at their ends
    assert not _same_drawn_line((100, 200), (150, 200), (140, 199), (220, 201), T)


def test_openings_on_different_wall_lines_are_not_merged():
    # perpendicular lines crossing at a corner
    assert not _same_drawn_line((100, 200), (160, 200), (130, 170), (130, 230), T)
    # crossing, same direction, but further apart than the narrowest line pair
    assert not _same_drawn_line((100, 190), (160, 214), (100, 214), (160, 190), T)


def test_partially_overlapping_spans_are_not_merged():
    # crossing pieces whose middles are more than half a span apart: two different stretches
    assert not _same_drawn_line((100, 198), (200, 202), (150, 202), (250, 198), T)


REAL = Path(__file__).resolve().parents[2] / "Test Cases"


@pytest.mark.skipif(not (REAL / "5.jpeg").exists(), reason="real test plans not available")
def test_5jpeg_door_d6_reported_once():
    from benchmark.real_plans import DEFAULT_DIR, load_plan
    from engine.opening_detection import classify_openings
    from engine.structure import analyze_structure

    spec = json.loads((Path(__file__).resolve().parents[1] / "benchmark/fixtures/real_plans.json").read_text())["plans"]["5.jpeg"]
    image = load_plan("5.jpeg", spec, DEFAULT_DIR)
    openings = classify_openings(image, analyze_structure(image))
    near = [o for o in openings if 575 <= o["center"][0] <= 631 and 300 <= o["center"][1] <= 322]
    assert len(near) == 1 and near[0]["type"] == "door"
