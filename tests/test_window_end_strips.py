"""Window frame strips (engine.opening_detection._frame_strips): jamb ticks near the ends of one glazed
gap split narrow strips off it; they belong to the pane next to them, not openings of their own.
Separately drawn openings next to each other stay separate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engine.opening_detection import _frame_strips
from engine.structure import OpeningCandidate

T = 4.0                                          # wall thickness, px
Y = 50.5                                         # wall axis


def _piece(x0: float, x1: float) -> OpeningCandidate:
    return OpeningCandidate("", (x0, Y), (x1, Y), T, "horizontal", 1.0, 2, "wall_gap")


def _thin(x0: int, x1: int) -> np.ndarray:
    """Two lines along the wall (the window's faces / glazing) from x0 to x1."""
    thin = np.zeros((100, 300), np.uint8)
    thin[int(Y) - 1, x0:x1 + 1] = 255
    thin[int(Y) + 2, x0:x1 + 1] = 255
    return thin


def test_strips_at_both_ends_of_one_glazing_belong_to_its_pane():
    # 16.png window W3: a 57 px gap split at ticks 9 px from each end
    pieces = [_piece(100, 109), _piece(109, 149), _piece(149, 157)]
    assert _frame_strips(pieces, _thin(100, 157)) == {0: 1, 2: 1}


def test_adjacent_windows_of_comparable_width_stay_separate():
    # 20.jpg: three windows side by side share the same glazing lines
    pieces = [_piece(20, 100), _piece(100, 190), _piece(190, 280)]
    assert _frame_strips(pieces, _thin(20, 280)) == {}


def test_a_narrow_piece_drawn_differently_stays_its_own_opening():
    # the glazing starts at the tick: the strip before it is a bare gap
    assert _frame_strips([_piece(100, 109), _piece(109, 149)], _thin(109, 149)) == {}


def test_a_narrow_opening_beyond_a_wall_pier_is_not_a_strip():
    # same lines, but a piece of wall between them: two openings
    pieces = [_piece(100, 109), _piece(115, 155)]
    assert _frame_strips(pieces, _thin(100, 155)) == {}


REAL = Path(__file__).resolve().parents[2] / "Test Cases"


@pytest.mark.skipif(not (REAL / "16.png").exists(), reason="real test plans not available")
def test_16png_window_w3_is_one_window_over_the_whole_wall_opening():
    from benchmark.real_plans import DEFAULT_DIR, load_plan
    from engine.opening_detection import classify_openings
    from engine.structure import analyze_structure

    spec = json.loads((Path(__file__).resolve().parents[1] / "benchmark/fixtures/real_plans.json").read_text())["plans"]["16.png"]
    image = load_plan("16.png", spec, DEFAULT_DIR)
    openings = classify_openings(image, analyze_structure(image))
    near = [o for o in openings if 95 <= o["center"][0] <= 162 and 345 <= o["center"][1] <= 363]
    assert [(o["type"], o["start"][0], o["end"][0]) for o in near] == [("window", 100, 157)]
