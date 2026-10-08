"""Object outlines on a wall axis are not opening symbols (furniture must not create partitions)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.structure import EXTERIOR, analyze_structure


def _space(s, x, y) -> int:
    return int(s.space_labels[y, x])


def _counter_plan():
    """One open room: a wall stub from the top, then a wide opening down to the bottom wall.
    A kitchen counter (thin outline) has its left edge exactly on the stub's axis."""
    img = np.full((420, 640, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 20), (620, 400), (0, 0, 0), 10)
    cv2.line(img, (320, 20), (320, 150), (0, 0, 0), 10)                 # wall stub
    cv2.rectangle(img, (320, 250), (390, 395), (140, 140, 140), 1)       # counter outline (grey, thin)
    return img


@pytest.mark.parametrize("flag,joined", [("1", True), ("0", False)])
def test_counter_edge_on_the_wall_axis_does_not_seal_the_opening(monkeypatch, flag, joined):
    monkeypatch.setenv("FLOORPLAN_OBJECT_EDGES", flag)
    s = analyze_structure(_counter_plan())
    left, right = _space(s, 150, 300), _space(s, 500, 300)
    assert left > 0 and right > 0
    assert (left == right) is joined


REAL_PLANS = Path(__file__).resolve().parents[1].parent / "Test Cases"


def _gt(name):
    plans = json.loads((Path(__file__).resolve().parents[1] / "benchmark/fixtures/real_plans.json").read_text())["plans"]
    return plans[name]


@pytest.mark.skipif(not (REAL_PLANS / "6.png").exists(), reason="real test plans not available")
@pytest.mark.parametrize("name", ["6.png", "11.png"])
def test_real_open_plans_with_counters_stay_one_space(monkeypatch, name):
    """GT 'must_share' zones (no wall between them, only counters / appliances) are one space."""
    monkeypatch.setenv("FLOORPLAN_OBJECT_EDGES", "1")
    spec = _gt(name)
    s = analyze_structure(cv2.imread(str(REAL_PLANS / name)))
    for group in spec["physical"]["must_share"]:
        spaces = {_space(s, *spec["rooms"][i]["point"]) for i in group}
        assert len(spaces) == 1 and 0 not in spaces, (name, [spec["rooms"][i]["name"] for i in group], spaces)


@pytest.mark.skipif(not (REAL_PLANS / "14.png").exists(), reason="real test plans not available")
def test_real_overhead_garage_door_still_closes_the_garage(monkeypatch):
    """The open overhead door is drawn jamb to jamb: opening evidence, not a room object."""
    monkeypatch.setenv("FLOORPLAN_OBJECT_EDGES", "1")
    spec = _gt("14.png")
    garage = next(r for r in spec["rooms"] if r["name"] == "Garage")
    s = analyze_structure(cv2.imread(str(REAL_PLANS / "14.png")))
    assert _space(s, *garage["point"]) > 0
