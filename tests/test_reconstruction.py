"""Hypothesis-based reconstruction (engine.reconstruction), rescaling back to the original frame
(engine.rescale), symbols read at a legible scale (engine.arch.symbol_scale)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine import reconstruction as R
from engine.arch.symbol_scale import transfer_types
from engine.rescale import to_original
from engine.structure import _along_walls, analyze_structure

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT.parent / "Test Cases"


def _gt(name):
    return json.loads((ROOT / "benchmark/fixtures/real_plans.json").read_text())["plans"][name]


def test_rescale_maps_pixel_geometry_and_keeps_real_units():
    r = {"image": {"width": 200, "height": 100},
         "rooms": [{"boundary": {"polygon": [{"x": 20, "y": 40}], "bbox": {"x": 20, "y": 40, "width": 60, "height": 30}},
                    "label_center": {"x": 10, "y": 8}, "area": {"value": 400, "unit": "px²"},
                    "dimensions": {"width": 12.5, "height": 10.0, "area": 125.0}}],
         "openings": [{"start": [10, 20], "end": [30, 20], "center": [20, 20], "width_pixels": 20,
                       "evidence": {"wall_thickness_px": 8.0, "width_in_wall_thicknesses": 2.5}}],
         "pixel_scale": {"pixels_per_unit": 40.0, "unit": "ft", "calibration_rooms": 3},
         "building": {"spaces": [{"area": 12.0, "area_px": 400, "center": [20, 20], "polygon": [[20, 40]]}],
                      "summary": {"spaces": 1}}}
    out = to_original(r, 2.0, 100, 50)
    room = out["rooms"][0]
    assert room["boundary"]["polygon"][0] == {"x": 10, "y": 20}
    assert room["boundary"]["bbox"] == {"x": 10, "y": 20, "width": 30, "height": 15}
    assert room["area"]["value"] == 100 and room["dimensions"] == {"width": 12.5, "height": 10.0, "area": 125.0}
    o = out["openings"][0]
    assert o["start"] == [5, 10] and o["width_pixels"] == 10 and o["evidence"]["wall_thickness_px"] == 4.0
    assert o["evidence"]["width_in_wall_thicknesses"] == 2.5
    assert out["pixel_scale"] == {"pixels_per_unit": 20.0, "unit": "ft", "calibration_rooms": 3}
    assert out["building"]["spaces"][0] == {"area": 12.0, "area_px": 100, "center": [10, 10], "polygon": [[10, 20]]}
    assert out["image"] == {"width": 100, "height": 50} and out["building"]["summary"] == {"spaces": 1}


def _plan(wall: int, scale: float = 1.0):
    img = np.full((int(420 * scale), int(640 * scale), 3), 255, np.uint8)
    s = lambda v: int(v * scale)                                    # noqa: E731
    cv2.rectangle(img, (s(20), s(20)), (s(620), s(400)), (0, 0, 0), wall)
    cv2.line(img, (s(320), s(20)), (s(320), s(170)), (0, 0, 0), wall)
    cv2.line(img, (s(320), s(240)), (s(320), s(400)), (0, 0, 0), wall)
    return img


def test_easy_plans_run_once_and_under_resolved_plans_branch():
    big = _plan(14)
    assert [h.name for h in R.generate(big, analyze_structure(big))] == ["default"]
    small = _plan(4)
    names = [h.name for h in R.generate(small, analyze_structure(small))]
    assert "normalised-scale" in names


def test_default_wins_ties_and_better_evidence_wins():
    img = _plan(14)
    s = analyze_structure(img)
    a = R.Hypothesis("default", 1.0, img, s, "")
    b = R.Hypothesis("other", 1.0, img, s, "")
    best, rep = R.choose([a, b], [("Bedroom", (150, 200)), ("Bath", (480, 200))], [])
    assert best is a and rep["chosen"] == "default"
    empty_img = np.full_like(img, 255)
    failed = R.Hypothesis("default", 1.0, empty_img, analyze_structure(empty_img), "")
    good = R.Hypothesis("wall-class", 1.0, img, s, "")
    best, rep = R.choose([failed, good], [("Bedroom", (150, 200))], [])
    assert best is good and rep["chosen"] == "wall-class" and failed.score == -1e9


def test_symbols_are_retyped_from_the_legible_reading():
    native = [{"type": "door", "type_label": "Door", "confidence": 0.7, "start": [10, 50], "end": [30, 50],
               "center": [20, 50], "evidence": {"symbol": "double swing"}}]
    legible = [{"type": "window", "type_label": "Window", "confidence": 0.8, "start": [31, 150], "end": [89, 150],
                "evidence": {"symbol": "glazing lines", "glazing_lines": 2}}]
    assert transfer_types(native, legible, 3.0, 4.0) == 1
    assert native[0]["type"] == "window" and native[0]["evidence"]["symbol"] == "glazing lines"
    assert native[0]["evidence"]["symbol_at_native_scale"]["type"] == "door"


def test_wall_directions():
    assert _along_walls(np.array([10.0, 0.4]), [0.0, 90.0]) and not _along_walls(np.array([10.0, 10.0]), [0.0, 90.0])


@pytest.mark.skipif(not (REAL / "7.png").exists(), reason="real test plans not available")
def test_real_under_resolved_plan_is_read_at_its_normalised_scale():
    from benchmark.real_plans import full_room_metrics
    from engine.analyzer import FloorPlanAnalyzer

    img = cv2.imread(str(REAL / "7.png"))
    r = FloorPlanAnalyzer().analyze(img, "7.png")
    assert r["reconstruction"]["chosen"] == "normalised-scale"
    assert r["image"] == {"width": img.shape[1], "height": img.shape[0]}
    m = full_room_metrics(r, _gt("7.png")["rooms"])
    assert m["recovered"] >= 8                                       # was 2 of 12 with the single reading
    for o in r["openings"]:
        assert 0 <= o["center"][0] <= img.shape[1] and 0 <= o["center"][1] <= img.shape[0]


@pytest.mark.skipif(not (REAL / "5.jpeg").exists(), reason="real test plans not available")
def test_real_wall_class_hypothesis_rescues_a_collapsed_reading():
    from engine.analyzer import FloorPlanAnalyzer

    img = cv2.resize(cv2.imread(str(REAL / "5.jpeg")), None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    r = FloorPlanAnalyzer().analyze(img, "5.jpeg")
    assert r["reconstruction"]["chosen"] == "wall-class"
    assert r["opening_count"] >= 10
