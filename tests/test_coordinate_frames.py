"""One coordinate frame for the whole result: walls of a deskewed plan (engine.structure.analyze_structure)
and door hinges of a plan analysed at a working scale (engine.rescale) end up in the original image's
pixels, like every other element."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.rescale import to_original
from engine.structure import analyze_structure


def _plan() -> np.ndarray:
    """Two rooms: outer walls, a partition with a door gap, a window gap in the top wall."""
    img = np.full((520, 760, 3), 255, np.uint8)
    cv2.rectangle(img, (60, 60), (700, 460), (0, 0, 0), 10)
    cv2.line(img, (380, 60), (380, 220), (0, 0, 0), 10)
    cv2.line(img, (380, 300), (380, 460), (0, 0, 0), 10)
    cv2.line(img, (160, 60), (260, 60), (255, 255, 255), 10)           # window gap
    return img


def _rotate(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    W, H = int(w * c + h * s), int(w * s + h * c)
    m[0, 2] += W / 2 - w / 2
    m[1, 2] += H / 2 - h / 2
    return cv2.warpAffine(img, m, (W, H), borderValue=(255, 255, 255))


def _on_walls(points, wall_mask) -> float:
    m = cv2.dilate((wall_mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    h, w = m.shape
    hits = [0 <= int(round(x)) < w and 0 <= int(round(y)) < h and m[int(round(y)), int(round(x))] > 0 for x, y in points]
    return sum(hits) / max(1, len(hits))


def _band_points(bands):
    return [tuple(np.add(b.p0, np.subtract(b.p1, b.p0) * t)) for b in bands for t in np.linspace(0.1, 0.9, 9)]


@pytest.mark.parametrize("deg", [0.0, 3.0, -4.0])
def test_wall_bands_are_in_the_original_frame_after_deskew(deg):
    s = analyze_structure(_rotate(_plan(), deg) if deg else _plan())
    if deg:
        assert abs(s.skew_degrees + deg) < 0.5          # the plan was deskewed
    assert s.bands and _on_walls(_band_points(s.bands), s.wall_mask) >= 0.95
    if s.work is not None:                              # the working frame keeps its own bands
        assert _on_walls(_band_points(s.work.bands), s.work.wall_mask) >= 0.95


def test_openings_spaces_and_walls_share_one_frame_after_deskew():
    s = analyze_structure(_rotate(_plan(), 3.0))
    ends = [np.array(p, float) for c in s.candidates for p in (c.start, c.end)]

    def to_wall(p):
        best = 1e9
        for b in s.bands:
            a, d = np.array(b.p0, float), np.subtract(b.p1, b.p0)
            t = np.clip(np.dot(p - a, d) / max(1e-9, np.dot(d, d)), 0, 1)
            best = min(best, float(np.linalg.norm(a + t * d - p)))
        return best
    assert ends and np.median([to_wall(p) for p in ends]) <= 3.0


def test_rescale_brings_hinges_and_run_lengths_back_to_the_original_frame():
    result = {"image": {"width": 300, "height": 200},
              "building": {"openings": [{"p0": [30, 60], "p1": [90, 60], "door": {"hinges": [[30, 60]], "leaf_angle": 90}}],
                           "wall_runs": [{"p0": [0, 0], "p1": [300, 0], "solid_length_px": 240, "open_length_px": 60}],
                           "summary": {"walls": 12}}}
    out = to_original(result, 3.0, 100, 67)
    o = out["building"]["openings"][0]
    assert o["p0"] == [10, 20] and o["door"]["hinges"] == [[10, 20]] and o["door"]["leaf_angle"] == 90
    run = out["building"]["wall_runs"][0]
    assert (run["solid_length_px"], run["open_length_px"]) == (80, 20)
    assert out["building"]["summary"]["walls"] == 12    # counts are not lengths


REAL = Path(__file__).resolve().parents[2] / "Test Cases"


@pytest.mark.skipif(not (REAL / "2.jpg").exists(), reason="real test plans not available")
def test_a_rotated_real_plan_keeps_its_walls_on_its_walls():
    from benchmark.real_plans import DEFAULT_DIR, load_plan
    from engine.arch import building as B
    from engine.analyzer import FloorPlanAnalyzer

    spec = json.loads((Path(__file__).resolve().parents[1] / "benchmark/fixtures/real_plans.json").read_text())["plans"]["2.jpg"]
    seen = {}
    orig = B.build_building

    def capture(structure, *a, **k):
        seen["s"] = structure
        return orig(structure, *a, **k)
    B.build_building = capture
    try:
        r = FloorPlanAnalyzer().analyze(_rotate(load_plan("2.jpg", spec, DEFAULT_DIR), 3.0), "2.jpg")
    finally:
        B.build_building = orig
    walls = r["building"]["walls"]
    pts = [tuple(np.add(w["p0"], np.subtract(w["p1"], w["p0"]) * t)) for w in walls for t in np.linspace(0.1, 0.9, 9)]
    assert abs(seen["s"].skew_degrees) > 1 and _on_walls(pts, seen["s"].wall_mask) >= 0.95
