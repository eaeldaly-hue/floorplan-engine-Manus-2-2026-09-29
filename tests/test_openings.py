"""Door / window / unknown classification: symbol-specific cases, the real sample, and a benchmark gate."""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmark.evaluate import OpeningStats, match
from benchmark.generator import Canvas, Style, draw_door, draw_window, family_specs, generate
from engine.opening_detection import classify_openings
from engine.structure import analyze_structure

ROOT = Path(__file__).resolve().parents[1]
T_EXT, T_INT = 24, 14


def two_rooms(door_style=None, window_style=None, door_on_exterior=False, line_px=2, interior_gap=True):
    """Two rooms side by side; one opening in the shared wall and one in the top exterior wall."""
    st = Style(line_px=line_px)
    cv = Canvas(1000, 720, st, random.Random(3))
    img = cv.img
    for (x0, y0, x1, y1, t) in ((100, 100, 900, 100, T_EXT), (100, 620, 900, 620, T_EXT), (100, 100, 100, 620, T_EXT),
                                (900, 100, 900, 620, T_EXT), (500, 100, 500, 620, T_INT)):
        cv2.rectangle(img, (min(x0, x1) - t // 2, min(y0, y1) - t // 2), (max(x0, x1) + t // 2, max(y0, y1) + t // 2), 0, -1)
    placed = []
    if door_style and interior_gap:
        p0, p1 = (500, 300), (500, 390)
        cv2.rectangle(img, (500 - T_INT // 2 - 1, 300), (500 + T_INT // 2 + 1, 390), 255, -1)
        draw_door(cv, p0, p1, T_INT, door_style, random.Random(5), side=1)
        placed.append(("door", p0, p1))
    if window_style or (door_style and door_on_exterior):
        p0, p1 = (250, 100), (400, 100)
        cv2.rectangle(img, (250, 100 - T_EXT // 2 - 1), (400, 100 + T_EXT // 2 + 1), 255, -1)
        if window_style:
            draw_window(cv, p0, p1, T_EXT, window_style, random.Random(6))
            placed.append(("window", p0, p1))
        else:
            draw_door(cv, p0, p1, T_EXT, door_style, random.Random(7), side=1)
            placed.append(("door", p0, p1))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), placed


def classified(image, placed):
    openings = classify_openings(image)
    gts = [{"kind": k, "p0": list(p0), "p1": list(p1), "wall_thickness_px": T_EXT} for k, p0, p1 in placed]
    pairs = match(openings, gts)
    return {j: openings[i] for i, j, _ in pairs}, openings


@pytest.mark.parametrize("style", ["swing", "leaf_only", "double", "outline", "sliding"])
def test_interior_door_styles_are_doors(style):
    image, placed = two_rooms(door_style=style)
    found, _ = classified(image, placed)
    assert 0 in found, f"{style} door not detected"
    assert found[0]["type"] == "door", found[0]["evidence"]
    assert found[0]["evidence"]["room_relation"] == "between_rooms"


@pytest.mark.parametrize("style", ["triple", "double", "sample", "weak"])
def test_window_styles_are_windows(style):
    image, placed = two_rooms(window_style=style)
    found, _ = classified(image, placed)
    assert 0 in found
    assert found[0]["type"] == "window", found[0]["evidence"]
    assert found[0]["evidence"]["room_relation"] == "room_to_exterior"


@pytest.mark.parametrize("style", ["swing", "double", "sliding"])
def test_exterior_doors_are_not_windows(style):
    image, placed = two_rooms(door_style=style, door_on_exterior=True, interior_gap=False)
    found, _ = classified(image, placed)
    assert found[0]["type"] == "door", found[0]["evidence"]


def test_plain_gap_is_unknown_not_a_door_or_window():
    image, _ = two_rooms()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[300:390, 500 - T_INT // 2 - 1:500 + T_INT // 2 + 2] = 255            # bare passage, no symbol
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    openings = classify_openings(image)
    [opening] = openings
    assert opening["type"] == "opening"
    assert opening["evidence"]["symbol"] == "bare gap"


def test_exterior_connection_alone_does_not_make_a_window():
    """A wide exterior gap closed only by one thin line is not called a window."""
    image, _ = two_rooms()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[100 - T_EXT // 2 - 1:100 + T_EXT // 2 + 2, 250:550] = 255
    cv2.line(gray, (250, 100 + T_EXT // 2), (550, 100 + T_EXT // 2), 160, 1)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    exterior = [o for o in classify_openings(image) if o["evidence"]["room_relation"] == "room_to_exterior"]
    assert exterior and all(o["type"] != "window" for o in exterior)


def test_api_fields_and_evidence_are_present():
    image, placed = two_rooms(door_style="swing", window_style="triple")
    openings = classify_openings(image)
    assert {o["type"] for o in openings} >= {"door", "window"}
    for o in openings:
        for key in ("id", "type", "type_label", "confidence", "start", "end", "center", "orientation", "width_pixels", "evidence"):
            assert key in o
        for key in ("symbol", "reason", "room_relation", "adjacent_space_ids", "arc_score", "glazing_lines", "face_lines"):
            assert key in o["evidence"]


def test_real_sample_matches_hand_labelled_openings():
    truth = json.loads((ROOT / "benchmark" / "fixtures" / "test_floorplan_openings.json").read_text())
    for o in truth["openings"]:
        o.setdefault("wall_thickness_px", 38.0 if o["id"].startswith("W") else 30.0)
    image = cv2.imread(str(ROOT / truth["image"]))
    stats = OpeningStats()
    stats.add(classify_openings(image, pixel_scale={"unit": "ft", "pixels_per_unit": 41.73}), truth["openings"])
    m = stats.metrics()
    assert m["candidate_recall"] == 1.0
    assert m["door_recall"] >= 16 / 17 and m["door_precision"] == 1.0
    assert m["window_recall"] == 1.0 and m["window_precision"] == 1.0
    assert m["passages_called_window"] == 0


# Regression floors (not targets): set a little below measured performance per family.
# Heavily degraded scans (blur + JPEG + speckle at low resolution) are a known weaker case.
GATES = {
    "door_swing": 0.9, "door_sliding": 0.85, "door_no_swing_outline": 0.9, "window_clear": 0.9,
    "window_weak": 0.9, "noise": 0.9, "holdout_scan": 0.75, "holdout_rotated": 0.75,
}


@pytest.mark.parametrize("family", sorted(GATES))
def test_synthetic_benchmark_gate(family):
    floor = GATES[family]
    stats = OpeningStats()
    for spec in family_specs(family, 2):
        plan = generate(spec)
        stats.add(classify_openings(plan.image, analyze_structure(plan.image)), plan.openings)
    m = stats.metrics()
    assert m["candidate_recall"] >= 0.95, m
    assert m["classification_accuracy"] >= floor, m
    for kind in ("door", "window"):
        if m[f"gt_{kind}s"]:
            assert m[f"{kind}_precision"] is None or m[f"{kind}_precision"] >= floor, m
            assert m[f"{kind}_recall"] >= floor - 0.05, m


def test_heavy_door_leaf_kept_as_wall_still_reads_as_a_door():
    """A leaf drawn as a thick stroke survives the wall filter; a free-standing stub of
    about the opening width from a jamb is still a door leaf (and not a wall)."""
    image, _ = two_rooms()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[300:390, 500 - T_INT // 2 - 1:500 + T_INT // 2 + 2] = 255
    cv2.rectangle(gray, (500 + T_INT // 2, 300), (500 + T_INT // 2 + 88, 307), 0, -1)   # 8 px leaf, opened 90°
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    s = analyze_structure(image)
    assert s.wall_mask[303, 560] > 0                       # the leaf really is in the wall mask
    [opening] = [o for o in classify_openings(image, s) if o["evidence"]["room_relation"] == "between_rooms"]
    assert opening["type"] == "door", opening["evidence"]
    assert "heavy" in opening["evidence"]["reason"]


def test_wall_stub_that_continues_is_not_a_door_leaf():
    """The same stub joined to another wall beyond its tip is a wall, not a leaf."""
    image, _ = two_rooms()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[300:390, 500 - T_INT // 2 - 1:500 + T_INT // 2 + 2] = 255
    cv2.rectangle(gray, (500 + T_INT // 2, 300), (900, 307), 0, -1)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for o in classify_openings(image):
        assert "heavy" not in o["evidence"]["reason"]


# --- Plan-level symbol conventions ------------------------------------------------------

def _outline_window(gray, x0, x1, y, t):
    gray[y - t // 2 - 1:y + t // 2 + 2, x0:x1] = 255
    cv2.rectangle(gray, (x0, y - t // 2), (x1, y + t // 2), 0, 1)


def test_outline_frames_are_windows_when_the_plan_draws_doors_with_swings():
    """No glazing-line windows anywhere, doors drawn with swing arcs: the repeated outline-only
    frames in exterior walls are this drawing's window symbol."""
    image, placed = two_rooms(door_style="swing")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for x0, x1 in ((180, 300), (600, 760)):
        _outline_window(gray, x0, x1, 620, T_EXT)
    _outline_window(gray, 250, 400, 100, T_EXT)
    openings = classify_openings(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    exterior = [o for o in openings if o["evidence"]["room_relation"] == "room_to_exterior"]
    assert len(exterior) == 3 and all(o["type"] == "window" for o in exterior), [o["evidence"] for o in exterior]
    assert all(o["evidence"].get("convention") == "outline windows" for o in exterior)
    assert [o["type"] for o in openings if o["evidence"]["room_relation"] == "between_rooms"] == ["door"]


def test_outline_exterior_door_stays_a_door_when_windows_use_glazing():
    image, _ = two_rooms(door_style="outline", door_on_exterior=True, interior_gap=False)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[620 - T_EXT // 2 - 1:620 + T_EXT // 2 + 2, 600:760] = 255
    for v in (-T_EXT // 2, 0, T_EXT // 2):
        cv2.line(gray, (600, 620 + v), (760, 620 + v), 0, 1)
    openings = classify_openings(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    kinds = sorted(o["type"] for o in openings if o["evidence"]["room_relation"] == "room_to_exterior")
    assert kinds == ["door", "window"], [o["evidence"] for o in openings]


REAL_PLANS = ROOT.parent / "Test Cases"


@pytest.mark.skipif(not (REAL_PLANS / "2.jpg").exists(), reason="real test plans not available")
def test_real_thin_wall_plan_rooms_doors_and_outline_windows():
    """Hand-labelled real plan with 5-6 px interior walls, swing doors and outline windows."""
    from benchmark.real_plans import FIXTURE, evaluate
    spec = json.loads(FIXTURE.read_text())["plans"]["2.jpg"]
    r = evaluate("2.jpg", spec, REAL_PLANS)
    assert r["rooms"]["separated"] == 5 and not r["catastrophic"]
    o = r["openings"]
    assert o["door_recall"] == 1.0 and o["door_precision"] == 1.0
    assert o["window_recall"] == 1.0 and o["window_precision"] == 1.0
