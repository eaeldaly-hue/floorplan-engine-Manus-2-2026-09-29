"""Architectural Plan Model: primitives, wall reconstruction across drawing styles, spaces,
openings, the analyzer fallback and its API payload."""

from __future__ import annotations

import json
import types

import cv2
import numpy as np
import pytest

from benchmark.evaluate import StructureStats
from benchmark.generator import generate
from benchmark.wall_styles import layout_specs, styled
from engine.plan import adapter
from engine.plan.model import Stroke
from engine.plan.primitives import extract_strokes, merge_strokes
from engine.plan.reconstruct import merge_collinear, reconstruct
from engine.plan.model import Wall


def _recall(style: str, k: int):
    p = generate(styled(layout_specs(4)[k], style))
    m = reconstruct(p.image)
    ss = StructureStats()
    ss.add(m._labels, p.room_labels, m._wall_mask, p.wall_mask)
    return m, ss.metrics()


# --- primitives -------------------------------------------------------------------------

def test_stroke_widths_and_centerlines():
    ink = np.zeros((200, 400), np.uint8)
    cv2.line(ink, (20, 50), (380, 50), 255, 2)          # thin line
    cv2.rectangle(ink, (20, 120), (380, 135), 255, -1)    # 16 px band
    strokes = [s for s in extract_strokes(ink) if s.length > 200]
    widths = sorted(round(s.width) for s in strokes)
    assert len(strokes) == 2
    assert widths[0] <= 3 and 14 <= widths[1] <= 17
    band = max(strokes, key=lambda s: s.width)
    assert abs((band.p0[1] + band.p1[1]) / 2 - 127.5) <= 1.5   # the centerline, not an edge


def test_thin_line_offset_from_its_edge_is_still_measured():
    # a 2 px line whose detected edge can sit a pixel off the ink (CAD sheets)
    ink = np.zeros((100, 600), np.uint8)
    ink[40:42, 10:590] = 255
    strokes = [s for s in extract_strokes(ink) if s.length > 400]
    assert strokes and all(s.width <= 3 for s in strokes)


def test_merge_strokes_handles_angle_wraparound():
    a = Stroke(p0=(0.0, 100.0), p1=(200.0, 100.01), width=3)      # ~0 deg
    b = Stroke(p0=(400.0, 100.0), p1=(201.0, 100.02), width=3)    # ~180 deg, same line
    merged = merge_strokes([a, b])
    assert len(merged) == 1
    assert merged[0].angle < 1 or merged[0].angle > 179


def test_collinear_pieces_across_a_crossing_wall_merge():
    pieces = [Wall("", (0, 100), (200, 100), 20, "hollow"), Wall("", (232, 100), (500, 100), 20, "hollow"),
              Wall("", (216, 0), (216, 300), 30, "hollow")]
    out = merge_collinear(pieces)
    horizontal = [w for w in out if abs(w.p0[1] - w.p1[1]) < 1]
    assert len(horizontal) == 1 and horizontal[0].length >= 499


def test_door_sized_gap_is_not_merged():
    pieces = [Wall("", (0, 100), (200, 100), 20, "hollow"), Wall("", (290, 100), (500, 100), 20, "hollow")]
    assert len(merge_collinear(pieces)) == 2


# --- reconstruction across drawing styles -------------------------------------------------

@pytest.mark.parametrize("style,minimum", [("filled", 0.85), ("hollow", 0.85), ("hatched", 0.8), ("thin", 0.75)])
def test_rooms_are_reconstructed_in_every_wall_style(style, minimum):
    recalls = []
    for k in (0, 1, 3):
        m, r = _recall(style, k)
        assert r["false_spaces"] == 0
        assert m.status == "ok"
        recalls.append(r["room_recall_iou70"])
    assert np.mean(recalls) >= minimum


def test_walls_carry_thickness_style_and_support():
    m, _ = _recall("hollow", 1)
    assert m.walls
    assert {w.style for w in m.walls} & {"hollow"}
    assert all(w.thickness > 0 and w.support is not None and w.support >= 0.7 for w in m.walls)
    assert len({round(c) for c in m.wall_classes}) >= 1


def test_openings_connect_spaces():
    m, _ = _recall("hollow", 1)
    assert any(o.kind == "door" for o in m.openings)
    ids = {s.id for s in m.spaces} | {"exterior"}
    assert m.adjacency and all(a in ids and b in ids for a, b, _ in m.adjacency)


def test_blank_image_reports_no_structure():
    m = reconstruct(np.full((400, 500, 3), 255, np.uint8))
    assert m.status == "no_structure" and not m.spaces and m.warnings


def test_labels_are_attached_but_never_create_spaces():
    blank = np.full((400, 500, 3), 255, np.uint8)
    m = reconstruct(blank, room_labels=[("KITCHEN", (250, 200))])
    assert not m.spaces                                          # a label alone fabricates nothing
    p = generate(styled(layout_specs(4)[1], "hollow"))
    depth = cv2.distanceTransform((p.room_labels == 1).astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)  # deepest point of ground-truth room 1
    m = reconstruct(p.image, room_labels=[("BEDROOM", (int(x), int(y)))])
    assert sum("BEDROOM" in s.labels for s in m.spaces) == 1
    assert len(m.spaces) == len(reconstruct(p.image).spaces)    # labels do not change the geometry


def test_sheet_frame_is_not_a_room():
    p = generate(styled(layout_specs(4)[1], "hollow"))
    img = p.image.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), (0, 0, 0), 3)    # a drawing border around the plan
    plain, framed = reconstruct(p.image), reconstruct(img)
    assert abs(len(framed.spaces) - len(plain.spaces)) <= 1
    assert max(s.area_px for s in framed.spaces) < 0.5 * h * w


# --- analyzer integration -------------------------------------------------------------------

def _structure(spaces, wall=True):
    mask = np.zeros((10, 10), np.uint8)
    if wall:
        mask[5] = 255
    return types.SimpleNamespace(wall_mask=mask, spaces=spaces)


def test_fallback_runs_only_when_the_legacy_structure_fails(monkeypatch):
    monkeypatch.setenv("FLOORPLAN_PLAN_MODEL", "fallback")
    image = np.full((200, 200, 3), 255, np.uint8)
    healthy = _structure([{"id": 1}])
    out, notes, structure = adapter.run(image, [], healthy)
    assert out is None and notes == [] and structure is healthy  # legacy healthy: untouched
    out, notes, _ = adapter.run(image, [], _structure([], wall=False))
    assert out is not None and out["role"] == "fallback" and notes


def test_off_and_shadow_modes(monkeypatch):
    image = np.full((200, 200, 3), 255, np.uint8)
    monkeypatch.setenv("FLOORPLAN_PLAN_MODEL", "off")
    failed = _structure([], wall=False)
    assert adapter.run(image, [], failed) == (None, [], failed)
    monkeypatch.setenv("FLOORPLAN_PLAN_MODEL", "shadow")
    healthy = _structure([{"id": 1}])
    out, notes, structure = adapter.run(image, [], healthy)
    assert out["role"] == "shadow" and notes == [] and structure is healthy


def test_payload_is_json_serialisable():
    p = generate(styled(layout_specs(4)[1], "hollow"))
    data = adapter.payload(reconstruct(p.image), "fallback")
    text = json.dumps(data)
    assert data["summary"]["spaces"] == len(data["spaces"]) > 0
    assert "thickness_px" in text and "connects" in text


def test_analyzer_attaches_plan_model_for_a_hollow_wall_plan(monkeypatch):
    from engine.analyzer import FloorPlanAnalyzer

    monkeypatch.setenv("FLOORPLAN_PLAN_MODEL", "fallback")
    p = generate(styled(layout_specs(4)[1], "hollow"))
    result = FloorPlanAnalyzer().analyze(p.image, source_name="hollow.png")
    assert result["plan_model"]["role"] == "fallback"
    assert result["plan_model"]["summary"]["spaces"] >= 4
    assert any("Plan Model" in w for w in result["warnings"])
    json.dumps(result["plan_model"])
    # the reconstructed spaces reach the regular response (rooms / unlabeled spaces, overlay)
    regions = result["unlabeled_spaces"] + [r for r in result["rooms"] if r.get("boundary")]
    assert len(regions) >= 4
    assert result["unlabeled_space_count"] == len(result["unlabeled_spaces"])


def test_analyzer_output_unchanged_when_the_legacy_structure_works(monkeypatch):
    from engine.analyzer import FloorPlanAnalyzer

    monkeypatch.setenv("FLOORPLAN_PLAN_MODEL", "fallback")
    p = generate(styled(layout_specs(4)[1], "filled"))
    result = FloorPlanAnalyzer().analyze(p.image, source_name="filled.png")
    assert "plan_model" not in result
    assert not any("Plan Model" in w for w in result["warnings"])
