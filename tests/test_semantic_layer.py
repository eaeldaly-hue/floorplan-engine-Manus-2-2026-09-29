"""Experimental semantic element layer: typing of vector elements, the structural image, the
feature flag, and the analyzer's structural-image input."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from engine.semantic import layer as SL
from ingest.pdf_vectors import VPath, parse_svg

H, W, T = 1100, 1500, 18
WALL_PEN = "rgb(0%, 0%, 0%)"
SYMBOL_PEN = "rgb(30%, 30%, 30%)"
DOOR_GAP = (300, 380)            # gap in the wall x = 550
WINDOW_GAP = (700, 820)          # gap in the top wall y = 150


def _page():
    """3 x 3 rooms of hollow walls (contours of the wall bands, one pen); a door with its swing arc
    and leaf; a window (two lines between the jambs); a sofa, a dimension line and a sheet frame
    in other pens."""
    band = np.zeros((H, W), np.uint8)
    xs, ys = [150, 550, 950, 1350], [150, 450, 750, 1000]
    for x in xs:
        cv2.rectangle(band, (x - T // 2, ys[0] - T // 2), (x + T // 2, ys[-1] + T // 2), 255, -1)
    for y in ys:
        cv2.rectangle(band, (xs[0] - T // 2, y - T // 2), (xs[-1] + T // 2, y + T // 2), 255, -1)
    band[DOOR_GAP[0]:DOOR_GAP[1], 550 - T // 2 - 1:550 + T // 2 + 2] = 0
    band[150 - T // 2 - 1:150 + T // 2 + 2, WINDOW_GAP[0]:WINDOW_GAP[1]] = 0
    paths = []
    contours, _ = cv2.findContours(band, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        pts = [tuple(map(float, p[0])) for p in c] + [tuple(map(float, c[0][0]))]
        paths.append(VPath("stroke", 3.0, WALL_PEN, [("L", a, b) for a, b in zip(pts, pts[1:])]))
    hx, hy, r = 550 + T / 2, float(DOOR_GAP[1]), float(DOOR_GAP[1] - DOOR_GAP[0])
    k = 0.5523 * r
    paths.append(VPath("stroke", 1.0, SYMBOL_PEN, [("C", (hx, hy - r), (hx + r, hy), [(hx + k, hy - r), (hx + r, hy - k), (hx + r, hy)])]))
    paths.append(VPath("stroke", 1.0, SYMBOL_PEN, [("L", (hx, hy), (hx + r, hy))]))
    for dy in (-3.0, 3.0):
        paths.append(VPath("stroke", 1.0, SYMBOL_PEN, [("L", (float(WINDOW_GAP[0]), 150 + dy), (float(WINDOW_GAP[1]), 150 + dy))]))
    sofa = [(250.0, 600.0), (420.0, 600.0), (420.0, 680.0), (250.0, 680.0), (250.0, 600.0)]
    paths.append(VPath("stroke", 1.0, SYMBOL_PEN, [("L", a, b) for a, b in zip(sofa, sofa[1:])]))
    paths.append(VPath("stroke", 0.5, "rgb(10%, 10%, 10%)", [("L", (150.0, 80.0), (1350.0, 80.0))]))
    frame = [(20.0, 20.0), (1480.0, 20.0), (1480.0, 1080.0), (20.0, 1080.0), (20.0, 20.0)]
    paths.append(VPath("stroke", 2.0, "rgb(10%, 10%, 10%)", [("L", a, b) for a, b in zip(frame, frame[1:])]))
    return paths


@pytest.fixture(scope="module")
def layer():
    return SL.build_layer(_page(), (H, W, 3))


def test_the_wall_pen_is_found_by_evidence(layer):
    assert layer.applicable
    assert [p[2] for p in layer.wall_pens] == [WALL_PEN]
    assert all(e.width == 3.0 for e in layer.elements if e.type == "WALL")


def test_door_arc_and_leaf_are_typed(layer):
    doors = [e for e in layer.elements if e.type == "DOOR"]
    assert len(doors) == 2
    assert any("arc" in e.evidence for e in doors) and any("leaf" in e.evidence for e in doors)


def test_window_lines_between_wall_ends_are_typed(layer):
    windows = [e for e in layer.elements if e.type == "WINDOW"]
    assert len(windows) == 2
    assert all(abs(e.geometry[0][1] - 150) <= 4 for e in windows)


def test_furniture_dimension_and_frame_stay_out_of_the_structure(layer):
    clean = cv2.cvtColor(layer.structural_image(), cv2.COLOR_BGR2GRAY)
    assert clean[600:681, 250:421][clean[600:681, 250:421] < 128].size == 0      # sofa
    assert (clean[78:83, 150:1350] < 128).sum() == 0                             # dimension line
    assert (clean[18:23, 20:1480] < 128).sum() == 0                              # frame
    assert (clean[150 - T // 2 - 2:150 - T // 2 + 3, 200:500] < 128).mean() > 0.3  # wall face kept


def test_structural_image_keeps_the_wall_drawing_exactly(layer):
    walls = [p for p in _page() if p.pen in layer.wall_pens]
    exact = SL._render_paths(walls, (H, W), 1.0) < 128
    clean = cv2.cvtColor(layer.structural_image(), cv2.COLOR_BGR2GRAY) < 128
    assert (exact & clean).sum() / exact.sum() > 0.99                            # nothing removed, nothing moved


def test_not_applicable_when_no_pen_draws_a_wall_network():
    lines = [VPath("stroke", 1.0, "rgb(0%, 0%, 0%)", [("L", (100.0 + 40 * i, 100.0), (100.0 + 40 * i, 900.0))]) for i in range(5)]
    out = SL.build_layer(lines, (1000, 1000, 3))
    assert not out.applicable and "no pen" in out.reason
    assert {e.type for e in out.elements} == {"OTHER"}


def test_svg_paths_are_mapped_to_the_render_frame():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="200pt" height="100pt" viewBox="0 0 200 100">'
           '<path fill="none" stroke-width="2" stroke="rgb(0%, 0%, 0%)" d="M 10 10 L 190 10 "/></svg>')
    (p,) = parse_svg(svg, (500, 1000, 3))
    assert p.segs[0][1] == pytest.approx((50.0, 50.0)) and p.segs[0][2] == pytest.approx((950.0, 50.0))
    assert p.width == pytest.approx(10.0)
    (q,) = parse_svg(svg, (500, 1000, 3), crop=(10.0, 0.0, 100.0, 50.0))       # crop box inside the media box
    assert q.segs[0][1] == pytest.approx((0.0, 100.0))


def test_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv("FLOORPLAN_STRUCTURAL_LAYER", raising=False)
    assert not SL.enabled() and not SL.enabled({})
    assert SL.enabled({"structural_layer": True})
    monkeypatch.setenv("FLOORPLAN_STRUCTURAL_LAYER", "on")
    assert SL.enabled() and not SL.enabled({"structural_layer": False})


def test_analyzer_uses_the_structural_image_for_geometry_only():
    from benchmark.generator import generate
    from benchmark.wall_styles import layout_specs, styled
    from engine.analyzer import FloorPlanAnalyzer

    p = generate(styled(layout_specs(4)[1], "filled"))
    analyzer = FloorPlanAnalyzer()
    plain = analyzer.analyze(p.image, "plan.png")
    same = analyzer.analyze(p.image, "plan.png", structure_image=p.image.copy())
    for key in ("room_count", "unlabeled_space_count", "opening_count", "door_count", "window_count"):
        assert plain[key] == same[key]
    assert [r["name"] for r in plain["rooms"]] == [r["name"] for r in same["rooms"]]
    with pytest.raises(ValueError):
        analyzer.analyze(p.image, "plan.png", structure_image=p.image[:-10])


# --- openings closed as drawn, poché walls, safety gate, drafting abbreviations ----------------

def test_door_and_window_gaps_are_closed(layer):
    kinds = {c["kind"] for c in layer.closures}
    assert {"door", "glazing"} <= kinds
    door = next(c for c in layer.closures if c["kind"] == "door")
    xs = sorted(v[0] for v in (door["p0"], door["p1"]))
    ys = [v[1] for v in (door["p0"], door["p1"])]
    assert xs[1] - xs[0] < 3 and all(DOOR_GAP[0] - 3 <= y <= DOOR_GAP[1] + 3 for y in ys)   # spans the gap, along the wall
    clean = cv2.cvtColor(layer.structural_image(closures=True), cv2.COLOR_BGR2GRAY)
    assert (clean[DOOR_GAP[0] + 10:DOOR_GAP[1] - 10, 550] < 128).all()                       # door gap sealed
    assert (clean[150, WINDOW_GAP[0] + 10:WINDOW_GAP[1] - 10] < 128).all()                   # window gap sealed
    open_ = cv2.cvtColor(layer.structural_image(), cv2.COLOR_BGR2GRAY)
    assert (open_[DOOR_GAP[0] + 10:DOOR_GAP[1] - 10, 550] > 128).all()                       # not without closures


def _poche_page():
    """Walls drawn as outline + 45-degree hatch in the same pen as the furniture (no wall pen)."""
    pen = "rgb(0%, 0%, 0%)"
    band = np.zeros((H, W), np.uint8)
    xs, ys = [150, 550, 950, 1350], [150, 450, 750, 1000]
    for x in xs:
        cv2.rectangle(band, (x - T // 2, ys[0] - T // 2), (x + T // 2, ys[-1] + T // 2), 255, -1)
    for y in ys:
        cv2.rectangle(band, (xs[0] - T // 2, y - T // 2), (xs[-1] + T // 2, y + T // 2), 255, -1)
    band[DOOR_GAP[0]:DOOR_GAP[1], 550 - T // 2 - 1:550 + T // 2 + 2] = 0
    paths = []
    contours, _ = cv2.findContours(band, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        pts = [tuple(map(float, p[0])) for p in c] + [tuple(map(float, c[0][0]))]
        paths.append(VPath("stroke", 1.0, pen, [("L", a, b) for a, b in zip(pts, pts[1:])]))
    hatch = []
    for k in range(-H, W, 6):                                         # clip 45-degree lines to the bands
        line = np.zeros((H, W), np.uint8)
        cv2.line(line, (k, 0), (k + H, H), 255, 1)
        cut = (line > 0) & (band > 0)
        n, lab = cv2.connectedComponents(cut.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            yy, xx = np.nonzero(lab == i)
            if len(xx) >= 2:
                a, b = np.argmin(xx), np.argmax(xx)
                hatch.append(("L", (float(xx[a]), float(yy[a])), (float(xx[b]), float(yy[b]))))
    paths.append(VPath("stroke", 1.0, pen, hatch))
    sofa = [(250.0, 600.0), (420.0, 600.0), (420.0, 680.0), (250.0, 680.0), (250.0, 600.0)]
    paths.append(VPath("stroke", 1.0, pen, [("L", a, b) for a, b in zip(sofa, sofa[1:])]))
    return paths, band


def test_poche_walls_without_a_wall_pen():
    paths, band = _poche_page()
    out = SL.build_layer(paths, (H, W, 3))
    assert out.applicable and "poché" in out.reason and not out.wall_pens
    clean = cv2.cvtColor(out.structural_image(), cv2.COLOR_BGR2GRAY) < 128
    core = cv2.erode(band, np.ones((5, 5), np.uint8)) > 0
    assert clean[core].mean() > 0.95                                  # wall bodies solid
    assert not clean[600:681, 250:421].any()                          # furniture in the same pen left out
    assert abs(out.wall_thickness - T) <= 5


def test_misaligned_vectors_are_not_used(layer):
    blank = np.full((H, W, 3), 255, np.uint8)                          # a render that does not show the vectors
    out = SL.build_layer(_page(), (H, W, 3), image=blank)
    assert not out.applicable and "do not match" in out.reason and out.closures == []


def test_drafting_abbreviations_only_in_their_context():
    from engine.analysis import room_lexicon
    from engine.analysis.room_labels import _is_name_token
    from engine.analyzer import _room_name

    tags = ["BDRM.#1", "BA-3", "KIT-2", "W/D", "STAIR#1"]
    assert [_room_name(t) for t in tags] == [None] * 5                 # production default unchanged
    assert not any(_is_name_token(t) for t in tags)
    with room_lexicon.abbreviations():
        assert [_room_name(t) for t in tags] == ["Bedroom", "Bathroom", "Kitchen", "Laundry", "Stairs"]
        assert all(_is_name_token(t) for t in tags)
        assert _room_name("A-3.6") is None and not _is_name_token("A-3.6")   # sheet references stay out
    assert _room_name("BDRM.#1") is None
