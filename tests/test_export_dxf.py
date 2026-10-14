"""DXF export (engine.export.dxf): layers, units, the coordinate policy, uncertain and invalid input."""

from __future__ import annotations

import io
import math
from pathlib import Path

import ezdxf
import pytest

from engine.export.dxf import ExportError, build_dxf, dxf_bytes

H = 100


def _model(**extra) -> dict:
    """A small plan: two walls, a hinged door, a window, an untyped opening, two rooms, one unnamed space."""
    building = {
        "walls": [{"id": "WL001", "p0": [10, 20], "p1": [90, 20], "thickness_px": 4.0, "exterior": True},
                  {"id": "WL002", "p0": [10, 20], "p1": [10, 80], "thickness_px": 4.0, "exterior": True}],
        "openings": [
            {"id": "D01", "type": "door", "confidence": 0.9, "p0": [30, 50], "p1": [50, 50],
             "door": {"operation": "hinged", "hinges": [[30, 50]], "swing_into": "space_1"}},
            {"id": "W01", "type": "window", "confidence": 0.8, "p0": [40, 20], "p1": [60, 20], "window": {}},
            {"id": "O01", "type": "opening", "confidence": 0.4, "p0": [70, 80], "p1": [85, 80], "passage": {}},
        ],
        "spaces": [{"id": "space_1", "kind": "room", "polygon": [[12, 22], [88, 22], [88, 78], [12, 78]]},
                   {"id": "space_2", "kind": "unnamed", "polygon": [[60, 82], [88, 82], [88, 98], [60, 98]]}],
    }
    result = {
        "source_name": "plan.png", "image": {"width": 120, "height": H}, "pixel_scale": None,
        "building": building,
        "openings": [{"id": "D01", "evidence": {"swing_side": 1, "wall_thickness_px": 4.0}},
                     {"id": "W01", "evidence": {"wall_thickness_px": 4.0}},
                     {"id": "O01", "evidence": {}}],
        "rooms": [
            {"id": "room-01", "name": "Kitchen", "space_id": "space_1", "label_center": {"x": 50, "y": 40},
             "boundary": {"method": "wall-region", "polygon": [{"x": 12, "y": 22}, {"x": 88, "y": 22}, {"x": 88, "y": 78}, {"x": 12, "y": 78}]},
             "area": {"value": 4256, "unit": "px²"}},
            {"id": "room-02", "name": "Dining", "space_id": "space_1", "label_center": {"x": 70, "y": 60},
             "boundary": {"method": "open-plan-zone", "polygon": [{"x": 60, "y": 50}, {"x": 88, "y": 50}, {"x": 88, "y": 78}, {"x": 60, "y": 78}]}},
        ],
    }
    result.update(extra)
    return result


def _reread(result) -> ezdxf.document.Drawing:
    data, _ = dxf_bytes(result)
    assert data.startswith(b"  0\r\nSECTION") or data.lstrip().startswith(b"0")
    doc = ezdxf.read(io.StringIO(data.decode("utf8")))     # independent re-read of the written file
    assert not doc.audit().errors
    return doc


def _layer(doc, name):
    return list(doc.modelspace().query(f'*[layer=="{name}"]'))


def test_a_plan_exports_every_element_on_its_layer():
    doc = _reread(_model())
    counts = {name: len(_layer(doc, name)) for name in ("WALLS", "WALL_CENTERLINES", "DOORS", "WINDOWS",
                                                        "OPENINGS_UNCLASSIFIED", "ROOM_BOUNDARIES",
                                                        "ROOM_BOUNDARIES_INFERRED", "SPACE_BOUNDARIES", "ROOM_LABELS")}
    assert counts == {"WALLS": 2, "WALL_CENTERLINES": 2, "DOORS": 3,       # span, leaf, swing arc
                      "WINDOWS": 2, "OPENINGS_UNCLASSIFIED": 1,            # frame + glazing axis
                      "ROOM_BOUNDARIES": 1, "ROOM_BOUNDARIES_INFERRED": 1,
                      "SPACE_BOUNDARIES": 1,                               # space_1 is drawn as the Kitchen
                      "ROOM_LABELS": 2}
    door = _layer(doc, "DOORS")[0]
    assert door.get_xdata("FLOORPLAN_ENGINE")[0] == (1000, "id=D01")


def test_uncalibrated_plans_are_unitless_and_keep_pixel_coordinates_with_y_up():
    doc, report = build_dxf(_model())
    assert report.mm_per_px is None and doc.header["$INSUNITS"] == 0
    wall = next(iter(doc.modelspace().query('LINE[layer=="WALL_CENTERLINES"]')))
    assert (wall.dxf.start.x, wall.dxf.start.y) == (10, H - 20) and (wall.dxf.end.x, wall.dxf.end.y) == (90, H - 20)
    outline = next(iter(doc.modelspace().query('LWPOLYLINE[layer=="WALLS"]')))
    ys = sorted({round(p[1], 6) for p in outline.get_points()})
    assert ys == [H - 22, H - 18]                          # +/- half the measured 4 px thickness


def test_printed_dimensions_calibrate_millimetres():
    doc, report = build_dxf(_model(pixel_scale={"unit": "m", "pixels_per_unit": 50.0}))
    assert report.mm_per_px == 20.0 and doc.header["$INSUNITS"] == 4
    wall = next(iter(doc.modelspace().query('LINE[layer=="WALL_CENTERLINES"]')))
    assert (wall.dxf.start.x, wall.dxf.start.y) == (200, (H - 20) * 20)
    ft, _ = build_dxf(_model(pixel_scale={"unit": "ft", "pixels_per_unit": 10.0}))
    assert ft.header["$INSUNITS"] == 4 and math.isclose(build_dxf(_model(pixel_scale={"unit": "ft", "pixels_per_unit": 10.0}))[1].mm_per_px, 30.48)


def test_an_estimated_scale_is_reported_not_applied():
    doc, report = build_dxf(_model(scale_estimate={"px_per_cm": 0.5, "source": "median of 5 door widths"}))
    assert report.mm_per_px is None and doc.header["$INSUNITS"] == 0
    assert any("Not applied" in n for n in report.notes)


def test_the_door_swings_to_the_side_read_from_the_drawing():
    # span (30,50)-(50,50), hinge at the start, swing_side +1: the image normal (-dy, dx) points to +y
    # (down in the image), i.e. lower Y in the drawing
    doc, _ = build_dxf(_model())
    leaf = [e for e in doc.modelspace().query('LINE[layer=="DOORS"]') if e.get_xdata("FLOORPLAN_ENGINE")[1][1] == "type=door leaf"][0]
    assert (leaf.dxf.start.x, leaf.dxf.start.y) == (30, H - 50)
    assert math.isclose(leaf.dxf.end.x, 30) and math.isclose(leaf.dxf.end.y, H - 70)
    arc = next(iter(doc.modelspace().query('ARC[layer=="DOORS"]')))
    assert (arc.dxf.center.x, arc.dxf.center.y, arc.dxf.radius) == (30, H - 50, 20)
    assert {round(arc.dxf.start_angle) % 360, round(arc.dxf.end_angle) % 360} == {270, 0}


def test_without_hinge_or_swing_side_a_door_is_its_span_only():
    m = _model()
    m["openings"][0]["evidence"]["swing_side"] = None
    doc, report = build_dxf(m)
    assert len(_layer(doc, "DOORS")) == 1
    assert any("D01" in n and "span only" in n for n in report.notes)


def test_missing_optional_fields_still_export():
    m = _model()
    del m["rooms"], m["openings"]                          # no room records, no opening evidence
    m["building"]["openings"][0].pop("door")
    doc = _reread(m)
    assert len(_layer(doc, "WINDOWS")) == 1                # glazing axis; no frame without a thickness
    assert len(_layer(doc, "SPACE_BOUNDARIES")) == 2 and not _layer(doc, "ROOM_LABELS")


def test_an_empty_model_is_a_valid_drawing_with_its_notes():
    m = _model()
    m["building"] = {"walls": [], "openings": [], "spaces": []}
    m["rooms"] = []
    doc = _reread(m)
    assert [e.dxf.layer for e in doc.modelspace()] == ["NOTES"]


def test_no_model_is_an_error():
    with pytest.raises(ExportError):
        build_dxf({"image": {"width": 10, "height": 10}})
    with pytest.raises(ExportError):
        build_dxf({"building": {"walls": []}})


def test_invalid_geometry_is_skipped_and_reported_never_repaired():
    m = _model()
    m["building"]["walls"].append({"id": "WL003", "p0": [5, 5], "p1": [5, 5], "thickness_px": 4.0})       # zero length
    m["building"]["walls"].append({"id": "WL004", "p0": [float("nan"), 5], "p1": [9, 5], "thickness_px": 4.0})
    m["building"]["walls"].append({"id": "WL005", "p0": [20, 90], "p1": [40, 90], "thickness_px": None})  # no thickness
    m["building"]["spaces"].append({"id": "space_3", "kind": "unnamed", "polygon": [[1, 1], [2, 2], [3, 3]]})  # collinear
    doc, report = build_dxf(m)
    skipped = {i for i, _ in report.skipped}
    assert {"WL003", "WL004", "WL005", "space_3"} <= skipped
    assert len(_layer(doc, "WALL_CENTERLINES")) == 3 and len(_layer(doc, "WALLS")) == 2   # WL005: axis only
    for e in doc.modelspace():
        pts = ([e.dxf.start, e.dxf.end] if e.dxftype() == "LINE" else
               [(p[0], p[1]) for p in e.get_points()] if e.dxftype() == "LWPOLYLINE" else [])
        assert all(math.isfinite(p[0]) and math.isfinite(p[1]) for p in pts)


REAL = Path(__file__).resolve().parents[1] / "test_floorplan.png"


@pytest.mark.skipif(not REAL.exists(), reason="sample plan not available")
def test_the_sample_plan_exports_its_model_exactly():
    import json

    import cv2

    from engine.analyzer import FloorPlanAnalyzer

    r = FloorPlanAnalyzer().analyze(cv2.imread(str(REAL)), "test_floorplan.png")
    r = json.loads(json.dumps({k: v for k, v in r.items() if not k.endswith("_png") and not k.startswith("_")}))
    doc = _reread(r)
    b, s = r["building"], build_dxf(r)[1].mm_per_px or 1.0
    walls = list(doc.modelspace().query('LINE[layer=="WALL_CENTERLINES"]'))
    assert len(walls) == len(b["walls"])
    model = {(tuple(w["p0"]), tuple(w["p1"])) for w in b["walls"]}
    back = {((round(e.dxf.start.x / s), round(H_ - e.dxf.start.y / s)), (round(e.dxf.end.x / s), round(H_ - e.dxf.end.y / s)))
            for e in walls for H_ in [r["image"]["height"]]}
    assert back == model
    n_open = sum(o["type"] == "opening" for o in b["openings"])
    assert len(_layer(doc, "OPENINGS_UNCLASSIFIED")) == n_open
