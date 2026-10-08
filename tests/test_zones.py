"""Functional zones inside structural spaces (engine.arch.zones) and the typed vector objects that
support them (engine.arch.objects)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.arch.objects import Obj, arcs, basins, circles
from engine.arch.zones import infer_zones

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT.parent / "Test Cases"
PPC = 2.0                                    # px per cm in the synthetic cases


def _space(w_cm=1200, h_cm=600):
    return {"id": "space_1", "names": [], "polygon": [[0, 0], [w_cm * PPC, 0], [w_cm * PPC, h_cm * PPC], [0, h_cm * PPC]]}


def _obj(i, kind, cx_cm, cy_cm, w_cm, h_cm, functions):
    x0, y0 = (cx_cm - w_cm / 2) * PPC, (cy_cm - h_cm / 2) * PPC
    return Obj(f"OB{i}", kind, (x0, y0, x0 + w_cm * PPC, y0 + h_cm * PPC), functions, 0.8, [kind])


def _zone_at(space, x_cm, y_cm):
    for z in space.get("zones", []):
        if cv2.pointPolygonTest(np.array(z["polygon"], np.float32).reshape(-1, 1, 2), (x_cm * PPC, y_cm * PPC), False) >= 0:
            return z["function"]
    return None


def test_open_space_with_kitchen_dining_living_objects_gets_three_zones():
    sp = _space()
    objs = [_obj(1, "range", 80, 60, 70, 60, {"kitchen": 1.0}), _obj(2, "refrigerator", 200, 60, 80, 70, {"kitchen": 1.0}),
            _obj(3, "sink", 140, 60, 80, 50, {"kitchen": 0.9}),
            _obj(4, "dining set", 600, 300, 200, 160, {"dining": 1.0}),
            _obj(5, "sofa", 1050, 450, 230, 90, {"living": 1.0}), _obj(6, "table", 1050, 330, 120, 60, {"living": 0.3})]
    summary = infer_zones([sp], (int(600 * PPC), int(1200 * PPC)), objs, [], PPC)
    assert summary["zones"] == 3 and summary["spaces_with_zones"] == 1
    assert _zone_at(sp, 150, 100) == "kitchen" and _zone_at(sp, 600, 300) == "dining" and _zone_at(sp, 1050, 420) == "living"
    for z in sp["zones"]:
        assert z["evidence"] and 0 < z["confidence"] <= 1 and z["area_m2"] > 0 and z["boundaries"]
    kitchen = next(z for z in sp["zones"] if z["function"] == "kitchen")
    assert {e["kind"] for e in kitchen["evidence"]} >= {"range", "refrigerator", "sink"}


def test_one_function_characterises_the_space_without_zones():
    sp = _space(500, 400)
    objs = [_obj(1, "bed", 250, 200, 160, 210, {"bedroom": 1.0})]
    infer_zones([sp], (int(400 * PPC), int(500 * PPC)), objs, [], PPC)
    assert "zones" not in sp and sp["function"]["function"] == "bedroom"


def test_weak_evidence_creates_nothing():
    sp = _space()
    objs = [_obj(1, "counter", 100, 60, 150, 60, {"kitchen": 0.4}), _obj(2, "seat", 600, 300, 45, 45, {"dining": 0.15}),
            _obj(3, "side table", 900, 300, 50, 50, {"living": 0.15})]
    infer_zones([sp], (int(600 * PPC), int(1200 * PPC)), objs, [], PPC)
    assert "zones" not in sp and "function" not in sp


def test_a_printed_label_overrides_a_contradicting_object():
    """A sofa group read as a dining set, under the printed word LIVING, is not a dining zone."""
    sp = _space()
    objs = [_obj(1, "range", 80, 60, 70, 60, {"kitchen": 1.0}), _obj(2, "refrigerator", 200, 60, 80, 70, {"kitchen": 1.0}),
            _obj(3, "dining set", 900, 350, 250, 200, {"dining": 1.0})]
    infer_zones([sp], (int(600 * PPC), int(1200 * PPC)), objs, [("Living", (900 * PPC, 350 * PPC))], PPC)
    functions = {z["function"] for z in sp["zones"]}
    assert functions == {"kitchen", "living"}


class _E:                                     # stand-in for a semantic-layer element
    def __init__(self, pts):
        self.geometry = pts


def _arc_elements(cx, cy, r, start_deg, n_quarters=4, step=90):
    out = []
    for q in range(n_quarters):
        a = np.radians(np.linspace(start_deg + q * step, start_deg + (q + 1) * step, 12))
        out.append(_E([(cx + r * math.cos(t), cy + r * math.sin(t)) for t in a]))
    return out


def test_circles_and_rounded_basins_from_vector_arcs():
    burners = [e for c in ((100, 100), (160, 100), (100, 160), (160, 160)) for e in _arc_elements(*c, 16, 0)]
    found = circles(arcs(burners), PPC)
    assert len(found) == 4 and all(abs(r - 16) < 1 for _, _, r, _ in found)
    # a basin: four quarter arcs at the corners of a 40 x 30 cm rectangle, bulging outward
    x0, y0, x1, y1, r = 400, 400, 400 + 80, 400 + 60, 8
    corners = [(x0, y0, 180), (x1, y0, 270), (x1, y1, 0), (x0, y1, 90)]
    elems = [_E([(cx + r * math.cos(t), cy + r * math.sin(t)) for t in np.radians(np.linspace(a, a + 90, 12))]) for cx, cy, a in corners]
    boxes = basins(arcs(elems), PPC)
    assert len(boxes) == 1
    bx0, by0, bx1, by1 = boxes[0]
    assert abs((bx1 - bx0) - 96) <= 2 and abs((by1 - by0) - 76) <= 2


@pytest.mark.skipif(not (REAL / "3.pdf").exists(), reason="real test plans not available")
def test_real_unlabelled_open_plan_has_kitchen_dining_living_zones():
    """The reported case: 3.pdf p5 (electrical sheet, no room names). One structural space holds
    the kitchen, dining and living areas; they are zones inferred from drawn objects alone."""
    from benchmark.holdout_inputs import load
    from engine.analyzer import FloorPlanAnalyzer
    from engine.cleaning.cleaner import clean

    gt = json.loads((ROOT / "benchmark/fixtures/zones.json").read_text())["plans"]["3.pdf:5"]
    image, evidence, _ = load(REAL / "3.pdf", 5)
    r = FloorPlanAnalyzer().analyze(image, "3.pdf", text_evidence=evidence or None,
                                    cleaner_candidate=lambda: clean(image, REAL / "3.pdf", 5, evidence, mode="vector"))
    spaces = r["building"]["spaces"]

    def space_of(x, y):
        return next(s for s in spaces if cv2.pointPolygonTest(np.array(s["polygon"], np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0)

    open_space = {space_of(*z["point"])["id"] for z in gt["zones"]}
    assert len(open_space) == 1                                      # one structural space, no wall between
    sp = space_of(*gt["zones"][0]["point"])
    assert sorted(z["function"] for z in sp["zones"]) == ["dining", "kitchen", "living"]
    for want in gt["zones"]:
        hit = [z for z in sp["zones"] if cv2.pointPolygonTest(np.array(z["polygon"], np.float32).reshape(-1, 1, 2),
                                                               tuple(float(v) for v in want["point"]), False) >= 0]
        assert hit and hit[0]["function"] == want["function"], want
        assert all(e["source"] == "object" for e in hit[0]["evidence"])     # no label on this sheet
    assert sum(len(s.get("zones", [])) for s in spaces) == 3                 # zones only where supported
