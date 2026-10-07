"""Benchmark N.1 measurement code: stress families and real-plan probes (no engine changes)."""

import dataclasses

import cv2
import numpy as np

from benchmark.generator import STRESS, family_specs, generate
from benchmark.real_plan_probes import api_probe, structural_probe
from engine.opening_detection import classify_openings
from engine.structure import analyze_structure


def test_stress_elements_do_not_change_ground_truth_walls():
    """Stress elements are drawn after the GT wall mask and from their own RNG: the same plan
    without them has the same walls, openings and rooms."""
    for name in STRESS:
        spec = family_specs(name, 1)[0]
        plain = dataclasses.replace(spec, style=dataclasses.replace(spec.style, dark_furniture=0.0, markers=0, hatch=0.0, stairs=0))
        a, b = generate(spec), generate(plain)
        assert np.array_equal(a.wall_mask, b.wall_mask), name
        assert a.openings == b.openings and np.array_equal(a.room_labels, b.room_labels), name
        assert not np.array_equal(a.image, b.image), name


def _plan():
    img = np.full((420, 620, 3), 255, np.uint8)
    cv2.rectangle(img, (40, 40), (580, 380), (0, 0, 0), 12)
    cv2.line(img, (300, 40), (300, 380), (0, 0, 0), 10)
    cv2.rectangle(img, (295, 180), (305, 250), (255, 255, 255), -1)          # door gap
    cv2.rectangle(img, (120, 150), (170, 200), (0, 0, 0), -1)                 # solid furniture block
    return img


def test_structural_probe_reports_furniture_walls_protected_pieces_and_recall():
    img = _plan()
    s = analyze_structure(img)
    openings = classify_openings(img, s)
    spec = {"rooms": [{"name": "A", "point": [200, 300]}, {"name": "B", "point": [450, 200]}],
            "openings": [{"id": "D1", "kind": "door", "p0": [300, 180], "p1": [300, 250], "wall_thickness_px": 10}],
            "nonstructural": [{"what": "block", "point": [145, 175]}],
            "protected": [{"what": "wall", "point": [300, 100]}]}
    p = structural_probe(s, spec, openings)
    assert p["under_segmentation"]["groups_sharing_a_space"] == 0
    assert p["openings_detail"]["recall_doors_windows"] == 1.0
    assert p["protected"]["present"] == 1
    fw = p["false_walls"]
    assert fw["nonstructural_points"] == 1 and fw["covered_by_wall"] in (0, 1)
    assert fw["covered_by_wall"] == int(s.wall_mask[175, 145] > 0)


def test_api_probe_counts_open_plan_groups_and_unsupported_boundaries():
    sq = lambda x0, y0, x1, y1: [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}]
    response = {"rooms": [
        {"id": "r1", "space_id": "s1", "boundary": {"method": "open-plan-shared", "polygon": sq(0, 0, 100, 100)}},
        {"id": "r2", "space_id": "s1", "boundary": {"method": "open-plan-shared", "polygon": sq(0, 0, 100, 100)}},
        {"id": "r3", "space_id": None, "boundary": {"method": "wall-ray-estimate", "polygon": sq(200, 0, 300, 100)}}]}
    spec = {"rooms": [{"point": [20, 20], "group": "open"}, {"point": [80, 80], "group": "open"}, {"point": [250, 50]}]}
    a = api_probe(response, spec)
    assert a["open_plan"] == {"groups": 1, "one_space": 1, "split": 0, "merged_with_other_group": 0,
                              "groups_detail": {"open": {"points": 2, "api_spaces": 1, "shares_with_other_group": False}}}
    assert a["unsupported_boundaries"] == 1


def test_holdout_freeze_detects_edited_ground_truth_and_changed_files(tmp_path):
    from benchmark.real_plans import check_frozen, file_digest, gt_digest
    (tmp_path / "a.png").write_bytes(b"plan-a")
    doc = {"plans": {"a.png": {"rooms": [{"point": [1, 2]}]}}}
    assert check_frozen(doc, tmp_path) == ["ground truth is not frozen yet"]
    doc["frozen"] = {"gt_sha256": gt_digest(doc), "images": {"a.png": file_digest(tmp_path / "a.png")}}
    assert check_frozen(doc, tmp_path) == []
    doc["plans"]["a.png"]["rooms"][0]["point"] = [5, 5]
    (tmp_path / "a.png").write_bytes(b"plan-a2")
    problems = check_frozen(doc, tmp_path)
    assert any("ground truth changed" in p for p in problems) and any("file changed" in p for p in problems)


def test_api_probe_keeps_unnamed_physical_spaces_separate_from_named_rooms():
    sq = lambda x0, y0, x1, y1: [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}]
    response = {"rooms": [{"id": "r1", "space_id": "s1", "boundary": {"method": "wall-region", "polygon": sq(0, 0, 100, 100)}}],
                "unlabeled_spaces": [{"id": "s2", "boundary": {"method": "wall-region", "polygon": sq(200, 0, 300, 100)}}]}
    spec = {"rooms": [{"point": [50, 50]}, {"point": [250, 50]}, {"point": [500, 500]}]}
    ph = api_probe(response, spec)["physical"]
    assert (ph["physical_spaces"], ph["physical_spaces_named"], ph["unnamed_spaces"], ph["named_rooms"]) == (2, 1, 1, 1)
    assert ph["gt_points_by_space"] == {"named_room": 1, "unnamed_space_only": 1, "no_space": 1}


def _holdout_case(tmp_path, sofa_wall: bool):
    """Open-plan space (kitchen + living) | bedroom, plus an unnamed closet; a dark sofa
    that (optionally) runs from wall to wall through the open-plan space."""
    img = np.full((420, 760, 3), 255, np.uint8)
    cv2.rectangle(img, (40, 40), (720, 380), (0, 0, 0), 10)
    cv2.line(img, (480, 40), (480, 380), (0, 0, 0), 10)
    cv2.rectangle(img, (475, 180), (485, 240), (255, 255, 255), -1)              # bedroom door
    cv2.rectangle(img, (600, 40), (720, 140), (0, 0, 0), 10)                      # closet in the bedroom
    cv2.rectangle(img, (640, 135), (680, 145), (255, 255, 255), -1)              # closet door
    if sofa_wall:
        cv2.rectangle(img, (230, 40), (262, 380), (0, 0, 0), -1)                  # "sofa" bar splitting the open plan
    cv2.imwrite(str(tmp_path / "h.png"), img)
    spec = {
        "spaces": [{"id": "S1", "kind": "open_plan", "points": [[120, 200], [380, 200]]},
                   {"id": "S2", "kind": "room", "points": [[560, 300]]},
                   {"id": "S3", "kind": "closet", "points": [[660, 90]]}],
        "rooms": [{"name": "Kitchen", "printed": "KITCHEN", "point": [120, 200], "space": "S1"},
                  {"name": "Living room", "printed": "LIVING", "point": [380, 200], "space": "S1"},
                  {"name": "Bedroom", "printed": "BEDROOM", "point": [560, 300], "space": "S2"}],
        "openings": [{"id": "D1", "kind": "door", "p0": [480, 180], "p1": [480, 240], "wall_thickness_px": 10}],
        "nonstructural": [{"kind": "furniture", "what": "sofa", "point": [246, 210], "bbox": [230, 40, 262, 380]}],
    }
    return spec


def test_holdout_metrics_separate_physical_spaces_zones_and_furniture_splits(tmp_path):
    from benchmark.holdout_metrics import normalize_spec
    from benchmark.real_plans import evaluate
    for sofa in (False, True):
        spec = normalize_spec(dict(_holdout_case(tmp_path, sofa), source=None))
        r = evaluate("h.png", spec, tmp_path)
        h = r["holdout"]
        ps = h["physical_spaces"]
        assert ps["gt"] == 3 and ps["unnamed_gt"] == 1
        if not sofa:
            assert ps["recovered"] == 3 and h["open_plan"] == {**h["open_plan"], "groups": 1, "one_space": 1, "split": 0}
            assert h["false_boundaries"]["split_boundaries"] == 0
            assert h["point_association"]["in_own_physical_space"] == 3
        else:
            assert ps["status"]["S1"] == "split" and h["open_plan"]["split"] == 1
            assert h["false_boundaries"]["open_plan_splits_nonstructural"] == 1
            assert h["nonstructural_as_wall"]["point_on_wall"] == 1
