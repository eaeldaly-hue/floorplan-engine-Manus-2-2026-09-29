"""PDF development pages scored against hand-placed ground truth (fixtures/pdf_plans.json).

Correctness first: a room counts as recovered only if its space holds no other room. Variants of
the pipeline run on the same render, text layer and OCR (benchmark OCR cache); only the geometry
path differs.

    python -m benchmark.pdf_plans [--variants A,B] [--plans "22.pdf:6,3.pdf:4"] [--save F] [--images DIR]

Variants: A = production pipeline; B = production pipeline on the experimental structural layer
(engine.semantic, vector pages only); further variants are registered in VARIANTS.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "benchmark" / "fixtures" / "pdf_plans.json"
PLANS_DIR = Path(os.environ.get("FLOORPLAN_REAL_PLANS", ROOT.parent / "Test Cases"))

CATEGORY = {"bedroom": "bedroom", "master bedroom": "bedroom", "bathroom": "bath", "bath": "bath", "powder room": "bath",
            "toilet": "bath", "walk-in closet": "closet", "closet": "closet", "linen": "closet", "laundry": "laundry",
            "storage": "storage", "pantry": "storage", "kitchen": "living", "living": "living", "living room": "living",
            "dining": "living", "dining room": "living", "family room": "living", "great room": "living",
            "corridor": "circulation", "hallway": "circulation", "hall": "circulation", "entry": "circulation",
            "foyer": "circulation", "elevator": "elevator", "stair": "stair", "stairs": "stair", "trash room": "service",
            "balcony": "outdoor", "terrace": "outdoor", "porch": "outdoor", "patio": "outdoor",
            "lobby": "lobby", "stairs": "stair", "electrical": "service", "mechanical": "service"}


def category(name: str | None) -> str:
    n = (name or "").strip().lower()
    return CATEGORY.get(n, n)


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------

def _analyze(image, name, evidence, structure_image=None, engine="legacy", cleaner=None):
    """Production analyzer; returns (response, final structure used for rooms)."""
    from engine import analyzer as analyzer_module
    from engine.analyzer import FloorPlanAnalyzer

    captured = {}
    original = analyzer_module.plan_adapter.finish

    def spy(pending, lines, structure):
        out = original(pending, lines, structure)
        captured["structure"] = out[2]
        return out
    analyzer_module.plan_adapter.finish = spy
    try:
        result = FloorPlanAnalyzer().analyze(image, name, text_evidence=evidence or None, structure_image=structure_image,
                                             structure_engine=engine, cleaner=cleaner)
    finally:
        analyzer_module.plan_adapter.finish = original
    return result, captured["structure"]


def variant_a(ctx):
    return _analyze(ctx["image"], ctx["name"], ctx["evidence"])


def _layer(ctx):
    from engine.semantic.layer import build_layer
    from ingest.pdf_vectors import page_paths

    if "layer" not in ctx:
        ctx["layer"] = build_layer(page_paths(ctx["pdf"], ctx["page"], ctx["image"].shape), ctx["image"].shape, ctx["evidence"])
    return ctx["layer"] if ctx["layer"].applicable else None


def variant_b(ctx):
    """structural layer, legacy structure (the first A/B experiment)"""
    layer = _layer(ctx)
    return None if layer is None else _analyze(ctx["image"], ctx["name"], ctx["evidence"], layer.structural_image(), "legacy")


def variant_c(ctx):
    """structural layer, Plan Model with opening evidence from the full page"""
    layer = _layer(ctx)
    return None if layer is None else _analyze(ctx["image"], ctx["name"], ctx["evidence"], layer.structural_image(), "plan")


def variant_d(ctx):
    """structural layer with openings closed as drawn (doors, glazing), legacy structure"""
    layer = _layer(ctx)
    return None if layer is None else _analyze(ctx["image"], ctx["name"], ctx["evidence"], layer.structural_image(closures=True), "legacy")


def variant_e(ctx):
    """Production cleaning path: Architectural Cleaner (vector layer, closures, gap closures) run
    by the analyzer; openings from the typed elements (engine.arch.openings)"""
    from engine.cleaning.cleaner import clean

    cleaned = {}

    def cleaner():
        cleaned["r"] = clean(ctx["image"], ctx["pdf"], ctx["page"], ctx["evidence"], mode="on")
        return cleaned["r"]
    result = _analyze(ctx["image"], ctx["name"], ctx["evidence"], cleaner=cleaner)
    return None if not cleaned["r"].applicable else result


VARIANTS = {"A": variant_a, "B": variant_b, "C": variant_c, "D": variant_d, "E": variant_e}


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _space_at(labels, x, y) -> int:
    h, w = labels.shape
    x, y = min(max(0, int(x)), w - 1), min(max(0, int(y)), h - 1)
    v = int(labels[y, x])
    if v > 0:
        return v
    win = labels[max(0, y - 6):y + 7, max(0, x - 6):x + 7]
    pos = win[win > 0]
    return int(np.bincount(pos).argmax()) if pos.size else 0


def score(spec: dict, response: dict, structure) -> dict:
    labels = structure.space_labels
    rooms = spec["rooms"]
    allowed = {frozenset(p) for p in spec.get("may_share", [])}
    groups = [r.get("group") or f"_{i}" for i, r in enumerate(rooms)]
    spaces = [_space_at(labels, *r["point"]) for r in rooms]
    by_space: dict[int, set] = {}
    for s, g, r in zip(spaces, groups, rooms):
        if s > 0:
            by_space.setdefault(s, set()).add(g)

    def conflicts(s, g):
        return [o for o in by_space.get(s, ()) if o != g and frozenset((o, g)) not in allowed]

    status = []
    for s, g, r in zip(spaces, groups, rooms):
        if s <= 0:
            status.append("missed")
        elif conflicts(s, g):
            status.append("merged")
        else:
            status.append("separated")
    interior = [i for i, r in enumerate(rooms) if r.get("kind", "room") != "outdoor"]
    outdoor = [i for i, r in enumerate(rooms) if r.get("kind") == "outdoor"]
    outdoor_merged = sum(1 for i in outdoor if spaces[i] > 0 and any(
        rooms[j].get("kind", "room") != "outdoor" for j in range(len(rooms)) if spaces[j] == spaces[i] and j != i))
    # splits: a room (group) whose points fall in different spaces
    group_spaces: dict[str, set] = {}
    for s, g in zip(spaces, groups):
        if s > 0:
            group_spaces.setdefault(g, set()).add(s)
    multi = {g for g in groups if groups.count(g) > 1}
    split = sum(1 for g in multi if len(group_spaces.get(g, ())) > 1)
    # names: a room record labelled inside the point's space with the right category
    named = {}
    for room in response["rooms"]:
        c = room.get("label_center") or {}
        if c.get("x") is None:
            continue
        k = _space_at(labels, c["x"], c["y"])
        if k > 0:
            named.setdefault(k, set()).add(category(room["name"]))
    printed = [i for i in interior if rooms[i].get("printed")]
    named_ok = sum(1 for i in printed if spaces[i] > 0 and status[i] == "separated"
                   and category(rooms[i]["name"]) in named.get(spaces[i], ()))
    # false spaces: spaces holding no ground-truth point, centred outside the building outline
    # (hull of the ground-truth points, grown by 3 % of the page diagonal)
    pts = np.array([r["point"] for r in rooms], np.int32)
    hull = np.zeros(labels.shape, np.uint8)
    cv2.fillPoly(hull, [cv2.convexHull(pts)], 1)
    grow = int(0.03 * np.hypot(*labels.shape)) | 1
    hull = cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))) > 0
    gt_spaces = {s for s in spaces if s > 0}
    false_outside = false_inside = 0
    for k, sp in enumerate(structure.spaces, 1):
        if k in gt_spaces:
            continue
        cx, cy = sp["center"]
        inside = hull[min(labels.shape[0] - 1, cy), min(labels.shape[1] - 1, cx)]
        false_outside += 0 if inside else 1
        false_inside += 1 if inside else 0
    n = len(interior)
    sep = sum(status[i] == "separated" for i in interior)
    return {
        "points": n,
        "separated": sep, "merged": sum(status[i] == "merged" for i in interior),
        "missed": sum(status[i] == "missed" for i in interior),
        "separation_rate": round(sep / n, 3) if n else None,
        "split_rooms": split, "multi_point_rooms": len(multi),
        "balcony_merged_with_interior": outdoor_merged, "balconies": len(outdoor),
        "named_correct": named_ok, "printed": len(printed),
        "spaces": len(structure.spaces), "spaces_outside_building": false_outside,
        "unmatched_spaces_inside": false_inside,
        "rooms_reported": response["room_count"],
        "rooms_with_boundary": sum(1 for r in response["rooms"] if r.get("boundary")),
        "doors": response["door_count"], "windows": response["window_count"],
        "per_point": status,
    }


def overlay(image, spec, response, structure, status) -> np.ndarray:
    vis = cv2.imdecode(np.frombuffer(response["overlay_png"], np.uint8), cv2.IMREAD_COLOR)
    if vis is None or vis.shape != image.shape:
        vis = image.copy()
    colors = {"separated": (0, 170, 0), "merged": (0, 0, 230), "missed": (0, 140, 255)}
    for r, s in zip(spec["rooms"], status):
        cv2.circle(vis, tuple(r["point"]), 14, colors[s], -1)
        cv2.circle(vis, tuple(r["point"]), 14, (255, 255, 255), 2)
    return vis


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variants", default="A,B")
    ap.add_argument("--plans", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--images", default=None)
    args = ap.parse_args()
    from benchmark import ocr_cache
    from benchmark.holdout_inputs import load

    ocr_cache.enable()
    doc = json.loads(FIXTURE.read_text())
    plans = args.plans.split(",") if args.plans else list(doc["plans"])
    out = {}
    for key in plans:
        spec = doc["plans"][key]
        pdf, page = key.split(":")
        image, evidence, _ = load(PLANS_DIR / pdf, int(page))
        ctx = {"pdf": PLANS_DIR / pdf, "page": int(page), "image": image, "evidence": evidence,
               "name": f"{Path(pdf).stem}-page-{page}"}
        for v in args.variants.split(","):
            t = time.perf_counter()
            res = VARIANTS[v](ctx)
            if res is None:
                print(f"{key:10s} {v}: not applicable", flush=True)
                continue
            response, structure = res
            m = score(spec, response, structure)
            m["seconds"] = round(time.perf_counter() - t, 1)
            out.setdefault(key, {})[v] = m
            print(f"{key:10s} {v}: " + " ".join(f"{k}={m[k]}" for k in m if k != "per_point"), flush=True)
            if args.images:
                d = Path(args.images)
                d.mkdir(parents=True, exist_ok=True)
                vis = overlay(image, spec, response, structure, m["per_point"])
                cv2.imwrite(str(d / f"{key.replace(':', '_p')}_{v}.png"), cv2.resize(vis, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA))
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(out, indent=1))
    print(ocr_cache.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
