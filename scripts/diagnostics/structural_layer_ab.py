"""A/B: the current pipeline (A) vs the current pipeline on the experimental structural layer (B)
for PDF pages. Same render, same text layer, same OCR (benchmark OCR cache); only the geometry
input differs.

    .venv/bin/python -m scripts.diagnostics.structural_layer_ab "../Test Cases/22.pdf" 5,6,7,8

Writes output/structural_layer/<pdf>_p<n>_*.png and ab_<pdf>.json.

Reference used for scoring (not by either pipeline): the building footprint, approximated by the
convex hull of the page's wall drawing (wall pen of the structural layer). A space whose centre
lies outside it is outside the building (a lower bound: patios and notches inside the hull count
as inside). "Merged" counts
spaces holding labels of two or more separately walled rooms (bedroom, bath, closet...).
Splits have no ground truth here; "interior unlabeled spaces" (unlabeled spaces inside the
footprint) is reported as a proxy (it also contains legitimate unlabeled rooms) and the overlays
are audited visually.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark import ocr_cache                                  # noqa: E402
from benchmark.holdout_inputs import load                        # noqa: E402
from engine import analyzer as analyzer_module                   # noqa: E402
from engine.analyzer import FloorPlanAnalyzer                    # noqa: E402
from engine.semantic.layer import build_layer, _render_paths    # noqa: E402
from ingest.pdf_vectors import page_paths                        # noqa: E402

OUT = ROOT / "output" / "structural_layer"
WALLED = {"Bedroom", "Bathroom", "Closet", "Walk-in closet", "Laundry", "Powder room", "Toilet", "Storage",
          "Pantry", "Mechanical", "Elevator", "Stair", "Utility"}
BOUNDED = {"wall-region", "open-plan-zone", "merged-space-zone", "passage-partition"}


def footprint(layer, shape) -> np.ndarray:
    """Convex hull of the page's main wall drawing (wall pen of the structural layer). Patios and
    notches inside the hull count as inside, so "spaces outside the building" is a lower bound."""
    walls = [p for p in layer._paths if p.pen in layer.wall_pens]
    ink = (_render_paths(walls, shape[:2], 1.0) < 128).astype(np.uint8)
    # the building is the largest group of wall drawing; legend samples and details lie apart
    k = max(3, int(round(6 * layer.wall_thickness)) | 1)
    grown = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grown)
    if n > 1:
        main = 1 + int(np.argmax([(ink[lab == i]).sum() for i in range(1, n)]))
        ink = ink & (lab == main)
    pts = cv2.findNonZero(ink)
    out = np.zeros(shape[:2], np.uint8)
    if pts is not None:
        cv2.fillPoly(out, [cv2.convexHull(pts)], 1)
    return out > 0


def run(analyzer, image, name, evidence, structure_image=None):
    captured = {}
    original = analyzer_module.plan_adapter.run

    def spy(img, lines, structure, **kw):
        out = original(img, lines, structure, **kw)
        captured["structure"] = out[2]
        captured["plan_model"] = out[0]
        return out
    analyzer_module.plan_adapter.run = spy
    try:
        t = time.perf_counter()
        result = analyzer.analyze(image, name, text_evidence=evidence or None, structure_image=structure_image,
                                  structure_engine="legacy")
        seconds = time.perf_counter() - t
    finally:
        analyzer_module.plan_adapter.run = original
    return result, captured["structure"], captured["plan_model"], seconds


def score(result, structure, plan_model, inside, seconds) -> dict:
    labels = structure.space_labels
    h, w = labels.shape
    spaces = structure.spaces
    outside = sum(1 for sp in spaces if not inside[min(h - 1, sp["center"][1]), min(w - 1, sp["center"][0])])
    by_space: dict = {}
    for room in result["rooms"]:
        c = room.get("label_center") or {}
        x, y = c.get("x"), c.get("y")
        if x is None:
            continue
        k = int(labels[min(h - 1, max(0, y)), min(w - 1, max(0, x))])
        if k > 0:
            by_space.setdefault(k, []).append(room["name"])
    merged = [names for names in by_space.values() if sum(n in WALLED for n in names) >= 2]
    labelled_spaces = set(by_space)
    interior_unlabeled = sum(1 for k, sp in enumerate(spaces, 1)
                             if k not in labelled_spaces and inside[min(h - 1, sp["center"][1]), min(w - 1, sp["center"][0])])
    pm = plan_model or {}
    kinds = [o["kind"] for o in pm.get("openings", [])]
    engine = "plan-model" if any(sp.get("source") == "plan-model" for sp in spaces) else "legacy"
    return {
        "engine": engine,
        "walls": len(pm.get("walls", [])) if engine == "plan-model" else len(structure.bands),
        "doors": result["door_count"] if engine == "legacy" else kinds.count("door"),
        "windows": result["window_count"] if engine == "legacy" else kinds.count("window"),
        "openings_other": result["unclassified_opening_count"] if engine == "legacy" else kinds.count("opening"),
        "spaces": len(spaces),
        "rooms": result["room_count"],
        "rooms_with_boundary": sum(1 for r in result["rooms"] if (r.get("boundary") or {}).get("method") in BOUNDED),
        "rooms_with_boundary_inside_building": sum(1 for r in result["rooms"]
                                                   if (r.get("boundary") or {}).get("method") in BOUNDED
                                                   and _inside_share(r["boundary"]["polygon"], inside) >= 0.9),
        "unlabeled_spaces": result["unlabeled_space_count"],
        "spaces_outside_building": outside,
        "merged_spaces": len(merged),
        "merged_labels": merged,
        "interior_unlabeled_spaces": interior_unlabeled,
        "seconds": round(seconds, 1),
    }


def _inside_share(polygon, inside) -> float:
    mask = np.zeros(inside.shape, np.uint8)
    pts = np.array([[p["x"], p["y"]] if isinstance(p, dict) else p for p in polygon], np.int32)
    cv2.fillPoly(mask, [pts], 1)
    area = mask.sum()
    return float((mask.astype(bool) & inside).sum() / area) if area else 0.0


def space_view(image, structure) -> np.ndarray:
    vis = (0.35 * image + 0.65 * 255).astype(np.uint8)
    rng = np.random.default_rng(3)
    labels = structure.space_labels
    for k in range(1, len(structure.spaces) + 1):
        vis[labels == k] = (0.45 * vis[labels == k] + 0.55 * rng.integers(60, 230, 3)).astype(np.uint8)
    vis[structure.wall_mask > 0] = (0, 0, 0)
    return vis


def main(pdf: str, pages: list[int]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ocr_cache.enable()
    analyzer = FloorPlanAnalyzer()
    pdf_path = Path(pdf)
    report = {}
    for n in pages:
        image, evidence, _ = load(pdf_path, n)
        name = f"{pdf_path.stem}-page-{n}"
        t = time.perf_counter()
        paths = page_paths(pdf_path, n, image.shape)
        layer = build_layer(paths, image.shape, evidence)
        layer._paths = paths
        layer_seconds = time.perf_counter() - t
        clean = layer.structural_image() if layer.applicable else None
        res_a, st_a, pm_a, sec_a = run(analyzer, image, name, evidence)
        inside = footprint(layer, image.shape) if layer.applicable else np.ones(image.shape[:2], bool)
        a = score(res_a, st_a, pm_a, inside, sec_a)
        if clean is not None:
            res_b, st_b, pm_b, sec_b = run(analyzer, image, name, evidence, structure_image=clean)
            b = score(res_b, st_b, pm_b, inside, sec_b + layer_seconds)
            b["layer_seconds"] = round(layer_seconds, 1)
        else:
            res_b, st_b, b = None, None, {"not_applicable": layer.reason}
        report[n] = {"A": a, "B": b, "layer": layer.summary()}
        print(f"p{n} A {json.dumps({k: v for k, v in a.items() if k != 'merged_labels'})}", flush=True)
        print(f"p{n} B {json.dumps({k: v for k, v in b.items() if k != 'merged_labels'})}", flush=True)
        stem = OUT / f"{pdf_path.stem}_p{n}"
        small = lambda im: cv2.resize(im, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
        decode = lambda png: cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(f"{stem}_1_original.png", small(image))
        if layer.applicable:
            cv2.imwrite(f"{stem}_2_elements.png", small(layer.debug_image(image)))
            cv2.imwrite(f"{stem}_3_structural.png", small(clean))
            cv2.imwrite(f"{stem}_4_spaces_B.png", small(space_view(image, st_b)))
            cv2.imwrite(f"{stem}_5_overlay_B.png", small(decode(res_b["overlay_png"])))
        cv2.imwrite(f"{stem}_4_spaces_A.png", small(space_view(image, st_a)))
        cv2.imwrite(f"{stem}_5_overlay_A.png", small(decode(res_a["overlay_png"])))
    (OUT / f"ab_{pdf_path.stem}.json").write_text(json.dumps(report, indent=1))
    print(ocr_cache.summary())


if __name__ == "__main__":
    main(sys.argv[1], [int(p) for p in sys.argv[2].split(",")])
