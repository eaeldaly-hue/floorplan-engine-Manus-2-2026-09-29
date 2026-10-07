"""Wall-style benchmark: the same synthetic layouts drawn in different wall conventions.

    .venv/bin/python -m benchmark.wall_styles                       # 12 layouts x 5 styles
    .venv/bin/python -m benchmark.wall_styles --count 4 --styles filled,hollow --images out/ws
    .venv/bin/python -m benchmark.wall_styles --save out/wall_styles.json

Every layout is generated once per style from the same seed. Layout, openings, room names and
all ground truth are identical across styles (checked on every run); only the way the wall band
is drawn changes, plus heavy furniture in `filled_furniture`. A score that drops from one style
to another is therefore caused by the drawing convention alone.

Styles: filled (the original generator), hollow (both wall faces outlined, band left white),
hatched (outline + 45-degree hatch), thin (one line per wall axis), filled_furniture (filled
walls, solid dark furniture against walls, furniture outlines, a stair run).

Runs the production structural pass and opening classifier (the calls the API makes, no OCR).
Measures walls (benchmark.wall_metrics), rooms / spaces (benchmark.evaluate.StructureStats) and
openings (benchmark.evaluate.OpeningStats). Synthetic evidence only: it is reported separately
from real plans and is never a real-world accuracy figure.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from benchmark.evaluate import OpeningStats, StructureStats
from benchmark.generator import PlanSpec, Style, generate
from benchmark.wall_metrics import WallStats

STYLES = {
    "filled": {"wall_render": "filled"},
    "hollow": {"wall_render": "hollow"},
    "hatched": {"wall_render": "hatched"},
    "thin": {"wall_render": "thin"},
    "filled_furniture": {"wall_render": "filled", "dark_furniture": 0.9, "furniture_touch": 0.7, "furniture": True,
                         "stairs": 1},
}

BASE = {"door_styles": ("swing", "leaf_only", "double", "outline", "sliding"),
        "window_styles": ("triple", "double", "sample"), "text": True}


def layout_specs(count: int = 12, salt: int = 0) -> list[PlanSpec]:
    """Fixed layouts over a range of scales, line weights and wall thicknesses."""
    scales = (0.55, 0.8, 1.05, 1.35)
    lines = (1, 2, 3)
    base = sum(map(ord, "wall_styles")) * 31 + 100_003 * salt
    specs = []
    for i in range(count):
        r = random.Random(base + i)
        s, lw = scales[i % len(scales)], lines[(i // len(scales)) % len(lines)]
        int_cm = max(r.choice((10.0, 12.0, 15.0)), (3.2 * lw + 2) / s)   # walls clearly thicker than symbol lines
        style = Style(px_per_cm=s, line_px=lw, ext_wall_cm=r.choice((22.0, 25.0, 30.0)), int_wall_cm=int_cm, **BASE)
        specs.append(PlanSpec(seed=base * 100 + i, style=style))
    return specs


def styled(spec: PlanSpec, style: str) -> PlanSpec:
    out = copy.deepcopy(spec)
    out.style = dataclasses.replace(out.style, **STYLES[style])
    return out


def gt_digest(plan) -> str:
    """Hash of everything that must be identical across styles."""
    h = hashlib.sha1()
    for a in (plan.wall_mask, plan.room_labels):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(json.dumps([o.to_dict() for o in plan.openings]).encode())
    h.update(json.dumps(sorted(plan.room_names.items())).encode())
    h.update(json.dumps(sorted(map(list, plan.adjacency))).encode())
    h.update(json.dumps([[list(map(float, p[0])), list(map(float, p[1])), *map(float, p[2:5]), p[5], p[6]]
                         for p in plan.wall_pieces]).encode())
    return h.hexdigest()


def run_engine(image: np.ndarray) -> dict:
    from engine.opening_detection import classify_openings
    from engine.structure import analyze_structure

    structure = analyze_structure(image)
    openings = classify_openings(image, structure, pixel_scale=None)
    return {"structure": structure, "openings": openings, "labels": structure.space_labels,
            "wall_mask": structure.wall_mask, "adjacency": structure.adjacency_pairs()}


def overlay(image: np.ndarray, plan, result: dict) -> np.ndarray:
    """Left: the drawing. Right: GT wall band (green) vs predicted wall (red), yellow = both."""
    gt, pred = plan.wall_mask > 0, result["wall_mask"] > 0
    vis = np.full_like(image, 255)
    vis[gt & ~pred] = (0, 170, 0)
    vis[pred & ~gt] = (0, 0, 230)
    vis[gt & pred] = (0, 200, 230)
    return np.hstack([image, vis])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--count", type=int, default=12, help="layouts")
    ap.add_argument("--styles", default=",".join(STYLES))
    ap.add_argument("--salt", type=int, default=0, help="> 0: fresh layouts for a one-off unbiased run")
    ap.add_argument("--save")
    ap.add_argument("--images", help="write each drawing and its wall overlay to this directory")
    args = ap.parse_args()
    styles = args.styles.split(",")
    specs = layout_specs(args.count, args.salt)
    stats = {s: (WallStats(), StructureStats(), OpeningStats()) for s in styles}
    seconds = {s: 0.0 for s in styles}
    per_plan = []
    digests: dict[int, str] = {}
    if args.images:
        Path(args.images).mkdir(parents=True, exist_ok=True)
    for spec in specs:
        for style in styles:
            plan = generate(styled(spec, style))
            digest = gt_digest(plan)
            if digests.setdefault(spec.seed, digest) != digest:
                raise SystemExit(f"ground truth differs between styles for layout {spec.seed} ({style})")
            t = time.perf_counter()
            result = run_engine(plan.image)
            seconds[style] += time.perf_counter() - t
            w, rooms, opens = stats[style]
            row = w.add(result["wall_mask"], plan, STYLES[style]["wall_render"], result["structure"].wall_thickness)
            rs = StructureStats()
            rs.add(result["labels"], plan.room_labels, result["wall_mask"], plan.wall_mask, result["adjacency"], plan.adjacency)
            rooms.merge(rs)
            os_ = OpeningStats()
            os_.add(result["openings"], plan.openings, f"{style}#{spec.seed}")
            opens.merge(os_)
            per_plan.append({"layout": spec.seed, "style": style, "size": list(plan.size),
                             "px_per_cm": spec.style.px_per_cm, "line_px": spec.style.line_px,
                             "walls": {k: row[k] for k in ("pixel_precision", "pixel_recall", "centerline_recall",
                                                           "segment_recall", "fabricated_share", "phantom_components",
                                                           "thickness_ratio")},
                             "rooms": {k: rs.metrics()[k] for k in ("rooms", "spaces", "room_recall_iou70", "merged_spaces",
                                                                     "split_rooms", "false_spaces")},
                             "openings": {k: os_.metrics()[k] for k in ("candidate_recall", "door_recall", "window_recall")}})
            if args.images:
                name = f"{spec.seed}_{style}"
                cv2.imwrite(str(Path(args.images) / f"{name}.png"), overlay(plan.image, plan, result))
        print(f"layout {spec.seed}: done", flush=True)

    cols = [("pix P", "walls", "pixel_precision"), ("pix R", "walls", "pixel_recall"), ("axis R", "walls", "centerline_recall"),
            ("seg R", "walls", "segment_recall"), ("fabr", "walls", "fabricated_share"),
            ("phant", "walls", "phantom_components_per_plan"), ("thick", "walls", "thickness_ratio_median"),
            ("room R", "rooms", "room_recall_iou70"), ("IoU", "rooms", "mean_best_room_iou"),
            ("merged", "rooms", "merged_spaces"), ("split", "rooms", "split_rooms"), ("false", "rooms", "false_spaces"),
            ("adj R", "rooms", "adjacency_recall"), ("cand R", "openings", "candidate_recall"),
            ("door R", "openings", "door_recall"), ("win R", "openings", "window_recall")]
    report = {"count": args.count, "salt": args.salt, "layouts": [s.seed for s in specs], "styles": {}, "plans": per_plan}
    print(f"\n{'style':<18}" + "".join(f"{c[0]:>8}" for c in cols) + f"{'sec':>7}")
    for style in styles:
        w, rooms, opens = stats[style]
        m = {"walls": w.metrics(), "rooms": rooms.metrics(), "openings": opens.metrics(), "seconds": round(seconds[style], 1)}
        report["styles"][style] = m

        def f(v):
            return "     –" if v is None else (f"{v:8.3f}" if isinstance(v, float) else f"{v:8d}")
        print(f"{style:<18}" + "".join(f(m[g][k]) for _, g, k in cols) + f"{seconds[style]:7.1f}")
    print("\nGT identical across styles for all", len(specs), "layouts")
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
