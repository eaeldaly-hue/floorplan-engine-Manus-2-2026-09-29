"""Run the opening + structure benchmark against an engine version.

    .venv/bin/python -m benchmark.run --engine new                 # all families, synthetic + real sample
    .venv/bin/python -m benchmark.run --engine baseline --count 3  # the pre-Phase-3 production code
    .venv/bin/python -m benchmark.run --engine new --families door_swing,window_weak --save out.json

Synthetic plans come from benchmark.generator (seeded, deterministic).
The real sample is scored against benchmark/fixtures/test_floorplan_openings.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from benchmark.evaluate import OpeningStats, StructureStats, polygons_to_labels
from benchmark.generator import FAMILIES, family_specs, generate

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Engine adapters: image -> {openings, spaces (label map), wall_mask, adjacency}
# ---------------------------------------------------------------------------

def run_baseline(image: np.ndarray) -> dict:
    """The production pipeline as it was before Phase 3 (kept for comparison)."""
    from benchmark.legacy_openings import detect_openings_legacy
    from engine.rooms.space_segmentation import SpaceSegmenter
    from benchmark.legacy_openings import LegacyWallMaskBuilder

    h, w = image.shape[:2]
    wall_mask = LegacyWallMaskBuilder(image).build()
    spaces = SpaceSegmenter(image, wall_mask=wall_mask, gap_close_ratio=0.0625,
                            min_area=max(1000, round(h * w * 0.0015)), max_image_area_ratio=0.80).detect_spaces()
    openings = detect_openings_legacy(image, None, spaces=spaces)
    index = {s["id"]: k for k, s in enumerate(spaces, 1)}
    adjacency = set()
    for o in openings:
        ids = o["evidence"].get("adjacent_space_ids") or []
        rel = o["evidence"].get("room_relation")
        if rel == "between_rooms" and len(ids) == 2:
            adjacency.add((index[ids[0]], index[ids[1]]))
        elif rel == "room_to_exterior" and ids:
            adjacency.add((index[ids[0]], 0))
    return {"openings": openings, "labels": polygons_to_labels(image.shape, spaces), "wall_mask": wall_mask, "adjacency": adjacency}


def run_new(image: np.ndarray) -> dict:
    from engine.structure import analyze_structure
    from engine.opening_detection import classify_openings

    structure = analyze_structure(image)
    openings = classify_openings(image, structure, pixel_scale=None)
    return {"openings": openings, "labels": structure.space_labels, "wall_mask": structure.wall_mask,
            "adjacency": structure.adjacency_pairs()}


ENGINES = {"baseline": run_baseline, "new": run_new}


# ---------------------------------------------------------------------------

def evaluate_family(engine, name: str, count: int, verbose: bool = False, salt: int = 0):
    ostats, sstats = OpeningStats(), StructureStats()
    elapsed = 0.0
    for spec in family_specs(name, count, salt):
        plan = generate(spec)
        started = time.perf_counter()
        result = engine(plan.image)
        elapsed += time.perf_counter() - started
        label = f"{name}#{spec.seed}"
        ostats.add(result["openings"], plan.openings, label)
        sstats.add(result["labels"], plan.room_labels, result["wall_mask"], plan.wall_mask,
                   result["adjacency"], plan.adjacency)
    return ostats, sstats, elapsed


def evaluate_sample(engine):
    truth = json.loads((ROOT / "benchmark" / "fixtures" / "test_floorplan_openings.json").read_text())
    image = cv2.imread(str(ROOT / truth["image"]))
    for o in truth["openings"]:
        o.setdefault("wall_thickness_px", 38.0 if o["id"].startswith("W") else 30.0)
    started = time.perf_counter()
    result = engine(image)
    elapsed = time.perf_counter() - started
    stats = OpeningStats()
    stats.add(result["openings"], truth["openings"], "test_floorplan.png")
    return stats, elapsed, result


def fmt(v):
    if v is None:
        return "   –  "
    if isinstance(v, float):
        return f"{v:6.3f}"
    return f"{v:6d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=sorted(ENGINES), default="new")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--count", type=int, default=5, help="plans per family")
    parser.add_argument("--no-sample", action="store_true")
    parser.add_argument("--errors", action="store_true", help="print individual errors")
    parser.add_argument("--salt", type=int, default=0,
                        help="> 0: fresh plans (new seeds) for a one-off unbiased validation run")
    parser.add_argument("--save")
    args = parser.parse_args()
    engine = ENGINES[args.engine]

    report = {"engine": args.engine, "count": args.count, "salt": args.salt, "families": {}}
    total_o, total_s = OpeningStats(), StructureStats()
    cols = ["candidate_recall", "door_precision", "door_recall", "window_precision", "window_recall",
            "classification_accuracy", "unknown_rate", "false_door_or_window", "missed_candidates"]
    print(f"engine={args.engine}  plans/family={args.count}")
    print(f"{'family':<24}" + "".join(f"{c[:10]:>11}" for c in cols) + f"{'room_rec':>10}{'roomIoU':>9}{'walls':>8}{'sec':>7}")
    for name in args.families.split(","):
        o, s, sec = evaluate_family(engine, name, args.count, salt=args.salt)
        total_o.merge(o); total_s.merge(s)
        om, sm = o.metrics(), s.metrics()
        report["families"][name] = {"openings": om, "structure": sm, "seconds": sec}
        print(f"{name:<24}" + "".join(f"{fmt(om[c]):>11}" for c in cols)
              + f"{fmt(sm['room_recall_iou70']):>10}{fmt(sm['mean_best_room_iou']):>9}{fmt(sm['wall_iou']):>8}{sec:7.1f}", flush=True)
        if args.errors:
            for e in o.errors[:20]:
                print("    ", e)
    om, sm = total_o.metrics(), total_s.metrics()
    report["overall"] = {"openings": om, "structure": sm}
    print(f"{'OVERALL':<24}" + "".join(f"{fmt(om[c]):>11}" for c in cols)
          + f"{fmt(sm['room_recall_iou70']):>10}{fmt(sm['mean_best_room_iou']):>9}{fmt(sm['wall_iou']):>8}")
    print("\nOverall openings:", json.dumps(om, indent=None))
    print("Overall structure:", json.dumps(sm, indent=None))

    if not args.no_sample:
        stats, sec, _ = evaluate_sample(engine)
        m = stats.metrics()
        report["sample"] = m
        print(f"\nReal sample test_floorplan.png ({sec:.1f} s):", json.dumps(m))
        if args.errors:
            for e in stats.errors:
                print("    ", e)
    if args.save:
        Path(args.save).write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
