"""Room-name (OCR + association) benchmark on generated plans that print room names.

Runs the full analyzer (OCR included) on fresh synthetic plans of the families that draw
room names and dimensions, and scores each named room:

  detected     a named room returned by the API has its label inside the room
  recognized   its OCR text equals the printed name (normalized, no fuzzy matching)
  associated   its returned boundary contains the room's interior point
  false labels named rooms returned by the API that match no printed room

    python -m benchmark.semantic_synthetic [--salt 21] [--count 2] [--save out.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from benchmark.generator import family_specs, generate
from benchmark.real_plans import _inside, normalize_label
from engine.analyzer import FloorPlanAnalyzer

FAMILIES = ("noise", "mixed", "holdout_mixed", "holdout_scan", "holdout_rotated", "holdout_large")


def score_plan(response: dict, plan) -> dict:
    labels = plan.room_labels
    rooms = [r for r in np.unique(labels) if r > 0 and r in plan.room_names]
    interior = {}
    for r in rooms:
        dist = cv2.distanceTransform((labels == r).astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        interior[r] = (int(x), int(y))
    grown = {r: cv2.dilate((labels == r).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0 for r in rooms}
    api = [a for a in response["rooms"] if a.get("label_center")]
    pairs = []
    for i, a in enumerate(api):
        cx, cy = a["label_center"]["x"], a["label_center"]["y"]
        for r in rooms:
            if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1] and grown[r][cy, cx]:
                same = normalize_label(a.get("label_text")) == normalize_label(plan.room_names[r])
                pairs.append((not same, i, r))
    pairs.sort()
    used_a, used_r, match = set(), set(), {}
    for _, i, r in pairs:
        if i in used_a or r in used_r:
            continue
        used_a.add(i); used_r.add(r); match[r] = i
    out = {"printed": len(rooms), "detected": len(match), "recognized": 0, "associated": 0, "named_correct": 0,
           "false_labels": len(api) - len(used_a)}
    for r, i in match.items():
        a = api[i]
        rec = normalize_label(a.get("label_text")) == normalize_label(plan.room_names[r])
        assoc = bool(a.get("boundary")) and _inside(a["boundary"]["polygon"], *interior[r])
        out["recognized"] += rec
        out["associated"] += assoc
        out["named_correct"] += rec and assoc
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--salt", type=int, default=21, help="fresh seeds (not used for tuning)")
    ap.add_argument("--count", type=int, default=2)
    ap.add_argument("--save")
    args = ap.parse_args()
    analyzer = FloorPlanAnalyzer()
    total = {k: 0 for k in ("printed", "detected", "recognized", "associated", "named_correct", "false_labels")}
    per = {}
    seconds = 0.0
    for fam in FAMILIES:
        fam_tot = dict.fromkeys(total, 0)
        for spec in family_specs(fam, args.count, args.salt):
            plan = generate(spec)
            t = time.perf_counter()
            response = analyzer.analyze(plan.image, f"{fam}-{spec.seed}")
            seconds += time.perf_counter() - t
            m = score_plan(response, plan)
            for k in total:
                total[k] += m[k]; fam_tot[k] += m[k]
        per[fam] = fam_tot
        print(f"{fam:<18}", fam_tot, flush=True)
    named = total["detected"] + total["false_labels"]
    summary = {**total,
               "label_recall": round(total["recognized"] / max(1, total["printed"]), 3),
               "label_precision": round(total["recognized"] / max(1, named), 3),
               "association_accuracy": round(total["named_correct"] / max(1, total["recognized"]), 3),
               "named_room_recovery": round(total["named_correct"] / max(1, total["printed"]), 3),
               "seconds": round(seconds, 1)}
    print("SUMMARY", json.dumps(summary))
    if args.save:
        Path(args.save).write_text(json.dumps({"summary": summary, "families": per, "salt": args.salt}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
