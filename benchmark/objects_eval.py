"""Furniture / fixture recognition against hand-labelled boxes (fixtures/objects.json).

A detection matches a ground-truth box when they overlap well (intersection >= 60 % of the smaller
box and IoU >= 0.15, so a basin of a double sink or a counter run around an island still match).

  correct      matched a box of a compatible kind
  wrong kind   matched a box, kind incompatible (a washbasin typed as a toilet)
  false        matched nothing (legend symbols, lamps, electrical symbols, notes)
  recall       ground-truth boxes found with a compatible kind

    python -m benchmark.objects_eval [--only 3.pdf:5] [--save F] [--verbose]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT.parent / "Test Cases"
FIXTURE = ROOT / "benchmark/fixtures/objects.json"

# detection kind -> ground-truth kinds it may stand for
COMPATIBLE = {
    "bed": {"bed"}, "range": {"range"}, "refrigerator": {"refrigerator"}, "dishwasher": {"dishwasher"},
    "sink": {"sink"}, "washbasin": {"washbasin"}, "toilet": {"toilet"}, "bathtub": {"bathtub", "shower"},
    "shower": {"shower", "bathtub"}, "counter": {"counter"}, "counter run": {"counter"},
    "dining set": {"dining set"}, "table": {"coffee table", "table", "desk"}, "coffee table": {"coffee table"},
    "sofa": {"sofa"}, "loveseat": {"sofa"}, "armchair": {"armchair", "seat"}, "seat": {"seat"}, "stool": {"seat"},
    "side table": {"nightstand", "end table"}, "nightstand": {"nightstand"}, "end table": {"end table"},
    "media / cabinet": {"media", "desk"}, "media": {"media"}, "wardrobe": {"wardrobe"}, "desk": {"desk"},
    "washer": {"washer"}, "dryer": {"dryer"}, "laundry appliance": {"washer", "dryer"},
}


def _overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: max(1, (r[2] - r[0]) * (r[3] - r[1]))  # noqa: E731
    return inter / min(area(a), area(b)), inter / (area(a) + area(b) - inter)


def score(detections: list[dict], gt: list[dict], region=None, ignore=()) -> dict:
    def centre(b):
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    if region:
        detections = [d for d in detections if region[0] <= centre(d["bbox"])[0] <= region[2] and region[1] <= centre(d["bbox"])[1] <= region[3]]
    detections = [d for d in detections if not any(_overlap(d["bbox"], g)[0] >= 0.6 for g in ignore)
                  or any(_overlap(d["bbox"], g["bbox"])[0] >= 0.6 for g in gt)]
    found = [False] * len(gt)
    rows, c = [], Counter()
    for d in detections:
        best = None
        for i, g in enumerate(gt):
            frac, iou = _overlap(d["bbox"], g["bbox"])
            if frac >= 0.6 and iou >= 0.15:
                ok = g["kind"] in COMPATIBLE.get(d["kind"], {d["kind"]})
                key = (ok, iou)
                if best is None or key > best[0]:
                    best = (key, i)
        if best is None:
            c["false"] += 1
            rows.append((d["id"], d["kind"], "false", None))
        elif best[0][0]:
            c["correct"] += 1
            found[best[1]] = True
            rows.append((d["id"], d["kind"], "correct", gt[best[1]]["kind"]))
        else:
            c["wrong_kind"] += 1
            rows.append((d["id"], d["kind"], "wrong kind", gt[best[1]]["kind"]))
    n = len(detections)
    missed = [g["kind"] for g, f in zip(gt, found) if not f]
    return {"detections": n, "correct": c["correct"], "wrong_kind": c["wrong_kind"], "false": c["false"],
            "precision": round(c["correct"] / max(1, n), 3), "gt": len(gt), "found": sum(found),
            "recall": round(sum(found) / max(1, len(gt)), 3), "missed": sorted(missed), "rows": rows}


def detect(case: str) -> list[dict]:
    from benchmark.holdout_inputs import load
    from engine.analyzer import FloorPlanAnalyzer
    from engine.cleaning.cleaner import clean

    name, _, page = case.partition(":")
    image, ev, _ = load(REAL / name, int(page) if page else None)
    cand = (lambda: clean(image, REAL / name, int(page), ev, mode="vector")) if page else None
    r = FloorPlanAnalyzer().analyze(image, name, text_evidence=ev or None, cleaner_candidate=cand)
    return (r.get("building") or {}).get("objects", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--save")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    plans = json.loads(FIXTURE.read_text())["plans"]
    out, tot = {}, Counter()
    for case, spec in plans.items():
        if args.only and case not in args.only.split(","):
            continue
        s = score(detect(case), spec["objects"], spec.get("region"), spec.get("ignore", ()))
        out[case] = s
        for k in ("detections", "correct", "wrong_kind", "false", "gt", "found"):
            tot[k] += s[k]
        print(f"{case:<10} precision {s['precision']:.2f} ({s['correct']}/{s['detections']}; wrong kind {s['wrong_kind']}, "
              f"false {s['false']})   recall {s['recall']:.2f} ({s['found']}/{s['gt']})   missed {s['missed']}")
        if args.verbose:
            for row in s["rows"]:
                if row[2] != "correct":
                    print("   ", *row)
    summary = {k: tot[k] for k in tot} | {"precision": round(tot["correct"] / max(1, tot["detections"]), 3),
                                          "recall": round(tot["found"] / max(1, tot["gt"]), 3)}
    print("TOTAL", json.dumps(summary))
    if args.save:
        Path(args.save).write_text(json.dumps({"summary": summary, "plans": out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
