"""Opening benchmark on the canonical test cases: doors, windows, passages (structure + classifier).

Every plan of fixtures/real_plans.json with opening ground truth ('openings' or 'openings_fixture').
Reports per plan and in total: door / window precision and recall, classification accuracy,
window<->door confusion, passages called doors, false candidates.

    python -m benchmark.openings_eval [--plans a.png,b.png] [--analyzer] [--save F]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.evaluate import OpeningStats
from benchmark.real_plans import DEFAULT_DIR as DEFAULT_PLANS, ROOT, load_plan

KEYS = ("door_precision", "door_recall", "window_precision", "window_recall", "classification_accuracy",
        "window_as_door", "door_as_window", "passages_called_door", "false_door_or_window", "false_candidates",
        "missed_candidates")


def gt_openings(spec: dict) -> list | None:
    if spec.get("openings"):
        return spec["openings"]
    if spec.get("openings_fixture"):
        gts = json.loads((ROOT / spec["openings_fixture"]).read_text())["openings"]
        for o in gts:
            o.setdefault("wall_thickness_px", 38.0 if o["id"].startswith("W") else 30.0)
        return gts
    return None


def run(plans: list[str] | None = None, analyze=None) -> dict:
    """`analyze(image) -> openings` (default: structure + classify_openings)."""
    from engine.opening_detection import classify_openings
    from engine.structure import analyze_structure

    analyze = analyze or (lambda image: classify_openings(image, analyze_structure(image)))
    doc = json.loads((ROOT / "benchmark/fixtures/real_plans.json").read_text())["plans"]
    total = OpeningStats()
    per = {}
    for name, spec in doc.items():
        gts = gt_openings(spec)
        if not gts or (plans and name not in plans) or not (DEFAULT_PLANS / name).exists():
            continue
        st = OpeningStats()
        st.add(analyze(load_plan(name, spec, DEFAULT_PLANS)), gts, name)
        total.merge(st)
        m = st.metrics()
        per[name] = {k: (round(m[k], 3) if isinstance(m[k], float) else m[k]) for k in KEYS}
    m = total.metrics()
    return {"plans": per, "total": {k: (round(m[k], 3) if isinstance(m[k], float) else m[k]) for k in KEYS + ("gt_doors", "gt_windows", "passages")},
            "errors": total.errors}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans")
    ap.add_argument("--save")
    ap.add_argument("--analyzer", action="store_true", help="openings as the full analyzer reports them (OCR cached)")
    args = ap.parse_args()
    analyze = None
    if args.analyzer:
        from benchmark import ocr_cache
        from engine.analyzer import FloorPlanAnalyzer
        ocr_cache.enable()
        analyze = lambda image: FloorPlanAnalyzer().analyze(image, "plan")["openings"]   # noqa: E731
    out = run(args.plans.split(",") if args.plans else None, analyze)
    for name, m in out["plans"].items():
        print(f"{name:18s} " + " ".join(f"{k}={v}" for k, v in m.items()))
    print("TOTAL", json.dumps(out["total"]))
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
