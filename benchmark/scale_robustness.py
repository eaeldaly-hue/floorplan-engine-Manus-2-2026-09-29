"""Robustness to drawing resolution: the same canonical plans analysed at other scales.

A plan does not change when it is exported at another resolution, so neither should its
reconstruction. Each dev plan (fixtures/real_plans.json) is resampled by each factor and analysed
by the full analyzer; GT room points are scaled with it. Reports API room recovery per factor.

    python -m benchmark.scale_robustness [--scales 0.8,1.25,1.5] [--plans a.png,b.png] [--save F]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from benchmark.real_plans import DEFAULT_DIR, FIXTURE, full_room_metrics, load_plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="0.8,1.25,1.5")
    ap.add_argument("--plans")
    ap.add_argument("--save")
    args = ap.parse_args()
    from engine.analyzer import FloorPlanAnalyzer

    doc = json.loads(FIXTURE.read_text())["plans"]
    scales = [float(v) for v in args.scales.split(",")]
    rows, tot = {}, {f: [0, 0, 0, 0, 0.0] for f in scales}
    for name, spec in doc.items():
        if (args.plans and name not in args.plans.split(",")) or not (DEFAULT_DIR / name).exists() or not spec.get("rooms"):
            continue
        img = load_plan(name, spec, DEFAULT_DIR)
        for f in scales:
            if img.shape[0] * img.shape[1] * f * f > 24_000_000:
                continue
            im = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
            rooms = [dict(r, point=[int(round(r["point"][0] * f)), int(round(r["point"][1] * f))]) for r in spec["rooms"]]
            t = time.perf_counter()
            resp = FloorPlanAnalyzer().analyze(im, name)
            m = full_room_metrics(resp, rooms)
            sec = time.perf_counter() - t
            rows.setdefault(name, {})[f] = {"recovered": m["recovered"], "merged": m["merged"], "missed": m["missed"],
                                            "points": m["points"], "seconds": round(sec, 1),
                                            "chosen": (resp.get("reconstruction") or {}).get("chosen", "default")}
            for i, k in enumerate(("recovered", "merged", "missed", "points")):
                tot[f][i] += m[k]
            tot[f][4] += sec
            print(f"{name:18s} x{f:<4} {m['recovered']}/{m['points']} merged {m['merged']} missed {m['missed']} "
                  f"{rows[name][f]['chosen']} {sec:.1f}s", flush=True)
    summary = {str(f): {"recovery": round(v[0] / max(1, v[3]), 3), "merged": v[1], "missed": v[2], "seconds": round(v[4], 1)}
               for f, v in tot.items()}
    print("SUMMARY", json.dumps(summary))
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
