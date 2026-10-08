"""Functional zones on the canonical test cases (engine.arch.zones through the analyzer).

Ground truth:
  fixtures/zones.json       explicit zone points (3.pdf p5: kitchen / dining / living, no labels)
  fixtures/pdf_plans.json   PDF pages: open-plan groups (Living / Dining / Kitchen of each unit)
  fixtures/real_plans.json  dev images: open-plan groups
A GT point of an open-plan function (kitchen, dining, living) whose structural space also holds a
point of another open-plan function must lie in a zone of its function ('zone points'). Points
whose function has a room of its own are not zone points. Zones in spaces holding no open-plan GT
are counted as unsupported.

    python -m benchmark.zones_eval [--only 3.pdf:5,22.pdf:6] [--save F]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from benchmark.real_plans import DEFAULT_DIR, FIXTURE, load_plan

ROOT = Path(__file__).resolve().parents[1]
FUNCTION = {"kitchen": "kitchen", "dining": "dining", "dining room": "dining", "breakfast nook": "dining",
            "living": "living", "living room": "living", "family": "living", "family room": "living",
            "great room": "living", "living/dining": "living|dining"}


def _function(name: str) -> str | None:
    return FUNCTION.get(name.strip().lower())


def _inside(poly, x, y) -> bool:
    return len(poly) >= 3 and cv2.pointPolygonTest(np.array(poly, np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0


def gt_points(key: str) -> list[dict]:
    zones = json.loads((ROOT / "benchmark/fixtures/zones.json").read_text())["plans"]
    if key in zones:
        return [{"function": z["function"], "point": z["point"], "name": z["function"]} for z in zones[key]["zones"]]
    pdf = json.loads((ROOT / "benchmark/fixtures/pdf_plans.json").read_text())["plans"]
    dev = json.loads(FIXTURE.read_text())["plans"]
    spec = pdf.get(key) or dev.get(key) or {}
    out = []
    for r in spec.get("rooms", []):
        f = _function(r["name"])
        if f:
            out.append({"function": f, "point": r["point"], "name": r["name"]})
    return out


def evaluate(key: str, response: dict) -> dict:
    b = response["building"]
    pts = gt_points(key)
    res = {"zone_points": 0, "correct": 0, "wrong_function": 0, "no_zone": 0, "zones": 0, "unsupported_zones": 0,
           "details": []}
    by_space: dict = {}
    for p in pts:
        sp = next((s for s in b["spaces"] if _inside(s["polygon"], *p["point"])), None)
        if sp is not None:
            by_space.setdefault(sp["id"], (sp, []))[1].append(p)
    for sid, (sp, ps) in by_space.items():
        functions = {f for p in ps for f in p["function"].split("|")}
        if len(functions) < 2:
            continue                                               # one function: a room, not zones
        for p in ps:
            res["zone_points"] += 1
            z = next((z for z in sp.get("zones", []) if _inside(z["polygon"], *p["point"])), None)
            if z is None:
                res["no_zone"] += 1
                verdict = "no zone"
            elif z["function"] in p["function"].split("|"):
                res["correct"] += 1
                verdict = "ok"
            else:
                res["wrong_function"] += 1
                verdict = f"zone is {z['function']}"
            res["details"].append({"space": sid, "gt": p["name"], "verdict": verdict})
    supported = {sid for sid, (sp, ps) in by_space.items() if len({f for p in ps for f in p["function"].split("|")}) >= 2}
    for s in b["spaces"]:
        for z in s.get("zones", []):
            res["zones"] += 1
            if s["id"] not in supported:
                res["unsupported_zones"] += 1
    return res


def analyze(key: str):
    from benchmark import ocr_cache
    from engine.analyzer import FloorPlanAnalyzer

    ocr_cache.enable()
    if ".pdf:" in key:
        from benchmark.holdout_inputs import load
        from engine.cleaning.cleaner import clean
        pdf, page = key.split(":")
        img, ev, _ = load(DEFAULT_DIR / pdf, int(page))
        return FloorPlanAnalyzer().analyze(img, key, text_evidence=ev or None,
                                           cleaner_candidate=lambda: clean(img, DEFAULT_DIR / pdf, int(page), ev, mode="vector"))
    dev = json.loads(FIXTURE.read_text())["plans"]
    return FloorPlanAnalyzer().analyze(load_plan(key, dev[key], DEFAULT_DIR), key)


def default_keys() -> list[str]:
    dev = json.loads(FIXTURE.read_text())["plans"]
    keys = ["3.pdf:5", "3.pdf:4", "22.pdf:5", "22.pdf:6", "22.pdf:7", "22.pdf:8"]
    keys += [k for k, v in dev.items() if (DEFAULT_DIR / k).exists() and any(r.get("group") for r in v.get("rooms", []))]
    return keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--save")
    args = ap.parse_args()
    keys = args.only.split(",") if args.only else default_keys()
    tot = {"zone_points": 0, "correct": 0, "wrong_function": 0, "no_zone": 0, "zones": 0, "unsupported_zones": 0}
    rows = {}
    for key in keys:
        r = analyze(key)
        m = evaluate(key, r)
        rows[key] = m
        for k in tot:
            tot[k] += m[k]
        print(f"{key:18s} zone points {m['correct']}/{m['zone_points']} wrong {m['wrong_function']} none {m['no_zone']} "
              f"zones {m['zones']} unsupported {m['unsupported_zones']} objects {r['building']['summary'].get('objects')}", flush=True)
    print("TOTAL", json.dumps(tot))
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps({"rows": rows, "total": tot}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
