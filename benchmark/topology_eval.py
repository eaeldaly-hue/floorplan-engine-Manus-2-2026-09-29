"""Opening topology on the canonical test cases: are doors and windows attached to the right walls
and spaces in the structured plan (response['building'])?

For every ground-truth door / window matched by an API opening:
  hosted            the opening sits on a wall run (opening-to-wall association)
  door_connects     a door joins two different spaces (not one GT room on both sides), or a space
                    and the exterior (opening-to-space association)
  window_exterior   a window faces the exterior from a space holding a GT room
  swing_resolved    a hinged door says which of its two spaces it swings into

    python -m benchmark.topology_eval [--save F]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from benchmark.evaluate import match
from benchmark.openings_eval import gt_openings
from benchmark.real_plans import DEFAULT_DIR, FIXTURE, load_plan


def _space_index(b: dict, shape) -> tuple[np.ndarray, list]:
    idx = np.zeros(shape[:2], np.int32)
    ids = []
    for k, sp in enumerate(b["spaces"], 1):
        if sp["polygon"]:
            cv2.fillPoly(idx, [np.array(sp["polygon"], np.int32).reshape(-1, 1, 2)], k)
        ids.append(sp["id"])
    return idx, ids


def evaluate(name: str, spec: dict, response: dict, shape) -> dict:
    gts = gt_openings(spec) or []
    b = response["building"]
    idx, ids = _space_index(b, shape)
    rooms = spec.get("rooms", [])
    groups: dict = {}
    for i, r in enumerate(rooms):
        x, y = r["point"]
        k = int(idx[min(shape[0] - 1, y), min(shape[1] - 1, x)])
        if k:
            groups.setdefault(ids[k - 1], set()).add(r.get("group") or f"_{i}")
    by_id = {o["id"]: o for o in b["openings"]}
    pairs = match(response["openings"], gts)
    out = {"doors": 0, "door_connects": 0, "windows": 0, "window_exterior": 0, "matched": 0, "hosted": 0,
           "hinged": 0, "swing_resolved": 0}
    for i, j, _ in pairs:
        g = gts[j]
        o = by_id.get(response["openings"][i]["id"])
        if o is None or g["kind"] not in ("door", "window"):
            continue
        out["matched"] += 1
        out["hosted"] += o["host_wall"] is not None
        c = [x for x in o["connects"]]
        inside = [x for x in c if x != "exterior"]
        if g["kind"] == "door":
            out["doors"] += 1
            ok = False
            if len(set(inside)) == 2:
                # two different spaces, not one GT room on both sides (spaces without a GT point are
                # real unlabelled rooms - closets, lobbies - and count)
                ga, gb = groups.get(inside[0], set()), groups.get(inside[1], set())
                ok = not (ga & gb)
            elif "exterior" in c and len(inside) == 1:
                ok = True
            out["door_connects"] += ok
            d = o.get("door") or {}
            if str(d.get("operation", "")).endswith("hinged"):
                out["hinged"] += 1
                out["swing_resolved"] += d.get("swing_into") in inside
        else:
            out["windows"] += 1
            # type-independent: one side is the exterior or an outdoor space, the other a room
            kinds = {sp["id"]: sp["kind"] for sp in b["spaces"]}
            outer = [x for x in c if x == "exterior" or kinds.get(x) == "outdoor"]
            rooms_ = [x for x in c if x not in outer]
            out["window_exterior"] += len(outer) == 1 and len(rooms_) == 1 and bool(groups.get(rooms_[0]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    args = ap.parse_args()
    from benchmark import ocr_cache
    from engine.analyzer import FloorPlanAnalyzer

    ocr_cache.enable()
    doc = json.loads(FIXTURE.read_text())["plans"]
    tot: dict = {}
    per = {}
    for name, spec in doc.items():
        if not gt_openings(spec) or not (DEFAULT_DIR / name).exists():
            continue
        img = load_plan(name, spec, DEFAULT_DIR)
        r = FloorPlanAnalyzer().analyze(img, name)
        m = evaluate(name, spec, r, img.shape)
        per[name] = m
        for k, v in m.items():
            tot[k] = tot.get(k, 0) + v
        print(f"{name:18s} " + " ".join(f"{k}={v}" for k, v in m.items()), flush=True)
    rates = {"opening_to_wall": round(tot["hosted"] / max(1, tot["matched"]), 3),
             "door_to_spaces": round(tot["door_connects"] / max(1, tot["doors"]), 3),
             "window_to_exterior": round(tot["window_exterior"] / max(1, tot["windows"]), 3),
             "swing_resolved": round(tot["swing_resolved"] / max(1, tot["hinged"]), 3)}
    print("TOTAL", json.dumps(tot), json.dumps(rates))
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps({"plans": per, "total": tot, "rates": rates}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
