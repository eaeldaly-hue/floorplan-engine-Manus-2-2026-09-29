"""Why did the engine get this plan wrong? Error report against the canonical ground truth.

    python -m scripts.diagnostics.explain_plan 14.png [19.jpg ...] [--out output/explain]

Per plan, every error is listed with the engine's own evidence:

  missed_opening      GT door / window with no API opening on it
  false_opening       API door / window where GT has none (symbol, relation, candidate source)
  wrong_type          door <-> window <-> passage confusions (symbol at native / legible scale)
  passage_as_door     GT passage (cased opening) typed as a door
  merged_rooms        GT rooms of different groups in one space, with the boundary that is missing:
                      the closest opening candidate / rejected gap between their points
  missed_room         GT room point in no space (wall, exterior)
  wrong_name          printed label read or associated wrongly
  ambiguous           hypotheses whose scores were close (engine.reconstruction)

Writes <plan>.json and <plan>.png (errors drawn over the plan) to --out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _poly_index(spaces, shape):
    idx = np.zeros(shape[:2], np.int32)
    for k, sp in enumerate(spaces, 1):
        if sp["polygon"]:
            cv2.fillPoly(idx, [np.array(sp["polygon"], np.int32).reshape(-1, 1, 2)], k)
    return idx


def explain(name: str, spec: dict, image: np.ndarray, r: dict) -> dict:
    from benchmark.evaluate import match
    from benchmark.openings_eval import gt_openings

    errors = []
    gts = gt_openings(spec) or []
    ops = r["openings"]
    pairs = match(ops, gts) if gts else []
    mp = {i: j for i, j, _ in pairs}
    mg = {j for _, j, _ in pairs}
    for j, g in enumerate(gts):
        if j not in mg and g["kind"] in ("door", "window"):
            errors.append({"kind": "missed_opening", "gt": g["id"], "gt_kind": g["kind"], "at": [g["p0"], g["p1"]],
                           "why": "no wall gap or line-bridged candidate on it (jambs not free wall ends, or the "
                                  "gap was sealed / rejected)"})
    for i, o in enumerate(ops):
        ev = o["evidence"]
        if i not in mp:
            if o["type"] in ("door", "window") and gts:
                errors.append({"kind": "false_opening", "id": o["id"], "type": o["type"], "at": [o["start"], o["end"]],
                               "symbol": ev.get("symbol"), "relation": ev.get("room_relation"),
                               "source": ev.get("candidate_source"), "reason": ev.get("reason")})
            continue
        g = gts[mp[i]]
        if g["kind"] == "opening" and o["type"] == "door":
            errors.append({"kind": "passage_as_door", "id": o["id"], "gt": g["id"], "symbol": ev.get("symbol"),
                           "at": [o["start"], o["end"]]})
        elif g["kind"] in ("door", "window") and o["type"] != g["kind"]:
            errors.append({"kind": "wrong_type", "id": o["id"], "gt": g["id"], "gt_kind": g["kind"], "type": o["type"],
                           "symbol": ev.get("symbol"), "native": ev.get("symbol_at_native_scale"),
                           "relation": ev.get("room_relation"), "at": [o["start"], o["end"]], "reason": ev.get("reason")})
    b = r["building"]
    idx = _poly_index(b["spaces"], image.shape)
    rooms = spec.get("rooms", [])
    owner: dict = {}
    for i, rm in enumerate(rooms):
        x, y = rm["point"]
        k = int(idx[min(image.shape[0] - 1, y), min(image.shape[1] - 1, x)])
        if not k:
            errors.append({"kind": "missed_room", "room": rm["name"], "at": rm["point"],
                           "why": "the point is on a wall, outside the building or in an unsealed leak"})
            continue
        owner.setdefault(k, []).append(i)
    for k, members in owner.items():
        groups = {rooms[i].get("group") or f"_{i}" for i in members}
        if len(groups) > 1:
            pts = np.array([rooms[i]["point"] for i in members], float)
            mid = pts.mean(axis=0)
            near = sorted(ops, key=lambda o: float(np.hypot(*(np.array(o["center"]) - mid))))[:2]
            errors.append({"kind": "merged_rooms", "space": b["spaces"][k - 1]["id"],
                           "rooms": [rooms[i]["name"] for i in members],
                           "nearest_openings": [{"id": o["id"], "type": o["type"], "symbol": o["evidence"].get("symbol")} for o in near],
                           "why": "no sealed boundary between these rooms: a wall the thickness filter did not keep, "
                                  "or a door / glazing not found as a candidate"})
    for i, rm in enumerate(rooms):
        if not rm.get("printed"):
            continue
        x, y = rm["point"]
        k = int(idx[min(image.shape[0] - 1, y), min(image.shape[1] - 1, x)])
        names = b["spaces"][k - 1]["names"] if k else []
        if k and not names:
            errors.append({"kind": "unnamed_room", "room": rm["name"], "printed": rm["printed"], "space": b["spaces"][k - 1]["id"],
                           "why": "label not read (size / contrast) or associated to another space"})
    rec = r.get("reconstruction") or {}
    cands = rec.get("candidates") or []
    if len(cands) > 1:
        sc = sorted(c["score"] for c in cands)
        if sc[-1] - sc[-2] < 1.0:
            errors.append({"kind": "ambiguous", "chosen": rec.get("chosen"),
                           "scores": {c["name"]: c["score"] for c in cands}, "evidence": {c["name"]: c["evidence"] for c in cands}})
    counts: dict = {}
    for e in errors:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    return {"plan": name, "counts": counts, "errors": errors, "reconstruction": rec.get("chosen", "default")}


COLOURS = {"missed_opening": (0, 140, 255), "false_opening": (255, 0, 255), "wrong_type": (0, 0, 255),
           "passage_as_door": (200, 120, 0), "missed_room": (0, 0, 0), "merged_rooms": (0, 0, 200)}


def draw(image: np.ndarray, rep: dict, r: dict) -> np.ndarray:
    vis = cv2.addWeighted(image, 0.55, np.full_like(image, 255), 0.45, 0)
    lw = max(2, int(max(image.shape[:2]) / 500))
    for sp in r["building"]["spaces"]:
        cv2.polylines(vis, [np.array(sp["polygon"], np.int32).reshape(-1, 1, 2)], True, (160, 160, 160), 1)
    for e in rep["errors"]:
        c = COLOURS.get(e["kind"], (0, 0, 0))
        at = e.get("at")
        if isinstance(at, list) and len(at) == 2 and isinstance(at[0], list):
            p0, p1 = tuple(int(v) for v in at[0]), tuple(int(v) for v in at[1])
            cv2.line(vis, p0, p1, c, 2 * lw)
            cv2.putText(vis, e["kind"].replace("_", " "), p0, cv2.FONT_HERSHEY_SIMPLEX, 0.35 * lw, c, max(1, lw // 2))
        elif isinstance(at, list) and len(at) == 2:
            cv2.drawMarker(vis, tuple(int(v) for v in at), c, cv2.MARKER_TILTED_CROSS, 8 * lw, lw)
    return vis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plans", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "output" / "explain"))
    args = ap.parse_args()
    from benchmark import ocr_cache
    from benchmark.real_plans import DEFAULT_DIR, FIXTURE, load_plan
    from engine.analyzer import FloorPlanAnalyzer

    ocr_cache.enable()
    doc = json.loads(FIXTURE.read_text())["plans"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.plans:
        spec = doc[name]
        img = load_plan(name, spec, DEFAULT_DIR)
        r = FloorPlanAnalyzer().analyze(img, name)
        rep = explain(name, spec, img, r)
        stem = name.replace(".", "_")
        (out / f"{stem}.json").write_text(json.dumps(rep, indent=1, default=str))
        cv2.imwrite(str(out / f"{stem}.png"), draw(img, rep, r))
        print(name, json.dumps(rep["counts"]), "reconstruction:", rep["reconstruction"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
