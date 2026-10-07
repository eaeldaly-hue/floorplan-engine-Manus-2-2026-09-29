"""Physical-space audit: structure (physical truth layer) vs the API's final room geometry.

    python scripts/diagnostics/space_audit.py [--only 5.jpeg,8.png] [--save out.json] [--fixture PATH] [--plans DIR]

For every labelled room point (fixture 'rooms', 'group' = one physical space when shared):

  structure   the engine's physical space at the point is exclusive to the point's group (ok),
              shared with another group (merged), or missing.
  api         the API region at the point (named room boundary, else unnamed space) holds no
              point of another group (ok) or does (merged); the boundary method is recorded.
  fidelity    when the structure is ok: IoU between the API region and the structural space.

Outcomes per point:
  kept        structure ok, API ok and the same geometry (IoU >= 0.7)
  reshaped    structure ok, API ok but different geometry (e.g. a dimension box replaced the room)
  broken      structure ok, API wrong (post-processing turned correct geometry into wrong geometry)
  repaired    structure wrong, API ok (post-processing split a merged space)
  still_wrong structure wrong, API wrong; 'as_open_plan' when the API calls the merged space an
              open plan although the points belong to different physical rooms (a merge reported
              as 'merged-space' is honest and not flagged)
Measurement only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.real_plans import DEFAULT_DIR, FIXTURE, load_plan, space_at  # noqa: E402


def _inside(poly, x, y, tol: float = 3.0) -> bool:
    """Point in polygon, allowing a few pixels for points on a thin line at a region's edge."""
    pts = np.array([[p["x"], p["y"]] if isinstance(p, dict) else p for p in poly], np.int32)
    return pts.size > 0 and cv2.pointPolygonTest(pts.reshape(-1, 1, 2), (float(x), float(y)), True) >= -tol
from engine.analyzer import FloorPlanAnalyzer  # noqa: E402
from engine.structure import analyze_structure  # noqa: E402

STRUCTURAL = ("wall-region", "open-plan-shared", "merged-space", "passage-partition")


def _poly_mask(shape, poly) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    pts = np.array([[p["x"], p["y"]] if isinstance(p, dict) else p for p in poly], np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def audit(name: str, spec: dict, plans_dir: Path) -> dict:
    image = load_plan(name, spec, plans_dir)
    s = analyze_structure(image)
    response = FloorPlanAnalyzer().analyze(image, name)
    rooms = spec.get("rooms", [])
    groups = [r.get("group") or f"_{i}" for i, r in enumerate(rooms)]
    labels = s.space_labels
    spaces = [space_at(labels, *r["point"]) for r in rooms]
    owners: dict[int, set] = {}
    for sp, g in zip(spaces, groups):
        if sp > 0:
            owners.setdefault(sp, set()).add(g)
    regions = []
    for r in response["rooms"]:
        if r.get("boundary"):
            regions.append({"kind": "room", "id": r["id"], "name": r["name"], "method": r["boundary"]["method"],
                            "poly": r["boundary"]["polygon"], "shares": bool(r.get("shares_space_with"))})
    for u in response.get("unlabeled_spaces", []):
        if u.get("boundary"):
            regions.append({"kind": "unnamed", "id": u["id"], "name": None, "method": u["boundary"]["method"],
                            "poly": u["boundary"]["polygon"], "shares": False})
    rows = []
    for i, (r, g, sp) in enumerate(zip(rooms, groups, spaces)):
        x, y = r["point"]
        struct = "missing" if sp <= 0 else "ok" if owners[sp] == {g} else "merged"
        hits = [reg for reg in regions if _inside(reg["poly"], x, y, 0.0)] or [reg for reg in regions if _inside(reg["poly"], x, y)]
        reg = (hits or [None])[0]
        if reg is None:
            api = "missing"
        else:
            foreign = any(groups[j] != g and _inside(reg["poly"], *q["point"], 0.0) for j, q in enumerate(rooms))
            api = "merged" if foreign else "ok"
        iou = None
        if struct == "ok" and reg is not None:
            a = labels == sp
            b = _poly_mask(labels.shape, reg["poly"])
            iou = round(float((a & b).sum()) / max(1, float((a | b).sum())), 3)
        if struct == "ok":
            outcome = "broken" if api != "ok" else ("kept" if (iou or 0) >= 0.7 else "reshaped")
        else:
            outcome = "repaired" if api == "ok" else "still_wrong"
        as_open_plan = bool(reg and api == "merged" and reg["method"] == "open-plan-shared")
        rows.append({"point": r["point"], "name": r.get("name"), "group": g, "structure": struct, "api": api,
                     "method": reg["method"] if reg else None, "region": reg["kind"] if reg else None, "iou": iou,
                     "outcome": outcome, "as_open_plan": as_open_plan})
    tally = {}
    for row in rows:
        tally[row["outcome"]] = tally.get(row["outcome"], 0) + 1
    return {"rows": rows, "tally": tally,
            "methods": {m: sum(1 for reg in regions if reg["method"] == m) for m in sorted({reg["method"] for reg in regions})}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only")
    ap.add_argument("--save")
    ap.add_argument("--fixture", default=str(FIXTURE))
    ap.add_argument("--plans", default=str(DEFAULT_DIR))
    args = ap.parse_args()
    plans = json.loads(Path(args.fixture).read_text())["plans"]
    only = set(args.only.split(",")) if args.only else None
    out, total = {}, {}
    for name, spec in plans.items():
        if (only and name not in only) or spec.get("format_only") or spec.get("out_of_scope") or not spec.get("rooms"):
            continue
        res = audit(name, spec, Path(args.plans))
        out[name] = res
        for k, v in res["tally"].items():
            total[k] = total.get(k, 0) + v
        bad = [f"{r['name']}@{r['point']}:{r['outcome']}({r['method']},iou={r['iou']}{',as-open-plan' if r['as_open_plan'] else ''})"
               for r in res["rows"] if r["outcome"] not in ("kept",)]
        print(f"{name:20s} {res['tally']}  {'; '.join(bad)}", flush=True)
    print("TOTAL", total)
    if args.save:
        Path(args.save).write_text(json.dumps({"total": total, "plans": out}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
