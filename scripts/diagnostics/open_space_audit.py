"""Open-space audit: is each labelled open-plan group one physical space, and if not, what splits it?

    python scripts/diagnostics/open_space_audit.py [--only 6.png] [--no-api] [--out DIR] [--save JSON]

For every open-plan group in the fixture (room points sharing a 'group'):

  structure  the engine spaces holding the group's points: one (correct), split (several spaces),
             merged (an engine space also holds another group), missed (point in no space).
  splitters  for every pair of engine spaces of one group: the boundary between them (pixels on
             the shortest separation) and what it is made of: wall-mask pixels (and the wall
             component's size), sealed bridges (sealed mask minus wall mask) with the nearest
             candidate's source / relation / jamb evidence.
  api        (unless --no-api) what the API returns for the group's labelled zones: boundary
             methods, and whether one returned region covers every point of the group.

Writes a crop per group (engine spaces coloured, bridges red, walls dark, zone points) for visual
classification. Measurement only.
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

from benchmark.holdout_metrics import _separation  # noqa: E402
from benchmark.real_plans import DEFAULT_DIR, FIXTURE, _inside, load_plan, space_at  # noqa: E402
from engine.structure import analyze_structure  # noqa: E402


def audit_plan(name, spec, plans_dir, with_api=True, structure=None, image=None):
    image = image if image is not None else load_plan(name, spec, plans_dir)
    s = structure or analyze_structure(image)
    L = s.space_labels
    T = float(s.wall_thickness)
    rooms = spec.get("rooms", [])
    groups = [r.get("group") or f"_{i}" for i, r in enumerate(rooms)]
    open_groups = sorted({g for g in groups if groups.count(g) > 1})
    sp = [space_at(L, *r["point"]) for r in rooms]
    owners: dict[int, set] = {}
    for k, g in zip(sp, groups):
        if k > 0:
            owners.setdefault(k, set()).add(g)
    wall = s.wall_mask > 0
    bridge = (s.sealed_mask > 0) & ~wall if s.sealed_mask is not None else np.zeros_like(wall)
    ncomp, comp, cstats, _ = cv2.connectedComponentsWithStats(wall.astype(np.uint8), connectivity=8)
    largest = int(np.argmax(cstats[1:, cv2.CC_STAT_AREA]) + 1) if ncomp > 1 else 0
    response = None
    if with_api:
        from engine.analyzer import FloorPlanAnalyzer
        response = FloorPlanAnalyzer().analyze(image, name)
    out = []
    for g in open_groups:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        ks = sorted({sp[i] for i in idx if sp[i] > 0})
        missed = any(sp[i] <= 0 for i in idx)
        merged = any(len(owners[k]) > 1 for k in ks)
        status = "missed" if not ks else "merged" if merged else "split" if len(ks) > 1 else ("partly_missed" if missed else "one_space")
        splitters = []
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                band = _separation(L, ks[a], ks[b], T)
                if band is None:
                    continue
                n = int(band.sum())
                w_px = band & wall
                b_px = band & bridge
                comps = np.unique(comp[w_px])
                comps = [int(c) for c in comps if c]
                ys, xs = np.nonzero(band)
                cx, cy = float(xs.mean()), float(ys.mean())
                near = sorted(s.candidates, key=lambda c: np.hypot(c.center[0] - cx, c.center[1] - cy))[:1]
                splitters.append({
                    "between": [ks[a], ks[b]], "at": [int(cx), int(cy)], "band_px": n,
                    "wall_share": round(float(w_px.sum()) / max(1, n), 2),
                    "bridge_share": round(float(b_px.sum()) / max(1, n), 2),
                    "wall_components": [{"area": int(cstats[c, cv2.CC_STAT_AREA]), "main": c == largest,
                                         "extent": int(max(cstats[c, cv2.CC_STAT_WIDTH], cstats[c, cv2.CC_STAT_HEIGHT]))} for c in comps[:4]],
                    "nearest_candidate": ({"source": near[0].source, "relation": near[0].relation, "width": round(near[0].width, 1),
                                           "jambs": list(near[0].jambs), "dist": round(float(np.hypot(near[0].center[0] - cx, near[0].center[1] - cy)), 1)}
                                          if near else None),
                })
        api = None
        if response is not None:
            zone_rows = []
            pts = [rooms[i]["point"] for i in idx]
            for r in response["rooms"]:
                b = r.get("boundary")
                lc = (r["label_center"]["x"], r["label_center"]["y"])
                # API rooms whose label sits in one of the group's engine spaces
                if space_at(L, *lc) not in ks:
                    continue
                covers = bool(b) and all(_inside(b["polygon"], *p) for p in pts)
                zone_rows.append({"name": r["name"], "method": b["method"] if b else None, "covers_group": covers,
                                  "space_id": r.get("space_id")})
            api = {"zones_reported": len(zone_rows), "methods": sorted({z["method"] for z in zone_rows if z["method"]}),
                   "one_region_covers_group": any(z["covers_group"] for z in zone_rows),
                   "distinct_spaces": len({z["space_id"] for z in zone_rows}), "zones": zone_rows}
        out.append({"plan": name, "group": g, "zones": [rooms[i].get("name") for i in idx], "points": [rooms[i]["point"] for i in idx],
                    "engine_spaces": ks, "status": status, "splitters": splitters, "api": api})
    return out, s, image


def render(rows, s, image, path, spec):
    if not rows:
        return
    L = s.space_labels
    rng = np.random.default_rng(5)
    col = rng.integers(80, 240, (int(L.max()) + 2, 3))
    c = image.copy()
    m = L > 0
    c[m] = (0.45 * c[m] + 0.55 * col[L[m]]).astype(np.uint8)
    c[s.wall_mask > 0] = (50, 50, 50)
    bridge = (s.sealed_mask > 0) & (s.wall_mask == 0)
    c[bridge] = (0, 0, 255)
    sc = max(1, round(max(image.shape[:2]) / 900))
    tiles = []
    for row in rows:
        pts = np.array(row["points"])
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        pad = int(max(0.35 * max(x1 - x0, y1 - y0), 12 * s.wall_thickness, 60))
        X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
        crop = c[Y0:y1 + pad, X0:x1 + pad].copy()
        for (px, py), z in zip(row["points"], row["zones"]):
            cv2.circle(crop, (px - X0, py - Y0), 4 * sc, (0, 0, 0), -1)
            cv2.putText(crop, str(z)[:10], (px - X0 + 5, py - Y0 - 4), 0, 0.4 * sc, (0, 0, 0), sc)
        for sp_ in row["splitters"]:
            cv2.drawMarker(crop, (sp_["at"][0] - X0, sp_["at"][1] - Y0), (255, 0, 255), cv2.MARKER_CROSS, 16 * sc, 2 * sc)
        f = 420 / max(crop.shape[:2])
        t = cv2.resize(crop, (max(1, int(crop.shape[1] * f)), max(1, int(crop.shape[0] * f))))
        tile = np.full((440, 420, 3), 255, np.uint8)
        tile[20:20 + t.shape[0], :t.shape[1]] = t
        cv2.putText(tile, f"{row['plan']} {row['group']}: {row['status']}", (3, 14), 0, 0.45, (200, 0, 0), 1)
        tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.full((440, 420, 3), 255, np.uint8))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only")
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "output" / "realplans" / "open_space"))
    ap.add_argument("--save")
    args = ap.parse_args()
    plans = json.loads(FIXTURE.read_text())["plans"]
    only = set(args.only.split(",")) if args.only else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    allrows = []
    for name, spec in plans.items():
        if (only and name not in only) or spec.get("format_only") or spec.get("out_of_scope") or not spec.get("rooms"):
            continue
        rows, s, image = audit_plan(name, spec, DEFAULT_DIR, with_api=not args.no_api)
        if not rows:
            continue
        render(rows, s, image, out_dir / f"{Path(name).stem}.png", spec)
        for r in rows:
            allrows.append(r)
            sp = "; ".join(f"@{x['at']} wall {x['wall_share']} bridge {x['bridge_share']} "
                           f"cand {x['nearest_candidate']['source'] + '/' + x['nearest_candidate']['relation'] if x['nearest_candidate'] else '-'}"
                           f" d={x['nearest_candidate']['dist'] if x['nearest_candidate'] else '-'} comps {[(c['extent'], c['main']) for c in x['wall_components']]}"
                           for x in r["splitters"])
            api = r["api"]
            apis = f" | API: {api['zones_reported']} zones, {api['methods']}, covers={api['one_region_covers_group']}" if api else ""
            print(f"{name:18s} {r['group']:6s} {r['status']:13s} zones={r['zones']} spaces={r['engine_spaces']}{apis}\n    {sp}", flush=True)
    tally = {}
    for r in allrows:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    print("TOTAL", len(allrows), tally,
          "API one region covers group:", sum(1 for r in allrows if r["api"] and r["api"]["one_region_covers_group"]))
    if args.save:
        Path(args.save).write_text(json.dumps(allrows, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
