"""Locate the leaks behind merged physical spaces.

    python scripts/diagnostics/leak_finder.py [--only 8.png,11.png] [--out output/realplans/leaks]

For every pair of labelled room groups that share one engine space (the engine merged two
physical rooms), the leak is the bottleneck of the widest path between them inside that space:
pixels are added in order of decreasing clearance (distance to the nearest non-space pixel)
until the two points connect; the last pixel added is the narrowest passage on the best path.

Each leak site is reported with its clearance and what lies there: labelled openings (fixture),
engine opening candidates (sealed), N.2-rejected candidates, and how much drawn ink outside the
wall mask surrounds it. A crop of every site is written for visual classification.
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
from engine.structure import analyze_structure  # noqa: E402


def bottleneck(region: np.ndarray, a: tuple[int, int], b: tuple[int, int], max_side: int = 400):
    """(x, y, clearance) of the narrowest passage on the widest path from a to b in region."""
    ys, xs = np.nonzero(region)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    reg = region[y0:y1, x0:x1].astype(np.uint8)
    scale = min(1.0, max_side / max(reg.shape))
    small = cv2.resize(reg, (max(1, int(reg.shape[1] * scale)), max(1, int(reg.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
    dist = cv2.distanceTransform(np.pad(small, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    h, w = small.shape
    pa = (min(h - 1, int((a[1] - y0) * scale)), min(w - 1, int((a[0] - x0) * scale)))
    pb = (min(h - 1, int((b[1] - y0) * scale)), min(w - 1, int((b[0] - x0) * scale)))

    def snap(p):
        if small[p]:
            return p
        yy, xx = np.nonzero(small)
        k = int(np.argmin((yy - p[0]) ** 2 + (xx - p[1]) ** 2))
        return int(yy[k]), int(xx[k])
    pa, pb = snap(pa), snap(pb)
    order = np.argsort(-dist.ravel(), kind="stable")
    parent = np.full(h * w, -1, np.int64)

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root
    ia, ib = pa[0] * w + pa[1], pb[0] * w + pb[1]
    for i in order:
        if not small.flat[i]:
            break
        parent[i] = i
        y, x = divmod(int(i), w)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and parent[yy * w + xx] >= 0:
                ra, rb = find(i), find(yy * w + xx)
                if ra != rb:
                    parent[ra] = rb
        if parent[ia] >= 0 and parent[ib] >= 0 and find(ia) == find(ib):
            return x0 + (x + 0.5) / scale, y0 + (y + 0.5) / scale, float(dist[y, x]) / scale
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only")
    ap.add_argument("--out", default=str(ROOT / "output" / "realplans" / "leaks"))
    ap.add_argument("--save")
    args = ap.parse_args()
    plans = json.loads(FIXTURE.read_text())["plans"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    report = {}
    tiles = []
    for name, spec in plans.items():
        if (only and name not in only) or spec.get("format_only") or spec.get("out_of_scope") or not spec.get("rooms"):
            continue
        image = load_plan(name, spec, DEFAULT_DIR)
        s = analyze_structure(image)
        L = s.space_labels
        T = float(s.wall_thickness)
        rooms = spec["rooms"]
        groups = [r.get("group") or f"_{i}" for i, r in enumerate(rooms)]
        sp = [space_at(L, *r["point"]) for r in rooms]
        thin = cv2.bitwise_and(s.symbol_ink, cv2.bitwise_not(cv2.dilate(s.wall_mask, np.ones((3, 3), np.uint8))))
        leaks = []
        done = set()
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                if sp[i] <= 0 or sp[i] != sp[j] or groups[i] == groups[j]:
                    continue
                key = (sp[i], groups[i], groups[j])
                if key in done:
                    continue
                done.add(key)
                hit = bottleneck(L == sp[i], tuple(rooms[i]["point"]), tuple(rooms[j]["point"]))
                if hit is None:
                    continue
                x, y, clear = hit
                r = max(2 * clear, 2 * T, 8)
                near = lambda c: np.hypot(c.center[0] - x, c.center[1] - y) <= r + 0.5 * c.width
                gt_open = [o["id"] for o in spec.get("openings", []) if np.hypot((o["p0"][0] + o["p1"][0]) / 2 - x, (o["p0"][1] + o["p1"][1]) / 2 - y) <= r + 0.5 * np.hypot(o["p1"][0] - o["p0"][0], o["p1"][1] - o["p0"][1])]
                yi, xi = int(y), int(x)
                win = (slice(max(0, yi - int(r)), yi + int(r) + 1), slice(max(0, xi - int(r)), xi + int(r) + 1))
                leaks.append({"rooms": [rooms[i].get("name"), rooms[j].get("name")], "at": [int(x), int(y)],
                              "passage_px": round(2 * clear, 1), "passage_in_walls": round(2 * clear / max(T, 1), 2),
                              "gt_openings": gt_open, "candidates": int(sum(near(c) for c in s.candidates)),
                              "rejected": int(sum(near(c) for c in s.rejected_candidates)),
                              "thin_ink": round(float((thin[win] > 0).mean()), 3)})
                half = int(max(6 * T, 4 * clear, 40))
                crop = image[max(0, yi - half):yi + half, max(0, xi - half):xi + half].copy()
                cm = s.wall_mask[max(0, yi - half):yi + half, max(0, xi - half):xi + half] > 0
                crop[cm] = (0.5 * crop[cm] + [0, 0, 110]).astype(np.uint8)
                for c in s.candidates:
                    cv2.line(crop, (int(c.start[0]) - max(0, xi - half), int(c.start[1]) - max(0, yi - half)),
                             (int(c.end[0]) - max(0, xi - half), int(c.end[1]) - max(0, yi - half)), (0, 170, 0), 1)
                cv2.circle(crop, (xi - max(0, xi - half), yi - max(0, yi - half)), max(3, int(clear)), (255, 0, 255), 1)
                tile = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_AREA if crop.shape[0] > 240 else cv2.INTER_NEAREST)
                cv2.putText(tile, f"{name} {len(tiles)}", (3, 12), 0, 0.4, (255, 0, 0), 1)
                tiles.append(tile)
                leaks[-1]["tile"] = len(tiles) - 1
        report[name] = leaks
        for lk in leaks:
            print(f"{name:12s} #{lk['tile']:<3d} {lk['rooms'][0]!s:>14s} | {lk['rooms'][1]!s:<14s} at {lk['at']} passage {lk['passage_px']}px "
                  f"({lk['passage_in_walls']} walls) gt={lk['gt_openings']} cands={lk['candidates']} rejected={lk['rejected']} thin_ink={lk['thin_ink']}")
    for k in range(0, len(tiles), 20):
        chunk = tiles[k:k + 20] + [np.full((240, 240, 3), 255, np.uint8)] * ((-len(tiles[k:k + 20])) % 5)
        rows = [np.hstack(chunk[r:r + 5]) for r in range(0, len(chunk), 5)]
        cv2.imwrite(str(out_dir / f"leaks_{k // 20}.png"), np.vstack(rows))
    if args.save:
        Path(args.save).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
