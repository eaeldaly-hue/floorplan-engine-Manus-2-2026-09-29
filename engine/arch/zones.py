"""Functional zones inside structural spaces.

  Space  structural region: bounded by walls and openings (engine.structure)
  Zone   functional region inside a space: kitchen, dining, living - bounded by implied edges
         (counters, islands, partial walls, furniture groups), not by walls

A zone is created only when the evidence supports it:
  - evidence items are typed objects (engine.arch.objects) and printed labels (OCR / PDF text) lying
    in the space; each proposes function weights
  - items of one function are clustered in space; a cluster's support is its summed evidence with
    diminishing returns for repeated items of one kind; it is accepted with enough support
    (several items, a label, or one decisive object: range, refrigerator, dishwasher, dining set,
    sofa) - a lone counter or chair never makes a zone
  - zones are made for the functions that share open space (kitchen, dining, living) and only when a
    space holds at least two accepted clusters; a space with one function gets that function (no
    zones); bedrooms, baths and laundries characterise whole spaces, never zones
Extents grow from each cluster's items through the space's floor (geodesic, all clusters at the same
pace, up to a reach limit) - floor no cluster reaches stays circulation. Each zone records the cues
along its boundaries: counter / fixture edge, partial wall, or implied (open).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

OPEN_FUNCTIONS = ("kitchen", "dining", "living")
DECISIVE = {"range", "refrigerator", "dishwasher", "dining set", "sofa", "loveseat", "bed", "bathtub"}
SPACE_DECISIVE = {"toilet", "dryer", "laundry appliance"}
LABEL_FUNCTION = {
    "kitchen": "kitchen", "kitchenette": "kitchen", "pantry": "kitchen",
    "dining": "dining", "dining room": "dining", "breakfast nook": "dining", "nook": "dining",
    "living": "living", "living room": "living", "family": "living", "family room": "living",
    "great room": "living", "den": "living", "sitting area": "living", "lounge": "living",
    "bedroom": "bedroom", "master bedroom": "bedroom", "bath": "bath", "bathroom": "bath",
    "laundry": "laundry", "laundry room": "laundry",
}
LABEL_WEIGHT = 1.5
CLUSTER_GAP_CM = 220.0
REACH_CM = 220.0
CELL_CM = 5.0


def _labels_enabled() -> bool:
    """FLOORPLAN_ZONE_LABELS=0: zones from drawn objects only (measures the label-free path)."""
    import os
    return os.environ.get("FLOORPLAN_ZONE_LABELS", "1").strip().lower() not in ("0", "off", "false", "no")


def label_function(name: str | None) -> str | None:
    return LABEL_FUNCTION.get((name or "").strip().lower())


def _support(items: list[dict]) -> float:
    """Summed evidence; repeated items of one kind add with diminishing returns."""
    by_kind: dict = {}
    for it in items:
        by_kind.setdefault(it["kind"], []).append(it["weight"])
    total = 0.0
    for ws in by_kind.values():
        ws = sorted(ws, reverse=True)
        total += ws[0] + 0.3 * sum(ws[1:3])
    return total


def _clusters(items: list[dict], ppc: float) -> list[list[dict]]:
    gap = CLUSTER_GAP_CM * ppc
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def box_gap(a, b):
        ax0, ay0, ax1, ay1 = a["bbox"]
        bx0, by0, bx1, by1 = b["bbox"]
        dx = max(0.0, max(ax0, bx0) - min(ax1, bx1))
        dy = max(0.0, max(ay0, by0) - min(ay1, by1))
        return math.hypot(dx, dy)

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if box_gap(items[i], items[j]) <= gap:
                parent[find(i)] = find(j)
    groups: dict = {}
    for i, it in enumerate(items):
        groups.setdefault(find(i), []).append(it)
    return list(groups.values())


def _accepted(items: list[dict]) -> bool:
    s = _support(items)
    decisive = any(it["kind"] in DECISIVE and it["weight"] >= 0.8 for it in items)
    labelled = any(it["kind"] == "label" for it in items)
    return s >= 1.0 and (len(items) >= 2 or decisive or labelled)


def _grow(free: np.ndarray, seeds: list[np.ndarray], steps: int) -> np.ndarray:
    """All seeds grow through free cells at the same pace (geodesic, 8-neighbour); 0 = unreached."""
    lab = np.zeros(free.shape, np.int32)
    for k, s in enumerate(seeds, 1):
        lab[s & free & (lab == 0)] = k
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(steps):
        unassigned = free & (lab == 0)
        if not unassigned.any():
            break
        claims = np.zeros(free.shape, np.int32)
        for k in range(1, len(seeds) + 1):
            reach = cv2.dilate((lab == k).astype(np.uint8), kernel) > 0
            new = reach & unassigned & (claims == 0)
            claims[new] = k
        if not claims.any():
            break
        lab[claims > 0] = claims[claims > 0]
    return lab


def infer_zones(spaces: list[dict], shape, objects: list, labels: list, ppc: float | None,
                wall_mask: np.ndarray | None = None) -> dict:
    """Add 'zones' (and 'function' when one function characterises the space) to building space
    records. `objects`: engine.arch.objects.Obj; `labels`: [(name, (x, y))]. Returns a summary."""
    summary = {"spaces_with_zones": 0, "zones": 0, "space_functions": 0, "scale_px_per_cm": ppc}
    if not ppc:
        return summary
    h, w = shape[:2]
    g = max(1, int(round(CELL_CM * ppc)))                   # px per grid cell
    gh, gw = (h + g - 1) // g, (w + g - 1) // g
    walls = None
    if wall_mask is not None and wall_mask.shape[:2] == (h, w):
        walls = cv2.resize((wall_mask > 0).astype(np.uint8), (gw, gh), interpolation=cv2.INTER_AREA) > 0
    for sp in spaces:
        poly = np.array(sp.get("polygon") or [], np.float64)
        if len(poly) < 3:
            continue
        m = np.zeros((gh, gw), np.uint8)
        cv2.fillPoly(m, [np.round(poly / g).astype(np.int32).reshape(-1, 1, 2)], 1)
        space = m > 0
        if not space.any():
            continue
        pts = poly.reshape(-1, 1, 2).astype(np.float32)

        def inside(x, y):
            return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

        items = []
        for o in objects:
            if not inside(*o.center):
                continue
            f = max(o.functions, key=o.functions.get)
            items.append({"source": "object", "id": o.id, "kind": o.kind, "function": f, "weight": o.functions[f],
                          "bbox": o.bbox, "why": "; ".join(o.evidence)})
        for name, (x, y) in (labels if _labels_enabled() else ()):
            f = label_function(name)
            if f and inside(x, y):
                half = 60 * ppc
                items.append({"source": "label", "id": name, "kind": "label", "function": f, "weight": LABEL_WEIGHT,
                              "bbox": (x - half, y - half, x + half, y + half), "why": f"printed label '{name}'"})
        # the drawing's own words win over geometry: an object typed as one function under a printed
        # label of another (a sofa group read as a dining set under 'LIVING') is down-weighted
        for it in items:
            if it["source"] != "object":
                continue
            x0, y0, x1, y1 = it["bbox"]
            for lb in items:
                if lb["source"] == "label" and lb["function"] != it["function"]:
                    lx, ly = (lb["bbox"][0] + lb["bbox"][2]) / 2, (lb["bbox"][1] + lb["bbox"][3]) / 2
                    if x0 <= lx <= x1 and y0 <= ly <= y1:
                        it["weight"] *= 0.3
                        it["why"] += f"; contradicted by the label '{lb['id']}'"
                        break
        clusters = []
        for f in sorted({it["function"] for it in items}):
            fitems = [it for it in items if it["function"] == f and it["weight"] >= 0.3]
            for cl in _clusters(fitems, ppc):
                if _accepted(cl):
                    clusters.append((f, cl, _support(cl)))
        open_clusters = [c for c in clusters if c[0] in OPEN_FUNCTIONS]
        if len(open_clusters) < 2:
            # a whole-space function needs a decisive object (or a label), never sinks / counters alone
            decided = [c for c in clusters if any(it["kind"] in DECISIVE or it["kind"] in SPACE_DECISIVE or it["kind"] == "label"
                                                  for it in c[1])]
            best = max(decided, key=lambda c: c[2], default=None)
            if best is not None and not sp.get("names"):          # a printed name already says what it is
                sp["function"] = {"function": best[0], "confidence": round(1 - math.exp(-best[2]), 2),
                                  "evidence": [{"kind": it["kind"], "id": it["id"], "weight": round(it["weight"], 2)} for it in best[1]]}
                summary["space_functions"] += 1
            continue
        # zone extents: grow from each cluster's items through the space's floor
        seeds = []
        for f, cl, _s in open_clusters:
            s = np.zeros((gh, gw), bool)
            for it in cl:
                x0, y0, x1, y1 = (int(round(v / g)) for v in it["bbox"])
                s[max(0, y0):max(0, y1) + 1, max(0, x0):max(0, x1) + 1] = True
            seeds.append(s & space)
        lab = _grow(space, seeds, int(REACH_CM / CELL_CM))
        zones = []
        for k, (f, cl, s) in enumerate(open_clusters, 1):
            zm = (lab == k).astype(np.uint8)
            if not zm.any():
                continue
            cs, _ = cv2.findContours(zm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cs, key=cv2.contourArea)
            c = cv2.approxPolyDP(c, 1.5, True)
            polygon = [[int(p[0][0] * g + g / 2), int(p[0][1] * g + g / 2)] for p in c]
            mom = cv2.moments(zm, binaryImage=True)
            cx, cy = (mom["m10"] / mom["m00"]) * g, (mom["m01"] / mom["m00"]) * g
            zones.append({"id": f"{sp['id']}.Z{k}", "function": f, "confidence": round(1 - math.exp(-s), 2),
                          "polygon": polygon, "center": [int(cx), int(cy)],
                          "area_m2": round(float(zm.sum()) * (CELL_CM / 100) ** 2, 1),
                          "evidence": [{"source": it["source"], "kind": it["kind"], "id": it["id"],
                                        "weight": round(it["weight"], 2), "why": it["why"]} for it in cl],
                          "boundaries": []})
        _boundaries(zones, lab, space, walls, open_clusters, g, ppc)
        sp["zones"] = zones
        sp["circulation_m2"] = round(float((space & (lab == 0)).sum()) * (CELL_CM / 100) ** 2, 1)
        summary["spaces_with_zones"] += 1
        summary["zones"] += len(zones)
    return summary


def _boundaries(zones, lab, space, walls, clusters, g, ppc) -> None:
    """For each zone: what separates it from its neighbours (other zones, circulation)."""
    fixture = np.zeros(lab.shape, bool)
    for f, cl, _ in clusters:
        for it in cl:
            if it["kind"] in ("counter", "counter run", "range", "sink", "refrigerator", "dishwasher", "media / cabinet"):
                x0, y0, x1, y1 = (int(round(v / g)) for v in it["bbox"])
                fixture[max(0, y0):y1 + 1, max(0, x0):x1 + 1] = True
    r = max(1, int(round(40 / CELL_CM)))
    fixture_near = cv2.dilate(fixture.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    wall_near = (cv2.dilate(walls.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0) if walls is not None else np.zeros(lab.shape, bool)
    k3 = np.ones((3, 3), np.uint8)
    ids = {k: z for k, z in enumerate(zones, 1)}
    for k, z in ids.items():
        mine = lab == k
        ring = (cv2.dilate(mine.astype(np.uint8), k3) > 0) & ~mine & space
        neigh = {}
        for other in np.unique(lab[ring]):
            other = int(other)
            cells = ring & (lab == other)
            n = int(cells.sum())
            if n == 0:
                continue
            cues = {"counter / fixture edge": int((cells & fixture_near).sum()),
                    "partial wall": int((cells & wall_near & ~fixture_near).sum())}
            cues["implied (open)"] = n - sum(cues.values())
            cue = max(cues, key=cues.get)
            to = ids[other]["id"] if other in ids else "circulation"
            neigh[to] = {"to": to, "cue": cue, "length_m": round(n * CELL_CM / 100, 1),
                         "cues": {c: round(v / n, 2) for c, v in cues.items() if v}}
        z["boundaries"] = sorted(neigh.values(), key=lambda b: -b["length_m"])
