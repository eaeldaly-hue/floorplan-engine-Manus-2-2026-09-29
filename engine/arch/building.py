"""The structured plan: Building -> envelope -> walls / openings -> spaces, with their relations.

The analyzer's records (rooms, unlabeled spaces, openings, topology) are views made for the
Workbench. This is the one architectural representation behind them, built from the final
geometry:

  buildings   each building on the sheet: footprint outline (engine.arch.envelope), area
  walls       wall runs (axis and diagonal bands) with thickness, and whether they bound the
              exterior (envelope walls) or divide the inside
  openings    doors / windows / unclassified openings, the wall they sit in, and the two places
              they connect (space ids or 'exterior'), found geometrically on both sides
  spaces      every final space: names written in it (none, one, several = open plan), area,
              the building it belongs to (or none: outside every building), its neighbours
              through openings and through shared walls
  graph       one adjacency list over spaces + 'exterior' (edges: door / window / opening / wall)

Confidence values are carried from the stages that measured them; nothing here invents
geometry or labels.
"""

from __future__ import annotations

import cv2
import numpy as np

from .envelope import estimate_envelope, plan_model_network

EXTERIOR = "exterior"
OUTDOOR = {"Porch", "Terrace", "Balcony", "Deck", "Patio", "Outdoor living"}


def _poly(points) -> np.ndarray:
    pts = [[p["x"], p["y"]] if isinstance(p, dict) else list(p) for p in points]
    return np.array(pts, np.int32).reshape(-1, 1, 2)


def _space_at(index: np.ndarray, ids: list, x: float, y: float) -> str | None:
    h, w = index.shape
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return EXTERIOR
    k = int(index[yi, xi])
    return ids[k - 1] if k > 0 else None


def build_building(structure, spaces: list, rooms: list, unlabeled: list, openings: list,
                   pixel_scale: dict | None = None, plan_model: dict | None = None) -> dict:
    shape = structure.space_labels.shape
    h, w = shape
    t = float(structure.wall_thickness or 4.0)

    # --- envelope -------------------------------------------------------------------------------
    if plan_model and plan_model.get("role") == "fallback":
        net, pts = plan_model_network(plan_model, shape)
        env = estimate_envelope(structure, network=net, opening_points=pts)
    else:
        env = estimate_envelope(structure)

    # --- final spaces, rasterised once ---------------------------------------------------------------
    index = np.zeros(shape, np.int32)                  # value k -> spaces[k-1]
    for k, sp in enumerate(spaces, 1):
        if sp.get("polygon"):
            cv2.fillPoly(index, [_poly(sp["polygon"])], k)
    ids = [sp["id"] for sp in spaces]
    names: dict = {}
    for r in rooms:
        if r.get("space_id"):
            names.setdefault(r["space_id"], []).append({"room_id": r["id"], "name": r["name"],
                                                        "confidence": r.get("label_confidence")})
    exterior_label = structure.space_labels < 0

    # share of each space inside each building, in one pass
    nb = len(env.buildings) if env.available else 0
    bidx = np.zeros(shape, np.int32)
    for j, b in enumerate(env.buildings if env.available else [], 1):
        bidx[b.mask & (bidx == 0)] = j
    counts = np.bincount((index * (nb + 1) + bidx).ravel(), minlength=(len(spaces) + 1) * (nb + 1))
    counts = counts.reshape(len(spaces) + 1, nb + 1)

    def building_of(k) -> str | None:
        total = counts[k].sum()
        if not nb or not total:
            return None
        j = int(np.argmax(counts[k, 1:])) + 1
        return env.buildings[j - 1].id if counts[k, j] >= 0.5 * total else None

    unit = pixel_scale.get("unit") if pixel_scale else None
    ppu = pixel_scale.get("pixels_per_unit") if pixel_scale else None
    space_records = []
    for k, sp in enumerate(spaces, 1):
        n = names.get(sp["id"], [])
        rec = {
            "id": sp["id"],
            "names": [x["name"] for x in n],
            "room_ids": [x["room_id"] for x in n],
            "kind": ("outdoor" if n and all(x["name"] in OUTDOOR for x in n) else
                     "open-plan" if len(n) > 1 else "room" if n else "unnamed"),
            "area_px": int(sp.get("area_pixels") or counts[k].sum()),
            "area": round(float(sp.get("area_pixels") or counts[k].sum()) / ppu ** 2, 1) if ppu else None,
            "area_unit": f"{unit}²" if ppu and unit else None,
            "center": [int(v) for v in sp.get("center", (0, 0))],
            "polygon": [[int(p[0]), int(p[1])] if not isinstance(p, dict) else [p["x"], p["y"]] for p in sp.get("polygon", [])],
            "building": building_of(k),
            "neighbors": {"through_openings": [], "through_walls": []},
        }
        space_records.append(rec)
    by_id = {r["id"]: r for r in space_records}

    # --- openings: what each one connects, found on both sides --------------------------------------
    opening_records = []
    edges = []
    for o in openings:
        p0, p1 = np.array(o["start"], float), np.array(o["end"], float)
        L = float(np.linalg.norm(p1 - p0))
        if L < 1:
            continue
        u = (p1 - p0) / L
        nrm = np.array([-u[1], u[0]])
        reach = 0.5 * float(o["evidence"].get("wall_thickness_px") or t) + max(4.0, 0.5 * t)
        sides = []
        for sign in (-1, 1):
            votes: dict = {}
            for f in (0.3, 0.5, 0.7):
                for depth in (1.0, 1.8):
                    q = p0 + u * L * f + sign * nrm * reach * depth
                    s = _space_at(index, ids, *q)
                    if s is None:
                        yi, xi = int(round(q[1])), int(round(q[0]))
                        s = EXTERIOR if (0 <= xi < w and 0 <= yi < h and exterior_label[yi, xi]) else None
                    if s is not None:
                        votes[s] = votes.get(s, 0) + 1
            sides.append(max(votes, key=votes.get) if votes else None)
        a, b = sides
        connects = [s for s in (a, b) if s is not None]
        rec = {"id": o["id"], "type": o["type"], "confidence": o.get("confidence"),
               "p0": [int(v) for v in o["start"]], "p1": [int(v) for v in o["end"]],
               "width_px": o.get("width_pixels"), "width": o.get("width_display"),
               "connects": connects,
               "role": ("exterior" if EXTERIOR in connects else "interior" if len(set(connects)) == 2
                        else "inside one space" if len(connects) == 2 else "unknown"),
               "host_wall": None}
        outdoor = [x for x in connects if x in by_id and by_id[x]["kind"] == "outdoor"]
        if rec["role"] == "interior" and len(outdoor) == 1:
            rec["role"] = "to outdoor space"               # a terrace / balcony door or window
        rec.update(_opening_object(o, a, b))
        if "window" in rec and len(outdoor) == 1 and EXTERIOR not in connects:
            rec["window"]["exterior"] = True
            rec["window"]["outdoor_space"] = outdoor[0]
            inside = [x for x in connects if x != outdoor[0]]
            rec["window"]["adjacent_space"] = inside[0] if inside else None
        opening_records.append(rec)
        if len(set(connects)) == 2:
            edges.append({"a": connects[0], "b": connects[1], "kind": o["type"], "via": o["id"]})
            for x, y in ((connects[0], connects[1]), (connects[1], connects[0])):
                if x in by_id and y not in by_id[x]["neighbors"]["through_openings"]:
                    by_id[x]["neighbors"]["through_openings"].append(y)

    # --- walls: bands, exterior or interior -----------------------------------------------------------
    labels = structure.space_labels
    env_mask = env.mask if env.available else None
    wall_records = []
    if plan_model and plan_model.get("role") == "fallback":
        runs = [(wl["p0"], wl["p1"], wl["thickness_px"], None) for wl in plan_model.get("walls", [])]
    else:
        runs = [(bd.p0, bd.p1, bd.thickness, bd.orientation) for bd in getattr(structure, "bands", []) or []]
    for i, (q0, q1, thickness, orientation) in enumerate(runs, 1):
        p0, p1 = np.array(q0, float), np.array(q1, float)
        L = float(np.linalg.norm(p1 - p0))
        if L < 2 * t:
            continue
        u = (p1 - p0) / L
        nrm = np.array([-u[1], u[0]])
        orientation = orientation or ("horizontal" if abs(u[1]) < 0.2 else "vertical" if abs(u[0]) < 0.2 else "diagonal")
        outside = 0
        samples = 0
        for f in (0.25, 0.5, 0.75):
            for sign in (-1, 1):
                q = p0 + u * L * f + sign * nrm * (0.5 * thickness + max(3.0, 0.5 * t))
                xi, yi = int(round(q[0])), int(round(q[1]))
                samples += 1
                if not (0 <= xi < w and 0 <= yi < h):
                    outside += 1
                elif labels[yi, xi] < 0 or (env_mask is not None and not env_mask[yi, xi] and labels[yi, xi] <= 0):
                    outside += 1
        wall_records.append({"id": f"WL{i:03d}", "p0": [int(v) for v in p0], "p1": [int(v) for v in p1],
                             "thickness_px": round(float(thickness), 1), "orientation": orientation,
                             "exterior": outside >= 2})

    runs_out = _wall_runs(wall_records, opening_records, t)
    for sp in space_records:
        sp["doors"], sp["windows"], sp["passages"] = [], [], []
    for o in opening_records:
        for sid in set(o["connects"]):
            if sid in by_id:
                by_id[sid]["doors" if o["type"] == "door" else "windows" if o["type"] == "window" else "passages"].append(o["id"])
    for sp in space_records:
        sp["exterior_openings"] = sum(1 for o in opening_records if sp["id"] in o["connects"] and o["role"] == "exterior")

    # --- shared walls between final spaces --------------------------------------------------------------
    k = max(3, int(round(1.5 * t)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    seen = set()
    for i in range(1, len(spaces) + 1):
        poly = spaces[i - 1].get("polygon")
        if not poly:
            continue
        x0, y0, bw, bh = cv2.boundingRect(_poly(poly))
        xa, ya, xb, yb = max(0, x0 - k), max(0, y0 - k), min(w, x0 + bw + k), min(h, y0 + bh + k)
        crop = index[ya:yb, xa:xb]
        grown = cv2.dilate((crop == i).astype(np.uint8), kernel) > 0
        for j in np.unique(crop[grown]):
            j = int(j)
            if j <= i or (i, j) in seen:
                continue
            seen.add((i, j))
            a, b = ids[i - 1], ids[j - 1]
            if b not in by_id[a]["neighbors"]["through_openings"]:
                edges.append({"a": a, "b": b, "kind": "wall", "via": None})
            by_id[a]["neighbors"]["through_walls"].append(b)
            by_id[b]["neighbors"]["through_walls"].append(a)

    buildings = [{"id": b.id, "polygon": b.polygon, "area_px": b.area_px, "opening_evidence": b.openings,
                  "spaces": [r["id"] for r in space_records if r["building"] == b.id]} for b in env.buildings]
    return {
        "schema": "floorplan.building/1",
        "buildings": buildings,
        "envelope": {"available": env.available,
                     "not_buildings": [{"reason": r, "bbox": bb} for r, bb in env.rejected]},
        "walls": wall_records,
        "wall_runs": runs_out,
        "openings": opening_records,
        "spaces": space_records,
        "graph": {"nodes": [r["id"] for r in space_records] + [EXTERIOR], "edges": edges},
        "summary": {
            "buildings": len(buildings),
            "spaces": len(space_records),
            "named_spaces": sum(r["kind"] != "unnamed" for r in space_records),
            "open_plan_spaces": sum(r["kind"] == "open-plan" for r in space_records),
            "spaces_outside_buildings": sum(r["building"] is None for r in space_records) if env.available else None,
            "walls": len(wall_records),
            "wall_runs": len(runs_out),
            "openings_hosted": sum(o["host_wall"] is not None for o in opening_records),
            "exterior_walls": sum(r["exterior"] for r in wall_records),
            "doors": sum(o["type"] == "door" for o in opening_records),
            "windows": sum(o["type"] == "window" for o in opening_records),
            "exterior_openings": sum(o["role"] == "exterior" for o in opening_records),
            "outdoor_space_openings": sum(o["role"] == "to outdoor space" for o in opening_records),
            "outdoor_spaces": sum(r["kind"] == "outdoor" for r in space_records),
            "interior_openings": sum(o["role"] == "interior" for o in opening_records),
        },
    }


OPERATION = {"swing arc": ("hinged", 1), "door leaf": ("hinged", 1), "double swing": ("double hinged", 2),
             "sliding panels": ("sliding", 1)}


def _opening_object(o: dict, side_a, side_b) -> dict:
    """Architectural description of one opening from the classifier's evidence. Side a lies along
    -normal of the span (start -> end), side b along +normal."""
    ev = o.get("evidence", {})
    sym = str(ev.get("symbol", "")).split(" (")[0]
    if o["type"] == "door":
        op, leaves = OPERATION.get(sym, ("unspecified", None))
        hinge = ev.get("hinge")
        side = ev.get("swing_side")
        into = None
        if op.endswith("hinged") and side in (-1, 1):
            into = side_b if side > 0 else side_a
        hinge_pts = ([o["start"]] if hinge == "start" else [o["end"]] if hinge == "end"
                     else [o["start"], o["end"]] if op == "double hinged" else [])
        return {"door": {"operation": op, "leaves": leaves, "hinges": [[int(v) for v in p] for p in hinge_pts],
                         "swing_into": into if op.endswith("hinged") else None,
                         "leaf_angle": ev.get("leaf_angle") if op.endswith("hinged") else None,
                         "symbol": ev.get("symbol")}}
    if o["type"] == "window":
        inside = [x for x in (side_a, side_b) if x not in (None, EXTERIOR)]
        return {"window": {"exterior": EXTERIOR in (side_a, side_b), "adjacent_space": inside[0] if inside else None,
                           "glazing_lines": ev.get("glazing_lines"), "symbol": ev.get("symbol")}}
    return {"passage": {"symbol": ev.get("symbol")}}


def _wall_runs(walls: list, openings: list, t: float) -> list:
    """Continuous architectural boundaries: collinear wall pieces joined across the openings they
    host (and across junction gaps), so a wall with a door is one wall, not two."""
    def line_of(p0, p1):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L = float(np.linalg.norm(d))
        u = d / L if L else np.array([1.0, 0.0])
        if u[0] < -1e-9 or (abs(u[0]) <= 1e-9 and u[1] < 0):
            u = -u
        n = np.array([-u[1], u[0]])
        return u, n, float(np.dot(p0, n)), sorted((float(np.dot(p0, u)), float(np.dot(p1, u))))

    items = []
    for wl in walls:
        u, n, off, (a, b) = line_of(wl["p0"], wl["p1"])
        items.append(("wall", wl, u, n, off, a, b))
    for o in openings:
        u, n, off, (a, b) = line_of(o["p0"], o["p1"])
        items.append(("opening", o, u, n, off, a, b))
    tol = max(3.0, 0.75 * t)
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(len(items)), key=lambda i: (round(np.degrees(np.arctan2(items[i][2][1], items[i][2][0]))), items[i][4], items[i][5]))
    for x, i in enumerate(order):
        ki, _, ui, _, oi, ai, bi = items[i]
        for j in order[x + 1:]:
            kj, _, uj, _, oj, aj, bj = items[j]
            if abs(float(np.dot(ui, uj))) < 0.995 or abs(oi - oj) > tol:
                continue
            gap = max(aj, ai) - min(bi, bj)
            if gap <= 1.5 * t:
                parent[find(i)] = find(j)
    groups: dict = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(items[i])
    runs = []
    for g in groups.values():
        wall_items = [x for x in g if x[0] == "wall"]
        if not wall_items:
            continue                                   # an opening with no wall on its line
        u, n, off = wall_items[0][2], wall_items[0][3], float(np.median([x[4] for x in wall_items]))
        a, b = min(x[5] for x in g), max(x[6] for x in g)
        rid = f"WR{len(runs) + 1:02d}"
        ops = [x[1] for x in g if x[0] == "opening"]
        for o in ops:
            o["host_wall"] = rid
        solid = sum(x[6] - x[5] for x in wall_items)
        open_len = sum(x[6] - x[5] for x in g if x[0] == "opening")
        p0, p1 = u * a + n * off, u * b + n * off
        runs.append({"id": rid, "p0": [int(round(v)) for v in p0], "p1": [int(round(v)) for v in p1],
                     "thickness_px": round(float(np.median([x[1]["thickness_px"] for x in wall_items])), 1),
                     "exterior": sum(x[1]["exterior"] for x in wall_items) * 2 > len(wall_items),
                     "segments": [x[1]["id"] for x in wall_items], "openings": [o["id"] for o in ops],
                     "doors": [o["id"] for o in ops if o["type"] == "door"],
                     "windows": [o["id"] for o in ops if o["type"] == "window"],
                     "solid_length_px": int(round(solid)), "open_length_px": int(round(open_len))})
    return runs
