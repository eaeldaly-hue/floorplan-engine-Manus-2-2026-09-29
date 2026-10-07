"""Benchmark N.1 measurements on real plans (measurement only; nothing here changes the engine).

Uses the hand-labelled points of fixtures/real_plans.json:

  rooms          points inside rooms; points sharing a 'group' are one open-plan area
  openings       door / window / 'opening' (real but ambiguous: passage, glazed door ...)
  nonstructural  points on furniture, fixtures and markers: must not be wall
  protected      points on short real structural pieces (piers, stubs, mullions): must stay wall

and reports, per plan:

  under_segmentation   GT room groups that share a structural space with another group
  over_segmentation    open-plan groups split over several spaces; spaces holding no GT point
                       (fragments or unlabelled small rooms such as closets)
  partitions           between_rooms candidates whose two sides hold points of the same GT group
                       (a partition inside one open-plan area: fabricated unless a real
                       partial wall exists there)
  false_walls          nonstructural points covered by the wall mask, the size of the wall
                       component they sit on, whether it is attached to the main wall network
                       and which wall class produced it
  openings_detail      real-opening recall (door+window, and including ambiguous openings),
                       false candidates broken down by relation / candidate source / symbol and
                       a probable cause (furniture jamb, pattern-like surroundings, other)
  protected            protected points present in the wall mask, and "at risk" under a
                       compact-component / isolated-jamb filter (a probe of the kind of rule
                       proposed for N.2, not an engine rule)

The causes are diagnostic probes, not ground truth: 'furniture jamb' means an endpoint of the
candidate touches a wall component that covers a labelled nonstructural point; 'pattern' means
the candidate's surroundings contain many short parallel strokes (hatching, treads, shelving).
"""

from __future__ import annotations

import cv2
import numpy as np

from benchmark.evaluate import match


def _covered(mask: np.ndarray, x: float, y: float, r: int = 3) -> bool:
    h, w = mask.shape
    x, y = int(round(x)), int(round(y))
    win = mask[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
    return bool(win.size and (win > 0).any())


def _component_at(comp: np.ndarray, x: float, y: float, r: int = 3) -> int:
    h, w = comp.shape
    x, y = int(round(x)), int(round(y))
    win = comp[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
    pos = win[win > 0]
    return int(np.bincount(pos).argmax()) if pos.size else 0


def _groups(rooms):
    return [r.get("group") or f"_{i}" for i, r in enumerate(rooms)]


def _space_of_points(labels, rooms):
    from benchmark.real_plans import space_at
    return [space_at(labels, *r["point"]) for r in rooms]


def _pattern_strokes(ink: np.ndarray, cx: float, cy: float, half: int) -> int:
    """Number of short parallel strokes crossed by scanlines through the candidate's
    surroundings (median over several rows and columns, best orientation)."""
    h, w = ink.shape
    x0, x1 = max(0, int(cx) - half), min(w, int(cx) + half)
    y0, y1 = max(0, int(cy) - half), min(h, int(cy) + half)
    patch = ink[y0:y1, x0:x1] > 0
    if patch.size == 0:
        return 0
    best = 0
    for arr in (patch, patch.T):
        rows = np.linspace(0, arr.shape[0] - 1, 7).astype(int)
        runs = [int(np.count_nonzero(np.diff(arr[r].astype(np.int8)) == 1)) for r in rows]
        best = max(best, int(np.median(runs)))
    return best


def structural_probe(s, spec: dict, openings: list[dict]) -> dict:
    rooms = spec.get("rooms", [])
    labels = s.space_labels
    wall = (s.wall_mask > 0).astype(np.uint8)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(wall, connectivity=8)
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1) if n > 1 else 0
    t = max(2.0, float(getattr(s.walls, "thinnest", 0) or s.wall_thickness))
    primary = None
    if s.walls is not None and getattr(s.walls, "primary", None) is not None and s.walls.primary.shape == wall.shape:
        primary = s.walls.primary > 0
    out: dict = {}

    # --- segmentation ---------------------------------------------------------------
    groups = _groups(rooms)
    spaces = _space_of_points(labels, rooms) if rooms else []
    by_space: dict[int, set] = {}
    for sp, g in zip(spaces, groups):
        if sp > 0:
            by_space.setdefault(sp, set()).add(g)
    merged_groups = sorted({g for v in by_space.values() if len(v) > 1 for g in v})
    multi = {g for g in groups if groups.count(g) > 1}
    split_groups = {g: sorted({sp for sp, gg in zip(spaces, groups) if gg == g and sp > 0}) for g in multi}
    split_groups = {g: v for g, v in split_groups.items() if len(v) > 1}
    img_area = labels.shape[0] * labels.shape[1]
    ids, areas = np.unique(labels[labels > 0], return_counts=True)
    orphan = [(int(i), int(a)) for i, a in zip(ids, areas) if int(i) not in by_space and a >= 0.002 * img_area]
    out["under_segmentation"] = {"groups": len(set(groups)), "groups_sharing_a_space": len(merged_groups),
                                 "merged_groups": merged_groups,
                                 "missed_points": sum(1 for sp in spaces if sp <= 0)}
    out["over_segmentation"] = {"open_plan_groups": len(multi), "open_plan_groups_split": len(split_groups),
                                "split_groups": {g: len(v) for g, v in split_groups.items()},
                                "spaces_without_gt_point": len(orphan),
                                "orphan_area_share": round(sum(a for _, a in orphan) / max(1, sum(areas)), 3)}

    # --- partitions inside one GT group -----------------------------------------------
    group_of_space = {sp: next(iter(g)) for sp, g in by_space.items() if len(g) == 1}
    intra = []
    for c in s.candidates:
        if c.relation != "between_rooms":
            continue
        a, b = c.sides
        ga, gb = group_of_space.get(a), group_of_space.get(b)
        if ga is not None and ga == gb:
            intra.append({"center": [int(c.center[0]), int(c.center[1])], "group": ga})
    out["partitions"] = {"between_rooms": sum(c.relation == "between_rooms" for c in s.candidates),
                         "inside_one_group": len(intra), "where": intra[:12]}

    # --- false walls (furniture / fixtures / markers) ------------------------------------
    bad_comps = set()
    fw = []
    for p in spec.get("nonstructural", []):
        x, y = p["point"]
        hit = _covered(wall, x, y)
        k = _component_at(comp, x, y) if hit else 0
        if k:
            bad_comps.add(k)
        fw.append({"what": p["what"], "wall": hit,
                   "component_area": int(stats[k, cv2.CC_STAT_AREA]) if k else 0,
                   "attached_to_main_walls": bool(k and k == largest),
                   "class": (None if not hit or primary is None else "primary" if _covered(primary.astype(np.uint8), x, y) else "secondary/line")})
    covered = [f for f in fw if f["wall"]]
    out["false_walls"] = {"nonstructural_points": len(fw), "covered_by_wall": len(covered),
                          "attached_to_main_walls": sum(f["attached_to_main_walls"] for f in covered),
                          "isolated": sum(not f["attached_to_main_walls"] for f in covered),
                          "from_primary_class": sum(f["class"] == "primary" for f in covered),
                          "false_wall_pixels": int(sum(stats[k, cv2.CC_STAT_AREA] for k in bad_comps if k != largest)),
                          "false_wall_pixel_share": round(float(sum(stats[k, cv2.CC_STAT_AREA] for k in bad_comps if k != largest)) / max(1, int(wall.sum())), 4),
                          "points": fw}

    # --- openings: recall and false candidates ---------------------------------------------
    gts = spec.get("openings") or []
    if gts:
        pairs = match(openings, gts)
        mp = {i for i, _, _ in pairs}
        mg = {j for _, j, _ in pairs}
        dw = [j for j, g in enumerate(gts) if g["kind"] in ("door", "window")]
        amb = [j for j, g in enumerate(gts) if g["kind"] == "opening"]
        # unlabelled small pieces detached from the wall network (furniture, symbols or real piers)
        small_comps = {k for k in range(1, n) if k != largest and max(stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]) <= 8 * t}
        false = []
        for i, o in enumerate(openings):
            if i in mp:
                continue
            ev = o["evidence"]
            ends = [o["start"], o["end"]]
            furniture = any(_component_at(comp, x, y, r=max(3, int(t))) in bad_comps for x, y in ends) if bad_comps else False
            strokes = _pattern_strokes(s.symbol_ink, o["center"][0], o["center"][1], int(max(2.0 * o["width_pixels"], 6 * t)))
            small = any(_component_at(comp, x, y, r=max(3, int(t))) in small_comps for x, y in ends)
            cause = ("furniture_jamb" if furniture else "pattern" if strokes >= 6
                     else "small_isolated_jamb" if small else "other")
            false.append({"center": o["center"], "type": o["type"], "relation": ev["room_relation"],
                          "source": ev["candidate_source"], "symbol": ev["symbol"], "cause": cause, "strokes": strokes})

        def tally(key):
            out_ = {}
            for f in false:
                out_[f[key]] = out_.get(f[key], 0) + 1
            return out_
        out["openings_detail"] = {
            "gt_doors_windows": len(dw), "gt_ambiguous": len(amb),
            "recall_doors_windows": round(sum(j in mg for j in dw) / max(1, len(dw)), 3),
            "recall_all_real": round(len(mg) / max(1, len(gts)), 3),
            "missed": [gts[j]["id"] for j in range(len(gts)) if j not in mg],
            "false_candidates": len(false),
            "false_by_relation": tally("relation"), "false_by_source": tally("source"),
            "false_by_type": tally("type"), "false_by_cause": tally("cause"),
            "false": false[:40],
        }

    # --- protected structural pieces ------------------------------------------------------
    jamb_comps = {}
    for o in openings:
        for x, y in (o["start"], o["end"]):
            k = _component_at(comp, x, y, r=max(3, int(t)))
            if k:
                jamb_comps[k] = jamb_comps.get(k, 0) + 1
    prot = []
    for p in spec.get("protected", []):
        x, y = p["point"]
        k = _component_at(comp, x, y)
        if not k:
            prot.append({"what": p["what"], "present": False, "at_risk": None})
            continue
        bw, bh = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        compact = max(bw, bh) <= 4 * t and stats[k, cv2.CC_STAT_AREA] <= 16 * t * t
        isolated = k != largest
        jamb = k in jamb_comps
        prot.append({"what": p["what"], "present": True, "compact": bool(compact), "isolated": bool(isolated),
                     "is_jamb": jamb, "at_risk": bool(jamb and (compact or (isolated and max(bw, bh) <= 8 * t)))})
    out["protected"] = {"points": len(prot), "present": sum(p["present"] for p in prot),
                        "at_risk": sum(bool(p["at_risk"]) for p in prot), "pieces": prot}
    return out


def api_probe(response: dict, spec: dict) -> dict:
    """Open-plan grouping and boundary support in the API response."""
    from benchmark.real_plans import _inside
    rooms = spec.get("rooms", [])
    groups = _groups(rooms)
    methods: dict[str, int] = {}
    for r in response["rooms"]:
        m = r["boundary"]["method"] if r.get("boundary") else "none"
        methods[m] = methods.get(m, 0) + 1
    polys = [(r["id"], r.get("space_id"), r["boundary"]["polygon"]) for r in response["rooms"] if r.get("boundary")]
    multi = sorted({g for g in groups if groups.count(g) > 1})
    single = split = merged = 0
    detail = {}
    for g in multi:
        pts = [r["point"] for r, gg in zip(rooms, groups) if gg == g]
        others = [r["point"] for r, gg in zip(rooms, groups) if gg != g]
        spaces = set()
        foreign = False
        for x, y in pts:
            hits = [(sid or rid) for rid, sid, poly in polys if _inside(poly, x, y)]
            spaces.add(hits[0] if hits else None)
            for rid, sid, poly in polys:
                if _inside(poly, x, y) and any(_inside(poly, *q) for q in others):
                    foreign = True
        spaces.discard(None)
        single += len(spaces) == 1
        split += len(spaces) > 1
        merged += foreign
        detail[g] = {"points": len(pts), "api_spaces": len(spaces), "shares_with_other_group": foreign}
    # physical spaces vs names: a room name is metadata; the space exists without it
    structural_ids = {r.get("space_id") for r in response["rooms"] if r.get("space_id") and r.get("boundary")
                      and (r.get("physical_space") or r["boundary"]["method"] in ("wall-region", "open-plan-shared", "merged-space", "passage-partition"))}
    unnamed = response.get("unlabeled_spaces", [])
    named_polys = [r["boundary"]["polygon"] for r in response["rooms"] if r.get("boundary")]
    unnamed_polys = [u["boundary"]["polygon"] for u in unnamed if u.get("boundary")]
    coverage = {"named_room": 0, "unnamed_space_only": 0, "no_space": 0}
    for r in rooms:
        x, y = r["point"]
        if any(_inside(poly, x, y) for poly in named_polys):
            coverage["named_room"] += 1
        elif any(_inside(poly, x, y) for poly in unnamed_polys):
            coverage["unnamed_space_only"] += 1
        else:
            coverage["no_space"] += 1
    physical = {"named_rooms": len(response["rooms"]),
                "named_rooms_with_boundary": sum(1 for r in response["rooms"] if r.get("boundary")),
                "physical_spaces_named": len(structural_ids),
                "unnamed_spaces": len(unnamed),
                "physical_spaces": len(structural_ids) + len(unnamed),
                "gt_points_by_space": coverage}
    unsupported = methods.get("wall-ray-estimate", 0) + methods.get("dimension-box-estimate", 0) + methods.get("dimension-only-estimate", 0)
    return {"boundary_methods": methods, "physical": physical,
            "unsupported_boundaries": unsupported,
            "dimension_snapped_boundaries": methods.get("dimension-box-wall-snapped", 0),
            "passage_partitions": methods.get("passage-partition", 0),
            "open_plan": {"groups": len(multi), "one_space": single, "split": split, "merged_with_other_group": merged,
                          "groups_detail": detail}}


__all__ = ["structural_probe", "api_probe"]
