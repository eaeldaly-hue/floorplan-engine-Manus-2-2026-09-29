"""Hold-out measurements (N.3a): physical spaces vs functional zones, open-plan grouping and
non-structural (furniture / fixture / stairs / hatching ...) induced errors.

Ground-truth schema and the meaning of each measure: docs/HOLDOUT_PROTOCOL.md. Measurement
only; nothing here changes the engine.
"""

from __future__ import annotations

import cv2
import numpy as np

from benchmark.evaluate import match


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def legacy_rooms(spec: dict) -> list[dict]:
    """Room points in the development-set format (group = physical space), so the existing
    room, name and opening measures run unchanged. A physical space with no functional entry
    contributes its first point as an unnamed room."""
    rooms = []
    named_spaces = set()
    for r in spec.get("rooms", []):
        rooms.append({"name": r.get("name"), "printed": r.get("printed"), "point": r["point"], "group": r["space"]})
        named_spaces.add(r["space"])
    for sp in spec.get("spaces", []):
        if sp["id"] not in named_spaces:
            rooms.append({"name": None, "printed": None, "point": sp["points"][0], "group": sp["id"]})
    return rooms


def normalize_spec(spec: dict) -> dict:
    if "spaces" not in spec:
        return spec
    return dict(spec, rooms=legacy_rooms(spec), _holdout_rooms=spec.get("rooms", []))


def _gt_points(spec: dict) -> dict[str, list]:
    """All interior points of each GT physical space (its own points + its zone points)."""
    pts = {sp["id"]: [tuple(p) for p in sp["points"]] for sp in spec.get("spaces", [])}
    for r in spec.get("_holdout_rooms", spec.get("rooms", [])):
        if r.get("space") in pts:
            pts[r["space"]].append(tuple(r["point"]))
    return pts


def _zones(spec: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in spec.get("_holdout_rooms", spec.get("rooms", [])):
        out.setdefault(r["space"], []).append(r)
    return out


def _open_plan_ids(spec: dict) -> set:
    zones = _zones(spec)
    return {sp["id"] for sp in spec.get("spaces", []) if sp.get("kind") == "open_plan" or len(zones.get(sp["id"], [])) >= 2}


def _ns_mask(shape, elements, typical_t, grow: float = 0.0) -> tuple[np.ndarray, list[np.ndarray]]:
    """Mask of labelled non-structural elements (bbox, or a small disc at the point)."""
    h, w = shape
    total = np.zeros((h, w), np.uint8)
    each = []
    r = max(3, int(round(0.5 * typical_t)))
    g = int(round(grow))
    for el in elements:
        m = np.zeros((h, w), np.uint8)
        if el.get("bbox"):
            x0, y0, x1, y1 = (int(v) for v in el["bbox"])
            cv2.rectangle(m, (x0 - g, y0 - g), (x1 + g, y1 + g), 1, -1)
        else:
            cv2.circle(m, tuple(int(v) for v in el["point"]), r + g, 1, -1)
        each.append(m.astype(bool))
        total |= m
    return total.astype(bool), each


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def in_labelled_region(spec: dict, x: float, y: float) -> bool:
    """Plans labelled only inside `roi` ([[x0, y0, x1, y1], ...], e.g. one plan of a multi-plan
    sheet): engine output outside it is neither credited nor counted as false."""
    rois = spec.get("roi")
    return not rois or any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in rois)


def _space_status(labels: np.ndarray, gt_pts: dict[str, list]) -> tuple[dict, dict]:
    from benchmark.real_plans import space_at
    eng = {sid: [space_at(labels, *p) for p in pts] for sid, pts in gt_pts.items()}
    owners: dict[int, set] = {}
    for sid, ids in eng.items():
        for e in ids:
            if e > 0:
                owners.setdefault(e, set()).add(sid)
    status = {}
    for sid, ids in eng.items():
        found = {e for e in ids if e > 0}
        missed = any(e <= 0 for e in ids)
        split = len(found) > 1
        merged = any(len(owners[e]) > 1 for e in found)
        status[sid] = ("missed" if missed and not found else "merged" if merged else "split" if split
                       else "partly_missed" if missed else "recovered")
    return status, {"engine_of": eng, "owners": owners}


def structural_holdout(s, spec: dict, openings: list[dict]) -> dict:
    labels = s.space_labels
    T = float(s.wall_thickness)
    gt_pts = _gt_points(spec)
    kinds = {sp["id"]: sp.get("kind", "room") for sp in spec.get("spaces", [])}
    status, aux = _space_status(labels, gt_pts)
    owners, engine_of = aux["owners"], aux["engine_of"]
    open_ids = _open_plan_ids(spec)
    zones = _zones(spec)
    printed = {sid for sid, zs in zones.items() if any(z.get("printed") for z in zs)}
    out: dict = {}

    # 1 / 3 / 7 / 8: physical spaces
    counts = {k: sum(v == k for v in status.values()) for k in ("recovered", "split", "merged", "missed", "partly_missed")}
    out["physical_spaces"] = {
        "gt": len(status), **counts,
        "recall": round(counts["recovered"] / max(1, len(status)), 3),
        "unnamed_gt": sum(1 for sid in status if sid not in printed),
        "unnamed_recovered": sum(1 for sid, st in status.items() if sid not in printed and st == "recovered"),
        "by_kind": {k: f"{sum(1 for sid in status if kinds[sid] == k and status[sid] == 'recovered')}/{sum(1 for sid in status if kinds[sid] == k)}"
                    for k in sorted(set(kinds.values()))},
        "status": status,
    }
    merged_engine = {e: sorted(o) for e, o in owners.items() if len(o) > 1}
    out["merges"] = {"engine_spaces_holding_several_gt_spaces": len(merged_engine),
                     "gt_spaces_merged": sum(len(v) for v in merged_engine.values()),
                     "groups": list(merged_engine.values())}

    # 4: room / zone point -> its physical space (an engine space that holds only this GT space)
    pts_ok = pts_all = 0
    for r in spec.get("_holdout_rooms", spec.get("rooms", [])):
        from benchmark.real_plans import space_at
        e = space_at(labels, *r["point"])
        pts_all += 1
        pts_ok += e > 0 and owners.get(e) == {r["space"]}
    out["point_association"] = {"points": pts_all, "in_own_physical_space": pts_ok,
                                "rate": round(pts_ok / max(1, pts_all), 3)}

    # 5 / 6 / 7: open-plan groups
    op = {sid: status[sid] for sid in open_ids if sid in status}
    out["open_plan"] = {"groups": len(op), "one_space": sum(v == "recovered" for v in op.values()),
                        "split": sum(v == "split" for v in op.values()),
                        "merged_with_other": sum(v == "merged" for v in op.values()),
                        "missed": sum(v in ("missed", "partly_missed") for v in op.values()), "status": op}

    # non-structural elements
    elements = spec.get("nonstructural", [])
    ns_any, ns_each = _ns_mask(labels.shape, elements, T)
    ns_near, ns_near_each = _ns_mask(labels.shape, elements, T, grow=max(2.0, 0.6 * T))
    wall = s.wall_mask > 0

    # 12: non-structural ink interpreted as wall
    from benchmark.real_plan_probes import _covered
    by_kind: dict[str, dict] = {}
    for el, m in zip(elements, ns_each):
        k = el.get("kind", "other")
        d = by_kind.setdefault(k, {"elements": 0, "point_on_wall": 0, "bbox_wall_share": []})
        d["elements"] += 1
        d["point_on_wall"] += bool(_covered(wall, *el["point"]))
        if el.get("bbox"):
            d["bbox_wall_share"].append(float(wall[m].mean()) if m.any() else 0.0)
    for d in by_kind.values():
        shares = d.pop("bbox_wall_share")
        d["mean_bbox_wall_share"] = round(float(np.mean(shares)), 3) if shares else None
    out["nonstructural_as_wall"] = {"elements": len(elements),
                                    "point_on_wall": sum(d["point_on_wall"] for d in by_kind.values()),
                                    "by_kind": by_kind}

    def kind_at(points) -> str | None:
        for x, y in points:
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= yi < labels.shape[0] and 0 <= xi < labels.shape[1]) or not ns_near[yi, xi]:
                continue
            for el, m in zip(elements, ns_near_each):
                if m[yi, xi]:
                    return el.get("kind", "other")
        return None

    # 10: false spaces (engine spaces with no GT point), furniture-induced or not
    gt_engine = {e for ids in engine_of.values() for e in ids if e > 0}
    false_spaces = []
    for e in np.unique(labels[labels > 0]):
        e = int(e)
        if e in gt_engine:
            continue
        region = labels == e
        ys, xs = np.nonzero(region)
        if not in_labelled_region(spec, float(xs.mean()), float(ys.mean())):
            continue
        area = int(region.sum())
        inside = float(ns_near[region].mean())
        ring = cv2.dilate(region.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~region
        ring_wall = ring & (s.sealed_mask > 0 if s.sealed_mask is not None and s.sealed_mask.shape == ring.shape else wall)
        bounded = float(ns_near[ring_wall].mean()) if ring_wall.any() else 0.0
        cause = kind_at([(float(xs.mean()), float(ys.mean()))]) if inside >= 0.5 else None
        if cause is None and bounded >= 0.4:
            hits = [(float((m & ring_wall).sum()), el.get("kind", "other")) for el, m in zip(elements, ns_near_each)]
            cause = max(hits)[1] if hits and max(hits)[0] > 0 else None
        false_spaces.append({"engine_space": e, "area": area, "inside_nonstructural": round(inside, 2),
                             "bounded_by_nonstructural": round(bounded, 2), "cause": cause})
    out["false_spaces"] = {"engine_spaces_without_gt_point": len(false_spaces),
                           "nonstructural_induced": sum(1 for f in false_spaces if f["cause"]),
                           "by_kind": _tally(f["cause"] for f in false_spaces if f["cause"]),
                           "spaces": false_spaces[:30]}

    # 11: false boundaries — sealed gaps ending on non-structural elements, and splits of one GT
    # space whose dividing boundary runs mainly through non-structural elements
    gaps = list(s.candidates) + ([] if getattr(s, "work", None) is not None else list(s.cracks))
    gap_hits = []
    for c in gaps:
        k = kind_at([c.start, c.end])
        if k:
            gap_hits.append(k)
    split_bounds = []
    for sid, st in status.items():
        found = sorted({e for e in engine_of[sid] if e > 0})
        if len(found) < 2:
            continue
        for i in range(len(found)):
            for j in range(i + 1, len(found)):
                band = _separation(labels, found[i], found[j], T)
                if band is None:
                    continue
                share = float(ns_near[band].mean())
                ys, xs = np.nonzero(band & ns_near) if share > 0 else ([], [])
                kind = kind_at([(float(xs[len(xs) // 2]), float(ys[len(ys) // 2]))]) if len(xs) else None
                split_bounds.append({"gt_space": sid, "open_plan": sid in open_ids, "share_nonstructural": round(share, 2),
                                     "nonstructural_induced": share >= 0.3, "kind": kind if share >= 0.3 else None})
    out["false_boundaries"] = {
        "sealed_gaps_on_nonstructural": len(gap_hits), "sealed_gaps_by_kind": _tally(gap_hits),
        "split_boundaries": len(split_bounds),
        "split_boundaries_nonstructural": sum(b["nonstructural_induced"] for b in split_bounds),
        "open_plan_splits_nonstructural": sum(b["nonstructural_induced"] and b["open_plan"] for b in split_bounds),
        "by_kind": _tally(b["kind"] for b in split_bounds if b["kind"]),
        "boundaries": split_bounds[:30],
    }

    # 13 / 14: openings
    gts = spec.get("openings") or []
    if gts:
        pairs = match(openings, gts)
        mp = {i for i, _, _ in pairs}
        causes = []
        for i, o in enumerate(openings):
            if i in mp or not in_labelled_region(spec, *o["center"]):
                continue
            causes.append(kind_at([o["start"], o["end"], o["center"]]) or "unattributed")
        mg = {j for _, j, _ in pairs}
        dw = [j for j, g in enumerate(gts) if g["kind"] in ("door", "window")]
        out["openings"] = {"gt_doors_windows": len(dw), "found_doors_windows": sum(j in mg for j in dw),
                           "gt_ambiguous": len(gts) - len(dw), "found_ambiguous": sum(j in mg for j in range(len(gts)) if j not in dw),
                           "false_candidates": len(causes), "false_by_cause": _tally(causes),
                           "missed": [gts[j]["id"] for j in range(len(gts)) if j not in mg]}
    return out


# ---------------------------------------------------------------------------
# API response
# ---------------------------------------------------------------------------

def api_holdout(response: dict, spec: dict) -> dict:
    from benchmark.real_plans import _inside, normalize_label
    gt_pts = _gt_points(spec)
    zones = _zones(spec)
    open_ids = _open_plan_ids(spec)
    printed = {sid for sid, zs in zones.items() if any(z.get("printed") for z in zs)}
    regions = []                       # (key, polygon, named, method)
    seen = set()
    from benchmark.physical_eval import physical_region
    for r in response["rooms"]:
        reg = physical_region(r)
        if not reg:
            continue
        key = reg[0][1]
        if key in seen:
            continue
        seen.add(key)
        regions.append((key, reg[1], True, "wall-region" if reg[2] else r["boundary"]["method"]))
    for u in response.get("unlabeled_spaces", []):
        if u.get("boundary") and u["id"] not in seen:
            seen.add(u["id"])
            regions.append((u["id"], u["boundary"]["polygon"], False, u["boundary"]["method"]))

    def region_of(x, y):
        hits = [reg for reg in regions if _inside(reg[1], x, y)]
        if not hits:
            return None
        structural = [reg for reg in hits if reg[3] in ("wall-region", "open-plan-shared", "merged-space", "passage-partition")]
        return (structural or hits)[0]

    where = {sid: [region_of(*p) for p in pts] for sid, pts in gt_pts.items()}
    owners: dict = {}
    for sid, regs in where.items():
        for reg in regs:
            if reg:
                owners.setdefault(reg[0], set()).add(sid)
    status = {}
    for sid, regs in where.items():
        keys = {reg[0] for reg in regs if reg}
        missing = any(reg is None for reg in regs)
        status[sid] = ("missed" if not keys else "merged" if any(len(owners[k]) > 1 for k in keys)
                       else "split" if len(keys) > 1 else "partly_missed" if missing else "recovered")
    counts = {k: sum(v == k for v in status.values()) for k in ("recovered", "split", "merged", "missed", "partly_missed")}
    unnamed = [sid for sid in status if sid not in printed]
    returned_unnamed = sum(1 for sid in unnamed if status[sid] == "recovered" and any(reg and not reg[2] for reg in where[sid]))

    # 9: functional zones of open-plan spaces
    api_rooms = [r for r in response["rooms"] if r.get("label_center")]
    diag = float(np.hypot(response["image"]["width"], response["image"]["height"]))
    R = max(60.0, 0.08 * diag)
    zone_rows = []
    for sid in sorted(open_ids):
        for z in zones.get(sid, []):
            if not z.get("printed"):
                continue
            cands = [r for r in api_rooms if normalize_label(r.get("label_text")) == normalize_label(z["printed"])
                     and np.hypot(r["label_center"]["x"] - z["point"][0], r["label_center"]["y"] - z["point"][1]) <= R]
            r = min(cands, key=lambda r: np.hypot(r["label_center"]["x"] - z["point"][0], r["label_center"]["y"] - z["point"][1])) if cands else None
            recognized = r is not None
            space_poly = (r.get("physical_space") or {}).get("polygon") or (r.get("boundary") or {}).get("polygon") if r else None
            covers = bool(space_poly) and all(_inside(space_poly, *p) for p in gt_pts[sid])
            shares = bool(r and (r.get("shares_space_with") or (r.get("physical_space") or {}).get("kind") == "open-plan"))
            zone_rows.append({"space": sid, "printed": z["printed"], "recognized": recognized,
                              "boundary_is_shared_space": covers, "reported_as_sharing": shares,
                              "identified": recognized and covers})
    methods: dict[str, int] = {}
    for r in response["rooms"]:
        m = r["boundary"]["method"] if r.get("boundary") else "none"
        methods[m] = methods.get(m, 0) + 1
    return {
        "physical_spaces": {"gt": len(status), **counts, "recall": round(counts["recovered"] / max(1, len(status)), 3),
                            "status": status},
        "api_counts": {"physical_spaces": len(regions), "named_regions": sum(1 for r in regions if r[2]),
                       "unnamed_spaces": sum(1 for r in regions if not r[2]), "named_rooms": len(response["rooms"])},
        "unnamed": {"gt_unnamed_spaces": len(unnamed), "recovered": sum(status[s] == "recovered" for s in unnamed),
                    "returned_as_unnamed_space": returned_unnamed},
        "open_plan": {"groups": sum(1 for s in open_ids if s in status),
                      "one_space": sum(status[s] == "recovered" for s in open_ids if s in status),
                      "split": sum(status[s] == "split" for s in open_ids if s in status),
                      "merged_with_other": sum(status[s] == "merged" for s in open_ids if s in status)},
        "zones": {"printed_zones": len(zone_rows), "recognized": sum(z["recognized"] for z in zone_rows),
                  "identified": sum(z["identified"] for z in zone_rows),
                  "reported_as_sharing": sum(z["reported_as_sharing"] for z in zone_rows), "rows": zone_rows},
        "boundaries": {"methods": methods,
                       "unsupported": methods.get("wall-ray-estimate", 0) + methods.get("dimension-box-estimate", 0)
                       + methods.get("dimension-only-estimate", 0),
                       "dimension_snapped": methods.get("dimension-box-wall-snapped", 0)},
    }


def _separation(labels: np.ndarray, a: int, b: int, T: float) -> np.ndarray | None:
    """Pixels on the shortest separation between engine spaces a and b (whatever its
    thickness: a wall, a sealed gap, or a thick piece of furniture). None if they are far apart."""
    da = cv2.distanceTransform((labels != a).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((labels != b).astype(np.uint8), cv2.DIST_L2, 3)
    total = da + db
    outside = (labels != a) & (labels != b)
    if not outside.any():
        return None
    gap = float(total[outside].min())
    if gap > 12 * max(T, 3.0):
        return None
    band = outside & (total <= gap + 2.0)
    return band if band.any() else None


def _tally(items) -> dict:
    out: dict = {}
    for k in items:
        out[k] = out.get(k, 0) + 1
    return out


def summarize_holdout(results: dict) -> dict:
    st = [r["holdout"] for r in results.values() if r.get("holdout")]
    api = [r["full"]["holdout"] for r in results.values() if r.get("full", {}).get("holdout")]
    if not st:
        return {}
    add = lambda rows, path: sum(_get(r, path) or 0 for r in rows)
    out = {"structure": {
        "physical_spaces": {k: add(st, f"physical_spaces.{k}") for k in ("gt", "recovered", "split", "merged", "missed", "partly_missed", "unnamed_gt", "unnamed_recovered")},
        "point_association": {k: add(st, f"point_association.{k}") for k in ("points", "in_own_physical_space")},
        "open_plan": {k: add(st, f"open_plan.{k}") for k in ("groups", "one_space", "split", "merged_with_other", "missed")},
        "merges": {k: add(st, f"merges.{k}") for k in ("engine_spaces_holding_several_gt_spaces", "gt_spaces_merged")},
        "false_spaces": {k: add(st, f"false_spaces.{k}") for k in ("engine_spaces_without_gt_point", "nonstructural_induced")},
        "false_boundaries": {k: add(st, f"false_boundaries.{k}") for k in ("sealed_gaps_on_nonstructural", "split_boundaries", "split_boundaries_nonstructural", "open_plan_splits_nonstructural")},
        "nonstructural_as_wall": {k: add(st, f"nonstructural_as_wall.{k}") for k in ("elements", "point_on_wall")},
        "openings": {k: add(st, f"openings.{k}") for k in ("gt_doors_windows", "found_doors_windows", "gt_ambiguous", "found_ambiguous", "false_candidates")},
        "false_openings_by_cause": _merge_tallies(r.get("openings", {}).get("false_by_cause", {}) for r in st),
    }}
    ps = out["structure"]["physical_spaces"]
    ps["recall"] = round(ps["recovered"] / max(1, ps["gt"]), 3)
    if api:
        out["api"] = {
            "physical_spaces": {k: add(api, f"physical_spaces.{k}") for k in ("gt", "recovered", "split", "merged", "missed", "partly_missed")},
            "api_counts": {k: add(api, f"api_counts.{k}") for k in ("physical_spaces", "named_regions", "unnamed_spaces", "named_rooms")},
            "unnamed": {k: add(api, f"unnamed.{k}") for k in ("gt_unnamed_spaces", "recovered", "returned_as_unnamed_space")},
            "open_plan": {k: add(api, f"open_plan.{k}") for k in ("groups", "one_space", "split", "merged_with_other")},
            "zones": {k: add(api, f"zones.{k}") for k in ("printed_zones", "recognized", "identified", "reported_as_sharing")},
            "boundaries": {k: add(api, f"boundaries.{k}") for k in ("unsupported", "dimension_snapped")},
        }
        a = out["api"]["physical_spaces"]
        a["recall"] = round(a["recovered"] / max(1, a["gt"]), 3)
    return out


def _get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _merge_tallies(rows) -> dict:
    out: dict = {}
    for row in rows:
        for k, v in row.items():
            out[k] = out.get(k, 0) + v
    return out


__all__ = ["normalize_spec", "legacy_rooms", "structural_holdout", "api_holdout", "summarize_holdout"]
