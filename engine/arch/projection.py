"""One reconstruction, several views.

The structured plan (engine.arch.building, with the typed objects of engine.arch.objects and the
functional zones of engine.arch.zones) is the canonical reconstruction. The flat API records the
Workbench's Rooms / Cleaned / Elements views read (`rooms`, `unlabeled_spaces`) predate it; this
module projects the structured plan onto them instead of inferring anything again:

  result["zones"]          every functional zone, with the structural space that holds it
  result["objects"]        every typed object (furniture, fixture, appliance), with its space / zone
  unlabeled space record   + "function" (a space characterised by its contents) and "zone_ids"
  room record              + "zone_id" when its label lies in a zone; an open-plan room (a label
                           sharing a structural space) takes that zone's extent as its boundary,
                           so a labelled zone has one geometry in every view

Three kinds of region stay distinct: a structural space (walls and openings), a room (a named
space, or a named part of a shared space) and a zone (an inferred functional part of a space,
bounded by implied edges, never by walls).
"""

from __future__ import annotations

import cv2
import numpy as np

ZONE_METHODS = ("open-plan-zone", "merged-space-zone")


def _inside(polygon, x: float, y: float) -> bool:
    if not polygon:
        return False
    pts = np.asarray([[p["x"], p["y"]] if isinstance(p, dict) else p for p in polygon], np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0


def _bbox(polygon) -> dict:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return {"x": int(min(xs)), "y": int(min(ys)), "width": int(max(xs) - min(xs)), "height": int(max(ys) - min(ys))}


def project(result: dict) -> None:
    """Add the structured plan's zones and objects to the flat API records (in place)."""
    from .zones import label_function

    b = result.get("building")
    if not b:
        return
    spaces = {s["id"]: s for s in b.get("spaces", [])}
    zones = [z | {"space_id": s["id"]} for s in b.get("spaces", []) for z in s.get("zones", [])]
    zone_of = {z["id"]: z for z in zones}

    for o in b.get("objects", []):
        cx, cy = o.get("center") or ((o["bbox"][0] + o["bbox"][2]) / 2, (o["bbox"][1] + o["bbox"][3]) / 2)
        o["space_id"] = next((s["id"] for s in spaces.values() if _inside(s.get("polygon"), cx, cy)), None)
        o["zone_id"] = next((z["id"] for z in zones if z["space_id"] == o["space_id"] and _inside(z["polygon"], cx, cy)), None)

    for z in zones:
        z["room_ids"] = []
    scale = result.get("pixel_scale") or {}
    for room in result.get("rooms", []):
        lc = room.get("label_center") or {}
        sp = spaces.get(room.get("space_id"))
        if sp is None or lc.get("x") is None:
            continue
        zone = next((z for z in zones if z["space_id"] == sp["id"] and _inside(z["polygon"], lc["x"], lc["y"])), None)
        if zone is None:
            continue
        room["zone_id"] = zone["id"]
        zone["room_ids"].append(room["id"])
        boundary = room.get("boundary") or {}
        if boundary.get("method") in ZONE_METHODS and label_function(room.get("name")) == zone["function"]:
            poly = [(int(p[0]), int(p[1])) for p in zone["polygon"]]
            room["boundary"] = boundary | {"polygon": [{"x": x, "y": y} for x, y in poly], "bbox": _bbox(poly),
                                           "zone_id": zone["id"]}
            area = room.get("area") or {}
            if area.get("source") == "zone-estimate":
                pixels = float(cv2.contourArea(np.asarray(poly, np.float32)))
                ppu = scale.get("pixels_per_unit")
                room["area"] = ({"value": round(pixels / ppu ** 2, 2), "unit": area.get("unit"), "source": "zone-estimate"}
                                if ppu else {"value": int(pixels), "unit": "px²", "source": "zone-estimate"})

    for u in result.get("unlabeled_spaces", []):
        sp = spaces.get(u["id"])
        if sp is None:
            continue
        if sp.get("function"):
            u["function"] = sp["function"]
        if sp.get("zones"):
            u["zone_ids"] = [z["id"] for z in sp["zones"]]
            u["circulation_m2"] = sp.get("circulation_m2")

    zs = b.get("zones_summary") or {}
    if zs.get("scale_px_per_cm"):
        # the structured plan's own scale (printed dimensions, or the plan's doors): reported
        # beside the legacy pixel_scale, which only printed room dimensions calibrate
        result["scale_estimate"] = {"px_per_cm": round(float(zs["scale_px_per_cm"]), 3), "source": zs.get("scale_source")}
    result["zones"] = [zone_of[z["id"]] for z in zones]
    result["zone_count"] = len(zones)
    result["objects"] = b.get("objects", [])
    result["object_count"] = len(result["objects"])
