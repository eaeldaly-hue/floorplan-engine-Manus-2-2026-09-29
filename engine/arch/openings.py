"""Openings from typed architectural elements, with topology against the final spaces.

When the Architectural Cleaner ran, every door and window it recognised (swing doors, glazing,
sliders, gaps holding a door or window symbol) is a typed opening with its jambs. The recognition
input seals them, so the legacy gap finder sees almost none of them on the cleaned plan; the
openings reported to the API therefore come from here, not from re-detection.

Topology: each opening connects the spaces on its two sides (sampled perpendicular to the opening,
beyond half the wall thickness, in the final space labels): between two rooms, room to exterior,
or one space on both sides. The architectural role follows from topology, not from the drawing
pattern alone: glazing on an exterior wall is a window; glazing-style panels between two rooms
are a sliding / bypass door; glass with one space on both sides is a partition (unclassified).
Gaps narrower than half the plan's door width (measured from its swing doors) are wall joints.
"""

from __future__ import annotations

import math

import numpy as np

KIND = {"door": "door", "gap-door": "door", "glazing": "window", "gap-window": "window"}
TYPE_LABELS = {"door": "Door", "window": "Window", "opening": "Unclassified opening"}
EXTERIOR = -1


def _side(labels, p, n, sign, reach) -> int:
    h, w = labels.shape
    votes: dict = {}
    for f in (1.0, 1.6, 2.4):
        x = int(round(p[0] + sign * n[0] * reach * f))
        y = int(round(p[1] + sign * n[1] * reach * f))
        v = int(labels[y, x]) if 0 <= x < w and 0 <= y < h else EXTERIOR
        votes[v] = votes.get(v, 0) + 1
    best = max(votes, key=votes.get)
    return EXTERIOR if best < 0 else best


def _relation(a: int, b: int) -> str:
    if a > 0 and b > 0:
        return "between_rooms" if a != b else "same_space"
    if (a > 0 and b in (EXTERIOR, 0)) or (b > 0 and a in (EXTERIOR, 0)):
        return "room_to_exterior"
    return "unknown"


def _dedupe(closures, t) -> list:
    """One record per opening: a door wins over glazing drawn in the same gap."""
    order = sorted(closures, key=lambda c: 0 if KIND.get(c["kind"]) == "door" else 1)
    kept = []
    for c in order:
        m = ((c["p0"][0] + c["p1"][0]) / 2, (c["p0"][1] + c["p1"][1]) / 2)
        if any(math.dist(m, ((k["p0"][0] + k["p1"][0]) / 2, (k["p0"][1] + k["p1"][1]) / 2)) <= max(6.0, 0.75 * t)
               for k in kept):
            continue
        kept.append(c)
    return kept


def typed_openings(cleaned, structure, pixel_scale=None) -> list[dict]:
    """API opening records (same schema as engine.opening_detection.classify_openings) for the
    cleaner's typed openings, related to the final spaces of `structure`."""
    labels = structure.space_labels
    t = float(cleaned.wall_thickness or structure.wall_thickness or 10.0)
    swings = [math.dist(c["p0"], c["p1"]) for c in cleaned.openings if c.get("kind") == "door"]
    door_w = float(np.median(swings)) if len(swings) >= 3 else None     # the plan's own door width
    out = []
    for c in _dedupe([c for c in cleaned.openings if c.get("kind") in KIND], t):
        kind = KIND[c["kind"]]
        if door_w and math.dist(c["p0"], c["p1"]) < 0.5 * door_w:
            continue                          # a wall joint (corner / junction gap), not an opening
        p0 = np.asarray(c["p0"], float)
        p1 = np.asarray(c["p1"], float)
        width = float(np.linalg.norm(p1 - p0))
        if width < 2:
            continue
        u = (p1 - p0) / width
        n = np.array([-u[1], u[0]])
        mid = (p0 + p1) / 2
        reach = 0.5 * t + 4.0
        sides = []
        for sign in (-1, 1):
            votes: dict = {}
            for f in (0.25, 0.5, 0.75):
                v = _side(labels, p0 + u * width * f, n, sign, reach)
                votes[v] = votes.get(v, 0) + 1
            sides.append(max(votes, key=votes.get))
        relation = _relation(*sides)
        ids = [structure.space_id(s) for s in sides if s and s > 0]
        role = ""
        if kind == "window" and relation == "between_rooms":
            kind, role = "door", "sliding / bypass door (glazing-style panels between two rooms)"
        elif kind == "window" and relation == "same_space":
            kind, role = "opening", "glass partition inside one space (e.g. shower screen)"
        orientation = "horizontal" if abs(u[0]) >= abs(u[1]) else "vertical"
        out.append({
            "id": "",
            "type": kind,
            "type_label": TYPE_LABELS[kind],
            "confidence": 0.8 if c["kind"] in ("door", "glazing") else 0.65,
            "geometry_confidence": 0.9,
            "start": [int(round(p0[0])), int(round(p0[1]))],
            "end": [int(round(p1[0])), int(round(p1[1]))],
            "center": [int(round(mid[0])), int(round(mid[1]))],
            "orientation": orientation,
            "width_pixels": int(round(width)),
            "width_display": (f"{width / pixel_scale['pixels_per_unit']:.1f} {pixel_scale['unit']}"
                              if pixel_scale and pixel_scale.get("pixels_per_unit") else None),
            "evidence": {
                "symbol": c.get("evidence", ""),
                "room_relation": relation,
                "adjacent_space_ids": [i for i in ids if i],
                "exterior_side": EXTERIOR in sides,
                "wall_thickness_px": round(t, 1),
                "width_in_wall_thicknesses": round(width / max(t, 1.0), 2),
                "arc_score": 1.0 if c["kind"] == "door" else 0.0,
                "glazing_lines": 2 if c["kind"] == "glazing" else 0,
                "detector": f"typed elements ({cleaned.source})",
                "typed_kind": c["kind"],
                "role": role,
                "reason": f"{kind} recognised by the Architectural Cleaner: {c.get('evidence', '')}"
                          + (f"; {role}" if role else ""),
            },
        })
    out.sort(key=lambda item: (item["type"], item["center"][1], item["center"][0]))
    counters = {"window": 0, "door": 0, "opening": 0}
    for item in out:
        counters[item["type"]] += 1
        item["id"] = f"{item['type'][0].upper()}{counters[item['type']]:02d}"
    return out
