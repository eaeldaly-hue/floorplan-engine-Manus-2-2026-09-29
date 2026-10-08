"""Opening symbols read where they are legible.

On an under-resolved plan (walls of a few pixels) a whole door is ~15 px wide: swing arcs, leaves,
glazing lines and wall faces lie within a pixel or two of each other, so the symbol tests see
'ink near the arc' everywhere and windows read as doors. The same plan read at a normalised scale
(engine.reconstruction, walls ~12 px) separates them again.

The structure keeps the reading it was chosen for (its candidates and spaces), but each opening's
*type* comes from the normalised reading when that reading has the same opening (same wall line,
overlapping span): one primitive, two representations, the legible one decides the symbol.
"""

from __future__ import annotations

import numpy as np


def _overlap(a: dict, b: dict, tol: float) -> float:
    pa0, pa1 = np.array(a["start"], float), np.array(a["end"], float)
    pb0, pb1 = np.array(b["start"], float), np.array(b["end"], float)
    la, lb = np.linalg.norm(pa1 - pa0), np.linalg.norm(pb1 - pb0)
    if la < 1e-6 or lb < 1e-6:
        return 0.0
    da, db = (pa1 - pa0) / la, (pb1 - pb0) / lb
    if abs(float(np.dot(da, db))) < 0.9:
        return 0.0
    n = np.array([-da[1], da[0]])
    if abs(float(np.dot((pb0 + pb1) / 2 - (pa0 + pa1) / 2, n))) > tol:
        return 0.0
    lo, hi = sorted((float(np.dot(pb0 - pa0, da)), float(np.dot(pb1 - pa0, da))))
    inter = max(0.0, min(la, hi) - max(0.0, lo))
    union = max(la, hi) - min(0.0, lo)
    return inter / max(union, 1e-6)


def transfer_types(openings: list[dict], legible: list[dict], scale: float, wall_t: float) -> int:
    """Re-type `openings` (original frame) from `legible` (classified on the image scaled by
    `scale`). Returns the number of openings whose type changed."""
    mapped = []
    for o in legible:
        mapped.append({**o, "start": [v / scale for v in o["start"]], "end": [v / scale for v in o["end"]]})
    tol = max(3.0, float(wall_t))
    changed = 0
    for o in openings:
        best, ov = None, 0.0
        for b in mapped:
            v = _overlap(o, b, tol)
            if v > ov:
                best, ov = b, v
        if best is None or ov < 0.5:
            continue
        ev = o["evidence"]
        ev["symbol_read_at_scale"] = round(scale, 2)
        ev["symbol_at_native_scale"] = {"type": o["type"], "symbol": ev.get("symbol")}
        if best["type"] != o["type"]:
            changed += 1
        o["type"] = best["type"]
        o["type_label"] = best.get("type_label", o.get("type_label"))
        o["confidence"] = best["confidence"]
        for key in ("symbol", "glazing_lines", "dashed_glazing", "arc_score", "double_arc_score", "leaf_score",
                    "door_swing_score", "sliding_panels", "reason", "hinge", "swing_side", "leaf_angle"):
            if key in best["evidence"]:
                ev[key] = best["evidence"][key]
        da = np.subtract(o["end"], o["start"]).astype(float)
        db = np.subtract(best["end"], best["start"]).astype(float)
        if float(np.dot(da, db)) < 0:                  # the legible candidate runs the other way
            if ev.get("hinge") in ("start", "end"):
                ev["hinge"] = "end" if ev["hinge"] == "start" else "start"
            if isinstance(ev.get("swing_side"), int):
                ev["swing_side"] = -ev["swing_side"]
    return changed


def renumber(openings: list[dict]) -> None:
    """Ids follow the type (D01, W01, O01), as engine.opening_detection assigns them."""
    openings.sort(key=lambda item: (item["type"], item["center"][1], item["center"][0]))
    counters = {"window": 0, "door": 0, "opening": 0}
    for item in openings:
        counters[item["type"]] = counters.get(item["type"], 0) + 1
        item["id"] = f"{item['type'][0].upper()}{counters[item['type']]:02d}"
