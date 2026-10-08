"""Door / window classification from structural candidates plus drawn-symbol evidence.

Candidates (wall gaps and line-bridged openings) come from engine.structure. Each one
is analysed in its own wall-aligned frame — u along the wall, v across it, so
diagonal openings are handled the same way — and several independent signals are
measured:

  band lines   lines drawn inside the wall band: wall-face lines, interior glazing
               lines (full or dashed), partial offset panels (sliding doors)
  swing arc    a quarter circle of radius ≈ width centred on either jamb, on either
               side of the wall; paired half-width arcs for double doors
  leaf         a straight door leaf from the hinge at 30–90° to the wall
  topology     room↔room, room↔exterior or unknown (from the sealed space map)
  width        relative to the wall thickness, and in real units when a scale is known

No single signal decides the type. In particular "touches the exterior" is never
enough to call something a window, and a door does not need a swing arc. When the
drawing does not support a decision the opening is returned as UNKNOWN (type
"opening") with the evidence that was found.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from engine.structure import EXTERIOR, OpeningCandidate, StructureResult, analyze_structure, thin_lines

TYPE_LABELS = {
    "door": "Door",
    "window": "Window",
    "opening": "Unknown opening",
}

# Real-world width bands (feet) used only when the plan scale is known.
DOOR_MAX_FT = 4.6          # single/double door leaf width upper bound
GARAGE_MIN_FT = 8.0        # overhead/garage doors


# ---------------------------------------------------------------------------
# Sampling in the opening's local frame
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    origin: np.ndarray   # opening start on the wall axis
    d: np.ndarray        # along the wall (start → end)
    n: np.ndarray        # across the wall
    w: float             # opening width
    t: float             # wall thickness

    @classmethod
    def of(cls, c: OpeningCandidate) -> "Frame":
        return cls(np.asarray(c.start, float), c.tangent, c.normal, c.width, max(3.0, c.thickness))

    def points(self, u, v) -> tuple[np.ndarray, np.ndarray]:
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        return self.origin[0] + self.d[0] * u + self.n[0] * v, self.origin[1] + self.d[1] * u + self.n[1] * v


def _sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, fill=0) -> np.ndarray:
    h, w = img.shape[:2]
    xi, yi = np.floor(np.asarray(xs) + 0.5).astype(int), np.floor(np.asarray(ys) + 0.5).astype(int)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.full(xi.shape, fill, dtype=img.dtype)
    out[inside] = img[yi[inside], xi[inside]]
    return out


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class BandLine:
    v: float               # offset across the wall (0 = wall axis)
    coverage: float        # fraction of the opening width that is inked
    first: float           # first / last inked position as a fraction of the width
    last: float
    interior: bool         # inside the band (not on a wall face)


def _band_matrix(thin: np.ndarray, f: Frame):
    """Ink in the wall band: rows = offsets across the wall (v), cols = positions along it (u).
    Each cell ORs samples at ±0.5 px across the wall, which tolerates the half-pixel wobble of
    rasterised slanted lines without merging neighbouring parallel lines."""
    u = np.linspace(0.06 * f.w, 0.94 * f.w, max(8, int(f.w)))
    v = np.arange(-0.5 * f.t - 2.0, 0.5 * f.t + 2.5, 1.0)
    uu, vv = np.meshgrid(u, v)
    ink = np.zeros(uu.shape, bool)
    slanted = min(abs(f.d[0]), abs(f.d[1])) > 0.2
    for dv in ((-0.5, 0.0, 0.5) if slanted else (0.0,)):
        xs, ys = f.points(uu, vv + dv)
        ink |= _sample(thin, xs, ys) > 0
    return ink, u, v


def band_lines(thin: np.ndarray, f: Frame) -> list[BandLine]:
    """Lines parallel to the wall drawn inside the opening's wall band.

    Inked rows are grouped into runs; a run is one line (centred on its weighted middle),
    several adjacent lines (split where coverage dips), or a filled bar (ignored).
    """
    if f.w < 4:
        return []
    ink, u, v = _band_matrix(thin, f)
    coverage = ink.mean(axis=1)
    max_rows = max(3, int(round(0.3 * f.t)))
    face_margin = max(1.5, 0.18 * f.t)
    segments: list[tuple[int, int]] = []
    k = 0
    while k < len(v):
        if coverage[k] < 0.25:
            k += 1
            continue
        j = k
        while j + 1 < len(v) and coverage[j + 1] >= 0.25:
            j += 1
        if j - k + 1 <= max_rows:
            segments.append((k, j))
        elif j - k + 1 <= max(5, int(round(0.55 * f.t))) and coverage[k:j + 1].min() >= 0.6 * coverage[k:j + 1].max():
            # two parallel lines too close to separate at this resolution (e.g. double glazing)
            mid = (k + j) // 2
            segments.extend([(k, mid), (mid + 1, j)])
        else:
            # split at dips; a long run without dips is a filled bar
            cut = [k]
            for r in range(k + 1, j):
                if coverage[r] < 0.6 * max(coverage[cut[-1]:r + 1].max(), coverage[r + 1:j + 1].max()):
                    if r - cut[-1] >= 1:
                        segments.append((cut[-1], r - 1))
                    cut.append(r + 1)
            if len(cut) > 1:
                segments.append((cut[-1], j))
        k = j + 1
    lines: list[BandLine] = []
    for a, b in segments:
        if b - a + 1 > max(max_rows + 1, 3):
            continue
        weights = coverage[a:b + 1]
        centre = float((v[a:b + 1] * weights).sum() / max(1e-9, weights.sum()))
        group = ink[a:b + 1].any(axis=0)
        cols = np.flatnonzero(group)
        lines.append(BandLine(
            v=centre,
            coverage=float(group.mean()),
            first=float(u[cols[0]] / f.w) if cols.size else 1.0,
            last=float(u[cols[-1]] / f.w) if cols.size else 0.0,
            interior=abs(centre) < 0.5 * f.t - face_margin,
        ))
    return lines


def sliding_shift(thin: np.ndarray, f: Frame) -> float:
    """Sliding-door signature: inside the wall band the ink sits on one side of the axis near
    one jamb and on the other side near the other jamb (two offset panels). Returns 0–1."""
    if f.w < 6:
        return 0.0
    ink, u, v = _band_matrix(thin, f)
    inner = np.abs(v) <= 0.5 * f.t + 1.0
    ink = ink[inner]
    vv = v[inner][:, None]
    counts = ink.sum(axis=0)
    if (counts > 0).mean() < 0.7:
        return 0.0
    centroid = np.where(counts > 0, (ink * vv).sum(axis=0) / np.maximum(counts, 1), 0.0)
    a = centroid[(u >= 0.08 * f.w) & (u <= 0.36 * f.w)]
    b = centroid[(u >= 0.64 * f.w) & (u <= 0.92 * f.w)]
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.size < 3 or b.size < 3:
        return 0.0
    ma, mb = float(np.median(a)), float(np.median(b))
    # The ink moves across the wall from one end to the other (panel A near one jamb, panel B
    # near the other). Judge the difference between the ends, not signs about the axis: a single
    # face line or a half-pixel axis offset biases both ends equally.
    shift = abs(ma - mb)
    tolerance = max(1.0, 0.15 * f.t)
    consistent = min((np.abs(a - ma) <= tolerance).mean(), (np.abs(b - mb) <= tolerance).mean())
    return float(min(1.0, shift / max(1.5, 0.25 * f.t)) * consistent)


def _near_ink(dist: np.ndarray, xs, ys, tol: float) -> np.ndarray:
    return _sample(dist, xs, ys, fill=np.float32(1e6)) <= tol


def _curve_support(near: np.ndarray) -> float:
    """Drawn fraction of a curve, discounted unless the support is continuous
    (scattered noise touches many sample points but never a long run of them)."""
    if near.size == 0:
        return 0.0
    longest, run = 0, 0
    for value in near:
        run = run + 1 if value else 0
        longest = max(longest, run)
    return float(near.mean()) * min(1.0, longest / (0.4 * near.size))


def _tol(r: float, line_width: float) -> float:
    return float(min(4.0, max(1.5, line_width + 0.5, 0.02 * r)))


def arc_scores(dist: np.ndarray, f: Frame, line_width: float) -> dict[str, Any]:
    """Best single-leaf swing arc and double-door arc pair (fraction of the arc that is drawn)."""
    best = {"arc": 0.0, "arc_side": 0, "arc_hinge": None, "double_arc": 0.0, "double_side": 0}
    phis = np.radians(np.linspace(10, 80, 26))
    for side in (-1, 1):
        for hinge_v in (side * 0.5 * f.t, 0.0):
            for h, e in ((0.0, 1.0), (f.w, -1.0)):
                for scale in (1.0, 0.94, 1.06):
                    r = f.w * scale
                    xs, ys = f.points(h + e * r * np.cos(phis), hinge_v + side * r * np.sin(phis))
                    score = _curve_support(_near_ink(dist, xs, ys, _tol(r, line_width)))
                    if score > best["arc"]:
                        best.update(arc=score, arc_side=side, arc_hinge="start" if h == 0 else "end")
            # double door: two half-width arcs from both jambs
            r = f.w / 2
            pair = []
            for h, e in ((0.0, 1.0), (f.w, -1.0)):
                xs, ys = f.points(h + e * r * np.cos(phis), hinge_v + side * r * np.sin(phis))
                pair.append(_curve_support(_near_ink(dist, xs, ys, _tol(r, line_width))))
            if min(pair) > best["double_arc"]:
                best.update(double_arc=min(pair), double_side=side)
    return best


# 30 deg leaves are kept: some drawings show doors barely open (measured: dropping 30 deg cut opening
# classification on the canonical test cases from 0.88 to 0.79)
LEAF_ANGLES = (90, 80, 70, 60, 45, 30)


def leaf_score(dist: np.ndarray, f: Frame, line_width: float) -> tuple[float, float]:
    """Straight door leaf of length ≈ width from a hinge, opened 30–90°. Returns (score, angle)."""
    best, angle = 0.0, 0.0
    rho = np.linspace(0.15, 0.95, 20)
    for side in (-1, 1):
        for h, e in ((0.0, 1.0), (f.w, -1.0)):
            for theta in LEAF_ANGLES:
                th = math.radians(theta)
                r = f.w * rho
                xs, ys = f.points(h + e * r * math.cos(th), side * (0.5 * f.t + r * math.sin(th)))
                score = _curve_support(_near_ink(dist, xs, ys, max(1.5, line_width + 0.5)))
                if score > best:
                    best, angle = score, theta
    return best, angle


def wall_leaf_score(wall_mask: np.ndarray, ink: np.ndarray, f: Frame) -> float:
    """Door leaf drawn as a heavy stroke that the thickness filter kept as wall.

    Accepted only as a free-standing stub from a jamb: on the wall mask along ≈ the full
    leaf length, clearly thinner than the opening's wall, and with nothing drawn beyond
    its tip (a real wall continues to another wall; a leaf ends in the room)."""
    if f.w < 8:
        return 0.0
    best = 0.0
    for side in (-1, 1):
        for h, e in ((0.0, 1.0), (f.w, -1.0)):
            for theta in (90, 80, 70, 60):
                th = math.radians(theta)
                du, dv = e * math.cos(th), side * math.sin(th)

                def at(r, off=0.0):
                    r = np.asarray(r, float)
                    return f.points(h + du * r - dv * off, side * 0.5 * f.t + dv * r + du * off)

                on = _sample(wall_mask, *at(np.linspace(0.15, 0.9, 16) * f.w)) > 0
                if on.mean() < 0.85 or best >= on.mean():
                    continue
                if (_sample(wall_mask, *at(np.linspace(0.85, 1.0, 6) * f.w)) > 0).mean() < 0.5:
                    continue                                   # stub much shorter than the opening
                if (_sample(ink, *at(np.linspace(1.12, 1.35, 8) * f.w)) > 0).mean() > 0.1:
                    continue                                   # continues: a wall, not a leaf
                offs = np.arange(-f.t, f.t + 0.5, 1.0)
                across = _sample(wall_mask, *at(np.full(offs.shape, 0.5 * f.w), offs)) > 0
                if across.sum() > 0.6 * f.t:
                    continue                                   # as thick as a wall
                best = float(on.mean())
    return best


def sliding_pattern(lines: list[BandLine]) -> bool:
    """Two offset panels inside the band, each covering part of the width and overlapping mid-way."""
    partial = [l for l in lines if l.interior and 0.25 <= l.coverage <= 0.8]
    for a in partial:
        for b in partial:
            if abs(a.v - b.v) < 1.5:
                continue
            if a.first <= 0.2 and 0.3 <= a.last <= 0.8 and b.last >= 0.8 and 0.2 <= b.first <= 0.7 and b.first < a.last:
                return True
    return False


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _width_ft(width_px: float, pixel_scale: dict | None) -> float | None:
    if not pixel_scale or not pixel_scale.get("pixels_per_unit"):
        return None
    value = width_px / float(pixel_scale["pixels_per_unit"])
    return value * 3.28084 if pixel_scale.get("unit") == "m" else value


def classify_candidate(c: OpeningCandidate, s: StructureResult, thin: np.ndarray, dist: np.ndarray,
                       pixel_scale: dict | None, raw_thin: np.ndarray | None = None) -> dict[str, Any]:
    f = Frame.of(c)
    lines = band_lines(thin, f)
    arcs = arc_scores(dist, f, s.line_width)
    leaf, leaf_angle = leaf_score(dist, f, s.line_width)
    heavy_leaf = False
    if max(leaf, arcs["arc"], arcs["double_arc"]) < 0.55 and c.source == "wall_gap":
        wl = wall_leaf_score(s.wall_mask, s.symbol_ink, f)
        if wl > leaf:
            leaf, heavy_leaf = wl, True
    shift = sliding_shift(raw_thin if raw_thin is not None else thin, f)
    sliding = sliding_pattern(lines) or shift >= 0.6

    interior_full = [l for l in lines if l.interior and l.coverage >= 0.7]
    interior_spread = [l for l in lines if l.interior and l.coverage >= 0.35 and l.first <= 0.2 and l.last >= 0.8]
    interior_any = [l for l in lines if l.interior and l.coverage >= 0.3]
    faces = [l for l in lines if not l.interior and l.coverage >= 0.6]
    any_line = max((l.coverage for l in lines), default=0.0)

    width_ft = _width_ft(c.width, pixel_scale)
    t_ref = max(s.wall_thickness, 1.0)
    relation = c.relation
    exterior = relation == "room_to_exterior"
    interior = relation == "between_rooms"

    door_symbol = max(arcs["arc"], arcs["double_arc"] * 1.05, leaf * 0.95)
    glazing = 0.0
    if interior_full:
        glazing = 0.92 if (faces or len(interior_full) >= 2) else 0.8
    elif interior_spread:
        glazing = 0.62

    kind, confidence, symbol, reason = "opening", 0.4, "none", ""
    single_line = c.source == "single_line"
    if single_line:
        # The band was placed on the only drawn line, so that line says nothing about glazing.
        glazing, sliding, interior_full, interior_spread, interior_any = 0.0, False, [], [], []
        faces = [l for l in lines if l.coverage >= 0.6][:1]
    if sliding:
        kind, confidence, symbol = "door", 0.85, "sliding panels"
        reason = "Two offset panels inside the wall band (sliding door)."
    elif door_symbol >= 0.55 and door_symbol >= glazing - 0.1:
        kind = "door"
        if arcs["double_arc"] * 1.05 >= max(arcs["arc"], leaf * 0.95):
            symbol = "double swing"
            reason = f"Two half-width swing arcs from both jambs ({arcs['double_arc']:.0%} drawn)."
        elif arcs["arc"] >= leaf * 0.95:
            symbol = "swing arc"
            reason = f"Swing arc of radius ≈ width from the {arcs['arc_hinge']} jamb ({arcs['arc']:.0%} drawn)."
        else:
            symbol = "door leaf"
            reason = (f"Door leaf drawn as a heavy free-standing stroke from a jamb ({leaf:.0%} of its length), no arc."
                      if heavy_leaf else f"Door leaf from a jamb at {leaf_angle:.0f}° ({leaf:.0%} drawn), no arc.")
        confidence = min(0.95, 0.6 + 0.35 * door_symbol + (0.05 if interior else 0.0))
    elif glazing >= 0.62:
        if interior and glazing < 0.8:
            kind, confidence, symbol = "opening", 0.45, "weak glazing"
            reason = "A faint line inside the wall band, but the opening connects two rooms; not enough for a window."
        else:
            kind = "window"
            symbol = "glazing lines" if glazing >= 0.8 else "dashed glazing line"
            parts = []
            if interior_full:
                parts.append(f"{len(interior_full)} glazing line(s) inside the wall band")
            elif interior_spread:
                parts.append("a dashed/faint line along the wall band")
            if faces:
                parts.append(f"{len(faces)} wall-face line(s)")
            parts.append({"room_to_exterior": "on an exterior wall", "between_rooms": "between two rooms"}.get(relation, "room side unknown"))
            reason = "Window: " + ", ".join(parts) + "."
            confidence = min(0.95, 0.45 + 0.5 * glazing - (0.15 if interior else 0.0))
    elif faces and not interior_any:
        symbol = "outline only"
        door_sized = (width_ft <= DOOR_MAX_FT) if width_ft is not None else (c.width <= 6.0 * max(c.thickness, t_ref))
        garage = (width_ft >= GARAGE_MIN_FT) if width_ft is not None else (c.width >= 14.0 * t_ref)
        if interior:
            kind, confidence = "door", 0.72
            reason = "Outlined gap with no glazing, connecting two rooms (door drawn without swing)."
        elif exterior and door_sized and len(faces) >= 2:
            kind, confidence = "door", 0.62
            reason = "Door-width outlined gap with no glazing on an exterior wall (entry door drawn without swing)."
        elif single_line and exterior and not garage:
            reason = "Open edge closed by a single drawn line on the exterior; could be an open side, a glazed wall or an undrawn door."
        elif exterior and garage:
            kind, confidence = "door", 0.58
            reason = "Very wide exterior gap closed by a single line (overhead/garage door)."
        else:
            reason = "Gap marked only by wall-face line(s); width and context do not separate a door, a glazed wall or an open edge."
    elif any_line < 0.2 and door_symbol < 0.35:
        symbol = "bare gap"
        reason = "Plain gap in the wall with no door or window symbol (passage or undrawn door)."
    else:
        reason = "Mixed or weak symbol evidence; not enough to decide."

    if relation in ("unknown", "same_space") and kind != "opening":
        confidence *= 0.85

    return {
        "type": kind,
        "confidence": round(float(confidence), 2),
        "evidence": {
            "symbol": symbol,
            "glazing_lines": len(interior_full),
            "dashed_glazing": bool(interior_spread and not interior_full),
            "face_lines": len(faces),
            "frame_marks": len(lines),
            "arc_score": round(arcs["arc"], 2),
            "double_arc_score": round(arcs["double_arc"], 2),
            "leaf_score": round(leaf, 2),
            "door_swing_score": round(door_symbol, 2),
            # leaf geometry in the opening's frame: hinge jamb ('start' / 'end' of the span), the side
            # of the wall the leaf swings to (+1 = along the candidate's normal), leaf opening angle
            "hinge": arcs["arc_hinge"] if arcs["arc"] >= arcs["double_arc"] * 1.05 else "both",
            "swing_side": int(arcs["double_side"] if arcs["double_arc"] * 1.05 > arcs["arc"] else arcs["arc_side"]),
            "leaf_angle": round(float(leaf_angle), 0),
            "sliding_panels": sliding,
            "sliding_shift": round(shift, 2),
            "room_relation": relation,
            "adjacent_space_ids": [sid for sid in (s.space_id(c.sides[0]), s.space_id(c.sides[1])) if sid],
            "exterior_side": EXTERIOR in c.sides,
            "wall_thickness_px": round(c.thickness, 1),
            "width_in_wall_thicknesses": round(c.width / max(c.thickness, 1.0), 2),
            "candidate_source": {"wall_gap": "gap in a wall", "line_pair": "outline lines between walls", "single_line": "single line between walls"}[c.source],
            "line_coverage": round(c.line_coverage, 2),
            "jamb_support": list(c.jambs),
            "detector": "geometry+symbols",
            "reason": reason,
        },
    }


STRONG_DOOR_SYMBOLS = {"swing arc", "double swing", "door leaf", "sliding panels"}
GLAZING_SYMBOLS = {"glazing lines", "dashed glazing line"}


def apply_plan_conventions(results: list[dict], candidates: list[OpeningCandidate], t_ref: float) -> None:
    """Plan-level symbol vocabulary: decide outline-only exterior openings from how the
    same drawing marks its doors and windows.

    Every plan has windows. When a drawing marks its doors with door symbols (swing, leaf, double,
    sliding) and outline-only frames are its dominant window symbol (more of them in exterior walls
    than glazing-line windows), those repeated outline-only exterior openings are its windows. Some
    windows elsewhere may still carry glazing lines and an interior passage may be an outline:
    drawings mix symbols. Where glazing-line windows are the norm, an outline-only exterior gap is
    an open side (porch, carport) and stays unclassified. (Measured on the canonical test cases:
    the earlier 'no glazing window anywhere and no interior outline' rule left 11 of 62 windows
    untyped.)
    """
    strong_doors = sum(r["evidence"]["symbol"] in STRONG_DOOR_SYMBOLS and r["type"] == "door" for r in results)
    exterior_outline = [i for i, (r, c) in enumerate(zip(results, candidates))
                        if r["evidence"]["symbol"] == "outline only" and r["evidence"]["room_relation"] == "room_to_exterior"
                        and c.source != "single_line" and r["evidence"]["face_lines"] >= 1
                        and c.width < 14.0 * max(c.thickness, t_ref)]
    glazing = sum(r["evidence"]["symbol"] in GLAZING_SYMBOLS and r["type"] == "window" for r in results)
    # the drawing's dominant window symbol decides: where glazing-line windows are the norm,
    # an outline-only exterior gap is an open side (porch, carport), not a window
    if strong_doors < 1 or len(exterior_outline) < 2 or glazing >= len(exterior_outline):
        return
    for i in exterior_outline:
        r = results[i]
        r["type"], r["confidence"] = "window", 0.62
        r["evidence"]["convention"] = "outline windows"
        r["evidence"]["reason"] = (
            f"Outlined frame in an exterior wall with no door symbol. This drawing marks doors with door "
            f"symbols ({strong_doors}) and shows no glazing-line windows, so its {len(exterior_outline)} "
            f"outline-only exterior openings are read as its window symbol.")


def classify_openings(image: np.ndarray, structure: StructureResult | None = None,
                      pixel_scale: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return openings in the API schema (id, type, confidence, geometry, evidence)."""
    public = structure if structure is not None else analyze_structure(image)
    # Evidence is measured in the frame the structure was analysed in (deskewed if needed);
    # geometry is reported in original image coordinates.
    s = public.work if public.work is not None else public
    thin = thin_lines(s.symbol_ink, s.wall_mask)
    dist = cv2.distanceTransform(cv2.bitwise_not(thin), cv2.DIST_L2, 3)
    raw_thin = thin
    results = [classify_candidate(c, s, thin, dist, pixel_scale, raw_thin) for c in s.candidates]
    apply_plan_conventions(results, s.candidates, max(s.wall_thickness, 1.0))
    openings = []
    for c_out, result in zip(public.candidates, results):
        c = c_out
        width_ft = _width_ft(c.width, pixel_scale)
        openings.append({
            "id": "",
            "type": result["type"],
            "type_label": TYPE_LABELS[result["type"]],
            "confidence": result["confidence"],
            "geometry_confidence": round(min(0.95, 0.55 + 0.2 * min(c.found_from, 2) + 0.2 * c.line_coverage), 2),
            "start": [int(round(c.start[0])), int(round(c.start[1]))],
            "end": [int(round(c.end[0])), int(round(c.end[1]))],
            "center": [int(round(c.center[0])), int(round(c.center[1]))],
            "orientation": c.orientation,
            "width_pixels": int(round(c.width)),
            "width_display": (
                f"{c.width / pixel_scale['pixels_per_unit']:.1f} {pixel_scale['unit']}"
                if pixel_scale and pixel_scale.get("pixels_per_unit") else None
            ),
            "evidence": result["evidence"],
        })
    openings.sort(key=lambda item: (item["type"], item["center"][1], item["center"][0]))
    counters = {"window": 0, "door": 0, "opening": 0}
    for item in openings:
        counters[item["type"]] += 1
        item["id"] = f"{item['type'][0].upper()}{counters[item['type']]:02d}"
    return openings


def detect_openings(image: np.ndarray, pixel_scale: dict[str, Any] | None = None, spaces=None,
                    structure: StructureResult | None = None) -> list[dict[str, Any]]:
    """Backward-compatible entry point (``spaces`` is ignored: topology comes from the structure)."""
    return classify_openings(image, structure, pixel_scale)


def draw_openings_overlay(image: np.ndarray, openings: list[dict[str, Any]]) -> bytes:
    """Draw a separate, readable openings layer over the original plan."""
    overlay = image.copy()
    colors = {
        "door": (50, 160, 65),
        "window": (220, 100, 25),
        "opening": (0, 145, 225),
    }
    thickness = max(4, min(8, image.shape[1] // 360))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    placed_labels: list[tuple[int, int, int, int]] = []
    for item in openings:
        color = colors[item["type"]]
        start = tuple(item["start"])
        end = tuple(item["end"])
        cv2.line(overlay, start, end, color, thickness, cv2.LINE_AA)
        radius = max(5, thickness // 2 + 2)
        cv2.circle(overlay, start, radius, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, end, radius, color, -1, cv2.LINE_AA)
        cx, cy = item["center"]
        label = item["id"]
        font_scale = 0.62
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        label_width = text_width + 10
        label_height = text_height + baseline + 8
        offsets = [
            (-label_width // 2, -34), (-label_width // 2, 48),
            (-label_width // 2 - 55, -24), (25, -24),
            (-label_width // 2 - 90, 42), (60, 42),
            (-label_width // 2, -68), (-label_width // 2, 78),
        ]
        placements = []
        for dx, dy in offsets:
            x = min(max(3, cx + dx), image.shape[1] - label_width - 3)
            y = min(max(label_height + 3, cy + dy), image.shape[0] - 3)
            rect = (x - 2, y - label_height, x + label_width + 2, y + 3)
            patch = gray[max(0, rect[1]):min(gray.shape[0], rect[3]), max(0, rect[0]):min(gray.shape[1], rect[2])]
            if patch.size == 0:
                continue
            collisions = sum(
                not (rect[2] < old[0] or rect[0] > old[2] or rect[3] < old[1] or rect[1] > old[3])
                for old in placed_labels
            )
            placements.append((float(np.mean(patch)) - collisions * 45, x, y, rect))
        if not placements:
            continue
        _, x, y, rect = max(placements, key=lambda entry: entry[0])
        placed_labels.append(rect)
        cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), -1)
        cv2.putText(overlay, label, (x + 5, y - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)

    success, encoded = cv2.imencode(".png", overlay)
    if not success:
        raise RuntimeError("Could not encode openings overlay")
    return encoded.tobytes()
