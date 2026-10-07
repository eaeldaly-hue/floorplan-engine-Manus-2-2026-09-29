"""Development-only view of the door/window evidence behind each opening decision.

Not part of the production path or the API. For every opening candidate it draws, in the
frame the classifier measured it (deskewed if the plan was rotated):

  * the wall band (width × wall thickness), outlined in the decision colour
    (door = orange, window = blue, unknown = grey);
  * the lines found inside the band (glazing = cyan, wall-face lines = dark blue);
  * the best swing arc / double arc / door leaf that was scored, in magenta, when its
    score is at least 0.3;
  * the opening id, its relation (R-R between rooms, R-E room to exterior) and the
    confidence.

It also writes a contact sheet with one enlarged crop per opening and its key scores,
and the full evidence as JSON.

    python -m scripts.diagnostics.opening_evidence_view [image] [--ppu 41.73] [--out output/evidence]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from engine.opening_detection import (
    Frame, arc_scores, band_lines, classify_candidate, leaf_score,
)
from engine.structure import analyze_structure, thin_lines

COLOURS = {"door": (0, 140, 255), "window": (220, 120, 0), "opening": (140, 140, 140)}
RELATION = {"between_rooms": "R-R", "room_to_exterior": "R-E", "same_space": "same", "unknown": "?"}


def _poly(f: Frame, us, vs) -> np.ndarray:
    xs, ys = f.points(us, vs)
    return np.stack([xs, ys], axis=1).round().astype(np.int32)


def _draw_band(img, f: Frame, colour, thickness=2):
    corners = _poly(f, [0, f.w, f.w, 0], [-0.5 * f.t, -0.5 * f.t, 0.5 * f.t, 0.5 * f.t])
    cv2.polylines(img, [corners], True, colour, thickness, cv2.LINE_AA)


def _draw_lines(img, f: Frame, lines):
    for line in lines:
        colour = (230, 200, 0) if line.interior else (140, 40, 0)
        pts = _poly(f, [line.first * f.w, line.last * f.w], [line.v, line.v])
        cv2.line(img, tuple(pts[0]), tuple(pts[1]), colour, 2, cv2.LINE_AA)


def _draw_symbol(img, f: Frame, arcs: dict, leaf: float, leaf_angle: float):
    """Redraw the best-scoring door symbol hypothesis (same parametrisation as the scorer)."""
    magenta = (200, 0, 200)
    phis = np.radians(np.linspace(10, 80, 26))
    side = arcs["arc_side"] or 1
    if arcs["double_arc"] >= 0.3 and arcs["double_arc"] >= arcs["arc"]:
        r, side = f.w / 2, arcs["double_side"] or 1
        for h, e in ((0.0, 1.0), (f.w, -1.0)):
            cv2.polylines(img, [_poly(f, h + e * r * np.cos(phis), side * r * np.sin(phis))], False, magenta, 2, cv2.LINE_AA)
    elif arcs["arc"] >= 0.3 and arcs["arc"] >= leaf:
        h, e = (0.0, 1.0) if arcs["arc_hinge"] == "start" else (f.w, -1.0)
        cv2.polylines(img, [_poly(f, h + e * f.w * np.cos(phis), side * f.w * np.sin(phis))], False, magenta, 2, cv2.LINE_AA)
    elif leaf >= 0.3:
        th = math.radians(leaf_angle)
        for s in (-1, 1):            # the scorer does not keep the side; draw both, faintly
            for h, e in ((0.0, 1.0), (f.w, -1.0)):
                pts = _poly(f, [h, h + e * f.w * math.cos(th)], [s * 0.5 * f.t, s * (0.5 * f.t + f.w * math.sin(th))])
                cv2.line(img, tuple(pts[0]), tuple(pts[1]), (230, 150, 230), 1, cv2.LINE_AA)


def _ids(results, candidates):
    order = sorted(range(len(results)), key=lambda i: (results[i]["type"], candidates[i].center[1], candidates[i].center[0]))
    counters = {"window": 0, "door": 0, "opening": 0}
    ids = [""] * len(results)
    for i in order:
        kind = results[i]["type"]
        counters[kind] += 1
        ids[i] = f"{kind[0].upper()}{counters[kind]:02d}"
    return ids


def render(image: np.ndarray, ppu: float | None = None):
    public = analyze_structure(image)
    s = public.work if public.work is not None else public
    thin = thin_lines(s.symbol_ink, s.wall_mask)
    dist = cv2.distanceTransform(cv2.bitwise_not(thin), cv2.DIST_L2, 3)
    scale = {"unit": "ft", "pixels_per_unit": ppu} if ppu else None

    base = cv2.cvtColor(s.gray, cv2.COLOR_GRAY2BGR)
    tint = base.copy()
    tint[s.wall_mask > 0] = (170, 210, 170)
    view = cv2.addWeighted(base, 0.55, tint, 0.45, 0)

    results, frames, extras = [], [], []
    for c in s.candidates:
        f = Frame.of(c)
        results.append(classify_candidate(c, s, thin, dist, scale, thin))
        frames.append(f)
        leaf, angle = leaf_score(dist, f, s.line_width)
        extras.append((band_lines(thin, f), arc_scores(dist, f, s.line_width), leaf, angle))
    ids = _ids(results, public.candidates)

    for f, r, (lines, arcs, leaf, angle), oid in zip(frames, results, extras, ids):
        _draw_symbol(view, f, arcs, leaf, angle)
        _draw_lines(view, f, lines)
        _draw_band(view, f, COLOURS[r["type"]])
        cx, cy = f.points(0.5 * f.w, 0.0)
        label = f"{oid} {RELATION.get(r['evidence']['room_relation'], '?')} {r['confidence']:.2f}"
        org = (int(cx) + 6, int(cy) - 6)
        cv2.putText(view, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(view, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOURS[r["type"]], 1, cv2.LINE_AA)

    sheet = _contact_sheet(view, frames, results, ids)
    records = [{"id": oid, "type": r["type"], "confidence": r["confidence"],
                "start_work_frame": [round(float(x), 1) for x in f.origin],
                "width_px": round(f.w, 1), "evidence": r["evidence"]}
               for f, r, oid in zip(frames, results, ids)]
    records.sort(key=lambda rec: rec["id"])
    return view, sheet, records, public.skew_degrees


def _contact_sheet(view, frames, results, ids, cell=260, cols=5):
    if not frames:
        return np.full((60, 400, 3), 255, np.uint8)
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    rows = math.ceil(len(order) / cols)
    text_h = 92
    sheet = np.full((rows * (cell + text_h), cols * cell, 3), 255, np.uint8)
    h, w = view.shape[:2]
    for k, i in enumerate(order):
        f, r = frames[i], results[i]
        cx, cy = f.points(0.5 * f.w, 0.0)
        half = int(max(f.w, 3 * f.t) * 0.9) + 10
        x0, y0 = max(0, int(cx) - half), max(0, int(cy) - half)
        x1, y1 = min(w, int(cx) + half), min(h, int(cy) + half)
        crop = view[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        z = cell / max(crop.shape[:2])
        crop = cv2.resize(crop, (int(crop.shape[1] * z), int(crop.shape[0] * z)), interpolation=cv2.INTER_NEAREST)
        oy, ox = (k // cols) * (cell + text_h), (k % cols) * cell
        sheet[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        ev = r["evidence"]
        text = [
            f"{ids[i]}  {r['type']}  conf {r['confidence']:.2f}  {RELATION.get(ev['room_relation'], '?')}",
            f"symbol: {ev['symbol']}",
            f"arc {ev['arc_score']:.2f}  dbl {ev['double_arc_score']:.2f}  leaf {ev['leaf_score']:.2f}",
            f"glaze {ev['glazing_lines']}  faces {ev['face_lines']}  shift {ev['sliding_shift']:.2f}",
            f"w/t {ev['width_in_wall_thicknesses']:.1f}  src {ev['candidate_source'].split()[0]}",
        ]
        for j, line in enumerate(text):
            cv2.putText(sheet, line, (ox + 4, oy + cell + 16 + 17 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        COLOURS[r["type"]] if j == 0 else (40, 40, 40), 1, cv2.LINE_AA)
        cv2.rectangle(sheet, (ox, oy), (ox + cell - 1, oy + cell + text_h - 1), (220, 220, 220), 1)
    return sheet


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("image", nargs="?", default="test_floorplan.png")
    ap.add_argument("--ppu", type=float, default=None, help="pixels per foot, if known")
    ap.add_argument("--out", default="output/evidence")
    args = ap.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    view, sheet, records, skew = render(image, args.ppu)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    cv2.imwrite(str(out / f"{stem}_evidence.png"), view)
    cv2.imwrite(str(out / f"{stem}_evidence_sheet.png"), sheet)
    (out / f"{stem}_evidence.json").write_text(json.dumps({"skew_degrees": skew, "openings": records}, indent=2))
    counts = {k: sum(1 for r in records if r["type"] == k) for k in ("door", "window", "opening")}
    print(f"{len(records)} openings {counts}; skew {skew:.2f}°; written to {out}/{stem}_evidence*.{{png,json}}")


if __name__ == "__main__":
    main()
