"""Hypothesis-based reconstruction: several readings of an ambiguous plan, the one that best
explains the independent evidence wins.

The structural pass makes early, irreversible choices - above all the wall class (which stroke
thickness is 'wall') and, implicitly, the working resolution. On most plans they are clear. On
some they are not:

  under-resolved   walls of 2-5 px: walls, door leaves, furniture and text strokes differ by a
                   pixel or less, so thickness cannot separate them (7.png: 2 px partitions next
                   to 4 px exterior walls). Read at a normalised scale (walls ~12 px) the classes
                   separate again. The scale comes from the plan's own measured wall thickness,
                   not from the image size.
  suspicious       the default reading collapses (no space, one space holding most of the floor,
                   a wall mask that floods or vanishes): the wall class is re-read from the
                   *cliff* of the erosion curve (engine.structure.cliff_wall_class).

Branching is triggered only by these structural signals, so easy plans run once. Each
hypothesis is a complete structure. They are compared, after OCR, on evidence the structure did
not use:

  room labels      every label lies in a space (not in a wall, not outside); two labels of rooms
                   that are normally walled off (bedroom, bath, closet...) do not share a space;
                   open-plan names (kitchen, living, dining...) may
  text blocks      name/dimension blocks found geometrically (read or not): each room holds one or
                   two; more in one space means rooms were merged, a block in a wall means one
                   was lost

score = -(2 x merged label pairs + 1.5 x lost labels + 0.5 x merged blocks + 0.25 x lost blocks).
The default wins ties. FLOORPLAN_HYPOTHESES=0 disables the search.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

OPEN_NAMES = {"Kitchen", "Dining", "Dining room", "Living", "Living room", "Family", "Family room", "Great room",
              "Breakfast nook", "Nook", "Foyer", "Entry", "Hall", "Hallway", "Sitting area", "Den", "Study",
              "Library", "Playroom", "Sunroom", "Stairs", "Room", "Balcony", "Terrace", "Porch", "Patio", "Deck",
              "Corridor", "Gallery", "Lobby", "Landing"}
UNDER_RESOLVED_T = 6.0          # px: thinner walls cannot be separated from lines by thickness
TARGET_T = 12.0                 # px: wall thickness of the normalised reading
MAX_SCALE = 3.5
MAX_PIXELS = 16_000_000
MIN_GAIN = 0.25                 # a hypothesis must beat the default by this much


def enabled() -> bool:
    return os.environ.get("FLOORPLAN_HYPOTHESES", "1").strip().lower() not in ("0", "off", "false", "no")


@dataclass
class Hypothesis:
    name: str
    scale: float
    image: np.ndarray
    structure: object
    reason: str
    wall_class: tuple | None = None
    score: float | None = None
    parts: dict = field(default_factory=dict)


def _suspicious(s) -> str | None:
    from engine.wall_inference import interior_collapsed

    wall = float((s.wall_mask > 0).mean())
    if not s.spaces:
        return "no enclosed space"
    if interior_collapsed(s.spaces):
        return "one space holds most of the floor"
    if wall > 0.3:
        return f"wall mask covers {wall:.0%} of the image"
    if wall < 0.005:
        return "almost no wall found"
    return None


def _cliff(image, s0, scale: float, name: str, why: str) -> Hypothesis | None:
    from engine.structure import _binarize, analyze_structure, cliff_wall_class, wall_class

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink, _ = _binarize(gray)
    c = cliff_wall_class(ink)
    if c is None or c[0] == s0.wall_kernel:
        return None
    with wall_class(c[0], c[1]):
        s = analyze_structure(image)
    return Hypothesis(name, scale, image, s, f"{why}; wall class re-read from the erosion cliff "
                      f"(kernel {c[0]}, walls ~{c[1]:.0f} px, {c[2]:.0%} of the ink)", wall_class=c[:2])


def generate(image: np.ndarray, s0) -> list[Hypothesis]:
    """The default reading plus the alternatives its structural signals call for."""
    from engine.structure import analyze_structure

    hyps = [Hypothesis("default", 1.0, image, s0, "single reading")]
    if not enabled():
        return hyps
    why = _suspicious(s0)
    if why:
        h = _cliff(image, s0, 1.0, "wall-class", why)
        if h:
            hyps.append(h)
    t = float(s0.wall_thickness or 0)
    h_px, w_px = image.shape[:2]
    if 0 < t <= UNDER_RESOLVED_T:
        f = float(np.clip(TARGET_T / t, 1.5, MAX_SCALE))
        f = min(f, (MAX_PIXELS / float(h_px * w_px)) ** 0.5)
        if f >= 1.5:
            big = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            s = analyze_structure(big)
            hyps.append(Hypothesis("normalised-scale", f, big, s,
                                   f"walls {t:.0f} px are under-resolved; read at x{f:.1f} (walls ~{t * f:.0f} px)"))
            why_s = _suspicious(s)
            if why_s:
                h = _cliff(big, s, f, "normalised-scale+wall-class", why_s)
                if h:
                    hyps.append(h)
    return hyps


def text_blocks(image: np.ndarray) -> list[tuple[float, float]]:
    """Centres of text blocks (name + dimensions of a room), found geometrically."""
    from engine.analysis.text_lines import find_text_lines

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    lines = find_text_lines(gray)
    if not lines:
        return []
    boxes = [(ln.x, ln.y, ln.x + ln.w, ln.y + ln.h, ln.glyph_h) for ln in lines]
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            g = max(a[4], b[4])
            if a[0] - g <= b[2] and b[0] - g <= a[2] and a[1] - 1.5 * g <= b[3] and b[1] - 1.5 * g <= a[3]:
                parent[find(i)] = find(j)
    groups: dict = {}
    for i, b in enumerate(boxes):
        groups.setdefault(find(i), []).append(b)
    out = []
    for g in groups.values():
        x0, y0 = min(b[0] for b in g), min(b[1] for b in g)
        x1, y1 = max(b[2] for b in g), max(b[3] for b in g)
        if x1 - x0 >= 3 * max(b[4] for b in g):
            out.append(((x0 + x1) / 2, (y0 + y1) / 2))
    return out


def score(h: Hypothesis, labels: list[tuple[str, tuple[float, float]]], blocks: list) -> float:
    s = h.structure
    if not s.spaces:
        h.score, h.parts = -1e9, {"spaces": 0}
        return h.score
    L = s.space_labels
    H, W = L.shape

    def space(x, y) -> int:
        x, y = int(x * h.scale), int(y * h.scale)
        if not (0 <= x < W and 0 <= y < H):
            return -1
        win = L[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
        v = win[win > 0]
        return int(np.bincount(v).argmax()) if v.size else int(L[y, x])

    lost = merged = 0
    per: dict = {}
    for name, (x, y) in labels:
        k = space(x, y)
        if k <= 0:
            lost += 1
        else:
            per.setdefault(k, []).append(name)
    for names in per.values():
        if any(n not in OPEN_NAMES for n in names):
            merged += len(names) - 1
    blost = 0
    bper: dict = {}
    for x, y in blocks:
        k = space(x, y)
        if k <= 0:
            blost += 1
        else:
            bper[k] = bper.get(k, 0) + 1
    bmerged = sum(max(0, c - 2) for c in bper.values())
    h.parts = {"labels_lost": lost, "labels_merged": merged, "blocks_lost": blost, "blocks_merged": bmerged,
               "spaces": len(s.spaces)}
    h.score = -(2.0 * merged + 1.5 * lost + 0.5 * bmerged + 0.25 * blost)
    return h.score


def choose(hyps: list[Hypothesis], labels, blocks) -> tuple[Hypothesis, dict]:
    if len(hyps) == 1:
        return hyps[0], {"hypotheses": 1, "chosen": "default"}
    for h in hyps:
        score(h, labels, blocks)
    best = hyps[0]
    for h in hyps[1:]:
        if h.score >= best.score + MIN_GAIN:
            best = h
    report = {
        "hypotheses": len(hyps),
        "chosen": best.name,
        "scale": round(best.scale, 2),
        "candidates": [{"name": h.name, "scale": round(h.scale, 2), "reason": h.reason, "score": round(h.score, 2),
                        "evidence": h.parts} for h in hyps],
    }
    return best, report


def scale_lines(lines, f: float):
    return [dataclasses.replace(ln, x=int(round(ln.x * f)), y=int(round(ln.y * f)),
                                width=int(round(ln.width * f)), height=int(round(ln.height * f))) for ln in lines]


def scale_ocr(ocr_result, f: float):
    """A view of the OCR result with word boxes in the scaled frame (only .boxes is read downstream)."""
    if ocr_result is None:
        return None
    boxes = [dataclasses.replace(b, x=int(round(b.x * f)), y=int(round(b.y * f)),
                                 width=int(round(b.width * f)), height=int(round(b.height * f))) for b in ocr_result.boxes]
    try:
        return dataclasses.replace(ocr_result, boxes=boxes)
    except TypeError:
        import types
        return types.SimpleNamespace(**{**vars(ocr_result), "boxes": boxes})
