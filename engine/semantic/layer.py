"""Semantic element layer (experimental): every drawn element with a type, its geometry, its
source and the evidence for the type. Structure-relevant types (WALL, COLUMN, DOOR, WINDOW) can be
rendered as a clean structural image that the existing geometry stages run on, while text,
dimensions and the original drawing stay available unchanged.

Sources today: PDF vector paths (ingest.pdf_vectors) and the PDF text layer. Types are assigned by
geometry and by evidence, never by a fixed pen width or a file-specific rule:

  WALL      paths of the pens that, drawn alone, reconstruct as walls (face pairs or filled bands,
            engine.plan) over most of their length and as a network of many walls - a pen that
            carries the wall structure (frames and outlines pair up too, but as a few long lines)
  COLUMN    filled shapes lying in the wall bands
  DOOR      arcs of about a quarter turn hinged at a wall (centre within a wall band), and the
            straight leaf from the hinge with the arc's radius
  WINDOW    straight lines in a wall line between two wall ends: both ends on a wall band, the wall
            continuing beyond both ends in the same line, the middle in the gap
  TEXT      words of the PDF text layer
  DIMENSION straight lines running along a dimension string of the text layer
  OTHER     everything else (furniture, fixtures, stroked text, annotation, hatch, grid, frame) -
            not separated further yet

When no pen carries the walls on its own (walls drawn in the same pen as other elements, e.g.
hatched walls), the layer says so (`applicable` False) and nothing is simplified.
"""

from __future__ import annotations

import copy
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

STRUCTURAL = ("WALL", "COLUMN", "DOOR", "WINDOW")
TYPE_COLORS = {"WALL": (30, 30, 30), "COLUMN": (90, 0, 90), "DOOR": (0, 160, 0), "WINDOW": (230, 120, 0),
               "TEXT": (0, 0, 220), "DIMENSION": (200, 0, 200), "OTHER": (175, 175, 175)}

PEN_SCALE = 0.5          # resolution of the pen-alone wall test
PEN_CANDIDATES = 12      # pens tested (by drawn length)
WALL_COVER = 0.6         # share of a pen's ink that must reconstruct as walls
MIN_WALLS = 4            # walls a pen alone must reconstruct to be a wall network (a small house has few)
MIN_ALIGNMENT = 0.9       # share of the typed wall ink that must lie on the page's ink
RICH_SHARE = 0.5         # further wall pens: at least this share of the richest pen's walls


def enabled(request_body: dict | None = None) -> bool:
    """Experimental, off by default: FLOORPLAN_STRUCTURAL_LAYER=on, or {"structural_layer": true}
    in a PDF page analysis request."""
    import os

    if request_body and request_body.get("structural_layer") is not None:
        return bool(request_body.get("structural_layer"))
    return os.environ.get("FLOORPLAN_STRUCTURAL_LAYER", "off").strip().lower() in ("1", "on", "true", "yes")


@dataclass
class Element:
    type: str
    geometry: list                      # polyline [(x, y), ...] (px) or box [(x0, y0), (x1, y1)]
    source: str                         # vector | text-layer
    width: float = 0.0                  # drawn pen width (px); 0 = fill / box
    fill: bool = False
    confidence: float = 0.0
    evidence: str = ""


@dataclass
class SemanticLayer:
    shape: tuple
    elements: list = field(default_factory=list)
    applicable: bool = False
    reason: str = ""
    wall_pens: list = field(default_factory=list)
    wall_thickness: float = 0.0
    pen_tests: list = field(default_factory=list)
    seconds: float = 0.0
    wall_body: np.ndarray | None = None     # interiors of outlined (hollow) walls: wall, not paper
    alignment: float | None = None          # share of typed wall ink on the page ink (with the page image)
    closures: list = field(default_factory=list)   # openings closed as drawn: {p0, p1, kind, evidence}

    def counts(self) -> dict:
        out = defaultdict(int)
        for e in self.elements:
            out[e.type] += 1
        return dict(out)

    def summary(self) -> dict:
        return {"applicable": self.applicable, "reason": self.reason, "counts": self.counts(),
                "wall_pens": [list(p) for p in self.wall_pens], "wall_thickness_px": round(self.wall_thickness, 1),
                "pen_tests": self.pen_tests, "seconds": round(self.seconds, 2),
                "alignment": None if self.alignment is None else round(self.alignment, 3),
                "wall_body_px": int(self.wall_body.sum()) if self.wall_body is not None else 0,
                "closures": {k: sum(c["kind"] == k for c in self.closures) for k in ("door", "glazing", "gap-door", "gap-window")}}

    def structural_image(self, closures: bool = False) -> np.ndarray:
        """Walls, columns, doors and windows exactly as drawn (same vectors, same pen widths) on a
        blank sheet, with the body of outlined walls filled (`wall_body`): a hollow wall's two face
        lines and the paper between them are one wall. Faces, thickness and openings are unchanged."""
        h, w = self.shape[:2]
        img = np.full((h, w), 255, np.uint8)
        if self.wall_body is not None:
            img[self.wall_body] = 0               # an outlined wall is solid wall between its faces
        for e in self.elements:
            if e.type in STRUCTURAL:
                _draw(img, e, 0)
        if closures:
            # an opening seals its room the way the closed door or the glazing does: a band of
            # wall thickness from jamb to jamb (the opening itself stays recorded in `closures`)
            t = max(3, int(round(self.wall_thickness)))
            for c in self.closures:
                cv2.line(img, tuple(int(round(v)) for v in c["p0"]), tuple(int(round(v)) for v in c["p1"]), 0, t)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def debug_image(self, original: np.ndarray) -> np.ndarray:
        """The original page faded, every element coloured by its type."""
        vis = (0.25 * original + 0.75 * 255).astype(np.uint8)
        for t in ("OTHER", "DIMENSION", "TEXT", "WINDOW", "COLUMN", "WALL", "DOOR"):
            for e in self.elements:
                if e.type == t:
                    _draw(vis, e, TYPE_COLORS[t], minimum=2 if t in ("DOOR", "WINDOW") else 1)
        return vis


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def _bezier(p0, c1, c2, p3, n=12) -> list:
    t = np.linspace(0, 1, n)[:, None]
    pts = ((1 - t) ** 3 * np.array(p0) + 3 * (1 - t) ** 2 * t * np.array(c1)
           + 3 * (1 - t) * t ** 2 * np.array(c2) + t ** 3 * np.array(p3))
    return [tuple(p) for p in pts]


def _polyline(seg) -> list:
    if seg[0] == "L":
        return [seg[1], seg[2]]
    c1, c2, p3 = seg[3]
    return _bezier(seg[1], c1, c2, p3)


def _draw(img, e: Element, color, minimum: int = 1, scale: float = 1.0) -> None:
    pts = np.round(np.array(e.geometry, float) * scale).astype(np.int32)
    if len(pts) == 0:
        return
    if e.fill:
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    elif e.source == "text-layer":
        cv2.rectangle(img, tuple(pts[0]), tuple(pts[-1]), color, max(1, minimum))
    else:
        cv2.polylines(img, [pts], False, color, max(minimum, int(round(e.width * scale))), cv2.LINE_AA)


def _path_elements(path, etype: str, conf: float, why: str) -> list:
    if path.kind == "fill":
        # one element per closed subpath (a compound fill holds many)
        out, poly, prev = [], [], None
        for seg in path.segs:
            pts = _polyline(seg)
            if prev is not None and math.dist(prev, pts[0]) > 0.5 and len(poly) >= 3:
                out.append(Element(etype, poly, "vector", 0.0, True, conf, why))
                poly = []
            poly += pts[:-1] if poly else pts[:-1]
            prev = pts[-1]
            if not poly:
                poly = [pts[0]]
        if prev is not None:
            poly.append(prev)
        if len(poly) >= 3:
            out.append(Element(etype, poly, "vector", 0.0, True, conf, why))
        return out
    return [Element(etype, _polyline(seg), "vector", path.width, False, conf, why) for seg in path.segs]


def _render_paths(paths, shape, scale) -> np.ndarray:
    img = np.full(shape, 255, np.uint8)
    for p in paths:
        for e in _path_elements(p, "X", 0, ""):
            _draw(img, e, 0, scale=scale)
    return img


# ---------------------------------------------------------------------------
# walls: pens that carry the structure
# ---------------------------------------------------------------------------

def _wall_test(paths, shape_small):
    """Reconstruct a pen drawn alone; share of its ink inside face-pair / filled-band walls."""
    from engine.plan.reconstruct import reconstruct, render_walls

    img = _render_paths(paths, shape_small, PEN_SCALE)
    ink = img < 128
    if ink.sum() < 50:
        return 0.0, None
    model = reconstruct(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    solid = [w for w in model.walls if w.style != "thin"]
    band = cv2.dilate(render_walls(solid, shape_small), np.ones((3, 3), np.uint8)) > 0
    return float((ink & band).sum() / ink.sum()), model


def _wall_band(model, shape) -> tuple[np.ndarray, float]:
    """Full-resolution wall bands of a reduced-resolution reconstruction."""
    from engine.plan.reconstruct import render_walls

    walls = []
    for w in model.walls:
        if w.style == "thin":
            continue
        w2 = copy.copy(w)
        w2.p0 = (w.p0[0] / PEN_SCALE, w.p0[1] / PEN_SCALE)
        w2.p1 = (w.p1[0] / PEN_SCALE, w.p1[1] / PEN_SCALE)
        w2.thickness = w.thickness / PEN_SCALE
        walls.append(w2)
    t = float(np.median([w.thickness for w in walls])) if walls else 0.0
    return render_walls(walls, shape[:2]) > 0, t


def _poche(paths, shape):
    """Walls drawn as poché - outline plus hatch, no dedicated pen: families of many short parallel
    off-axis segments (the hatch) whose closed region forms long bands; the outline segments lying
    on those bands. Returns (band mask, wall thickness, {(path, segment)} of wall segments) or None.
    Dimension ticks and arrows are short diagonals too, but isolated: they never form a band."""
    h, w = shape[:2]
    diag = math.hypot(h, w)
    families: dict = defaultdict(list)
    for pi, p in enumerate(paths):
        if p.kind != "stroke":
            continue
        for si, seg in enumerate(p.segs):
            if seg[0] != "L":
                continue
            L = math.dist(seg[1], seg[2])
            ang = math.degrees(math.atan2(seg[2][1] - seg[1][1], seg[2][0] - seg[1][0])) % 180.0
            off_axis = min(ang % 90.0, 90.0 - ang % 90.0)
            if 2.0 < L <= 0.02 * diag and off_axis >= 20.0:
                families[int(round(ang / 2.0)) % 90].append((pi, si))
    hatch = [k for fam in families.values() if len(fam) >= 150 for k in fam]
    if not hatch:
        return None
    ink = np.zeros((h, w), np.uint8)
    for pi, si in hatch:
        seg = paths[pi].segs[si]
        cv2.line(ink, tuple(int(round(v)) for v in seg[1]), tuple(int(round(v)) for v in seg[2]), 1, 2)
    k = max(5, int(round(0.004 * diag)) | 1)
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed)
    band = np.zeros((h, w), bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if max(bw, bh) >= 6 * k and area >= 4 * k * k:
            band |= lab == i
    if band.sum() < 0.002 * h * w:
        return None
    dist = cv2.distanceTransform(band.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = dist[(dist > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))]
    t_wall = float(2 * np.median(ridge)) if ridge.size else float(k)
    near = cv2.dilate(band.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    wall_segs = set()                         # hatch and outline segments lying on the bands
    for pi, p in enumerate(paths):
        if p.kind != "stroke":
            continue
        for si, seg in enumerate(p.segs):
            if seg[0] != "L":
                continue
            pts = [(seg[1][0] + (seg[2][0] - seg[1][0]) * f, seg[1][1] + (seg[2][1] - seg[1][1]) * f) for f in np.linspace(0, 1, 7)]
            if all(0 <= int(round(y)) < h and 0 <= int(round(x)) < w and near[int(round(y)), int(round(x))] for x, y in pts):
                wall_segs.add((pi, si))
    return band, t_wall, wall_segs


def _wall_body(wall_paths, shape, band, t_wall, model=None) -> np.ndarray:
    """Paper enclosed by the wall pen's own outlines, narrower than half a wall thickness and lying
    at the reconstructed walls: the inside of a wall drawn as an outline. Room interiors, door
    gaps and anything not closed by the wall drawing itself never qualify."""
    ink = _render_paths(wall_paths, shape, 1.0) < 128
    paper = (~ink).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(paper, connectivity=4)
    dist = cv2.distanceTransform(paper, cv2.DIST_L2, 3)
    h, w = shape[:2]
    radius = np.zeros(n)
    np.maximum.at(radius, lab.ravel(), dist.ravel())
    k = max(3, int(round(2 * t_wall)) | 1)          # near a reconstructed wall (reduced resolution misses short ones)
    near = cv2.dilate(band.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    on_band = np.zeros(n)
    np.add.at(on_band, lab.ravel(), near.ravel().astype(np.float64))
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        if radius[i] <= 0.5 * t_wall and on_band[i] >= 0.5 * area:
            keep[i] = True
    body = keep[lab]
    # the inside of every outlined wall, from its own faces: two parallel wall-pen lines at most a
    # wall thickness apart, overlapping along their length, bound the wall body between them
    # (the outline need not be closed by the wall pen - another pen may abut it)
    faces = np.zeros(shape[:2], np.uint8)
    segs = []
    for p in wall_paths:
        for seg in p.segs:
            if seg[0] == "L" and math.dist(seg[1], seg[2]) >= 0.5 * t_wall:
                segs.append((seg[1], seg[2], p.width))
    by_angle: dict = defaultdict(list)
    for a_, b_, wd in segs:
        ang = math.degrees(math.atan2(b_[1] - a_[1], b_[0] - a_[0])) % 180.0
        by_angle[int(round(ang)) % 180].append((a_, b_, wd, ang))
    for key, items in by_angle.items():
        pool = items + by_angle.get((key + 1) % 180, []) + by_angle.get((key - 1) % 180, [])
        for i, (a_, b_, wd, ang) in enumerate(items):
            L = math.dist(a_, b_)
            u = ((b_[0] - a_[0]) / L, (b_[1] - a_[1]) / L)
            n = (-u[1], u[0])
            for c_, d_, wd2, ang2 in pool:
                if (c_, d_) == (a_, b_):
                    continue
                off = (c_[0] - a_[0]) * n[0] + (c_[1] - a_[1]) * n[1]
                off2 = (d_[0] - a_[0]) * n[0] + (d_[1] - a_[1]) * n[1]
                if not (0.5 * (wd + wd2) + 1.0 < off <= 1.15 * t_wall and abs(off2 - off) <= 1.5):
                    continue                                   # one side only: each pair is seen once
                s0, s1 = sorted([(c_[0] - a_[0]) * u[0] + (c_[1] - a_[1]) * u[1], (d_[0] - a_[0]) * u[0] + (d_[1] - a_[1]) * u[1]])
                lo, hi = max(0.0, s0), min(L, s1)
                if hi - lo < max(0.5 * t_wall, 0.3 * min(L, s1 - s0)):
                    continue
                o = 0.5 * (off + off2)
                quad = [(a_[0] + u[0] * lo, a_[1] + u[1] * lo), (a_[0] + u[0] * hi, a_[1] + u[1] * hi),
                        (a_[0] + u[0] * hi + n[0] * o, a_[1] + u[1] * hi + n[1] * o), (a_[0] + u[0] * lo + n[0] * o, a_[1] + u[1] * lo + n[1] * o)]
                cv2.fillPoly(faces, [np.round(np.array(quad)).astype(np.int32)], 1)
    body |= (faces > 0) & (paper > 0)
    return body


# ---------------------------------------------------------------------------
# doors: quarter arcs hinged at a wall, and their leaves
# ---------------------------------------------------------------------------

def _arc(seg):
    """(centre, radius, start angle, end angle) of a cubic that is a circular arc, else None."""
    p0, p3 = np.array(seg[1]), np.array(seg[2])
    c1, c2 = np.array(seg[3][0]), np.array(seg[3][1])
    t0, t3 = c1 - p0, p3 - c2
    if np.hypot(*t0) < 1e-6 or np.hypot(*t3) < 1e-6:
        return None
    n0, n3 = np.array([-t0[1], t0[0]]), np.array([-t3[1], t3[0]])
    a = np.array([n0, -n3]).T
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    s = np.linalg.solve(a, p3 - p0)
    c = p0 + s[0] * n0
    r0, r3 = np.hypot(*(p0 - c)), np.hypot(*(p3 - c))
    if r0 < 1 or abs(r0 - r3) > 0.08 * r0:
        return None
    mid = np.array(_bezier(seg[1], seg[3][0], seg[3][1], seg[2], 3)[1])
    if abs(np.hypot(*(mid - c)) - r0) > 0.08 * r0:
        return None
    a0, a3 = math.atan2(*(p0 - c)[::-1]), math.atan2(*(p3 - c)[::-1])
    sweep = abs((math.degrees(a3 - a0) + 180) % 360 - 180)
    return (float(c[0]), float(c[1])), float((r0 + r3) / 2), sweep


def _doors(paths, wall_pens, band_dist, t_wall):
    arcs = []           # (path index, seg index, centre, r, sweep)
    for pi, p in enumerate(paths):
        if p.pen in wall_pens or p.kind != "stroke":
            continue
        for si, seg in enumerate(p.segs):
            if seg[0] == "C":
                a = _arc(seg)
                if a and 5 <= a[2] <= 100:
                    arcs.append((pi, si, *a))
    # one door swing may be split into several cubics: group by centre and radius
    groups: dict = defaultdict(list)
    for k, (_, _, c, r, sweep) in enumerate(arcs):
        groups[(round(c[0] / 4), round(c[1] / 4), round(math.log(r) / 0.05))].append(k)
    h, w = band_dist.shape
    doors, used = [], set()
    for key, members in groups.items():
        if any(k in used for k in members):
            continue
        near = [k for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dr in (-1, 0, 1)
                for k in groups.get((key[0] + dx, key[1] + dy, key[2] + dr), []) if k not in used]
        sweep = sum(arcs[k][4] for k in near)
        c, r = arcs[near[0]][2], arcs[near[0]][3]
        if not 60 <= sweep <= 100 or not 1.5 * t_wall <= r <= 15 * t_wall:
            continue
        cx, cy = int(round(c[0])), int(round(c[1]))
        if not (0 <= cx < w and 0 <= cy < h) or band_dist[cy, cx] > 0.5 * t_wall + 4:
            continue                          # hinge not at a wall
        used.update(near)
        doors.append((c, r, near))
    return arcs, doors


# ---------------------------------------------------------------------------
# windows: lines in the wall line between two wall ends
# ---------------------------------------------------------------------------

def _is_window(a, b, band, band_dist, t) -> bool:
    h, w = band.shape
    L = math.dist(a, b)
    if L < 1.5 * t:
        return False
    for p in (a, b):
        x, y = int(round(p[0])), int(round(p[1]))
        if not (0 <= x < w and 0 <= y < h) or band_dist[y, x] > 0.6 * t + 2:
            return False
    u = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
    mids = [(a[0] + u[0] * L * f, a[1] + u[1] * L * f) for f in np.linspace(0.2, 0.8, 9)]
    on = [band[min(h - 1, max(0, int(round(y)))), min(w - 1, max(0, int(round(x))))] for x, y in mids]
    if np.mean(on) > 0.4:
        return False                          # the middle is wall: not a gap
    for p, sgn in ((a, -1), (b, 1)):          # the wall continues beyond both ends, in this line
        beyond = [(p[0] + sgn * u[0] * t * f, p[1] + sgn * u[1] * t * f) for f in (1.0, 1.5, 2.0, 2.5)]
        hit = [0 <= int(round(y)) < h and 0 <= int(round(x)) < w and band[int(round(y)), int(round(x))] for x, y in beyond]
        if np.mean(hit) < 0.75:
            return False
    return True


def _alongside(a, b, band, t) -> bool:
    """Does the line a-b run beside a wall (wall band just to one side over most of its middle)?
    An opening's closure crosses a gap and has no wall beside it; an open door leaf, a dimension
    line or a counter edge lies along a wall."""
    h, w = band.shape
    L = math.dist(a, b)
    if L < 1:
        return False
    n = (-(b[1] - a[1]) / L, (b[0] - a[0]) / L)
    hits = 0
    fs = np.linspace(0.25, 0.75, 7)
    for f in fs:
        m = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        for sgn in (1, -1):
            x, y = int(round(m[0] + sgn * n[0] * (0.5 * t + 3))), int(round(m[1] + sgn * n[1] * (0.5 * t + 3)))
            if 0 <= x < w and 0 <= y < h and band[y, x]:
                hits += 1
                break
    return hits >= 0.5 * len(fs)


def _dominant_directions(lines, t, share: float = 0.05) -> list:
    """The building's wall directions: length-weighted 2-degree histogram of wall lines at least two
    wall thicknesses long (hatch strokes are shorter), peaks carrying >= `share` of the length."""
    hist = np.zeros(90)
    for a, b in lines:
        L = math.dist(a, b)
        if L >= 2 * t:
            hist[int((math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0) // 2) % 90] += L
    if hist.sum() <= 0:
        return [0.0, 90.0]
    smooth = hist + np.roll(hist, 1) + np.roll(hist, -1)
    return [2.0 * k + 1.0 for k in range(90) if hist[k] > 0 and smooth[k] >= share * hist.sum()
            and smooth[k] >= max(smooth[(k - 1) % 90], smooth[(k + 1) % 90])]


def _glazing(paths, wall_pens, door_segs, band, band_dist, t, wall_dir=None) -> list:
    """Glazing, sliding panels and windows: two or more parallel thin lines spanning a gap in the
    walls - the middle off the walls and with no wall beside it, each end on a wall (or closed door)
    or on another glazing run (corner windows). Every run needs at least one end on a wall, so
    double-lined furniture (tubs, rugs) cannot chain into a closure. Counters, dimension lines and
    furniture edges touching walls are single lines or run along a wall."""
    h, w = band.shape
    # straight pieces long enough to span a gap, in path order (vectorised below)
    pieces = []
    for pi, p in enumerate(paths):
        if p.pen in wall_pens or p.kind != "stroke":
            continue
        pts = [q for seg in p.segs for q in (seg[1], seg[2])]
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        if len(p.segs) >= 3 and math.dist(p.segs[0][1], p.segs[-1][2]) <= 1.0 and max(max(xs) - min(xs), max(ys) - min(ys)) <= 4 * t:
            continue                          # a small closed shape: a tag, a symbol, a fixture - not glazing
        for si, seg in enumerate(p.segs):
            if seg[0] != "L" or (pi, si) in door_segs:
                continue
            a, b = seg[1], seg[2]
            L = math.dist(a, b)
            if L >= 1.5 * t:
                pieces.append((a, b, L))
    cands = []
    if pieces:
        # the middle of the piece must be off the walls: 9 samples, as many on wall as before
        A = np.array([q[0] for q in pieces], float)
        B = np.array([q[1] for q in pieces], float)
        f = np.linspace(0.2, 0.8, 9)
        mx = A[:, 0, None] + (B[:, 0, None] - A[:, 0, None]) * f[None, :]
        my = A[:, 1, None] + (B[:, 1, None] - A[:, 1, None]) * f[None, :]
        xi = np.minimum(w - 1, np.maximum(0, np.round(mx).astype(int)))
        yi = np.minimum(h - 1, np.maximum(0, np.round(my).astype(int)))
        on_wall = band[yi, xi].mean(axis=1) > 0.4
        for (a, b, L), off in zip(pieces, on_wall):
            if off:
                continue
            if _alongside(a, b, band, t):
                continue                      # runs along a wall face (dimension line, counter, shelf)
            ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
            cands.append((a, b, L, ang))
    parent = list(range(len(cands)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    # Only lines within 2 degrees, at most t apart and overlapping can pair. A qualifying pair has
    # points within t + M*sin(2 deg) of each other (M: the other line's length), so each line's box
    # grown by t/2 + 0.04*(its own length) meets its partner's: lines are binned on a grid by these
    # grown boxes and compared within shared cells, after the angle test - the same pairs as
    # comparing all of them.
    cell = max(64.0, 4.0 * (t + 2.0))
    bins: dict = defaultdict(list)
    spans = []
    for k, (a, b, L, _ang) in enumerate(cands):
        m = 0.5 * t + 0.04 * L + 1.0
        gx0, gx1 = int((min(a[0], b[0]) - m) // cell), int((max(a[0], b[0]) + m) // cell)
        gy0, gy1 = int((min(a[1], b[1]) - m) // cell), int((max(a[1], b[1]) + m) // cell)
        spans.append((gx0, gx1, gy0, gy1))
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                bins[(gx, gy)].append(k)
    bins = {key: np.asarray(v, int) for key, v in bins.items()}
    angs = np.array([c[3] for c in cands], float)

    def neighbours(i):
        gx0, gx1, gy0, gy1 = spans[i]
        parts = [bins[(gx, gy)] for gx in range(gx0, gx1 + 1) for gy in range(gy0, gy1 + 1)]
        js = np.unique(np.concatenate(parts)) if parts else np.zeros(0, int)
        js = js[js > i]
        da = np.abs(angs[i] - angs[js]) % 180.0
        return js[np.minimum(da, 180.0 - da) <= 2.0 + 1e-9].tolist()
    for i in range(len(cands)):
        a, b, L, ang = cands[i]
        u = ((b[0] - a[0]) / L, (b[1] - a[1]) / L)
        for j in neighbours(i):
            c, d, M, ang2 = cands[j]
            da = abs(ang - ang2) % 180.0
            if min(da, 180 - da) > 2.0:
                continue
            off = abs((c[0] - a[0]) * -u[1] + (c[1] - a[1]) * u[0])
            if off > t or off < 0.5:
                continue
            s0, s1 = sorted([(c[0] - a[0]) * u[0] + (c[1] - a[1]) * u[1], (d[0] - a[0]) * u[0] + (d[1] - a[1]) * u[1]])
            if min(L, s1) - max(0.0, s0) >= 0.8 * min(L, M):
                parent[find(j)] = find(i)
    groups: dict = defaultdict(list)
    for i in range(len(cands)):
        groups[find(i)].append(i)
    runs = []
    for members in groups.values():
        if len(members) >= 2:
            a, b, _, _ = max((cands[i] for i in members), key=lambda c: c[2])
            runs.append((a, b, len(members)))

    def on_wall(q):
        x, y = int(round(q[0])), int(round(q[1]))
        return 0 <= x < w and 0 <= y < h and band_dist[y, x] <= 0.6 * t + 2
    ends_wall = [(on_wall(a), on_wall(b)) for a, b, _ in runs]
    keep = [ea or eb for ea, eb in ends_wall]            # at least one end on a wall
    if wall_dir is not None:
        # glazing follows the building's wall directions (tags and symbols drawn at other angles
        # are not glazing); wall_dir: the plan's dominant wall directions in degrees
        for i, (a, b, _) in enumerate(runs):
            ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
            if keep[i] and not any(min(abs(ang - d) % 180.0, 180.0 - abs(ang - d) % 180.0) <= 5.0 for d in wall_dir):
                keep[i] = False
    changed = True
    while changed:                                         # the other end: wall, or a kept run's end
        changed = False
        for i, (a, b, _) in enumerate(runs):
            if not keep[i]:
                continue
            ok = []
            for q, on in ((a, ends_wall[i][0]), (b, ends_wall[i][1])):
                ok.append(on or any(keep[j] and j != i and min(math.dist(q, runs[j][0]), math.dist(q, runs[j][1])) <= t
                                    for j in range(len(runs))))
            if not all(ok):
                keep[i] = False
                changed = True
    return [{"p0": a, "p1": b, "kind": "glazing", "evidence": f"{n} parallel lines across a wall gap"}
            for (a, b, n), k in zip(runs, keep) if k]


# ---------------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------------

def build_layer(paths, image_shape, text_boxes=(), image: np.ndarray | None = None) -> SemanticLayer:
    """`image` (optional): the rendered page. When given, gaps between the layer's walls are also
    closed where the page draws an opening symbol in them (engine.plan gap finder: sliding panels,
    bifold leaves, window lines, swing arcs), in addition to the typed doors and glazing."""
    t_start = time.perf_counter()
    shape = tuple(image_shape[:2])
    layer = SemanticLayer(shape=shape)
    for b in text_boxes:
        layer.elements.append(Element("TEXT", [(b.x, b.y), (b.x + b.width, b.y + b.height)], "text-layer",
                                      confidence=0.99, evidence=f"text layer '{b.text}'"))
    if not paths:
        layer.reason = "no vector drawing on this page"
        layer.seconds = time.perf_counter() - t_start
        return layer

    by_pen: dict = defaultdict(list)
    for p in paths:
        by_pen[p.pen].append(p)
    amount = {pen: sum(p.length() * max(1.0, p.width) for p in ps) for pen, ps in by_pen.items()}
    small = (int(shape[0] * PEN_SCALE), int(shape[1] * PEN_SCALE))
    tests = []
    pens = sorted(amount, key=lambda k: -amount[k])[:PEN_CANDIDATES]
    # each pen is reconstructed on its own: independent tests, run concurrently, read in pen order
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(pens)))) as pool:
        results = list(pool.map(lambda pen: _wall_test(by_pen[pen], small), pens))
    for pen, (cover, model) in zip(pens, results):
        walls = sum(w.style != "thin" for w in model.walls) if model else 0
        tests.append((pen, cover, walls))
        layer.pen_tests.append({"pen": list(pen), "wall_cover": round(cover, 2), "walls": walls})
    # A building's walls form a network of many walls. Frames, title-block rules, site outlines and
    # stacked dimension strings can also pair up, but as a handful of long lines: among the pens
    # that reconstruct as walls, keep the richest network and pens comparable to it.
    passing = [(pen, cover, walls) for pen, cover, walls in tests if cover >= WALL_COVER and walls >= MIN_WALLS]
    richest = max((walls for _, _, walls in passing), default=0)
    wall_pens = [pen for pen, _, walls in passing if walls >= RICH_SHARE * richest]
    wall_segs: set = set()
    if wall_pens:
        cover, model = _wall_test([p for pen in wall_pens for p in by_pen[pen]], small)
        band, t_wall = _wall_band(model, shape)
        if not band.any() or t_wall <= 0:
            layer.reason = "the wall pens did not reconstruct as walls together"
            layer.seconds = time.perf_counter() - t_start
            return layer
        wall_paths = [p for pen in wall_pens for p in by_pen[pen]]
        layer.wall_body = _wall_body(wall_paths, shape, band, t_wall, model)
        # the walls as drawn: reconstructed bands, plus the wall pen's own ink and wall bodies (short
        # jamb stubs and piers the reduced-resolution reconstruction does not keep)
        band = band | (_render_paths(wall_paths, shape, 1.0) < 128) | layer.wall_body
        layer.reason = f"walls are the pen(s) that reconstruct as walls on their own (cover {cover:.2f})"
    else:
        poche = _poche(paths, shape)
        if poche is None:
            best = max(tests, key=lambda t: t[1]) if tests else (None, 0.0, 0)
            layer.reason = (f"no pen draws a wall network on its own (best wall cover {best[1]:.2f}, {best[2]} walls) "
                            "and no wall poché (hatched wall bodies) was found - not simplified")
            layer.elements += [e for p in paths for e in _path_elements(p, "OTHER", 0.0, "untyped")]
            layer.seconds = time.perf_counter() - t_start
            return layer
        band, t_wall, wall_segs = poche
        layer.wall_body = band.copy()
        layer.reason = (f"walls are the plan's poché: hatched wall bodies ({len(wall_segs)} hatch and outline "
                        f"segments, thickness {t_wall:.0f}px)")
    band_dist = cv2.distanceTransform((~band).astype(np.uint8), cv2.DIST_L2, 3)
    layer.applicable = True
    layer.wall_pens = wall_pens
    layer.wall_thickness = t_wall

    arcs, doors = _doors(paths, set(wall_pens), band_dist, t_wall)
    door_segs = {(arcs[k][0], arcs[k][1]) for _, _, ks in doors for k in ks}
    hinges = [(c, r) for c, r, _ in doors]
    grid: dict = defaultdict(list)
    for i, (c, r) in enumerate(hinges):
        grid[(int(c[0] // 32), int(c[1] // 32))].append(i)

    def leaf_of(a, b):
        for p, q in ((a, b), (b, a)):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for i in grid.get((int(p[0] // 32) + dx, int(p[1] // 32) + dy), []):
                        c, r = hinges[i]
                        if math.dist(p, c) <= max(4.0, 0.08 * r) and 0.75 * r <= math.dist(p, q) <= 1.25 * r:
                            return r
        return None

    # closures: each door closed from its hinge to the arc end at the opposite jamb. The open leaf
    # rests along a wall; the closed leaf spans the opening, i.e. it crosses a gap in the walls.
    near_band = band_dist <= 2.0
    hh, ww = band.shape

    def band_cover(a, b):
        pts = [(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f) for f in np.linspace(0.15, 0.85, 15)]
        return float(np.mean([near_band[min(hh - 1, max(0, int(round(y)))), min(ww - 1, max(0, int(round(x))))] for x, y in pts]))
    for c, r, ks in doors:
        ends = [pt for k in ks for pt in (paths[arcs[k][0]].segs[arcs[k][1]][1], paths[arcs[k][0]].segs[arcs[k][1]][2])]
        ends = [e for e in ends if 0.6 * r <= math.dist(c, e) <= 1.6 * r and not _alongside(c, e, band, t_wall)]
        if not ends:
            continue

        def end_dist(p):
            x, y = int(round(p[0])), int(round(p[1]))
            return band_dist[y, x] if 0 <= x < ww and 0 <= y < hh else 1e9
        # the closed leaf lands on the opposite jamb; the open leaf ends in the room
        landing = [e for e in ends if end_dist(e) <= 0.6 * t_wall + 4]
        e = min(landing, key=end_dist) if landing else min(ends, key=lambda e: band_cover(c, e))
        if band_cover(c, e) <= 0.3:
            layer.closures.append({"p0": c, "p1": e, "kind": "door", "evidence": f"swing arc r {r:.0f}px"})
    # glazing may span from a wall to a closed door (door with side lights) as well as wall to wall
    closed = band.copy().astype(np.uint8)
    for cl in layer.closures:
        cv2.line(closed, tuple(int(round(v)) for v in cl["p0"]), tuple(int(round(v)) for v in cl["p1"]), 1, max(3, int(round(t_wall))))
    closed_dist = cv2.distanceTransform((closed == 0).astype(np.uint8), cv2.DIST_L2, 3)
    wall_lines = [(seg[1], seg[2]) for pi, p in enumerate(paths) for si, seg in enumerate(p.segs)
                  if seg[0] == "L" and (p.pen in wall_pens or (pi, si) in wall_segs)]
    wall_dir = _dominant_directions(wall_lines, t_wall)
    layer.closures += _glazing(paths, set(wall_pens), door_segs, band, closed_dist, t_wall, wall_dir)

    dim_boxes = [b for b in text_boxes if b.kind == "DIMENSION"]
    for pi, p in enumerate(paths):
        if p.pen in wall_pens:
            layer.elements += _path_elements(p, "WALL", 0.9, f"wall pen {p.pen[1]:.2f}px")
            continue
        if wall_segs and p.kind == "stroke" and any((pi, si) in wall_segs for si in range(len(p.segs))):
            rest = []
            for si, seg in enumerate(p.segs):
                if (pi, si) in wall_segs:
                    layer.elements.append(Element("WALL", _polyline(seg), "vector", p.width, False, 0.8, "poché hatch / outline"))
                else:
                    rest.append(si)
            if not rest:
                continue
        if p.kind == "fill":
            poly = np.array([q for seg in p.segs for q in _polyline(seg)], np.int32)
            if len(poly) >= 3:
                x, y, w, h = cv2.boundingRect(poly)
                inside = band[max(0, y):y + h, max(0, x):x + w]
                if inside.size and inside.mean() >= 0.6 and max(w, h) <= 6 * t_wall:
                    layer.elements += _path_elements(p, "COLUMN", 0.6, "filled shape in a wall band")
                    continue
            layer.elements += _path_elements(p, "OTHER", 0.0, "fill")
            continue
        for si, seg in enumerate(p.segs):
            if (pi, si) in wall_segs:
                continue
            pts = _polyline(seg)
            if (pi, si) in door_segs:
                layer.elements.append(Element("DOOR", pts, "vector", p.width, False, 0.8, "swing arc hinged at a wall"))
                continue
            if seg[0] == "L":
                a, b = seg[1], seg[2]
                r = leaf_of(a, b) if hinges else None
                if r:
                    layer.elements.append(Element("DOOR", pts, "vector", p.width, False, 0.7, f"leaf from a door hinge (r {r:.0f}px)"))
                    continue
                if _is_window(a, b, band, band_dist, t_wall):
                    layer.elements.append(Element("WINDOW", pts, "vector", p.width, False, 0.6, "line in a wall line between two wall ends"))
                    continue
                if dim_boxes and _along_dimension(a, b, dim_boxes):
                    layer.elements.append(Element("DIMENSION", pts, "vector", p.width, False, 0.6, "runs along a dimension string"))
                    continue
            layer.elements.append(Element("OTHER", pts, "vector", p.width, False, 0.0, "untyped"))
    _drop_isolated_walls(layer, band, t_wall)
    if image is not None:
        # safety: the typed walls must lie on the page's own ink (vectors mapped correctly, no
        # clipping or patterns that the SVG conversion renders differently)
        page_ink = cv2.dilate((cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 160).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        walls = np.full(shape, 255, np.uint8)
        for e in layer.elements:
            if e.type == "WALL":
                _draw(walls, e, 0)
        wall_ink = walls < 128
        layer.alignment = float((wall_ink & page_ink).sum() / max(1, wall_ink.sum()))
        if layer.alignment < MIN_ALIGNMENT:
            layer.applicable = False
            layer.reason = (f"the PDF's vector walls do not match the rendered page (alignment {layer.alignment:.2f}); "
                            "not simplified")
            layer.closures = []
            layer.seconds = time.perf_counter() - t_start
            return layer
        layer.closures += _gap_closures(layer, image)
    layer.seconds = time.perf_counter() - t_start
    return layer


def _drop_isolated_walls(layer: SemanticLayer, band: np.ndarray, t: float, share: float = 0.05) -> None:
    """A building's walls form a network. Wall drawing far from every large network - wall-type
    samples in a legend, details, key plans - is not this plan's structure: retyped OTHER (kept in
    the representation, left out of the cleaned plan). Several buildings of similar size all stay."""
    h, w = band.shape
    net = band.astype(np.uint8)
    for c in layer.closures:                  # openings connect the walls they close
        cv2.line(net, tuple(int(round(v)) for v in c["p0"]), tuple(int(round(v)) for v in c["p1"]), 1, max(3, int(round(t))))
    k = max(3, int(round(3 * t)) | 1)
    grown = cv2.dilate(net, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grown)
    if n <= 2:
        return
    ink = np.bincount(lab[band], minlength=n)
    keep = ink >= share * ink[1:].max()
    keep[0] = False
    # a small cluster near a kept network belongs to it (piers, stubs between wide glazing)
    big = keep[lab]
    # 'within reach of a kept network': the structuring element is symmetric, so dilating each small
    # cluster in its own window and testing it against the kept networks is the same test as
    # dilating the (page-sized) networks - without a 20t-wide dilation of the whole page
    k2 = int(20 * t) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    r = k2 // 2
    # the Euclidean distance to the kept networks decides every cluster clearly inside or outside the
    # (discrete, near-circular) reach; only a cluster on the boundary is tested with the element itself
    dist = cv2.distanceTransform((~big).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    for i in range(1, n):
        if keep[i]:
            continue
        x, y, bw, bh = stats[i, :4]
        nearest = float(dist[y:y + bh, x:x + bw][lab[y:y + bh, x:x + bw] == i].min())
        if nearest <= r - 1.5:
            keep[i] = True
            continue
        if nearest > r + 1.5:
            continue
        x0, y0, x1, y1 = max(0, x - r), max(0, y - r), min(w, x + bw + r), min(h, y + bh + r)
        own = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        pad = cv2.copyMakeBorder(own, r, r, r, r, cv2.BORDER_CONSTANT, value=0)
        grown_i = cv2.dilate(pad, se)[r:-r, r:-r] if r else cv2.dilate(own, se)
        if (grown_i.astype(bool) & big[y0:y1, x0:x1]).any():
            keep[i] = True
    for e in layer.elements:
        if e.type not in ("WALL", "COLUMN"):
            continue
        pts = np.array(e.geometry, float)
        cx, cy = int(np.clip(pts[:, 0].mean(), 0, w - 1)), int(np.clip(pts[:, 1].mean(), 0, h - 1))
        if not keep[lab[cy, cx]]:
            e.type, e.evidence = "OTHER", "isolated wall-like drawing (legend, detail)"
    if layer.wall_body is not None:
        layer.wall_body &= keep[lab]
    def at(p):
        return lab[int(np.clip(p[1], 0, h - 1)), int(np.clip(p[0], 0, w - 1))]
    # an opening belongs to the network its jambs (ends) sit on; a doorway's middle is paper
    layer.closures = [c for c in layer.closures if not any(at(p) > 0 and not keep[at(p)] for p in (c["p0"], c["p1"]))]


def _gap_closures(layer: SemanticLayer, image: np.ndarray) -> list:
    """Openings the typed symbols did not explain: gaps between the layer's wall ends that hold an
    opening symbol on the page. Gaps without a symbol (passages) stay open."""
    from engine.plan.reconstruct import reconstruct

    model = reconstruct(layer.structural_image(closures=True), evidence_image=image)
    out = []
    t = layer.wall_thickness
    radii = [float(np.hypot(c["p1"][0] - c["p0"][0], c["p1"][1] - c["p0"][1])) for c in layer.closures if c["kind"] == "door"]
    door_w = float(np.median(radii)) if radii else None       # the plan's own door width, from its swing arcs
    for o in model.openings:
        if o.kind not in ("door", "window"):
            continue
        why = (o.provenance[0] if o.provenance else "").lower()
        if o.kind == "door" and door_w:
            strong = ("double" in why or "sliding" in why
                      or ("swing arc" in why and float(why.split("(")[1].split(")")[0]) >= 0.7))
            within_wall = any("collinear" in p_ for p_ in o.provenance)
            leaf_limit = (2.2 if within_wall else 1.3) * door_w          # bifold / pair of leaves in one wall line
            limit = 2.2 * door_w if strong else (leaf_limit if "leaf" in why else 0.0)
            if "sliding" in why and within_wall:
                limit = 3.5 * door_w          # bypass / multi-panel sliders span most of a wall
            if o.width > limit:
                continue                      # weak symbol for its width: counters, stools, furniture
        mid = ((o.p0[0] + o.p1[0]) / 2, (o.p0[1] + o.p1[1]) / 2)
        if any(_seg_point_dist(mid, c["p0"], c["p1"]) <= t for c in layer.closures + out):
            continue
        out.append({"p0": o.p0, "p1": o.p1, "kind": f"gap-{o.kind}", "evidence": "; ".join(o.provenance)})
    return out


def _seg_point_dist(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy or 1e-9
    f = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(p[0] - a[0] - f * dx, p[1] - a[1] - f * dy)


def _along_dimension(a, b, boxes) -> bool:
    L = math.dist(a, b)
    if L < 4:
        return False
    horizontal = abs(b[0] - a[0]) >= abs(b[1] - a[1])
    for box in boxes:
        long_side = max(box.width, box.height)
        if horizontal != (box.width >= box.height):
            continue
        cx, cy = box.x + box.width / 2, box.y + box.height / 2
        th = min(box.width, box.height)
        if horizontal:
            if min(a[0], b[0]) - th <= cx <= max(a[0], b[0]) + th and abs((a[1] + b[1]) / 2 - cy) <= 1.5 * th and L >= long_side:
                return True
        else:
            if min(a[1], b[1]) - th <= cy <= max(a[1], b[1]) + th and abs((a[0] + b[0]) / 2 - cx) <= 1.5 * th and L >= long_side:
                return True
    return False
