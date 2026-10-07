"""Seeded synthetic floor-plan generator with exact ground truth.

Layouts are built in centimetres: a rectangular footprint is partitioned into
rooms (with occasional corridors), optionally made L-shaped by dropping a
corner room, and optionally given an irregular (L-shaped) room by merging two
rooms. Walls are derived from the room rectangles (interior = between two
rooms, exterior = one side outside). Doors connect rooms along a spanning tree
of the adjacency graph, plus exterior doors, windows and bare passages.

Everything is then rendered at a chosen pixel scale with a chosen symbol style
and optional drawing noise. Ground truth (openings, wall mask, room label map,
room adjacency) is recorded exactly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np

DOOR_STYLES = ("swing", "leaf_only", "double", "outline", "sliding")
WINDOW_STYLES = ("triple", "double", "sample", "weak")


@dataclass
class Style:
    px_per_cm: float = 0.45
    line_px: int = 2
    ext_wall_cm: float = 25.0
    int_wall_cm: float = 12.0
    wall_gray: int = 0
    symbol_gray: int = 0
    door_styles: tuple = ("swing",)
    window_styles: tuple = ("triple",)
    antialias: bool = True
    # noise
    text: bool = False
    furniture: bool = False
    dimension_lines: bool = False
    speckle: float = 0.0        # fraction of pixels flipped
    blur: float = 0.0           # gaussian sigma in px
    jpeg: int = 0               # JPEG quality (0 = off)
    background: int = 255
    rotate_deg: float = 0.0
    # stress elements (benchmark N.1); all off by default and drawn with their own RNG, so
    # existing families generate exactly the same plans
    dark_furniture: float = 0.0     # probability per room of solid dark furniture blocks
    furniture_tone: int = 40        # gray level of those blocks
    furniture_touch: float = 0.5    # share of blocks placed against a wall
    markers: int = 0                # coloured annotation dots inside rooms
    hatch: float = 0.0              # probability per room of a tile-hatched floor patch
    stairs: int = 0                 # stair runs (closely spaced parallel treads)
    # How the wall band is drawn (benchmark.wall_styles). Layout, openings and ground truth do
    # not depend on it and it uses no randomness, so the same seed gives the same plan in
    # every style. "filled" is the original rendering.
    wall_render: str = "filled"     # filled | hollow | hatched | thin


@dataclass
class PlanSpec:
    seed: int
    footprint_cm: tuple[float, float] = (1300.0, 950.0)
    min_room_cm: float = 280.0
    corridor_prob: float = 0.25
    drop_corner_prob: float = 0.35
    merge_room_prob: float = 0.3
    windows_per_room: tuple[int, int] = (1, 2)
    exterior_doors: int = 1
    passage_prob: float = 0.15
    adjacent_window_prob: float = 0.25
    door_width_cm: tuple[float, float] = (80.0, 95.0)
    window_width_cm: tuple[float, float] = (70.0, 220.0)
    double_door_width_cm: tuple[float, float] = (130.0, 170.0)
    corner_door_prob: float = 0.25
    style: Style = field(default_factory=Style)


@dataclass
class GTOpening:
    id: str
    kind: str          # door | window | opening
    style: str
    p0: tuple[float, float]
    p1: tuple[float, float]
    width_px: float
    wall_thickness_px: float
    exterior: bool
    connects: tuple    # (room_id, room_id|'exterior')

    def to_dict(self):
        return {
            "id": self.id, "kind": self.kind, "style": self.style,
            "p0": [round(self.p0[0], 1), round(self.p0[1], 1)], "p1": [round(self.p1[0], 1), round(self.p1[1], 1)],
            "width_px": round(self.width_px, 1), "wall_thickness_px": round(self.wall_thickness_px, 1),
            "exterior": self.exterior, "connects": list(self.connects),
        }


@dataclass
class Plan:
    image: np.ndarray
    openings: list[GTOpening]
    wall_mask: np.ndarray        # uint8 0/255, walls without openings
    room_labels: np.ndarray      # int32, 0 = exterior/wall, k = room k
    room_names: dict[int, str]
    adjacency: set               # {(a, b)} rooms connected by a door/passage, b may be 0 = exterior
    px_per_cm: float
    spec: PlanSpec
    # Wall axes in pixels: (p0, p1, thickness, extension at p0, extension at p1, room_a, room_b)
    wall_pieces: list = field(default_factory=list)

    @property
    def size(self):
        return self.image.shape[1], self.image.shape[0]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _partition(rng: random.Random, spec: PlanSpec):
    W, H = spec.footprint_cm
    leaves = []
    corridor_flags = []

    def split(x0, y0, x1, y1, depth):
        w, h = x1 - x0, y1 - y0
        m = spec.min_room_cm
        can_v = w >= 2 * m
        can_h = h >= 2 * m
        if depth >= 4 or not (can_v or can_h) or (depth >= 2 and rng.random() < 0.25):
            leaves.append((x0, y0, x1, y1))
            corridor_flags.append(False)
            return
        vertical = can_v and (not can_h or w >= h)
        if rng.random() < spec.corridor_prob and (w if vertical else h) >= 2 * m + 120:
            cw = rng.uniform(110, 150)
            pos = rng.uniform(m, (w if vertical else h) - m - cw)
            if vertical:
                split(x0, y0, x0 + pos, y1, depth + 1)
                leaves.append((x0 + pos, y0, x0 + pos + cw, y1)); corridor_flags.append(True)
                split(x0 + pos + cw, y0, x1, y1, depth + 1)
            else:
                split(x0, y0, x1, y0 + pos, depth + 1)
                leaves.append((x0, y0 + pos, x1, y0 + pos + cw)); corridor_flags.append(True)
                split(x0, y0 + pos + cw, x1, y1, depth + 1)
            return
        pos = rng.uniform(m, (w if vertical else h) - m)
        pos = round(pos / 5) * 5
        if vertical:
            split(x0, y0, x0 + pos, y1, depth + 1)
            split(x0 + pos, y0, x1, y1, depth + 1)
        else:
            split(x0, y0, x1, y0 + pos, depth + 1)
            split(x0, y0 + pos, x1, y1, depth + 1)

    split(0.0, 0.0, W, H, 0)
    return leaves, corridor_flags


def _layout(rng: random.Random, spec: PlanSpec):
    W, H = spec.footprint_cm
    leaves, corridors = _partition(rng, spec)
    room_of = list(range(1, len(leaves) + 1))  # leaf index -> room id

    # Drop one corner room → L-shaped footprint.
    if len(leaves) >= 5 and rng.random() < spec.drop_corner_prob:
        corner = [i for i, (x0, y0, x1, y1) in enumerate(leaves)
                  if not corridors[i] and (x0 == 0 or x1 == W) and (y0 == 0 or y1 == H)]
        if corner:
            room_of[rng.choice(corner)] = 0

    # Merge two partially-adjacent rooms into one irregular (L-shaped) room.
    if rng.random() < spec.merge_room_prob:
        pairs = []
        for i, a in enumerate(leaves):
            for j, b in enumerate(leaves):
                if j <= i or room_of[i] == 0 or room_of[j] == 0 or corridors[i] or corridors[j]:
                    continue
                shared = _shared_edge(a, b)
                if shared and shared[2] - shared[1] < min(_span(a, shared[3]), _span(b, shared[3])) - 50:
                    pairs.append((i, j))
        if pairs:
            i, j = rng.choice(pairs)
            room_of[j] = room_of[i]
    return leaves, room_of, corridors


def _span(rect, orientation):
    x0, y0, x1, y1 = rect
    return (x1 - x0) if orientation == "h" else (y1 - y0)


def _shared_edge(a, b):
    """(coord, lo, hi, orientation) of the edge shared by two rectangles, if any."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if ay1 == by0 or ay0 == by1:
        lo, hi = max(ax0, bx0), min(ax1, bx1)
        if hi - lo > 1:
            return (ay1 if ay1 == by0 else ay0, lo, hi, "h")
    if ax1 == bx0 or ax0 == bx1:
        lo, hi = max(ay0, by0), min(ay1, by1)
        if hi - lo > 1:
            return (ax1 if ax1 == bx0 else ax0, lo, hi, "v")
    return None


def _wall_pieces(leaves, room_of):
    """Elementary wall pieces: (orientation, coord, lo, hi, room_a, room_b) with 0 = outside."""
    pieces = []
    for orient in ("h", "v"):
        lines = {}
        for i, (x0, y0, x1, y1) in enumerate(leaves):
            if orient == "h":
                lines.setdefault(y0, []).append((x0, x1)); lines.setdefault(y1, []).append((x0, x1))
            else:
                lines.setdefault(x0, []).append((y0, y1)); lines.setdefault(x1, []).append((y0, y1))
        for c, intervals in lines.items():
            cuts = sorted({v for iv in intervals for v in iv})
            for lo, hi in zip(cuts, cuts[1:]):
                mid = (lo + hi) / 2
                side_a = side_b = 0  # a = above/left, b = below/right
                for i, (x0, y0, x1, y1) in enumerate(leaves):
                    if orient == "h" and x0 <= mid <= x1:
                        if y1 == c: side_a = room_of[i]
                        if y0 == c: side_b = room_of[i]
                    if orient == "v" and y0 <= mid <= y1:
                        if x1 == c: side_a = room_of[i]
                        if x0 == c: side_b = room_of[i]
                if side_a == side_b:
                    continue  # no wall (outside on both sides, or merged room)
                pieces.append((orient, c, lo, hi, side_a, side_b))
    return pieces


# ---------------------------------------------------------------------------
# Rendering helpers (all in pixel space)
# ---------------------------------------------------------------------------

class Canvas:
    def __init__(self, width, height, style: Style, rng: random.Random):
        self.img = np.full((height, width), style.background, dtype=np.uint8)
        self.style = style
        self.rng = rng
        self.lt = cv2.LINE_AA if style.antialias else cv2.LINE_8

    def line(self, p, q, gray=None, width=None):
        gray = self.style.symbol_gray if gray is None else gray
        width = self.style.line_px if width is None else width
        cv2.line(self.img, _ip(p), _ip(q), int(gray), int(max(1, width)), self.lt)

    def poly(self, points, gray):
        cv2.fillPoly(self.img, [np.round(np.asarray(points)).astype(np.int32)], int(gray), self.lt)

    def arc(self, center, radius, start_deg, end_deg, gray=None, width=None, dashed=False):
        gray = self.style.symbol_gray if gray is None else gray
        width = self.style.line_px if width is None else width
        steps = max(12, int(abs(end_deg - start_deg) / 3))
        pts = [(center[0] + radius * math.cos(math.radians(a)), center[1] + radius * math.sin(math.radians(a)))
               for a in np.linspace(start_deg, end_deg, steps)]
        for k, (p, q) in enumerate(zip(pts, pts[1:])):
            if dashed and k % 3 == 2:
                continue
            self.line(p, q, gray, width)


def _ip(p):
    return (int(round(p[0])), int(round(p[1])))


def _frame(p0, p1):
    """Unit tangent and normal for a wall axis segment."""
    d = np.subtract(p1, p0).astype(float)
    t = d / max(1e-9, np.linalg.norm(d))
    n = np.array([-t[1], t[0]])
    return t, n


def _at(p0, t, n, u, v):
    return (p0[0] + t[0] * u + n[0] * v, p0[1] + t[1] * u + n[1] * v)


def _band(p0, t, n, u0, u1, half):
    return [_at(p0, t, n, u0, -half), _at(p0, t, n, u1, -half), _at(p0, t, n, u1, half), _at(p0, t, n, u0, half)]


def draw_door(cv: Canvas, p0, p1, thick, style, rng, side=None):
    t, n = _frame(p0, p1)
    w = float(np.linalg.norm(np.subtract(p1, p0)))
    half = thick / 2
    lw = cv.style.line_px
    side = side if side is not None else rng.choice((-1, 1))
    hinge_at_start = rng.random() < 0.5
    if style in ("swing", "leaf_only"):
        u_h = 0.0 if hinge_at_start else w
        hinge = _at(p0, t, n, u_h, side * half)
        tip = _at(p0, t, n, u_h, side * (half + w))
        leaf_w = max(lw + 1, thick * 0.18)
        cv.poly([_at(p0, t, n, u_h - leaf_w / 2, side * half), _at(p0, t, n, u_h + leaf_w / 2, side * half),
                 _at(p0, t, n, u_h + leaf_w / 2, side * (half + w)), _at(p0, t, n, u_h - leaf_w / 2, side * (half + w))],
                cv.style.symbol_gray)
        if style == "swing":
            a_tip = math.degrees(math.atan2(tip[1] - hinge[1], tip[0] - hinge[0]))
            other = _at(p0, t, n, w - u_h, side * half)
            a_other = math.degrees(math.atan2(other[1] - hinge[1], other[0] - hinge[0]))
            delta = (a_other - a_tip + 540) % 360 - 180
            cv.arc(hinge, w, a_tip, a_tip + delta, width=max(1, lw - 1))
        # jamb lines
        cv.line(_at(p0, t, n, 0, -half), _at(p0, t, n, 0, half), width=max(1, lw - 1))
        cv.line(_at(p0, t, n, w, -half), _at(p0, t, n, w, half), width=max(1, lw - 1))
    elif style == "double":
        for u_h, sgn in ((0.0, 1), (w, -1)):
            hinge = _at(p0, t, n, u_h, side * half)
            leaf_w = max(lw + 1, thick * 0.15)
            cv.poly([_at(p0, t, n, u_h - leaf_w / 2, side * half), _at(p0, t, n, u_h + leaf_w / 2, side * half),
                     _at(p0, t, n, u_h + leaf_w / 2, side * (half + w / 2)), _at(p0, t, n, u_h - leaf_w / 2, side * (half + w / 2))],
                    cv.style.symbol_gray)
            tip = _at(p0, t, n, u_h, side * (half + w / 2))
            mid = _at(p0, t, n, w / 2, side * half)
            a_tip = math.degrees(math.atan2(tip[1] - hinge[1], tip[0] - hinge[0]))
            a_mid = math.degrees(math.atan2(mid[1] - hinge[1], mid[0] - hinge[0]))
            delta = (a_mid - a_tip + 540) % 360 - 180
            cv.arc(hinge, w / 2, a_tip, a_tip + delta, width=max(1, lw - 1))
    elif style == "outline":
        gray = rng.choice((0, 120, 170))
        for v in (-half, half):
            cv.line(_at(p0, t, n, 0, v), _at(p0, t, n, w, v), gray, max(1, lw - 1))
        cv.line(_at(p0, t, n, 0, -half), _at(p0, t, n, 0, half), gray, max(1, lw - 1))
        cv.line(_at(p0, t, n, w, -half), _at(p0, t, n, w, half), gray, max(1, lw - 1))
    elif style == "sliding":
        pw = max(lw + 2, thick * 0.22)
        for (a, b), v in (((0.0, 0.56 * w), -thick * 0.18), ((0.44 * w, w), thick * 0.18)):
            corners = [_at(p0, t, n, a, v - pw / 2), _at(p0, t, n, b, v - pw / 2), _at(p0, t, n, b, v + pw / 2), _at(p0, t, n, a, v + pw / 2)]
            for k in range(4):
                cv.line(corners[k], corners[(k + 1) % 4], width=max(1, lw - 1))
        cv.line(_at(p0, t, n, 0, -half), _at(p0, t, n, 0, half), width=max(1, lw - 1))
        cv.line(_at(p0, t, n, w, -half), _at(p0, t, n, w, half), width=max(1, lw - 1))
    elif style == "garage":
        cv.line(_at(p0, t, n, 0, -half), _at(p0, t, n, w, -half), 150, max(1, lw - 1))


def draw_window(cv: Canvas, p0, p1, thick, style, rng):
    t, n = _frame(p0, p1)
    w = float(np.linalg.norm(np.subtract(p1, p0)))
    half = thick / 2
    lw = max(1, cv.style.line_px - 1)
    if style == "triple":
        for v in (-half, 0.0, half):
            cv.line(_at(p0, t, n, 0, v), _at(p0, t, n, w, v), width=lw)
        for u in (0.0, w):
            cv.line(_at(p0, t, n, u, -half), _at(p0, t, n, u, half), width=lw)
    elif style == "double":
        for v in (-half, half):
            cv.line(_at(p0, t, n, 0, v), _at(p0, t, n, w, v), width=lw)
        for v in (-thick / 6, thick / 6):
            cv.line(_at(p0, t, n, 0, v), _at(p0, t, n, w, v), width=lw)
        for u in (0.0, w):
            cv.line(_at(p0, t, n, u, -half), _at(p0, t, n, u, half), width=lw)
    elif style == "sample":
        for v in (-half, 0.0, half):
            cv.line(_at(p0, t, n, 0, v), _at(p0, t, n, w, v), width=lw)
        jw = max(2.0, thick * 0.12)
        for u0, u1 in ((0.0, jw), (w - jw, w)):
            cv.poly(_band(p0, t, n, u0, u1, half), cv.style.symbol_gray)
    elif style == "weak":
        gray = rng.choice((140, 170, 190))
        seg = max(6.0, w / rng.choice((5, 7, 9)))
        u = 0.0
        while u < w:
            cv.line(_at(p0, t, n, u, 0.0), _at(p0, t, n, min(w, u + seg * 0.75), 0.0), gray, lw)
            u += seg


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------

ROOM_NAMES = ("BEDROOM", "KITCHEN", "BATH", "OFFICE", "LIVING ROOM", "DINING", "CLOSET", "LAUNDRY", "STUDY", "DEN")


def generate(spec: PlanSpec) -> Plan:
    rng = random.Random(spec.seed)
    st = spec.style
    s = st.px_per_cm
    leaves, room_of, corridors = _layout(rng, spec)
    pieces = _wall_pieces(leaves, room_of)
    W, H = spec.footprint_cm
    margin = int(round(max(60, 140 * s)))
    width, height = int(round(W * s)) + 2 * margin, int(round(H * s)) + 2 * margin
    to_px = lambda x, y: (margin + x * s, margin + y * s)  # noqa: E731

    cv = Canvas(width, height, st, rng)
    wall_mask = np.zeros((height, width), np.uint8)
    t_ext, t_int = st.ext_wall_cm * s, st.int_wall_cm * s

    # Room label map (rooms filled first, walls carved out afterwards).
    labels = np.zeros((height, width), np.int32)
    for i, rect in enumerate(leaves):
        if room_of[i]:
            x0, y0 = to_px(rect[0], rect[1]); x1, y1 = to_px(rect[2], rect[3])
            labels[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))] = room_of[i]

    # Walls: one band per elementary piece, extended at both ends to close joints.
    def thickness_of(piece):
        return t_ext if (piece[4] == 0 or piece[5] == 0) else t_int

    def end_extension(orient, x, y):
        """Half the thickness of perpendicular walls through (x, y) — closes L/T joints flush."""
        best = 0.0
        for q in pieces:
            if q[0] == orient:
                continue
            if q[0] == "h" and q[1] == y and q[2] <= x <= q[3]:
                best = max(best, thickness_of(q) / 2)
            if q[0] == "v" and q[1] == x and q[2] <= y <= q[3]:
                best = max(best, thickness_of(q) / 2)
        return best

    piece_px = []
    wall_axes = []
    for piece in pieces:
        orient, c, lo, hi, a, b = piece
        thick = thickness_of(piece)
        if orient == "h":
            p0, p1 = to_px(lo, c), to_px(hi, c)
            e0, e1 = end_extension("h", lo, c), end_extension("h", hi, c)
        else:
            p0, p1 = to_px(c, lo), to_px(c, hi)
            e0, e1 = end_extension("v", c, lo), end_extension("v", c, hi)
        piece_px.append((orient, p0, p1, thick, a, b))
        wall_axes.append((p0, p1, thick, e0, e1, a, b))
        t, n = _frame(p0, p1)
        band = _band(p0, t, n, -e0, float(np.linalg.norm(np.subtract(p1, p0))) + e1, thick / 2)
        cv2.fillPoly(wall_mask, [np.round(np.asarray(band)).astype(np.int32)], 255, cv2.LINE_8)
    labels[wall_mask > 0] = 0
    _draw_walls(cv, wall_mask, wall_axes, st, t_int)

    # --- openings -------------------------------------------------------
    openings: list[GTOpening] = []
    occupied: dict[int, list[tuple[float, float]]] = {}
    adjacency = set()

    def place(k, width_cm, kind, style, corner=False):
        orient, p0, p1, thick, a, b = piece_px[k]
        length = float(np.linalg.norm(np.subtract(p1, p0)))
        wpx = width_cm * s
        end_margin = (t_ext / 2 + (2 * s if corner else 25 * s))
        free = length - 2 * end_margin - wpx
        if free < 0:
            return False
        for _ in range(12):
            if corner:
                u0 = end_margin if rng.random() < 0.5 else length - end_margin - wpx
            else:
                u0 = end_margin + rng.uniform(0, free)
            if all(u0 + wpx + 30 * s < lo or u0 > hi + 30 * s for lo, hi in occupied.get(k, [])):
                break
        else:
            return False
        occupied.setdefault(k, []).append((u0, u0 + wpx))
        t, n = _frame(p0, p1)
        q0, q1 = _at(p0, t, n, u0, 0), _at(p0, t, n, u0 + wpx, 0)
        # clear the wall band, then draw the symbol
        _clear_opening(cv, p0, t, n, u0, u0 + wpx, thick)
        if kind == "door":
            # +v points to room a for vertical pieces and to room b for horizontal ones;
            # exterior doors swing inwards.
            side_a = -1 if orient == "h" else 1
            side = -side_a if a == 0 else (side_a if b == 0 else None)
            draw_door(cv, q0, q1, thick, style, rng, side=side)
        elif kind == "window":
            draw_window(cv, q0, q1, thick, style, rng)
        oid = f"{kind[0].upper()}{sum(o.kind == kind for o in openings) + 1}"
        connects = (a or "exterior", b or "exterior")
        openings.append(GTOpening(oid, kind, style, q0, q1, wpx, thick, a == 0 or b == 0, connects))
        adjacency.add(tuple(sorted((a, b))))  # any opening connects its two sides (0 = exterior)
        return True

    interior = [k for k, (_, p0, p1, th, a, b) in enumerate(piece_px) if a and b]
    exterior = [k for k, (_, p0, p1, th, a, b) in enumerate(piece_px) if not (a and b)]

    # Doors along a spanning tree of rooms (+ some extra), on interior pieces.
    rooms = sorted({r for r in room_of if r})
    by_pair: dict[tuple, list[int]] = {}
    for k in interior:
        _, _, _, _, a, b = piece_px[k]
        by_pair.setdefault(tuple(sorted((a, b))), []).append(k)
    connected = {rooms[0]}
    pairs = list(by_pair)
    rng.shuffle(pairs)
    while len(connected) < len(rooms):
        progress = False
        for pair in pairs:
            if (pair[0] in connected) != (pair[1] in connected):
                k = max(by_pair[pair], key=lambda i: np.linalg.norm(np.subtract(piece_px[i][2], piece_px[i][1])))
                passage = rng.random() < spec.passage_prob
                if passage:
                    ok = place(k, rng.uniform(110, 170), "opening", "bare")
                else:
                    style = rng.choice(st.door_styles)
                    wcm = rng.uniform(*spec.double_door_width_cm) if style in ("double", "sliding") else rng.uniform(*spec.door_width_cm)
                    ok = place(k, wcm, "door", style, corner=rng.random() < spec.corner_door_prob)
                if ok:
                    connected.update(pair)
                    progress = True
        if not progress:
            break

    # Exterior doors.
    ext_long = sorted(exterior, key=lambda i: -np.linalg.norm(np.subtract(piece_px[i][2], piece_px[i][1])))
    for k in ext_long[: spec.exterior_doors]:
        style = rng.choice(st.door_styles)
        wcm = rng.uniform(*spec.double_door_width_cm) if style in ("double", "sliding") else rng.uniform(*spec.door_width_cm)
        place(k, wcm, "door", style)

    # Windows on exterior pieces of each room.
    for room in rooms:
        candidates = [k for k in exterior if room in (piece_px[k][4], piece_px[k][5])]
        rng.shuffle(candidates)
        for k in candidates[: rng.randint(*spec.windows_per_room)]:
            style = rng.choice(st.window_styles)
            if rng.random() < spec.adjacent_window_prob:
                wcm = rng.uniform(70, 110)
                if place(k, wcm, "window", style):
                    last = openings[-1]
                    orient, p0, p1, thick, a, b = piece_px[k]
                    t, n = _frame(p0, p1)
                    u_end = float(np.dot(np.subtract(last.p1, p0), t))
                    pier = rng.uniform(20, 40) * s
                    u0, u1 = u_end + pier, u_end + pier + wcm * s
                    length = float(np.linalg.norm(np.subtract(p1, p0)))
                    if u1 < length - t_ext / 2 - 10 * s and all(u1 + 5 * s < lo or u0 > hi + 5 * s for lo, hi in occupied[k][:-1]):
                        occupied[k].append((u0, u1))
                        q0, q1 = _at(p0, t, n, u0, 0), _at(p0, t, n, u1, 0)
                        _clear_opening(cv, p0, t, n, u0, u1, thick)
                        draw_window(cv, q0, q1, thick, style, rng)
                        openings.append(GTOpening(f"W{sum(o.kind == 'window' for o in openings) + 1}", "window", style,
                                                  q0, q1, wcm * s, thick, True, last.connects))
            else:
                place(k, rng.uniform(*spec.window_width_cm), "window", style)

    # GT wall mask excludes opening gaps.
    for o in openings:
        t, n = _frame(o.p0, o.p1)
        band = _band(o.p0, t, n, 0, o.width_px, o.wall_thickness_px / 2 + 1.5)
        cv2.fillPoly(wall_mask, [np.round(np.asarray(band)).astype(np.int32)], 0, cv2.LINE_8)

    names = {}
    for i, r in enumerate(room_of):
        if r and r not in names:
            names[r] = "CORRIDOR" if corridors[i] else rng.choice(ROOM_NAMES)
    _noise(cv, rng, leaves, room_of, names, to_px, s, spec)
    srng = random.Random(spec.seed * 7919 + 17)
    _stress(cv, srng, leaves, room_of, to_px, s, spec)
    image = cv.img
    if st.rotate_deg:
        image, wall_mask, labels, openings = _rotate(image, wall_mask, labels, openings, st.rotate_deg, st.background)
    image = _degrade(image, st, rng)
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if st.markers:
        _markers(bgr, srng, leaves, room_of, to_px, s, st)
    return Plan(bgr, openings, wall_mask, labels, names, adjacency, s, spec, wall_axes)


# ---------------------------------------------------------------------------
# Wall drawing styles (benchmark.wall_styles): deterministic, no randomness
# ---------------------------------------------------------------------------

WALL_RENDERS = ("filled", "hollow", "hatched", "thin")


def _draw_walls(cv: Canvas, wall_mask: np.ndarray, axes: list, st: Style, t_int: float) -> None:
    """Draw the wall band in the style's convention. The band itself (wall_mask) is the same
    ground truth in every style."""
    if st.wall_render == "filled":
        cv.img[wall_mask > 0] = st.wall_gray
        return
    lw = max(1, st.line_px)
    if st.wall_render == "thin":
        # one line along each wall axis, a little heavier than the symbol lines; axes meet at
        # their intersections, so no extension is needed
        for p0, p1, _thick, _e0, _e1, _a, _b in axes:
            cv.line(p0, p1, st.wall_gray, max(2, lw + 1))
        return
    if st.wall_render == "hatched":
        # 45-degree hatch inside the band, spaced about half an interior wall apart
        step = max(lw + 3, int(round(0.5 * t_int)))
        hatch = np.full_like(cv.img, 255)
        h, w = hatch.shape
        for k in range(-h, w, step):
            cv2.line(hatch, (k, 0), (k + h, h), int(st.wall_gray), max(1, lw - 1), cv.lt)
        inside = cv2.erode(wall_mask, np.ones((3, 3), np.uint8)) > 0
        cv.img[inside] = np.minimum(cv.img[inside], hatch[inside])
    # hollow and hatched: the outline of the band (both faces of every wall)
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(cv.img, contours, -1, int(st.wall_gray), lw, cv.lt)


def _clear_opening(cv: Canvas, p0, t, n, u0: float, u1: float, thick: float) -> None:
    """Remove the wall drawing where an opening is. Outlined walls are closed by a jamb line at
    both ends of the opening, as CAD drawings do."""
    st = cv.style
    if st.wall_render == "filled":
        clear = _band(p0, t, n, u0, u1, thick / 2 + 1.5)
        cv2.fillPoly(cv.img, [np.round(np.asarray(clear)).astype(np.int32)], st.background, cv2.LINE_8)
        return
    lw = max(1, st.line_px)
    clear = _band(p0, t, n, u0, u1, thick / 2 + lw / 2 + 1.5)
    cv2.fillPoly(cv.img, [np.round(np.asarray(clear)).astype(np.int32)], st.background, cv2.LINE_8)
    if st.wall_render in ("hollow", "hatched"):
        for u in (u0, u1):
            cv.line(_at(p0, t, n, u, -thick / 2), _at(p0, t, n, u, thick / 2), st.wall_gray, lw)


def _room_rects(leaves, room_of, to_px, s, st):
    inset = (st.ext_wall_cm / 2 + 6) * s
    for i, rect in enumerate(leaves):
        if room_of[i]:
            x0, y0 = to_px(rect[0], rect[1]); x1, y1 = to_px(rect[2], rect[3])
            yield x0 + inset, y0 + inset, x1 - inset, y1 - inset


def _stress(cv: Canvas, rng, leaves, room_of, to_px, s, spec):
    """Stress elements that are not walls: solid dark furniture (some against walls), tile
    hatching and stair runs. Drawn inside rooms; the GT wall mask is unaffected."""
    st = cv.style
    for x0, y0, x1, y1 in _room_rects(leaves, room_of, to_px, s, st):
        w, h = x1 - x0, y1 - y0
        if w < 120 * s or h < 120 * s:
            continue
        if st.dark_furniture and rng.random() < st.dark_furniture:
            for _ in range(rng.randint(1, 3)):
                bw, bh = rng.uniform(35, 110) * s, rng.uniform(35, 70) * s
                if rng.random() < 0.5:
                    bw, bh = bh, bw
                if rng.random() < st.furniture_touch:          # against one of the room's walls
                    side = rng.randrange(4)
                    ax = x0 if side == 0 else x1 - bw if side == 1 else rng.uniform(x0, max(x0, x1 - bw))
                    ay = y0 if side == 2 else y1 - bh if side == 3 else rng.uniform(y0, max(y0, y1 - bh))
                    if side in (0, 1):
                        ax += (-1 if side == 0 else 1) * 6 * s
                    else:
                        ay += (-1 if side == 2 else 1) * 6 * s
                else:
                    ax, ay = rng.uniform(x0 + 25 * s, max(x0 + 25 * s, x1 - bw - 25 * s)), rng.uniform(y0 + 25 * s, max(y0 + 25 * s, y1 - bh - 25 * s))
                shape = rng.random()
                if shape < 0.6:
                    cv2.rectangle(cv.img, _ip((ax, ay)), _ip((ax + bw, ay + bh)), st.furniture_tone, -1, cv.lt)
                elif shape < 0.8:
                    r = min(bw, bh) / 2
                    cv2.circle(cv.img, _ip((ax + r, ay + r)), int(r), st.furniture_tone, -1, cv.lt)
                else:                                             # plant-like star
                    cx, cy, r = ax + min(bw, bh) / 2, ay + min(bw, bh) / 2, min(bw, bh) / 2
                    pts = [(cx + (r if k % 2 == 0 else r * 0.45) * np.cos(k * np.pi / 6), cy + (r if k % 2 == 0 else r * 0.45) * np.sin(k * np.pi / 6)) for k in range(12)]
                    cv2.fillPoly(cv.img, [np.round(np.asarray(pts)).astype(np.int32)], st.furniture_tone, cv.lt)
        if st.hatch and rng.random() < st.hatch:
            hw, hh = min(w * 0.6, rng.uniform(90, 200) * s), min(h * 0.6, rng.uniform(90, 200) * s)
            ax, ay = rng.uniform(x0, x1 - hw), rng.uniform(y0, y1 - hh)
            step = max(4.0, rng.uniform(10, 18) * s)
            g = rng.choice((90, 130, 170))
            for k in np.arange(0, hw, step):
                cv2.line(cv.img, _ip((ax + k, ay)), _ip((ax + k, ay + hh)), g, 1, cv.lt)
            for k in np.arange(0, hh, step):
                cv2.line(cv.img, _ip((ax, ay + k)), _ip((ax + hw, ay + k)), g, 1, cv.lt)
    if st.stairs:
        rects = list(_room_rects(leaves, room_of, to_px, s, st))
        rng.shuffle(rects)
        for x0, y0, x1, y1 in rects[:st.stairs]:
            if x1 - x0 < 150 * s or y1 - y0 < 220 * s:
                continue
            sw, sl = rng.uniform(90, 110) * s, rng.uniform(200, 280) * s
            ax, ay = x0 + 6 * s, rng.uniform(y0, max(y0, y1 - sl))
            cv2.rectangle(cv.img, _ip((ax, ay)), _ip((ax + sw, ay + sl)), 0, max(1, st.line_px - 1), cv.lt)
            tread = rng.uniform(22, 28) * s
            for k in np.arange(tread, sl, tread):
                cv2.line(cv.img, _ip((ax, ay + k)), _ip((ax + sw, ay + k)), 0, max(1, st.line_px - 1), cv.lt)


def _markers(bgr, rng, leaves, room_of, to_px, s, st):
    """Coloured annotation dots (sensor/legend markers) inside rooms."""
    rects = list(_room_rects(leaves, room_of, to_px, s, st))
    colours = ((40, 40, 230), (230, 160, 30), (60, 170, 60), (30, 30, 150))
    for _ in range(st.markers):
        if not rects:
            break
        x0, y0, x1, y1 = rng.choice(rects)
        r = max(4, int(rng.uniform(8, 14) * s))
        cx, cy = rng.uniform(x0 + r, max(x0 + r, x1 - r)), rng.uniform(y0 + r, max(y0 + r, y1 - r))
        cv2.circle(bgr, _ip((cx, cy)), r, rng.choice(colours), -1, cv2.LINE_AA)


def _rotate(image, wall_mask, labels, openings, degrees, background):
    """Rotate the drawing (as a slightly skewed scan) and all ground truth with it."""
    h, w = image.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    image = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=int(background))
    wall_mask = cv2.warpAffine(wall_mask, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    labels = cv2.warpAffine(labels.astype(np.float32), m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0).astype(np.int32)

    def tf(p):
        x, y = p
        return (m[0, 0] * x + m[0, 1] * y + m[0, 2], m[1, 0] * x + m[1, 1] * y + m[1, 2])

    rotated = [GTOpening(o.id, o.kind, o.style, tf(o.p0), tf(o.p1), o.width_px, o.wall_thickness_px, o.exterior, o.connects)
               for o in openings]
    return image, wall_mask, labels, rotated


def _noise(cv: Canvas, rng, leaves, room_of, names, to_px, s, spec):
    st = cv.style
    drawn = set()
    for i, rect in enumerate(leaves):
        r = room_of[i]
        if not r:
            continue
        x0, y0 = to_px(rect[0], rect[1]); x1, y1 = to_px(rect[2], rect[3])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if st.text and r not in drawn:
            drawn.add(r)
            scale = max(0.35, 0.0018 * s * 450)
            cv2.putText(cv.img, names[r], (int(cx - 60 * scale * 1.5), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, max(1, st.line_px - 1), cv.lt)
            dims = f"{(rect[2]-rect[0])/30.48:.0f}'x{(rect[3]-rect[1])/30.48:.0f}'"
            cv2.putText(cv.img, dims, (int(cx - 40 * scale * 1.5), int(cy + 30 * scale * 1.5)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.85, 0, 1, cv.lt)
        if st.furniture:
            inset = (st.ext_wall_cm + 40) * s
            fx0, fy0, fx1, fy1 = x0 + inset, y0 + inset, x1 - inset, y1 - inset
            if fx1 - fx0 > 60 * s and fy1 - fy0 > 60 * s:
                for _ in range(rng.randint(1, 3)):
                    w = rng.uniform(40, 160) * s; h = rng.uniform(40, 120) * s
                    ax = rng.uniform(fx0, max(fx0, fx1 - w)); ay = rng.uniform(fy0, max(fy0, fy1 - h))
                    kind = rng.random()
                    if kind < 0.6:
                        cv2.rectangle(cv.img, _ip((ax, ay)), _ip((ax + w, ay + h)), 0, max(1, st.line_px - 1), cv.lt)
                        if kind < 0.3:
                            cv2.line(cv.img, _ip((ax, ay + h * 0.25)), _ip((ax + w, ay + h * 0.25)), 0, 1, cv.lt)
                    else:
                        cv2.circle(cv.img, _ip((ax + w / 2, ay + h / 2)), int(min(w, h) / 2), 0, max(1, st.line_px - 1), cv.lt)
    if st.dimension_lines:
        W, H = spec.footprint_cm
        off = 45 * s
        x0, y0 = to_px(0, 0); x1, y1 = to_px(W, H)
        cv2.line(cv.img, _ip((x0, y0 - off)), _ip((x1, y0 - off)), 0, 1, cv.lt)
        for x in (x0, x1):
            cv2.line(cv.img, _ip((x, y0 - off - 8)), _ip((x, y0 - off + 8)), 0, 1, cv.lt)
        cv2.putText(cv.img, f"{W/30.48:.0f}'", _ip(((x0 + x1) / 2, y0 - off - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv.lt)
        cv2.line(cv.img, _ip((x0 - off, y0)), _ip((x0 - off, y1)), 0, 1, cv.lt)


def _degrade(img: np.ndarray, st: Style, rng: random.Random) -> np.ndarray:
    out = img
    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    if st.blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), st.blur)
    if st.speckle > 0:
        mask = np_rng.random(out.shape) < st.speckle
        out = out.copy()
        out[mask] = np.where(out[mask] > 127, 0, 255).astype(np.uint8)
    if st.jpeg:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, st.jpeg])
        out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return out


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------

def _variants(base_seed: int, count: int):
    """Controlled variations of scale / line weight / wall thickness."""
    scales = (0.55, 0.8, 1.05, 1.35)
    lines = (1, 2, 3)
    for i in range(count):
        r = random.Random(base_seed * 1000 + i)
        s = scales[i % len(scales)]
        lw = lines[(i // 2) % len(lines)]
        int_cm = r.choice((10.0, 12.0, 15.0))
        # keep walls clearly thicker than symbol lines
        int_cm = max(int_cm, (3.2 * lw + 2) / s)
        yield i, s, lw, r.choice((22.0, 25.0, 30.0)), int_cm


FAMILIES = {
    # name: (style overrides, spec overrides)
    "door_swing": ({"door_styles": ("swing",)}, {}),
    "door_double": ({"door_styles": ("double",)}, {}),
    "door_leaf_only": ({"door_styles": ("leaf_only",)}, {}),
    "door_no_swing_outline": ({"door_styles": ("outline",)}, {}),
    "door_sliding": ({"door_styles": ("sliding",)}, {}),
    "door_narrow": ({"door_styles": ("swing", "outline")}, {"door_width_cm": (60.0, 70.0)}),
    "door_wide": ({"door_styles": ("swing", "outline")}, {"door_width_cm": (100.0, 115.0)}),
    "door_exterior": ({"door_styles": ("swing", "outline", "double")}, {"exterior_doors": 3}),
    "window_clear": ({"window_styles": ("triple", "double")}, {}),
    "window_sample_style": ({"window_styles": ("sample",)}, {}),
    "window_weak": ({"window_styles": ("weak",)}, {}),
    "window_widths": ({"window_styles": ("triple", "sample")}, {"window_width_cm": (50.0, 320.0), "windows_per_room": (2, 3)}),
    "window_adjacent": ({"window_styles": ("triple", "sample")}, {"adjacent_window_prob": 0.9}),
    "noise": ({"door_styles": DOOR_STYLES, "window_styles": ("triple", "double", "sample"), "text": True,
               "furniture": True, "dimension_lines": True, "speckle": 0.002, "blur": 0.6, "jpeg": 70}, {}),
    "ambiguous_passages": ({"door_styles": ("swing", "outline")}, {"passage_prob": 0.7}),
    "mixed": ({"door_styles": DOOR_STYLES, "window_styles": ("triple", "double", "sample", "weak"), "text": True,
               "furniture": True}, {}),
}


ALL_DOORS = DOOR_STYLES
ALL_WINDOWS = ("triple", "double", "sample", "weak")

# Stress families (benchmark N.1): elements that are not walls. Synthetic evidence only —
# they use this generator's own drawing style and are reported separately from real plans.
STRESS = {
    "stress_dark_furniture": ({"door_styles": DOOR_STYLES, "window_styles": ("triple", "double", "sample"),
                               "dark_furniture": 0.8, "furniture_touch": 0.5}, {}),
    "stress_markers": ({"door_styles": DOOR_STYLES, "window_styles": ("triple", "double"), "markers": 8}, {}),
    "stress_hatch": ({"door_styles": DOOR_STYLES, "window_styles": ("triple", "double", "sample"), "hatch": 0.5}, {}),
    "stress_stairs": ({"door_styles": DOOR_STYLES, "window_styles": ("triple", "double"), "stairs": 2}, {}),
}
FAMILIES_STRESS_HOLDOUT = {
    "holdout_stress": ({"door_styles": DOOR_STYLES, "window_styles": ("triple", "double", "sample"), "dark_furniture": 0.6,
                        "furniture_tone": 90, "furniture_touch": 0.7, "markers": 5, "hatch": 0.35, "stairs": 1,
                        "text": True}, {}),
}

# Hold-out families: different seeds, scales, line weights, noise and skew than the
# tuning families above. Used to check that results generalise.
HOLDOUT = {
    "holdout_mixed": ({"door_styles": ALL_DOORS, "window_styles": ALL_WINDOWS, "text": True, "furniture": True}, {},
                      (0.45, 0.7, 1.2, 1.6), (1, 2, 3, 4)),
    "holdout_scan": ({"door_styles": ALL_DOORS, "window_styles": ("triple", "double", "sample"), "text": True,
                      "furniture": True, "dimension_lines": True, "speckle": 0.004, "blur": 1.0, "jpeg": 55,
                      "background": 238}, {}, (0.7, 0.95, 1.25), (1, 2, 3)),
    "holdout_rotated": ({"door_styles": ALL_DOORS, "window_styles": ("triple", "double", "sample"), "text": True,
                         "rotate_deg": 2.0}, {}, (0.8, 1.1, 1.4), (1, 2)),
    "holdout_large": ({"door_styles": ALL_DOORS, "window_styles": ALL_WINDOWS, "text": True, "furniture": True},
                      {"footprint_cm": (2100.0, 1500.0), "min_room_cm": 300.0}, (0.9, 1.3), (2, 3)),
}


def family_specs(name: str, count: int = 6, salt: int = 0) -> list[PlanSpec]:
    """`salt` > 0 gives a fresh, never-tuned-on set of plans with the same family settings."""
    if name in HOLDOUT:
        return holdout_specs(name, count, salt)
    if name in FAMILIES_STRESS_HOLDOUT:
        style_over, spec_over = FAMILIES_STRESS_HOLDOUT[name]
        base = 7919 + sum(map(ord, name)) * 13 + 100_003 * salt
        specs = []
        for i in range(count):
            r = random.Random(base + i)
            scale, lw = (0.6, 0.9, 1.2, 1.5)[i % 4], (1, 2, 3)[(i // 4) % 3]
            int_cm = max(r.choice((9.0, 12.0, 15.0)), (3.2 * lw + 2) / scale)
            style = Style(px_per_cm=scale, line_px=lw, ext_wall_cm=r.choice((20.0, 25.0, 30.0)), int_wall_cm=int_cm, **style_over)
            specs.append(PlanSpec(seed=base * 10 + i, style=style, **spec_over))
        return specs
    style_over, spec_over = FAMILIES[name] if name in FAMILIES else STRESS[name]
    specs = []
    base = sum(map(ord, name)) + 100_003 * salt
    for i, s, lw, ext_cm, int_cm in _variants(base, count):
        style = Style(px_per_cm=s, line_px=lw, ext_wall_cm=ext_cm, int_wall_cm=int_cm, **style_over)
        specs.append(PlanSpec(seed=base * 100 + i, style=style, **spec_over))
    return specs


def holdout_specs(name: str, count: int, salt: int = 0) -> list[PlanSpec]:
    style_over, spec_over, scales, lines = HOLDOUT[name]
    base = 7919 + sum(map(ord, name)) * 13 + 100_003 * salt
    specs = []
    for i in range(count):
        r = random.Random(base + i)
        s, lw = scales[i % len(scales)], lines[(i // len(scales)) % len(lines)]
        int_cm = max(r.choice((9.0, 11.0, 14.0, 18.0)), (3.2 * lw + 2) / s)
        over = dict(style_over)
        if "rotate_deg" in over:
            over["rotate_deg"] = r.choice((-1, 1)) * r.uniform(0.8, 4.0)
        style = Style(px_per_cm=s, line_px=lw, ext_wall_cm=r.choice((20.0, 24.0, 28.0, 35.0)), int_wall_cm=int_cm, **over)
        specs.append(PlanSpec(seed=base * 10 + i, style=style, **spec_over))
    return specs
