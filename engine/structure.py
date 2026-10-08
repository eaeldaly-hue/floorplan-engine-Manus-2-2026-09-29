"""Structural geometry: walls → wall gaps (opening candidates) → sealed spaces → topology.

One pass per image, shared by room analysis and opening classification.

1. Ink and stroke widths. The erosion kernel that separates wall strokes from
   thin drawing lines is derived from the measured stroke-width distribution
   (thin-line mode vs. thinnest wall mode), not from the image size.
2. Wall mask by thickness filtering only. Fragments are *not* bridged: a wall
   interrupted by a door or window is genuinely disconnected.
3. Wall segments: axis-aligned bands (row/column runs) plus diagonal bands
   (Hough with gap-preserving settings).
4. Opening candidates: from every free wall end (not a corner or junction),
   march along the wall axis until the next wall face. This finds collinear
   gaps, gaps that end at a perpendicular wall, and gaps in diagonal walls.
   Wide gaps are kept only if a drawn line spans them.
5. Spaces: every gap is sealed with a wall-thickness bridge exactly where the
   gap is, the exterior is flood-filled from the border, and the remaining
   regions are kept as spaces if they are large and wide enough to be rooms.
6. Topology: which space (or the exterior) lies on each side of every gap, and
   which spaces share a wall.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from engine.walls.segments import WallSegmentExtractor

EXTERIOR = -1  # label value for the exterior region in StructureResult.space_labels


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class WallBand:
    p0: tuple[float, float]
    p1: tuple[float, float]
    thickness: float
    orientation: str  # horizontal | vertical | diagonal

    @property
    def direction(self) -> np.ndarray:
        d = np.subtract(self.p1, self.p0).astype(float)
        return d / max(1e-9, float(np.linalg.norm(d)))


@dataclass
class OpeningCandidate:
    id: str
    start: tuple[float, float]          # gap endpoints on the wall axis
    end: tuple[float, float]
    thickness: float                    # host wall thickness (px)
    orientation: str
    line_coverage: float                # fraction of the gap spanned by drawn lines inside the wall band
    found_from: int = 1                 # 1 = one wall end, 2 = both ends agree, 0 = drawn lines only
    source: str = "wall_gap"            # wall_gap | line_pair | single_line
    sides: tuple = (None, None)         # space label on each side (−n, +n); EXTERIOR = −1; 0 = none
    relation: str = "unknown"           # between_rooms | room_to_exterior | same_space | unknown
    jambs: tuple = ()                   # structural support of each end (see _jamb_support)

    @property
    def width(self) -> float:
        return float(np.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1]))

    @property
    def wall_thickness(self) -> float:  # compatibility with the older candidate objects
        return self.thickness

    @property
    def center(self) -> tuple[float, float]:
        return ((self.start[0] + self.end[0]) / 2, (self.start[1] + self.end[1]) / 2)

    @property
    def tangent(self) -> np.ndarray:
        d = np.subtract(self.end, self.start).astype(float)
        return d / max(1e-9, float(np.linalg.norm(d)))

    @property
    def normal(self) -> np.ndarray:
        t = self.tangent
        return np.array([-t[1], t[0]])


@dataclass
class StructureResult:
    gray: np.ndarray
    ink: np.ndarray                  # dark pixels (walls + lines), 0/255
    symbol_ink: np.ndarray           # permissive ink incl. light-gray symbol lines, 0/255
    wall_mask: np.ndarray            # walls only, 0/255
    wall_kernel: int
    line_width: float
    wall_thickness: float            # typical (median) wall thickness, px
    bands: list[WallBand]
    candidates: list[OpeningCandidate]   # gaps wide enough to be openings
    cracks: list[OpeningCandidate]       # small gaps (sealed, not reported)
    sealed_mask: np.ndarray
    space_labels: np.ndarray         # int32: k > 0 space k, EXTERIOR outside, 0 wall/noise
    spaces: list[dict]               # analyzer-compatible records (id, bbox, area_pixels, center, polygon)
    wall_adjacency: set = field(default_factory=set)
    skew_degrees: float = 0.0
    # When the drawing was deskewed, the analysis ran on a rotated copy ("work" frame).
    # Public fields above are in original image coordinates; `work` holds the rotated-frame
    # result (used by the classifier) and `to_original` maps work points back.
    work: "StructureResult | None" = None
    to_original: object = None
    walls: object = None                 # engine.wall_inference.WallInference (evidence for diagnostics)
    rejected_candidates: list = field(default_factory=list)   # gaps with an unsupported jamb (not sealed, not reported)

    def space_id(self, label: int) -> str | None:
        if label is None or label <= 0:
            return None
        return self.spaces[label - 1]["id"]

    def adjacency_pairs(self) -> set:
        """{(a, b)} of space labels connected by an opening; b = 0 means exterior."""
        pairs = set()
        for c in self.candidates:
            a, b = c.sides
            if c.relation == "between_rooms":
                pairs.add(tuple(sorted((a, b))))
            elif c.relation == "room_to_exterior":
                pairs.add((max(a, b), 0))
        return pairs


# ---------------------------------------------------------------------------
# 1–2. Ink, stroke widths, wall mask
# ---------------------------------------------------------------------------

def despeckle(mask: np.ndarray, min_extent: int = 6, min_area: int = 8) -> np.ndarray:
    """Remove isolated specks (scan noise, JPEG artefacts) while keeping thin lines.

    Connectivity is judged after a small closing, so faint dotted lines (anti-aliased
    1-px diagonals break into 2–3 px pieces) count as one long component."""
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    keep = np.zeros(count, bool)
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep[1:] = (extent[1:] >= min_extent) & (stats[1:, cv2.CC_STAT_AREA] >= min_area)
    return np.where(keep[labels] & (mask > 0), 255, 0).astype(np.uint8)


def _binarize(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    background = float(np.percentile(gray, 90))
    symbol_threshold = max(60.0, background - 35.0)
    symbol_ink = np.where(gray < symbol_threshold, 255, 0).astype(np.uint8)
    return ink, despeckle(np.maximum(ink, symbol_ink))


def _ridge_widths(mask: np.ndarray) -> np.ndarray:
    """Stroke width (2·distance) at local maxima of the distance transform."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ridge = (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8))) & (dist >= 1.0)
    return np.round(2.0 * dist[ridge]).astype(int)


def _surviving_area_at(ink: np.ndarray, k: int) -> float:
    """Wall-mask area left by the thickness filter with kernel k."""
    if k <= 1:
        return float((ink > 0).sum())
    eroded = cv2.erode(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    restored = cv2.bitwise_and(cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (k + 2, k + 2))), ink)
    return float((restored > 0).sum())


def _surviving_area(ink: np.ndarray, kmax: int) -> np.ndarray:
    return np.asarray([_surviving_area_at(ink, k) for k in range(1, kmax + 1)])


def estimate_wall_kernel(ink: np.ndarray) -> tuple[int, float, float]:
    """(erosion kernel, thin-line width, thinnest wall width) — see kernel_search."""
    kernel, line_width, thinnest, _ = kernel_search(ink)
    return kernel, line_width, thinnest


# A wall-class hypothesis (engine.reconstruction): when set, the thickness filter uses this
# (kernel, thinnest wall width) instead of choosing it from the surviving-area curve.
_WALL_CLASS: contextvars.ContextVar = contextvars.ContextVar("floorplan_wall_class", default=None)


@contextlib.contextmanager
def wall_class(kernel: int, thinnest: float):
    token = _WALL_CLASS.set((int(kernel), float(thinnest)))
    try:
        yield
    finally:
        _WALL_CLASS.reset(token)


def cliff_wall_class(ink: np.ndarray, kmax: int = 40) -> tuple[int, float, float] | None:
    """(kernel, thinnest, ink share lost) of the strongest wall class by its *cliff*.

    Strokes of one thickness vanish together: as the erosion kernel grows past a wall class's
    width, a large share of the ink disappears within one or two kernel steps. Anti-aliased or
    JPEG walls erode gradually before that, so 'nearly flat plateaus' can be missed or found on
    a tail of filled furniture; the cliff is the steadier mark of the dominant wall width."""
    widths = _ridge_widths(ink)
    line_width = float(np.argmax(np.bincount(widths, minlength=8)[1:7]) + 1) if widths.size else 1.0
    h, w = ink.shape
    kmax = int(min(kmax, max(8, 0.04 * max(h, w))))
    a = [_surviving_area_at(ink, k) for k in range(1, kmax + 1)]
    if not a or a[0] <= 0:
        return None
    best = None
    for k in range(int(line_width) + 3, kmax + 1):        # a[k-1] = area at kernel k
        loss = (a[k - 2] - a[k - 1]) + ((a[k - 1] - a[k]) if k < kmax else 0.0)
        share = loss / a[0]
        if best is None or share > best[2]:
            best = (k, k, share)
    if best is None or best[2] < 0.1:
        return None
    k_end = float(best[0] - 1)                             # widest kernel that still keeps the class
    return max(3, int(round(0.55 * k_end))), k_end, float(best[2])


def kernel_search(ink: np.ndarray) -> tuple[int, float, float, list[float]]:
    """(erosion kernel, thin-line width, thinnest wall width, surviving area per kernel).

    As the thickness-filter kernel grows, thin strokes (lines, text, blurred symbols)
    vanish first — the surviving area drops — then the area holds on a plateau while
    only walls remain, until the kernel exceeds the thinnest walls. The kernel is chosen
    on the earliest plateau (>= 3 nearly flat steps) that ends in a clear drop. Robust to blur and
    noise, unlike stroke-width modes. Areas are computed lazily, so the search stops
    as soon as that plateau is confirmed.
    """
    h, w = ink.shape
    widths = _ridge_widths(ink)
    line_width = float(np.argmax(np.bincount(widths, minlength=8)[1:7]) + 1) if widths.size else 1.0
    forced = _WALL_CLASS.get()
    if forced is not None:
        return forced[0], line_width, forced[1], []
    kmax = max(8, int(0.04 * max(h, w)))
    areas: list[float] = []

    def area(i: int) -> float:          # index i <-> kernel i + 1
        while len(areas) <= i:
            areas.append(_surviving_area_at(ink, len(areas) + 1))
        return areas[i]

    def drop(i: int) -> float:
        return (area(i - 1) - area(i)) / max(area(i - 1), 1.0)

    def alive(i: int) -> bool:
        return area(i) >= 0.01 * area(0)

    flat, event = 0.025, 0.04   # a plateau step loses < 2.5 %; walls vanishing lose >= 4 % (over 2 steps)
    i = 1
    while i < kmax:
        if drop(i) >= flat or not alive(i):
            if not alive(i):
                break
            i += 1
            continue
        j = i
        while j + 1 < kmax and drop(j + 1) < flat and alive(j + 1):
            j += 1
        # walls of one thickness can vanish over two kernel steps (anti-aliased edges); interior
        # walls may be a small share of all wall area, so even a modest drop marks them
        after = area(min(j + 2, kmax - 1))
        ends_in_drop = j + 1 < kmax and (area(j) - after) / max(area(j), 1.0) >= event
        if j - i + 1 >= 3 and ends_in_drop:
            k_start, k_end = i + 1, j + 1
            return max(3, k_start, int(round(0.55 * k_end))), line_width, float(k_end), areas
        # Thin interior walls only a few pixels thicker than the lines give a short plateau.
        # Accept two steps when they are very flat, start clearly above the line width
        # (so thick drawing lines cannot qualify) and end in a strong drop.
        if (j - i + 1 == 2 and max(drop(i), drop(j)) < 0.01 and i + 1 >= line_width + 2
                and j + 1 < kmax and (area(j) - after) / max(area(j), 1.0) >= 0.1):
            return max(3, i + 1), line_width, float(j + 1), areas
        i = j + 1
    fallback = max(3, int(round(math.hypot(w, h) * 0.0035)))
    return fallback, line_width, float(2 * fallback), areas


def build_wall_mask(ink: np.ndarray, kernel: int) -> np.ndarray:
    k = max(3, int(kernel))
    eroded = cv2.erode(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    restored = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (k + 2, k + 2)))
    walls = cv2.bitwise_and(restored, ink)
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Door/window jambs are often drawn as short dark blocks thinner than the walls.
    # Keep medium-thickness blobs that are attached to a wall and compact.
    mk = max(3, int(round(0.4 * k)))
    if mk < k:
        medium = cv2.bitwise_and(cv2.dilate(cv2.erode(ink, np.ones((mk, mk), np.uint8)), np.ones((mk + 2, mk + 2), np.uint8)), ink)
        medium = cv2.bitwise_and(medium, cv2.bitwise_not(walls))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(medium, connectivity=8)
        touching = np.unique(labels[(cv2.dilate(walls, np.ones((3, 3), np.uint8)) > 0) & (labels > 0)])
        keep_blob = np.zeros(count, bool)
        for i in touching:
            area, bw, bh = stats[i, cv2.CC_STAT_AREA], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            solid = area >= 0.45 * bw * bh  # filled block, not an outline glyph (axis-aligned bbox; lenient for slanted jambs)
            keep_blob[i] = area <= 3 * k * k and max(bw, bh) <= 3 * k and solid
        if keep_blob.any():
            walls[keep_blob[labels]] = 255
    # Drop specks: anything smaller than a short piece of the thinnest wall.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(walls, connectivity=8)
    min_area = max(20, int(1.2 * k * k))
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _thinnest_wall(plateau_thinnest: float, typical_t: float) -> float:
    """The plateau end over-states the width of walls whose dark core is narrower than their
    anti-aliased outline; the measured typical wall width bounds the thinnest real wall."""
    return min(plateau_thinnest, typical_t)


def _typical_thickness(wall_mask: np.ndarray, fallback: float) -> float:
    widths = _ridge_widths(wall_mask)
    widths = widths[widths >= 2]
    return float(np.median(widths)) if widths.size else fallback


# ---------------------------------------------------------------------------
# 3. Wall bands
# ---------------------------------------------------------------------------

def _axis_bands(wall_mask: np.ndarray, kernel: int, thick_max: float) -> list[WallBand]:
    h, w = wall_mask.shape
    extractor = WallSegmentExtractor(wall_mask, min_thickness=max(2, kernel - 1))
    # Runs must be longer than the thickest wall, otherwise a vertical wall's rows
    # would be read as short horizontal walls (and vice versa).
    extractor.min_length = max(int(1.4 * thick_max) + 2, 3 * kernel)
    extractor.max_thickness = max(extractor.max_thickness, int(1.3 * thick_max) + 2)
    bands = []
    for s in extractor.detect_horizontal() + extractor.detect_vertical():
        bands.append(WallBand(tuple(map(float, s["start"])), tuple(map(float, s["end"])), float(s["thickness"]), s["orientation"]))
    return bands


def hough_tiles(img: np.ndarray, tile: int = 512, overlap: int = 160, theta: float = np.pi / 180, **params) -> np.ndarray | None:
    """HoughLinesP over overlapping tiles. On large images the probabilistic transform drops
    short lines when many unrelated points compete; local accumulation keeps them."""
    h, w = img.shape[:2]
    step = tile - overlap
    found = []
    for y0 in range(0, max(1, h - overlap), step):
        for x0 in range(0, max(1, w - overlap), step):
            sub = img[y0:y0 + tile, x0:x0 + tile]
            if not sub.any():
                continue
            lines = cv2.HoughLinesP(sub, 1, theta, **params)
            if lines is not None:
                found.append(lines[:, 0, :] + np.array([x0, y0, x0, y0]))
    if not found:
        return None
    return np.concatenate(found)[:, None, :]


def _diagonal_bands(wall_mask: np.ndarray, thickness: float) -> list[WallBand]:
    """Diagonal wall bands from straight slanted *edges* of the wall mask.

    Lines fitted to a filled band run corner-to-corner across it; the band's
    edges, however, are exact straight lines along its direction.
    """
    edges = cv2.morphologyEx(wall_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    min_len = max(15, int(round(1.2 * thickness)))
    lines = hough_tiles(edges, threshold=max(10, int(min_len * 0.5)), minLineLength=min_len, maxLineGap=3)
    if lines is None:
        return []
    found = []
    for x1, y1, x2, y2 in lines[:, 0, :].astype(float):
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0
        if min(angle, 180 - angle, abs(angle - 90)) <= 15:
            continue
        p0, p1 = np.array([x1, y1]), np.array([x2, y2])
        length = float(np.linalg.norm(p1 - p0))
        d = (p1 - p0) / length
        n = np.array([-d[1], d[0]])
        s = np.linspace(0.2 * length, 0.8 * length, 7)
        # which side of the edge is wall?
        side = 1.0 if _cross_occupancy(wall_mask, p0, d, n, s, [3.0]).mean() >= _cross_occupancy(wall_mask, p0, d, n, s, [-3.0]).mean() else -1.0
        across = np.arange(1.0, 2.5 * thickness, 1.0) * side
        occ = _cross_occupancy(wall_mask, p0, n, d, across, s)  # per offset, averaged along the edge
        solid = occ >= 0.6
        if not solid[:3].any():
            continue
        exits = np.flatnonzero(~solid[1:])
        width = float(exits[0] + 2) if exits.size else float(len(across))
        if width < 3 or width > 2.2 * thickness:
            continue
        shift = side * width / 2
        q0, q1 = p0 + n * shift, p1 + n * shift
        found.append(_centre_on_band(wall_mask, q0, q1, thickness))
    found = [f for f in found if f is not None]
    # Merge parallel, overlapping detections of the same band (both edges, split edges).
    found.sort(key=lambda f: -math.hypot(f[2] - f[0], f[3] - f[1]))
    merged: list[list[float]] = []
    for f in found:
        p0, p1 = np.array(f[:2]), np.array(f[2:4])
        d = (p1 - p0) / max(1e-9, np.linalg.norm(p1 - p0))
        for m in merged:
            q0, q1 = np.array(m[:2]), np.array(m[2:4])
            e = (q1 - q0) / max(1e-9, np.linalg.norm(q1 - q0))
            if abs(float(np.dot(d, e))) < 0.985:
                continue
            nrm = np.array([-e[1], e[0]])
            if abs(float(np.dot((p0 + p1) / 2 - q0, nrm))) > 0.5 * max(m[4], f[4]) + 2:
                continue
            u = sorted((float(np.dot(p0 - q0, e)), float(np.dot(p1 - q0, e))))
            L = float(np.linalg.norm(q1 - q0))
            if u[1] < -0.4 * thickness or u[0] > L + 0.4 * thickness:
                continue
            lo, hi = min(0.0, u[0]), max(L, u[1])
            m[0:4] = [*(q0 + e * lo), *(q0 + e * hi)]
            m[4] = max(m[4], f[4])
            break
        else:
            merged.append(list(f))
    return [WallBand((m[0], m[1]), (m[2], m[3]), max(3.0, m[4]), "diagonal") for m in merged]


def _centre_on_band(mask, p0, p1, thickness):
    """Shift a Hough line onto the medial axis of the band it lies in; reject non-bands."""
    length = float(np.linalg.norm(p1 - p0))
    d = (p1 - p0) / max(1e-9, length)
    n = np.array([-d[1], d[0]])
    v = np.arange(-2.0 * thickness, 2.0 * thickness + 1, 1.0)
    s = np.linspace(0.25 * length, 0.75 * length, 7)
    profile = _cross_occupancy(mask, p0, n, d, v, s)  # occupancy for each v (across), averaged over s
    solid = profile >= 0.6
    zero = int(np.argmin(np.abs(v)))
    if not solid[zero]:
        near = np.flatnonzero(solid)
        if near.size == 0:
            return None
        zero = int(near[np.argmin(np.abs(near - zero))])
    lo = hi = zero
    while lo > 0 and solid[lo - 1]:
        lo -= 1
    while hi < len(v) - 1 and solid[hi + 1]:
        hi += 1
    width = float(v[hi] - v[lo] + 1)
    if width < 2 or width > 2.2 * thickness or lo == 0 or hi == len(v) - 1:
        return None  # not a band of wall thickness (line through a larger blob)
    shift = (v[lo] + v[hi]) / 2
    q0, q1 = p0 + n * shift, p1 + n * shift
    outside = max(profile[max(0, lo - 7):max(0, lo - 2)].max(initial=0), profile[hi + 3:hi + 8].max(initial=0))
    if outside > 0.25:
        return None
    return [q0[0], q0[1], q1[0], q1[1], width]


# ---------------------------------------------------------------------------
# 4. Gaps along wall axes
# ---------------------------------------------------------------------------

def _sample(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    xi, yi = np.floor(np.asarray(xs) + 0.5).astype(int), np.floor(np.asarray(ys) + 0.5).astype(int)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros(xi.shape, bool)
    out[inside] = mask[yi[inside], xi[inside]] > 0
    return out


def _cross_occupancy(mask, origin, d, n, s_values, v_values) -> np.ndarray:
    """Mean mask occupancy of cross-sections at distances s along d (v across)."""
    s = np.asarray(s_values, float)[:, None]
    v = np.asarray(v_values, float)[None, :]
    xs = origin[0] + d[0] * s + n[0] * v
    ys = origin[1] + d[1] * s + n[1] * v
    return _sample(mask, xs, ys).mean(axis=1)


def _parallel_strokes(mask, origin, d, n, s_values, lo, hi) -> bool:
    """The ink between two bridging lines is more parallel lines (a glazing band: frame, glass,
    sill, sliding panels), not text or hatching: across the band, every inked 1-px row is one or
    two long runs along the opening, and at least one runs (nearly) its whole length. Text and
    hatching ink rows in many short pieces."""
    s = np.asarray(s_values, float)
    if max(abs(d[0]), abs(d[1])) < 0.97:
        return False                                  # glazing runs along the plan's wall axes
    # Hough pieces of an axis-aligned band come out slightly tilted: judge the rows on the axis
    mid = np.asarray(origin, float) + d * s.mean()
    d = np.array([np.sign(d[0]), 0.0]) if abs(d[0]) >= abs(d[1]) else np.array([0.0, np.sign(d[1])])
    n = np.array([-d[1], d[0]]) * (1.0 if float(np.dot(n, [-d[1], d[0]])) >= 0 else -1.0)
    origin = mid - d * s.mean()
    # a glazing band stands alone in its wall; parallel lines that go on beyond it are a pattern
    # (floor planks, hatching, stair treads)
    width = max(3.0, hi - lo)
    outside = np.concatenate([np.arange(lo - 2.0 - 1.5 * width, lo - 2.0), np.arange(hi + 3.0, hi + 3.0 + 1.5 * width)])
    ox = origin[0] + d[0] * s[:, None] + n[0] * outside[None, :]
    oy = origin[1] + d[1] * s[:, None] + n[1] * outside[None, :]
    for col in _sample(mask, ox, oy).T:
        runs = _runs(col)
        if runs and max(b - a for a, b in runs) >= 0.5 * len(s):
            return False
    v = np.arange(np.floor(lo) - 1.0, np.ceil(hi) + 2.0)
    xs = origin[0] + d[0] * s[:, None] + n[0] * v[None, :]
    ys = origin[1] + d[1] * s[:, None] + n[1] * v[None, :]
    occ = _sample(mask, xs, ys)                     # rows = positions along, cols = 1-px rows across
    n_s = occ.shape[0]
    full = 0
    for col in occ.T:
        share = float(col.mean())
        if share <= 0.15:
            continue
        runs = _runs(col)
        # a drawn line (or a sliding panel's line, covering part of the opening) is one or two
        # long runs; text and hatching are many short ones
        if len(runs) > 2 or max(b - a for a, b in runs) < 0.3 * n_s:
            return False
        full += share >= 0.85
    return full >= 1


def _axis_end_faces(wall_mask: np.ndarray, kernel: int, thick_max: float) -> list[tuple]:
    """Wall end faces of axis-aligned walls: (end point, direction, face length)."""
    m = wall_mask > 0
    h, w = m.shape
    faces = []
    specs = (
        ((1.0, 0.0), np.pad(m[:, :-1] & ~m[:, 1:], ((0, 0), (0, 1)))),
        ((-1.0, 0.0), np.pad(m[:, 1:] & ~m[:, :-1], ((0, 0), (1, 0)))),
        ((0.0, 1.0), np.pad(m[:-1, :] & ~m[1:, :], ((0, 1), (0, 0)))),
        ((0.0, -1.0), np.pad(m[1:, :] & ~m[:-1, :], ((1, 0), (0, 0)))),
    )
    lo, hi = max(3, int(0.6 * kernel)), 2.2 * thick_max
    for d, trans in specs:
        horizontal_face = d[1] == 0  # face spans y for ±x directions
        grow = np.ones((1, 3) if horizontal_face else (3, 1), np.uint8)
        merged = cv2.dilate(trans.astype(np.uint8), grow)
        count, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
        for i in range(1, count):
            x, y, bw, bh = stats[i, :4]
            length = bh if horizontal_face else bw
            if not lo <= length <= hi:
                continue
            ex = x + (bw - 1) / 2
            ey = y + (bh - 1) / 2
            faces.append((np.array([ex, ey]), np.array(d), float(length)))
    return faces


def _refine_end(mask, end, d, n, t) -> np.ndarray | None:
    """Move an approximate band end to the last cross-section that is still wall."""
    s = np.arange(-0.8 * t, 1.2 * t + 1, 1.0)
    occ = _cross_occupancy(mask, end, d, n, s, np.linspace(-0.3 * t, 0.3 * t, 5))
    solid = occ >= 0.5
    if not solid[: max(1, int(0.5 * t))].any():
        return None
    last = None
    for i, value in enumerate(solid):
        if value:
            last = i
        elif last is not None and i - last > 2:
            break
    return np.asarray(end, float) + d * s[last]


def _is_free_end(mask, end, d, n, t) -> bool:
    # The band must really end here (look just past the end; narrow cracks still count as ends).
    beyond = _cross_occupancy(mask, end, d, n, np.array([1.5, 2.5]), np.linspace(-0.3 * t, 0.3 * t, 5))
    if beyond.mean() >= 0.35:
        return False
    # Corner / junction: another wall overlaps the end of the band, or continues from it at an angle.
    s = np.linspace(-0.5 * t, 0.6 * t, 7)
    for side in (-1, 1):
        v = side * np.linspace(0.5 * t + 0.15 * t + 2, 0.5 * t + 0.9 * t, 5)
        if _cross_occupancy(mask, end, d, n, s, v).mean() >= 0.3:
            return False
    return True


def _aligned_band_behind(mask, end, d, n, t) -> bool:
    """The wall behind an end face is a band along d (rejects stair-steps of diagonal walls)."""
    depth = min(_depth_behind(mask, end, d, n, t), 1.0 * t)
    if depth < 3:
        return False
    s = -np.linspace(0.2 * depth, depth, 5)
    # sample the band's core: on thin anti-aliased walls (4-5 px) +-0.4 t lands on the edge
    # pixels, and real door jambs were rejected as "not a band"
    half = max(0.5, min(0.4 * t, 0.5 * t - 1.0))
    inside = _cross_occupancy(mask, end, d, n, s, np.linspace(-half, half, 7)).mean()
    edges = [_cross_occupancy(mask, end, d, n, s, side * np.array([0.5 * t + 2, 0.5 * t + 4])).mean() for side in (-1, 1)]
    return inside >= 0.85 and max(edges) <= 0.4


def _depth_behind(mask, end, d, n, t) -> float:
    s = -np.arange(0, 3 * t + 1, 1.0)
    occ = _cross_occupancy(mask, end, d, n, s, np.linspace(-0.3 * t, 0.3 * t, 5))
    gaps = np.flatnonzero(occ < 0.6)
    return float(gaps[0]) if gaps.size else float(3 * t)


def _width_behind(mask, end, d, n, t) -> float:
    """Wall width measured across the band a little behind an end face."""
    best = 0.0
    for back in (1.0 * t, 1.5 * t, 2.0 * t):
        v = np.arange(-1.5 * t - 2, 1.5 * t + 3, 1.0)
        xs = end[0] - d[0] * back + n[0] * v
        ys = end[1] - d[1] * back + n[1] * v
        inside = _sample(mask, xs, ys) > 0
        if not inside.any():
            continue
        # contiguous run through the axis
        c = len(v) // 2
        if not inside[c]:
            continue
        lo = c
        while lo > 0 and inside[lo - 1]:
            lo -= 1
        hi = c
        while hi < len(v) - 1 and inside[hi + 1]:
            hi += 1
        best = max(best, float(hi - lo + 1))
    return best


def _scan_gap(mask, end, d, n, t, max_steps) -> float | None:
    """Distance to the next wall face along d."""
    s = np.arange(1, max_steps + 1, 1.0)
    occ = _cross_occupancy(mask, end, d, n, s, np.linspace(-0.3 * t, 0.3 * t, 5))
    hit = np.flatnonzero((occ[:-1] >= 0.6) & (occ[1:] >= 0.6))
    if hit.size == 0:
        return None
    gap = float(s[hit[0]] - 1.0)
    return gap if gap >= 2 else None


def _band_profile(symbol_ink, start, d, n, t, u):
    """Symbol-ink matrix along a gap: rows = positions u along the wall, cols = offsets across it."""
    v = np.arange(-0.5 * t - 2, 0.5 * t + 3, 1.0)
    uu = np.asarray(u, float)[:, None]
    xs = start[0] + d[0] * uu + n[0] * v[None, :]
    ys = start[1] + d[1] * uu + n[1] * v[None, :]
    return _sample(symbol_ink, xs, ys), v


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], flags, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2], edges[1::2]))


def _split_gap(symbol_ink, start, d, n, t, gap) -> list[tuple[float, float]]:
    """Split one wall gap into separately drawn openings (adjacent windows, window + plain gap, …)."""
    u = np.arange(0.0, gap + 1.0, 1.0)
    profile, v = _band_profile(symbol_ink, start, d, n, t, u)
    if gap < 2.4 * t:
        return [(0.0, gap)]
    inner = np.abs(v) <= 0.5 * t - max(2.0, 0.15 * t)
    cuts = set()
    # (a) frame / jamb / mullion lines: one continuous stroke across the wall thickness,
    #     away from the gap ends (not several parallel lines that happen to add up)
    band = profile[:, np.abs(v) <= 0.5 * t]
    longest = np.zeros(len(band), int)
    for i, row in enumerate(band):
        runs = _runs(row)
        longest[i] = max((b - a for a, b in runs), default=0)
    across = longest >= 0.75 * band.shape[1]
    long_rows = band.mean(axis=0) >= 0.6          # rows that are long lines along the whole gap
    def cluttered(c: float) -> bool:
        """Opening symbols are organised lines; text or furniture crossing the gap is scattered
        ink. A cut is not trusted where the band around it is cluttered."""
        lo, hi = max(0, int(c - 1.2 * t)), min(len(band), int(c + 1.2 * t) + 1)
        window = band[lo:hi]
        if window.size == 0 or not (~long_rows).any():
            return False
        return float(window[:, ~long_rows].mean()) > 0.12
    for a, b in _runs(across):
        if b - a > max(4.0, 0.4 * t):
            continue  # not a thin perpendicular line: the band is just full of parallel lines
        c = (a + b - 1) / 2
        if not 1.2 * t <= c <= gap - 1.2 * t:
            continue
        # A frame/mullion line is isolated; text strokes have other ink right next to them.
        near = [k for k in list(range(int(a) - 5, int(a) - 1)) + list(range(int(b) + 1, int(b) + 5)) if 0 <= k < len(band)]
        clutter = band[near][:, ~long_rows].mean() if near and (~long_rows).any() else 0.0
        if clutter < 0.2:
            cuts.add(float(c))
    # (b) a long line inside the band (glazing) that starts or stops well inside the gap
    lines = profile[:, inner].any(axis=1)
    closed = lines.copy()
    tol = max(2, int(0.6 * t))
    for a, b in _runs(~lines):
        if b - a <= tol and a > 0 and b < len(lines):
            closed[a:b] = True
    for a, b in _runs(closed):
        if b - a < 2 * t:
            continue
        for edge in (a, b):
            if 1.2 * t <= edge <= gap - 1.2 * t:
                cuts.add(float(edge))
    cuts = {c for c in cuts if not cluttered(c)}
    if not cuts:
        return [(0.0, gap)]
    points = [0.0]
    for c in sorted(cuts):
        if c - points[-1] >= 1.2 * t:
            points.append(c)
    if gap - points[-1] < 1.2 * t:
        points[-1] = gap
    else:
        points.append(gap)
    return list(zip(points[:-1], points[1:]))


def _line_coverage(symbol_ink, start, d, n, t, u0, u1) -> float:
    width = u1 - u0
    u = np.linspace(u0 + 0.1 * width, u1 - 0.1 * width, max(5, int(width * 0.8)))
    profile, _ = _band_profile(symbol_ink, start, d, n, t, u)
    return float(profile.any(axis=1).mean())


def object_edges_enabled() -> bool:
    import os
    return os.environ.get("FLOORPLAN_OBJECT_EDGES", "1").strip().lower() not in ("0", "off", "false", "no")


class _ObjectOutlines:
    """Lines along a wall gap that belong to an object drawn inside a room, not to the opening.

    A threshold, glazing or sliding-panel symbol lies inside the wall band. A counter, an
    appliance, a cabinet or a stair box can also have one edge on the wall axis - its other
    edges then go into the room and it closes with a back edge parallel to the gap, well
    beyond the wall's thickness. Such a component is an object outline: it says nothing about
    an opening there, and must not turn a wide gap into a 'drawn' (sealed) one.
    """

    def __init__(self, symbol_ink: np.ndarray):
        self.count, self.labels, self.stats, _ = cv2.connectedComponentsWithStats(
            (symbol_ink > 0).astype(np.uint8), connectivity=8)
        self._verdict: dict = {}

    def _profile(self, start, d, n, t, u0, u1) -> np.ndarray:
        """Component ids along the gap (rows = positions along it, cols = offsets across the band),
        sampled exactly like _line_coverage."""
        width = u1 - u0
        u = np.linspace(u0 + 0.1 * width, u1 - 0.1 * width, max(5, int(width * 0.8)))
        v = np.arange(-0.5 * t - 2, 0.5 * t + 3, 1.0)
        xs = start[0] + d[0] * u[:, None] + n[0] * v[None, :]
        ys = start[1] + d[1] * u[:, None] + n[1] * v[None, :]
        h, w = self.labels.shape
        xi, yi = np.floor(xs + 0.5).astype(int), np.floor(ys + 0.5).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        out = np.zeros(xi.shape, np.int32)
        out[ok] = self.labels[yi[ok], xi[ok]]
        return out

    def is_object(self, k, start, d, n, t, gap) -> bool:
        key = (k, round(float(start[0])), round(float(start[1])), round(float(d[0]), 2), round(float(d[1]), 2))
        if key in self._verdict:
            return self._verdict[key]
        x, y, bw, bh, _ = (int(v) for v in self.stats[k])
        ys, xs = np.nonzero(self.labels[y:y + bh, x:x + bw] == k)
        px, py = xs + x - start[0], ys + y - start[1]
        uu = px * d[0] + py * d[1]
        vv = px * n[0] + py * n[1]
        verdict = False
        lo, hi = float(uu.min()), float(uu.max())
        if lo >= -1.5 * t and hi <= gap + 1.5 * t and hi - lo >= 0.85 * gap:
            # spans exactly the opening, jamb to jamb: drawn for it (an overhead garage door,
            # a door leaf drawn open in plan) - opening evidence, not a room object
            self._verdict[key] = False
            return False
        for side in (1, -1):
            far = side * vv >= 3.0 * t
            if far.sum() < 3 * t:
                continue
            # a back edge: many pixels at one depth, spread along the gap direction
            bins = np.round(side * vv[far] / 2.0).astype(int)
            best = np.bincount(bins - bins.min()).argmax() + bins.min()
            on = np.abs(np.round(side * vv[far] / 2.0) - best) <= 1
            span = float(np.ptp(uu[far][on])) if on.any() else 0.0
            if span >= max(3.0 * t, 0.3 * gap):
                verdict = True
                break
        self._verdict[key] = verdict
        return verdict

    def coverage(self, start, d, n, t, u0, u1, gap) -> tuple[float, int]:
        """(line coverage without object outlines, number of object outlines on the gap)."""
        prof = self._profile(start, d, n, t, u0, u1)
        objects = [k for k in set(np.unique(prof).tolist()) - {0} if self.is_object(k, start, d, n, t, gap)]
        hit = (prof > 0) & ~np.isin(prof, objects) if objects else prof > 0
        return float(hit.any(axis=1).mean()), len(objects)


def thin_lines(symbol_ink: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    """Drawing lines that are not part of a wall (wall edges excluded)."""
    return cv2.bitwise_and(symbol_ink, cv2.bitwise_not(cv2.dilate(wall_mask, np.ones((5, 5), np.uint8))))


def _touches_wall(wall_mask, point, direction, reach) -> bool:
    s = np.arange(0.0, reach + 1.0, 1.0)
    return bool(_cross_occupancy(wall_mask, point, direction, np.array([-direction[1], direction[0]]), s, [-1.0, 0.0, 1.0]).max() > 0)


def _continues_wall(wall_mask, point, direction, typical_t) -> bool:
    """A wall face or axis continues the line beyond this endpoint (wall runs along the line)."""
    n = np.array([-direction[1], direction[0]])
    s = np.arange(3.0, 2.0 * typical_t, 1.0)
    for offset in np.arange(-0.6 * typical_t, 0.6 * typical_t + 1, 2.0):
        if _cross_occupancy(wall_mask, point, direction, n, s, [offset]).mean() >= 0.8:
            return True
    return False


def _trace(thin, wall_mask, point, d, reach, limit) -> tuple[np.ndarray, bool]:
    """Follow a drawn line from `point` along d while ink continues; report whether it ends on a wall."""
    n = np.array([-d[1], d[0]])
    s = np.arange(1.0, limit, 1.0)
    ink = _cross_occupancy(thin, point, d, n, s, [-1.5, -0.5, 0.5, 1.5]) > 0
    misses = 0
    last = 0.0
    for step, has_ink in zip(s, ink):
        if has_ink:
            last, misses = step, 0
        else:
            misses += 1
            if misses > 5:
                break
    end = np.asarray(point, float) + d * last
    return end, _touches_wall(wall_mask, end, d, reach)


def _line_bridged_gaps(wall_mask, thin, typical_t, existing) -> list[tuple]:
    """Openings drawn as thin lines spanning from wall to wall (outlines, frames, open edges).

    Used for openings that are not a gap along a wall axis, e.g. doors set between two
    wall corners at an angle. Pairs of parallel bridging lines one wall-thickness apart
    form an opening band; a single line counts only if it continues a wall line.
    """
    min_len = max(10, int(round(1.0 * typical_t)))
    closed = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    lines = hough_tiles(closed, threshold=max(8, int(0.6 * min_len)), minLineLength=min_len, maxLineGap=4)
    if lines is None:
        return []
    reach = max(6.0, 0.35 * typical_t)
    bridging = []
    for x1, y1, x2, y2 in lines[:, 0, :].astype(float):
        a, b = np.array([x1, y1]), np.array([x2, y2])
        d = (b - a) / max(1e-9, float(np.linalg.norm(b - a)))
        # Hough returns faint lines in pieces: trace each piece to the true ends of the drawn line.
        b, touch_b = _trace(closed, wall_mask, b, d, reach, 32 * typical_t)
        a, touch_a = _trace(closed, wall_mask, a, -d, reach, 32 * typical_t)
        length = float(np.linalg.norm(b - a))
        if touch_a and touch_b and min_len <= length <= 32 * typical_t:
            bridging.append((a, b, d, length))
    # de-duplicate (Hough returns several segments per drawn line)
    unique = []
    for a, b, d, length in sorted(bridging, key=lambda r: -r[3]):
        n = np.array([-d[1], d[0]])
        if any(abs(float(np.dot(d, e))) > 0.99 and abs(float(np.dot((a + b) / 2 - p, np.array([-e[1], e[0]])))) <= 2.5
               and abs(float(np.dot((a + b) / 2 - (p + q) / 2, e))) <= 0.5 * L for p, q, e, L in unique):
            continue
        unique.append((a, b, d, length))
    pattern = _repeated_lines(unique, typical_t, wall_mask)
    unique = [u for k, u in enumerate(unique) if k not in pattern]
    used = set()
    found = []
    for i, (a, b, d, length) in enumerate(unique):
        if i in used:
            continue
        n = np.array([-d[1], d[0]])
        partner = None
        for j, (a2, b2, d2, length2) in enumerate(unique):
            if j == i or j in used or abs(float(np.dot(d, d2))) < 0.99:
                continue
            gap = float(np.dot((a2 + b2) / 2 - (a + b) / 2, n))
            along = abs(float(np.dot((a2 + b2) / 2 - (a + b) / 2, d)))
            if 0.4 * typical_t <= abs(gap) <= 1.8 * typical_t and along <= 0.3 * max(length, length2) and abs(length - length2) <= 0.3 * max(length, length2):
                if partner is None or abs(gap) > abs(partner[1]):
                    partner = (j, gap)
        if partner is not None:
            j, gap = partner
            a2, b2 = unique[j][0], unique[j][1]
            lo, hi = sorted((0.0, gap))
            inner_v = np.linspace(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo), 5)
            u = np.linspace(0.15 * length, 0.85 * length, max(6, int(length / 2)))
            fill = _cross_occupancy(thin, a, d, n, u, inner_v).mean()
            if fill > 0.22 and not _parallel_strokes(thin, a, d, n, u, lo, hi):
                continue  # densely inked between the lines: text or hatching, not an opening outline
            used.update((i, j))
            centre_shift = n * gap / 2
            start, stop = a + centre_shift, b + centre_shift
            thickness = abs(gap)
        else:
            if not (_continues_wall(wall_mask, b, d, typical_t) or _continues_wall(wall_mask, a, -d, typical_t)):
                continue
            # A boundary line has paper on both sides; the edge of a filled shape (a thick
            # door leaf, a dark bar) has ink on one side.
            s_mid = np.linspace(0.2 * length, 0.8 * length, 9)
            sides = [_cross_occupancy(thin, a, d, n, s_mid, side * np.array([3.0, 4.0])).mean() for side in (-1, 1)]
            if max(sides) > 0.3:
                continue
            start, stop, thickness = a, b, typical_t
        # extend to the wall faces the line touches
        start = start - d * reach * 0.5
        stop = stop + d * reach * 0.5
        mid = (start + stop) / 2
        if any(abs(float(np.dot(mid - np.asarray(c[0]), np.array([-c[2][1], c[2][0]])))) <= 0.7 * max(c[3], thickness)
               and abs(float(np.dot(mid - (np.asarray(c[0]) + np.asarray(c[1])) / 2, c[2]))) <= 0.5 * c[4]
               for c in existing):
            continue  # already found as an axis gap
        found.append((start, stop, thickness, partner is not None))
    return found


def _merge_through_fragments(wall_mask, merged: list[list]) -> list[list]:
    """Join consecutive gaps on one wall line that are separated only by a short fragment
    thinner than the wall (a symbol that survived the thickness filter, e.g. blurred sliding
    panels). A real pier between adjacent windows spans the full wall thickness."""
    gaps = [m for m in merged if len(m) < 6 or m[5] == "wall_gap"]
    others = [m for m in merged if not (len(m) < 6 or m[5] == "wall_gap")]
    used = set()
    out = []
    for i, a in enumerate(gaps):
        if i in used:
            continue
        start, stop = np.asarray(a[0], float), np.asarray(a[1], float)
        t = a[2]
        changed = True
        while changed:
            changed = False
            d = (stop - start) / max(1e-9, np.linalg.norm(stop - start))
            n = np.array([-d[1], d[0]])
            for j, b in enumerate(gaps):
                if j == i or j in used:
                    continue
                b0, b1 = np.asarray(b[0], float), np.asarray(b[1], float)
                e = (b1 - b0) / max(1e-9, np.linalg.norm(b1 - b0))
                if abs(float(np.dot(d, e))) < 0.98 or abs(float(np.dot((b0 + b1) / 2 - start, n))) > 0.5 * max(t, b[2]):
                    continue
                ub = sorted((float(np.dot(b0 - start, d)), float(np.dot(b1 - start, d))))
                L = float(np.linalg.norm(stop - start))
                between = ub[0] - L
                if not (0.0 < between <= 4.0 * t):
                    continue
                # thickness of the fragment between the two gaps (thin = symbol, full = pier)
                s_mid = np.linspace(L + 0.25 * between, L + 0.75 * between, 3)
                v = np.arange(-0.6 * t, 0.6 * t + 1, 1.0)
                prof = _cross_occupancy(wall_mask, start, n, d, v, s_mid)
                thickness = float((prof >= 0.5).sum())
                if thickness >= (0.75 * t if between <= 1.5 * t else 0.6 * t):
                    continue
                stop = start + d * ub[1]
                a = [start, stop, max(t, b[2]), max(a[3], b[3]), max(a[4], b[4])] + a[5:]
                used.add(j)
                changed = True
        out.append([start, stop] + list(a[2:]))
    return out + others


# ---------------------------------------------------------------------------
# Jamb verification: is each end of a gap held by structure?
# ---------------------------------------------------------------------------
#
# A gap is the space between two pieces of the wall mask. When one of those pieces is not
# structure (a furniture block, a plant, a marker, title text) the gap is not an opening, and
# sealing it would create a partition that is not on the drawing. Pieces are judged locally,
# at the gap, from the geometry around the end — never by deleting wall components globally:
#
#   wall network   the piece belongs to a wall component that spans many wall thicknesses
#   wall run       the piece continues behind the end as a straight band of wall width
#   junction       a wall meets the piece at an angle (corner / T, stub on a wall)
#   wall line      a short piece of wall width centred on the gap axis, with a wall run further
#                  along the same axis (pier between openings, mullion, short jamb)
#
# Short real pieces (piers, mullions, stubs) satisfy one of these. Many real pieces do not
# (free-standing columns, bay-window piers, posts of walls drawn as outlines), so missing
# structural evidence alone never rejects a gap: an unsupported jamb is rejected only when the
# plan's wall tone (engine.wall_tone) also says the piece is not drawn like the walls. On
# plans where walls and furniture share one tone there is no such evidence and nothing is
# rejected.

def _enter_jamb(mask, e, g, n, t, reach) -> float | None:
    s = np.arange(0.0, reach + 1.0, 1.0)
    occ = _cross_occupancy(mask, e, g, n, s, np.linspace(-0.3 * t, 0.3 * t, 5))
    hit = np.flatnonzero(occ >= 0.5)
    return float(s[hit[0]]) if hit.size else None


def _across(mask, p, n, limit) -> float:
    """Width of the wall run through p, measured along n (0 if p is not wall)."""
    v = np.arange(-limit, limit + 1.0, 1.0)
    inside = _sample(mask, p[0] + n[0] * v, p[1] + n[1] * v) > 0
    c = len(v) // 2
    if not inside[c]:
        return 0.0
    lo = hi = c
    while lo > 0 and inside[lo - 1]:
        lo -= 1
    while hi < len(v) - 1 and inside[hi + 1]:
        hi += 1
    return float(hi - lo + 1)


def _branch(mask, p, g, n, depth, width, t_eff, wide) -> bool:
    """A wall meets the piece at an angle: on one side, a band of wall thickness leaves the
    piece's side face and runs for at least two wall thicknesses without interruption."""
    s = np.arange(-0.2 * t_eff, depth + 0.3 * t_eff + 1.0, 1.0)
    v_len = 2.0 * t_eff
    for side in (-1, 1):
        v = side * np.arange(0.5 * width + 1, 0.5 * width + 1 + v_len, 1.0)
        xs = p[0] + g[0] * s[:, None] + n[0] * v[None, :]
        ys = p[1] + g[1] * s[:, None] + n[1] * v[None, :]
        rows = (_sample(mask, xs, ys) > 0).all(axis=1)            # the branch crosses this row fully
        runs = _runs(rows)
        if any(0.4 * t_eff <= b - a <= wide for a, b in runs):
            return True
    return False


def _jamb_support(mask, comp, extent, e, g, t, typical_t) -> tuple[bool, str, np.ndarray | None]:
    """Structural support of the piece of wall that ends a gap at e (g points into the piece)."""
    g = np.asarray(g, float)
    n = np.array([-g[1], g[0]])
    t_eff = max(float(t), 0.6 * typical_t)
    wide = 1.6 * max(float(t), typical_t)          # wider than this across the axis: not a wall band
    s_in = _enter_jamb(mask, e, g, n, t_eff, max(3.0, 0.6 * t_eff))
    if s_in is None:
        return True, "end not on wall (not verified)", None
    p = np.asarray(e, float) + g * s_in
    depth = _depth_behind(mask, p, -g, n, t_eff)
    positions = np.unique(np.clip(np.r_[0.5 * depth, np.arange(0.5 * t_eff, depth, max(1.0, 0.5 * t_eff))], 0, max(0.0, depth - 1)))
    width = min(_across(mask, p + g * s, n, 6 * wide) for s in positions)
    if width <= wide:
        if depth >= 3 * t_eff:
            return True, "wall run", p
        xi, yi = int(round(p[0])), int(round(p[1]))
        h, w = comp.shape
        k = int(comp[min(max(yi, 0), h - 1), min(max(xi, 0), w - 1)])
        if k and extent[k] >= 8 * typical_t:
            return True, "wall network", p
        if _branch(mask, p, g, n, depth, width, t_eff, wide):
            return True, "junction", p
        # further along the axis (through openings and past further piers of wall width) the
        # wall line reaches a wall run or the wall network
        far = np.arange(depth + 1.0, depth + 40 * t_eff, 1.0)
        occ = _cross_occupancy(mask, p, g, n, far, np.linspace(-0.3 * t_eff, 0.3 * t_eff, 5))
        solid = (occ[:-1] >= 0.6) & (occ[1:] >= 0.6)
        for a0, a1 in _runs(solid)[:6]:
            if far[a0] - depth > 16 * t_eff:
                break
            q = p + g * (far[a0] + 1.0)
            qk = int(comp[min(max(int(round(q[1])), 0), h - 1), min(max(int(round(q[0])), 0), w - 1)])
            run = (_depth_behind(mask, q, -g, n, t_eff) >= 3 * t_eff and _across(mask, q + g * 1.5 * t_eff, n, 6 * wide) <= wide)
            network = qk and extent[qk] >= 8 * typical_t and width <= 1.35 * typical_t
            if run or network:
                return True, "wall line (pier / mullion)", p
            if _across(mask, q + g * min(0.5 * (a1 - a0), t_eff), n, 6 * wide) > wide:
                break                                          # not another wall-width piece
        return False, f"short detached piece ({depth:.0f} px deep, {width:.0f} px across) not on a wall line", p
    if depth <= wide:
        across = _across(mask, p + g * 0.5 * depth, n, 8 * t_eff)
        if across >= 4 * t_eff:
            return True, "wall face (corner / T)", p
    return False, f"piece wider than a wall band ({depth:.0f} px deep, {width:.0f} px across)", p


def _repeated_lines(unique, typical_t, wall_mask=None) -> set:
    """Indices of bridging lines that belong to a set of repeated parallel lines (stair treads,
    bath / shower hatching, shelving): at least four parallel, overlapping lines of similar
    length, with no wall between them, spread over an area (more than five wall thicknesses).
    Opening outlines are two or three lines within one wall thickness, folding-door leaves stay
    within a few thicknesses of the opening, and door symbols of neighbouring rooms are
    separated by a wall."""
    m = len(unique)
    parent = list(range(m))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    offsets = {}
    for i, (a, b, d, length) in enumerate(unique):
        n = np.array([-d[1], d[0]])
        for j in range(i + 1, m):
            a2, b2, d2, length2 = unique[j]
            if abs(float(np.dot(d, d2))) < 0.99 or min(length, length2) < 0.6 * max(length, length2):
                continue
            off = float(np.dot((a2 + b2) / 2 - (a + b) / 2, n))
            if abs(off) > 4 * typical_t:
                continue
            u = sorted((float(np.dot(a2 - a, d)), float(np.dot(b2 - a, d))))
            if min(u[1], length) - max(u[0], 0.0) < 0.5 * min(length, length2):
                continue
            if wall_mask is not None:
                m0, m1 = (a + b) / 2, (a2 + b2) / 2
                k = np.linspace(0.0, 1.0, max(3, int(np.linalg.norm(m1 - m0)) + 1))
                if (_sample(wall_mask, m0[0] + (m1[0] - m0[0]) * k, m0[1] + (m1[1] - m0[1]) * k) > 0).any():
                    continue
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(m):
        groups.setdefault(find(i), []).append(i)
    out = set()
    for members in groups.values():
        if len(members) < 4:
            continue
        a, b, d, _ = unique[members[0]]
        n = np.array([-d[1], d[0]])
        offs = [float(np.dot((unique[k][0] + unique[k][1]) / 2 - a, n)) for k in members]
        if max(offs) - min(offs) > 5.0 * typical_t:      # a pattern covers an area, not one opening's frame / leaves
            out.update(members)
    return out


def _piece_tone(comp, p, g, t, typical_t, image_lab, tone) -> float | None:
    """Distance from the plan's wall tone of the jamb itself: the piece's pixels in the strip
    on the gap axis, one wall thickness wide, where the gap ends (furniture pressed against the
    far side of a real pier does not count). None: no evidence."""
    if tone is None or image_lab is None or p is None:
        return None
    h, w = comp.shape
    x, y = int(round(p[0])), int(round(p[1]))
    if not (0 <= x < w and 0 <= y < h) or comp[y, x] == 0:
        return None
    g = np.asarray(g, float)
    n = np.array([-g[1], g[0]])
    t_eff = max(float(t), 0.6 * typical_t)
    half = max(1.0, (0.35 if t_eff >= 5 else 0.5) * t_eff)       # stay off anti-aliased edges
    ss, vv = np.meshgrid(np.arange(0.0, 1.5 * t_eff + 1.0, 1.0), np.arange(-half, half + 0.5, 1.0))
    xs = np.clip(np.round(p[0] + g[0] * ss + n[0] * vv).astype(int), 0, w - 1).ravel()
    ys = np.clip(np.round(p[1] + g[1] * ss + n[1] * vv).astype(int), 0, h - 1).ravel()
    keep = comp[ys, xs] == comp[y, x]
    px = image_lab[ys[keep], xs[keep]]
    if len(px) >= 12:
        # anti-aliasing mixes ink towards the paper: judge the more ink-like half of the strip
        px = px[px[:, 0] <= np.median(px[:, 0])]
    return tone.distance(px)


_DOOR_DIST: dict = {}


def _drawn_door(symbol_ink, m) -> bool:
    """A long plain gap found from one wall end only is kept when a door is drawn in it: a swing
    arc of radius ~ its width, or two half-width arcs (double door). The door is evidence for the
    opening - and so for the wall it interrupts - where the second jamb was not recognised."""
    if not DOOR_RESCUE:
        return False
    from engine.opening_detection import Frame, arc_scores

    key = id(symbol_ink)
    if key not in _DOOR_DIST:
        _DOOR_DIST.clear()
        _DOOR_DIST[key] = cv2.distanceTransform(255 - (symbol_ink > 0).astype(np.uint8) * 255, cv2.DIST_L2, 3)
    start, stop, t = np.asarray(m[0], float), np.asarray(m[1], float), float(m[2])
    w = float(np.linalg.norm(stop - start))
    if w < 4 * t:
        return False
    d = (stop - start) / w
    a = arc_scores(_DOOR_DIST[key], Frame(start, d, np.array([-d[1], d[0]]), w, max(3.0, t)), 1.0)
    return max(a["arc"], a["double_arc"]) >= 0.75


DOOR_RESCUE = True


def _wall_directions(bands, min_share: float = 0.03) -> list[float]:
    """Directions (degrees mod 180) the plan's walls take, weighted by length."""
    weights: dict = {}
    total = 0.0
    for b in bands:
        d = np.subtract(b.p1, b.p0).astype(float)
        L = float(np.hypot(*d))
        if L <= 0:
            continue
        a = int(round(math.degrees(math.atan2(d[1], d[0])) % 180.0)) % 180
        weights[a] = weights.get(a, 0.0) + L
        total += L
    if total <= 0:
        return []
    return [float(a) for a, w in weights.items() if w >= min_share * total]


def _along_walls(v, dirs: list[float], tol: float = 8.0) -> bool:
    a = math.degrees(math.atan2(float(v[1]), float(v[0]))) % 180.0
    return any(min(abs(a - d), 180.0 - abs(a - d)) <= tol for d in dirs)


def find_gaps(wall_mask, symbol_ink, bands, typical_t, kernel, thick_max, thinnest=None, rejected=None, tone=None, image_lab=None,
              corner_ends=None) -> tuple[list[OpeningCandidate], list[OpeningCandidate]]:
    """Wall gaps from wall end faces (and diagonal band ends). `corner_ends` adds band ends that
    stop at a wall corner (see _terminal_band_ends): they have no end face of their own and are
    exempt from the end-face guards; callers must confirm what they produce (_seal_corner_doors)."""
    h, w = wall_mask.shape
    thinnest = float(thinnest) if thinnest else typical_t
    symbol_ink = thin_lines(symbol_ink, wall_mask)
    outlines = _ObjectOutlines(symbol_ink) if object_edges_enabled() else None
    ends = _axis_end_faces(wall_mask, kernel, thick_max)
    for band in bands:
        if band.orientation != "diagonal":
            continue
        d = band.direction
        ends.append((np.asarray(band.p1, float), d, band.thickness))
        ends.append((np.asarray(band.p0, float), -d, band.thickness))

    n_regular = len(ends)
    ends = ends + list(corner_ends or [])
    raw = []
    for index, (end, d, t) in enumerate(ends):
        t = max(3.0, t)
        n = np.array([-d[1], d[0]])
        refined = _refine_end(wall_mask, end, d, n, t)
        if refined is None:
            continue
        corner = index >= n_regular
        if not corner and not _is_free_end(wall_mask, refined, d, n, t):
            continue
        if not corner and not _aligned_band_behind(wall_mask, refined, d, n, t):
            continue
        max_steps = int(min(32 * t, 0.6 * min(h, w)))
        gap = _scan_gap(wall_mask, refined, d, n, t, max_steps)
        if gap is None:
            continue
        coverage = _line_coverage(symbol_ink, refined, d, n, t, 0.0, gap)
        if outlines is not None and coverage >= 0.5:
            coverage = outlines.coverage(refined, d, n, t, 0.0, gap, gap)[0]
        long_plain = False
        # clearly thinner than the thinnest real walls — judged on the wall's cross-section just
        # behind the end (end faces of anti-aliased walls are often shorter than the wall)
        thin_piece = max(t, _width_behind(wall_mask, refined, d, n, t)) < 0.7 * thinnest
        if coverage < 0.5:
            # A plain gap must start from a real wall run, not from the side of a short pier.
            scale_t = max(t, 0.8 * typical_t)
            if gap > 22 * scale_t or _depth_behind(wall_mask, refined, d, n, t) < 1.5 * t:
                continue
            # Long plain gaps (wide passages, double doors drawn without a band symbol) are kept
            # only if the wall end on the other side faces back (checked after merging).
            long_plain = gap > 13 * scale_t
        # plain gaps from pieces much thinner than the plan's walls (often symbols, e.g. a
        # thick-drawn door leaf) must be confirmed from the other end as well
        raw.append((refined + d * 0.5, refined + d * (gap + 0.5), t, coverage, long_plain or (coverage < 0.5 and thin_piece)))

    # Merge detections of the same gap found from both ends.
    merged: list[list] = []
    long_plain_ids = set()
    for start, stop, t, cov, long_plain in sorted(raw, key=lambda r: -np.linalg.norm(r[1] - r[0])):
        mid = (start + stop) / 2
        d = (stop - start) / max(1e-9, np.linalg.norm(stop - start))
        for m in merged:
            e = (m[1] - m[0]) / max(1e-9, np.linalg.norm(m[1] - m[0]))
            if abs(float(np.dot(d, e))) < 0.95:
                continue
            nrm = np.array([-e[1], e[0]])
            if abs(float(np.dot(mid - m[0], nrm))) > 0.6 * max(t, m[2]):
                continue
            L = float(np.linalg.norm(m[1] - m[0]))
            u = sorted((float(np.dot(start - m[0], e)), float(np.dot(stop - m[0], e))))
            overlap = min(u[1], L) - max(u[0], 0.0)
            if overlap >= 0.5 * min(L, u[1] - u[0]):
                m[4] += 1
                m[2] = max(m[2], t)
                break
        else:
            merged.append([start, stop, t, cov, 1])
            if long_plain:
                long_plain_ids.add(id(merged[-1]))
    merged = [m for m in merged if not (id(m) in long_plain_ids and m[4] < 2 and not _drawn_door(symbol_ink, m))]

    # A wall end that is already the jamb of another opening does not also start a
    # plain, one-sided gap in a different direction (e.g. across a hallway).
    def _is_jamb_of_other(m):
        start, stop, t, cov, votes = m[:5]
        if votes != 1 or cov >= 0.5:
            return False
        d = (stop - start) / max(1e-9, np.linalg.norm(stop - start))
        for o in merged:
            if o is m:
                continue
            e = (o[1] - o[0]) / max(1e-9, np.linalg.norm(o[1] - o[0]))
            if abs(float(np.dot(d, e))) > 0.94:
                continue
            if min(np.linalg.norm(start - o[0]), np.linalg.norm(start - o[1])) <= 0.9 * max(t, o[2]):
                return True
        return False

    existing = []
    for start, stop, t, cov, votes in merged:
        L = float(np.linalg.norm(stop - start))
        existing.append((start, stop, (stop - start) / max(1e-9, L), t, L))
    wall_dirs = _wall_directions(bands)
    for start, stop, t, paired in _line_bridged_gaps(wall_mask, symbol_ink, typical_t, existing):
        if wall_dirs and not _along_walls(stop - start, wall_dirs) and float(np.linalg.norm(stop - start)) > 10.0 * typical_t:
            # at an angle no wall takes, an opening is a door set across a corner (door-sized);
            # longer lines there are hatch or furniture, not openings
            continue
        cov = _line_coverage(symbol_ink, start, (stop - start) / np.linalg.norm(stop - start),
                             np.array([-(stop - start)[1], (stop - start)[0]]) / np.linalg.norm(stop - start), t, 0.0, float(np.linalg.norm(stop - start)))
        merged.append([start, stop, t, max(cov, 0.6), 0, "line_pair" if paired else "single_line"])
    merged = [m for m in merged if not _is_jamb_of_other(m)]

    merged = _merge_through_fragments(wall_mask, merged)

    _, comp, cstats, _ = cv2.connectedComponentsWithStats((wall_mask > 0).astype(np.uint8), connectivity=8)
    extent = np.maximum(cstats[:, cv2.CC_STAT_WIDTH], cstats[:, cv2.CC_STAT_HEIGHT]).astype(float)

    candidates, cracks = [], []
    for start, stop, t, cov, votes, *rest in merged:
        source = rest[0] if rest else "wall_gap"
        gap = float(np.linalg.norm(stop - start))
        d = (stop - start) / max(1e-9, gap)
        n = np.array([-d[1], d[0]])
        orient = "horizontal" if abs(d[1]) < 0.2 else "vertical" if abs(d[0]) < 0.2 else "diagonal"
        if gap < max(1.1 * t, 6.0):
            cracks.append(OpeningCandidate("", tuple(map(float, start)), tuple(map(float, stop)), t, orient, cov, votes, source))
            continue
        jambs, refuted = [], False
        for e, g in ((start, -d), (stop, d)):
            ok, reason, piece = _jamb_support(wall_mask, comp, extent, e, g, t, typical_t)
            if not ok:
                dist = _piece_tone(comp, piece, g, t, typical_t, image_lab, tone)
                if dist is not None and dist >= 1.0:
                    refuted = True
                    reason += f"; tone differs from the walls ({dist:.1f}x tolerance)"
                else:
                    reason += "; kept: " + ("tone matches the walls" if dist is not None else "no tone evidence")
            jambs.append(reason)
        if refuted:
            if rejected is not None:
                rejected.append(OpeningCandidate("", tuple(map(float, start)), tuple(map(float, stop)), t, orient, cov, votes,
                                                 source, jambs=tuple(jambs)))
            continue
        parts = _split_gap(symbol_ink, start, d, n, t, gap) if source == "wall_gap" else [(0.0, gap)]
        for u0, u1 in parts:
            a, b = start + d * u0, start + d * u1
            cov_part = _line_coverage(symbol_ink, start, d, n, t, u0, u1) if source == "wall_gap" else cov
            if outlines is not None and source == "wall_gap" and cov_part >= 0.5:
                cov_part = outlines.coverage(start, d, n, t, u0, u1, gap)[0]
            candidates.append(OpeningCandidate("", tuple(map(float, a)), tuple(map(float, b)), t, orient, cov_part, votes, source,
                                               jambs=tuple(jambs)))
    candidates.sort(key=lambda c: (round(c.center[1] / 10), c.center[0]))
    for k, c in enumerate(candidates, 1):
        c.id = f"gap-{k:02d}"
    return candidates, cracks


def _same_opening(a: OpeningCandidate, b: OpeningCandidate) -> bool:
    """Two candidates describing one opening: parallel, on one wall axis, overlapping by at least
    half of the shorter (extents found from different ends differ by a few pixels)."""
    da, db = np.asarray(a.tangent), np.asarray(b.tangent)
    if abs(float(np.dot(da, db))) < 0.95:
        return False
    if abs(float(np.dot(np.subtract(b.center, a.center), a.normal))) > 0.75 * max(a.thickness, b.thickness, 3.0):
        return False
    ua = sorted((0.0, a.width))
    ub = sorted((float(np.dot(np.subtract(b.start, a.start), da)), float(np.dot(np.subtract(b.end, a.start), da))))
    overlap = min(ua[1], ub[1]) - max(ua[0], ub[0])
    return overlap >= 0.5 * min(a.width, b.width)


def _terminal_band_ends(wall_mask, bands) -> list[tuple]:
    """Ends of axis wall bands where the wall really stops (no wall continues along the axis).
    Many are ordinary end faces; the ones that matter here stop at a corner, where the wall
    turns: such a jamb has no end face of its own, so a door next to it is never scanned."""
    out = []
    for band in bands:
        if band.orientation == "diagonal":
            continue
        d = band.direction
        t = max(3.0, band.thickness)
        for p, dd in ((np.asarray(band.p1, float), d), (np.asarray(band.p0, float), -d)):
            nn = np.array([-dd[1], dd[0]])
            beyond = _cross_occupancy(wall_mask, p, dd, nn, np.array([2.0, 3.0, 4.0]), np.linspace(-0.3 * t, 0.3 * t, 5))
            if beyond.mean() < 0.35:
                out.append((p, dd, t))
    return out


def _seal_corner_doors(image, prelim, bands, typical_t, band_kernel, thick_max, thinnest, tone, image_lab):
    """Doors whose jamb is a wall corner. Corner ends produce many gaps (counters, fixture
    outlines, open-plan edges), so a corner gap is sealed only when the drawing shows a door
    there: a swing arc (or double swing) of a door-sized width, measured in this plan's own door
    width; a clean gap or one confirmed from both jambs (not a seal along fixture lines); both
    jambs drawn like the walls (wall tone, when the plan has a reliable one); and a seal that
    separates two rooms, not a sliver smaller than a closet. Returns the accepted candidates."""
    from engine.opening_detection import classify_candidate        # local: opening_detection imports this module
    wall_mask, symbol_ink = prelim.wall_mask, prelim.symbol_ink
    cands, _ = find_gaps(wall_mask, symbol_ink, bands, typical_t, band_kernel, thick_max, thinnest, None, tone, image_lab,
                         corner_ends=_terminal_band_ends(wall_mask, bands))
    new = [c for c in cands if not any(_same_opening(c, o) for o in list(prelim.candidates) + list(prelim.cracks))]
    if not new:
        return []
    thin = thin_lines(symbol_ink, wall_mask)
    dist = cv2.distanceTransform(cv2.bitwise_not(thin), cv2.DIST_L2, 3)
    doors = []
    for c in prelim.candidates:
        r = classify_candidate(c, prelim, thin, dist, None, thin)
        if r["type"] == "door" and r["evidence"]["arc_score"] >= 0.6:
            doors.append(c.width)
    if len(doors) < 3:
        return []
    door = float(np.median(doors))
    _, comp, cstats, _ = cv2.connectedComponentsWithStats((wall_mask > 0).astype(np.uint8), connectivity=8)
    extent = np.maximum(cstats[:, cv2.CC_STAT_WIDTH], cstats[:, cv2.CC_STAT_HEIGHT]).astype(float)
    accepted = []
    for c in new:
        assign_relations([c], prelim.space_labels)
        ev = classify_candidate(c, prelim, thin, dist, None, thin)["evidence"]
        single = ev["arc_score"] >= 0.6 and 0.6 * door <= c.width <= 1.5 * door
        double = ev["double_arc_score"] >= 0.6 and 1.2 * door <= c.width <= 2.5 * door
        clean = c.found_from >= 2 or c.line_coverage < 0.5
        if not ((single or double) and clean):
            continue
        d = np.asarray(c.tangent)
        tones = []
        for e, g in ((np.asarray(c.start), -d), (np.asarray(c.end), d)):
            _, _, piece = _jamb_support(wall_mask, comp, extent, e, g, c.thickness, typical_t)
            tones.append(_piece_tone(comp, piece, g, c.thickness, typical_t, image_lab, tone))
        if any(x is not None and x >= 1.0 for x in tones):
            continue                                   # a jamb not drawn like the walls (counter, fixture)
        accepted.append(c)
    keep = []
    for c in accepted:                                 # a door joins two rooms
        _, labels, _ = segment_spaces(wall_mask, list(prelim.candidates) + keep + [c], prelim.cracks, typical_t)
        assign_relations([c], labels)
        sizes = [float((labels == k).sum()) for k in c.sides if k and k > 0]
        if c.relation == "between_rooms" and sizes and min(sizes) >= (1.2 * door) ** 2:
            c.jambs = tuple(c.jambs) + ("corner door: swing arc on a door-sized corner gap",)
            keep.append(c)
    return keep


# ---------------------------------------------------------------------------
# 5. Sealing and spaces
# ---------------------------------------------------------------------------

def _bridge(mask, cand: OpeningCandidate) -> None:
    d = cand.tangent
    n = cand.normal
    half = cand.thickness / 2 + 1.0
    a = np.asarray(cand.start) - d * 1.0
    b = np.asarray(cand.end) + d * 1.0
    poly = np.array([a - n * half, b - n * half, b + n * half, a + n * half])
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)


def segment_spaces(wall_mask, candidates, cracks, typical_t) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    h, w = wall_mask.shape
    sealed = wall_mask.copy()
    for cand in list(candidates) + list(cracks):
        _bridge(sealed, cand)
    sealed = cv2.morphologyEx(sealed, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    free = cv2.copyMakeBorder(cv2.bitwise_not(sealed), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    flood = free.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 128)
    exterior = (flood == 128)[1:-1, 1:-1]
    interior = ((free == 255)[1:-1, 1:-1] & ~exterior).astype(np.uint8)

    labels = np.zeros((h, w), np.int32)
    labels[exterior] = EXTERIOR
    count, comp, stats, centroids = cv2.connectedComponentsWithStats(interior, connectivity=4)
    min_area = max(400.0, (3.0 * typical_t) ** 2)
    min_clear = max(4.0, 0.75 * typical_t)
    dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
    spaces = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        region = comp == i
        if float(dist[region].max()) < min_clear:
            continue  # too narrow to be a room (sliver between parallel walls, symbol interior, …)
        k = len(spaces) + 1
        labels[region] = k
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        polygon = cv2.approxPolyDP(contour, max(1.5, 0.002 * cv2.arcLength(contour, True)), True)
        x, y, bw, bh = (int(v) for v in stats[i, :4])
        spaces.append({
            "id": f"space_{k}",
            "bbox": {"x": x, "y": y, "width": bw, "height": bh},
            "area_pixels": area,
            "center": (int(centroids[i][0]), int(centroids[i][1])),
            "polygon": [(int(p[0][0]), int(p[0][1])) for p in polygon],
        })
    return sealed, labels, spaces


# ---------------------------------------------------------------------------
# 6. Topology
# ---------------------------------------------------------------------------

def _side_label(labels, point, n, side, t) -> int:
    votes = {}
    for depth in (0.5 * t + 3, 0.5 * t + 0.6 * t + 3, 0.5 * t + 1.2 * t + 3):
        x = point[0] + n[0] * side * depth
        y = point[1] + n[1] * side * depth
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < labels.shape[1] and 0 <= yi < labels.shape[0]:
            value = int(labels[yi, xi])
        else:
            value = EXTERIOR
        if value != 0:
            votes[value] = votes.get(value, 0) + 1
    return max(votes, key=votes.get) if votes else 0


def assign_relations(candidates, labels) -> None:
    for cand in candidates:
        n = cand.normal
        sides = []
        for side in (-1, 1):
            votes = {}
            for f in (0.3, 0.5, 0.7):
                p = np.asarray(cand.start) + cand.tangent * cand.width * f
                value = _side_label(labels, p, n, side, cand.thickness)
                votes[value] = votes.get(value, 0) + 1
            sides.append(max(votes, key=votes.get))
        a, b = sides
        cand.sides = (a, b)
        if a > 0 and b > 0:
            cand.relation = "between_rooms" if a != b else "same_space"
        elif (a > 0 and b == EXTERIOR) or (b > 0 and a == EXTERIOR):
            cand.relation = "room_to_exterior"
        else:
            cand.relation = "unknown"


def wall_adjacency(labels, spaces, thick_max) -> set:
    """Pairs of spaces separated only by a wall (reach = thickest wall + margin)."""
    pairs = set()
    reach = max(3, int(round(1.15 * thick_max + 2)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    h, w = labels.shape
    for k, space in enumerate(spaces, 1):
        # dilate only within the space's bounding box + reach (identical result, much cheaper)
        b = space["bbox"]
        x0, y0 = max(0, b["x"] - reach - 1), max(0, b["y"] - reach - 1)
        x1, y1 = min(w, b["x"] + b["width"] + reach + 1), min(h, b["y"] + b["height"] + reach + 1)
        roi = labels[y0:y1, x0:x1]
        grown = cv2.dilate((roi == k).astype(np.uint8), kernel) > 0
        for other in np.unique(roi[grown]):
            if other > k:
                pairs.add((k, int(other)))
    return pairs


# ---------------------------------------------------------------------------

def estimate_skew(wall_mask: np.ndarray, typical_t: float) -> float:
    """Dominant wall angle (degrees) relative to the image axes, from straight wall edges."""
    edges = cv2.morphologyEx(wall_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    min_len = max(30, int(4 * typical_t))
    # Fine angular resolution: HoughLinesP walks segments along its quantised angle bins.
    lines = hough_tiles(edges, theta=np.pi / 1440, threshold=max(20, int(0.5 * min_len)), minLineLength=min_len, maxLineGap=3)
    if lines is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in lines[:, 0, :].astype(float):
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        a = (a + 45.0) % 90.0 - 45.0          # fold onto the nearest axis
        if abs(a) <= 20.0:
            angles.append(a)
            weights.append(math.hypot(x2 - x1, y2 - y1))
    if len(angles) < 4:
        return 0.0
    order = np.argsort(angles)
    cum = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(angles)[order][np.searchsorted(cum, cum[-1] / 2)])


def analyze_structure(image: np.ndarray, deskew: bool = True) -> StructureResult:
    """Structural analysis in original image coordinates (deskewing internally if needed)."""
    first = _analyze_frame(image)
    if not deskew:
        return first
    skew = estimate_skew(first.wall_mask, first.wall_thickness)
    if abs(skew) < 0.35:
        first.skew_degrees = skew
        return first

    h, w = image.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)   # rotating by +skew aligns the walls
    cos, sin = abs(rot[0, 0]), abs(rot[0, 1])
    W, H = int(math.ceil(w * cos + h * sin)), int(math.ceil(w * sin + h * cos))
    rot[0, 2] += W / 2 - w / 2
    rot[1, 2] += H / 2 - h / 2
    inv = cv2.invertAffineTransform(rot)
    border = (255, 255, 255) if image.ndim == 3 else 255
    rotated = cv2.warpAffine(image, rot, (W, H), flags=cv2.INTER_LINEAR, borderValue=border)
    work = _analyze_frame(rotated)
    work.skew_degrees = skew

    def to_original(point):
        x, y = point
        return (float(inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]), float(inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]))

    def back(mask, interp=cv2.INTER_NEAREST, dtype=None):
        src = mask.astype(np.float32) if dtype is not None else mask
        out = cv2.warpAffine(src, inv, (w, h), flags=interp, borderValue=0)
        return out.astype(dtype) if dtype is not None else out

    candidates = []
    for c in work.candidates:
        candidates.append(OpeningCandidate(c.id, to_original(c.start), to_original(c.end), c.thickness, c.orientation,
                                           c.line_coverage, c.found_from, c.source, c.sides, c.relation, c.jambs))
    rejected = [OpeningCandidate(c.id, to_original(c.start), to_original(c.end), c.thickness, c.orientation,
                                 c.line_coverage, c.found_from, c.source, c.sides, c.relation, c.jambs)
                for c in work.rejected_candidates]
    spaces = []
    for sp in work.spaces:
        poly = [tuple(int(round(v)) for v in to_original(pt)) for pt in sp["polygon"]]
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        cx, cy = to_original(sp["center"])
        spaces.append({**sp, "polygon": poly, "center": (int(round(cx)), int(round(cy))),
                       "bbox": {"x": min(xs), "y": min(ys), "width": max(xs) - min(xs) + 1, "height": max(ys) - min(ys) + 1}})
    return StructureResult(
        gray=first.gray, ink=first.ink, symbol_ink=first.symbol_ink, wall_mask=back(work.wall_mask),
        wall_kernel=work.wall_kernel, line_width=work.line_width, wall_thickness=work.wall_thickness,
        bands=work.bands, candidates=candidates, cracks=work.cracks, sealed_mask=back(work.sealed_mask),
        space_labels=back(work.space_labels, dtype=np.int32), spaces=spaces, wall_adjacency=work.wall_adjacency,
        skew_degrees=skew, work=work, to_original=to_original, rejected_candidates=rejected,
    )


def _analyze_frame(image: np.ndarray) -> StructureResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ink, symbol_ink = _binarize(gray)
    from engine.wall_inference import infer_walls      # local import: wall_inference builds on this module
    walls = infer_walls(ink)
    kernel, line_width, thinnest = walls.kernel, walls.line_width, walls.thinnest
    wall_mask = walls.mask
    band_kernel = walls.min_kernel                       # smallest accepted wall class
    typical_t = _typical_thickness(wall_mask, thinnest)
    thinnest = _thinnest_wall(thinnest, typical_t)
    widths = _ridge_widths(wall_mask)
    thick_max = float(np.percentile(widths, 95)) if widths.size else 2 * typical_t
    bands = _axis_bands(wall_mask, band_kernel, thick_max) + _diagonal_bands(wall_mask, typical_t)
    from engine.wall_tone import learn_wall_tone
    colour = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    tone = learn_wall_tone(colour, wall_mask, typical_t)
    image_lab = cv2.cvtColor(colour, cv2.COLOR_BGR2LAB)
    rejected: list = []
    candidates, cracks = find_gaps(wall_mask, symbol_ink, bands, typical_t, band_kernel, thick_max, thinnest, rejected,
                                   tone, image_lab)
    sealed, labels, spaces = segment_spaces(wall_mask, candidates, cracks, typical_t)
    # Fallback for walls drawn with ordinary line weight: when one space holds most of the
    # floor area, look for straight lines that behave like walls and re-segment with them.
    from engine.wall_inference import interior_collapsed, recover_line_walls
    if interior_collapsed(spaces):
        line_walls, evidence = recover_line_walls(ink, wall_mask, typical_t)
        walls.line_evidence = evidence
        # One line can split a space at most once; recovering rooms needs a network of lines.
        if sum(e["decision"] == "wall" for e in evidence) >= 2:
            mask2 = cv2.bitwise_or(wall_mask, line_walls)
            thin2 = min(thinnest, 3.0)
            bands2 = _axis_bands(mask2, 3, thick_max) + _diagonal_bands(mask2, typical_t)
            rejected2: list = []
            cand2, cracks2 = find_gaps(mask2, symbol_ink, bands2, typical_t, 3, thick_max, thin2, rejected2, tone, image_lab)
            sealed2, labels2, spaces2 = segment_spaces(mask2, cand2, cracks2, typical_t)
            # Keep the recovered structure only if it adds real rooms, not a stray split.
            floor = sum(sp["area_pixels"] for sp in spaces)
            def substantial(sps):
                return sum(sp["area_pixels"] >= 0.01 * floor for sp in sps)
            if substantial(spaces2) >= substantial(spaces) + 2:
                walls.line_walls = line_walls
                walls.mask = wall_mask = mask2
                walls.min_kernel = band_kernel = 3
                walls.thinnest = thinnest = thin2
                bands, candidates, cracks, rejected = bands2, cand2, cracks2, rejected2
                sealed, labels, spaces = sealed2, labels2, spaces2
    assign_relations(candidates, labels)
    prelim = StructureResult(
        gray=gray, ink=ink, symbol_ink=symbol_ink, wall_mask=wall_mask, wall_kernel=kernel,
        line_width=line_width, wall_thickness=typical_t, bands=bands, candidates=candidates, cracks=cracks,
        sealed_mask=sealed, space_labels=labels, spaces=spaces, walls=walls)
    corner_doors = _seal_corner_doors(image, prelim, bands, typical_t, band_kernel, thick_max, thinnest, tone, image_lab)
    if corner_doors:
        candidates = list(candidates) + corner_doors
        sealed, labels, spaces = segment_spaces(wall_mask, candidates, cracks, typical_t)
        assign_relations(candidates, labels)
    return StructureResult(
        gray=gray, ink=ink, symbol_ink=symbol_ink, wall_mask=wall_mask, wall_kernel=kernel,
        line_width=line_width, wall_thickness=typical_t, bands=bands, candidates=candidates, cracks=cracks,
        sealed_mask=sealed, space_labels=labels, spaces=spaces,
        wall_adjacency=wall_adjacency(labels, spaces, thick_max), walls=walls, rejected_candidates=rejected,
    )
