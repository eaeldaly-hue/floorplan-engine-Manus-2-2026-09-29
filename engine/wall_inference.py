"""Multi-scale structural wall inference.

A floor plan can contain several stroke classes at once — 1–2 px drawing lines, 5 px
interior walls, 12 px exterior walls — so there is no single "wall thickness". The
primary thickness filter (engine.structure.kernel_search) finds the earliest clear
plateau of the surviving-area curve and reliably captures the dominant wall class.
When thinner walls are only a few pixels thicker than the drawing lines, their plateau
is too short to be chosen, and they are erased together with the lines.

This module recovers such secondary wall classes without lowering the global threshold:

1. Secondary scale. On the same surviving-area curve, look below the primary kernel for
   a kernel on a locally flat stretch that still holds clearly more ink than the primary
   mask (a stroke population thicker than the lines but thinner than the primary walls).
2. Candidates. Strokes kept by that smaller kernel but not by the primary one.
3. Structural validation. A candidate component is accepted as wall only if it behaves
   like architecture:
     * it contains a long straight run (walls are long and straight; text, arrowheads,
       fixtures and furniture glyphs are not);
     * it connects to the primary wall structure, directly or through other accepted
       candidates (interior walls meet the building's walls; dimension lines, legends
       and free-standing furniture do not);
     * it is not collinear fill inside a thicker wall's line (window glazing, frames,
       sliding panels and thresholds drawn in a wall gap continue the thick wall's axis
       inside its thickness — real thin walls meet thick walls at an angle).
   Rejected candidates keep a reason code, so the decision can be inspected.

No pixel is ever added that is not ink in the drawing: accepted walls are the drawn
strokes themselves, never connecting lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from engine import structure as st

# Reason codes for rejected secondary candidates (diagnostics).
REASONS = {1: "no long straight run", 2: "not connected to wall structure", 3: "collinear fill in a thicker wall",
           4: "hinged at a wall end (door leaf)"}


@dataclass
class WallInference:
    mask: np.ndarray                 # final wall mask: primary ∪ accepted secondary walls (0/255)
    kernel: int                      # primary (thick-class) kernel
    min_kernel: int                  # smallest kernel of an accepted wall class (for band extraction)
    line_width: float
    thinnest: float
    primary: np.ndarray              # primary wall mask
    secondary_kernel: int | None = None
    accepted: np.ndarray | None = None   # secondary pixels accepted as wall
    rejected: np.ndarray | None = None   # secondary candidate pixels, value = reason code
    classes: list[dict] = field(default_factory=list)
    components: list[dict] = field(default_factory=list)
    line_walls: np.ndarray | None = None       # walls recovered from line-weight strokes (fallback)
    line_evidence: list[dict] = field(default_factory=list)


def _areas_up_to(ink: np.ndarray, areas: list[float], k: int) -> list[float]:
    areas = list(areas)
    while len(areas) < k:
        areas.append(st._surviving_area_at(ink, len(areas) + 1))
    return areas


def secondary_kernel(ink: np.ndarray, areas: list[float], primary: int) -> int | None:
    """Kernel of a stroke class between the drawing lines and the primary walls, or None.

    areas[i] is the surviving ink area at kernel i + 1. A secondary class exists when, below
    the primary kernel, the curve has a locally flat step (the class is stable there) and the
    ink surviving at that kernel exceeds the primary mask's by a meaningful share.
    """
    if primary <= 4:
        return None
    areas = _areas_up_to(ink, areas, primary + 1)
    total = max(areas[0], 1.0)
    base = areas[primary - 1]
    best, best_drop = None, 0.03
    for k in range(3, primary):
        here, nxt = areas[k - 1], areas[k]
        drop = (here - nxt) / max(here, 1.0)
        extra = (here - base) / total
        if extra >= 0.01 and drop < best_drop:
            best, best_drop = k, drop
    return best


def _long_runs(mask: np.ndarray, length: int) -> np.ndarray:
    """Pixels on horizontal or vertical straight runs of at least `length` px."""
    length = max(3, int(length))
    h = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
    return cv2.bitwise_or(h, v)


def _collinear_fill(primary: np.ndarray, x: int, y: int, w: int, h: int, own_t: float) -> bool:
    """True if a straight thin run continues a thicker wall's axis inside its thickness.

    The run's bounding box (x, y, w, h) is probed just beyond both ends along its axis.
    """
    H, W = primary.shape
    horizontal = w >= h
    reach = int(max(4, 3 * own_t))
    if horizontal:
        cy = y + h // 2
        ends = [(x - 1, -1), (x + w, 1)]
        for ex, sgn in ends:
            for d in range(1, reach):
                px = ex + sgn * d
                if not (0 <= px < W):
                    break
                col = primary[:, px]
                if col[cy] > 0:
                    # thickness of the primary wall at this column around cy
                    top = cy
                    while top > 0 and col[top - 1] > 0:
                        top -= 1
                    bot = cy
                    while bot < H - 1 and col[bot + 1] > 0:
                        bot += 1
                    thick = bot - top + 1
                    if thick >= 1.5 * own_t and thick <= 6 * own_t + 30:
                        return True
                    break
    else:
        cx = x + w // 2
        ends = [(y - 1, -1), (y + h, 1)]
        for ey, sgn in ends:
            for d in range(1, reach):
                py = ey + sgn * d
                if not (0 <= py < H):
                    break
                row = primary[py, :]
                if row[cx] > 0:
                    left = cx
                    while left > 0 and row[left - 1] > 0:
                        left -= 1
                    right = cx
                    while right < W - 1 and row[right + 1] > 0:
                        right += 1
                    thick = right - left + 1
                    if thick >= 1.5 * own_t and thick <= 6 * own_t + 30:
                        return True
                    break
    return False


def infer_walls(ink: np.ndarray) -> WallInference:
    kernel, line_width, thinnest, areas = st.kernel_search(ink)
    primary = st.build_wall_mask(ink, kernel)
    result = WallInference(mask=primary, kernel=kernel, min_kernel=kernel, line_width=line_width,
                           thinnest=thinnest, primary=primary,
                           classes=[{"kernel": kernel, "role": "primary", "pixels": int((primary > 0).sum())}])
    k2 = secondary_kernel(ink, areas, kernel)
    if k2 is None:
        return result

    near_primary = cv2.dilate(primary, np.ones((3, 3), np.uint8))
    own_t = float(k2 + 1)
    min_run = int(max(8 * own_t, 0.025 * min(ink.shape)))
    # Cheap first look: if no stroke of this class forms a long straight run outside the
    # primary walls, the class holds no walls (text, symbols, fixtures) — stop here.
    core = cv2.erode(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (k2, k2)))
    quick = cv2.bitwise_and(cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_RECT, (k2 + 2, k2 + 2))), ink)
    quick = cv2.bitwise_and(quick, cv2.bitwise_not(near_primary))
    if not _long_runs(quick, min_run).any():
        result.secondary_kernel = k2
        result.classes.append({"kernel": k2, "role": "secondary", "candidates": None, "accepted": 0,
                               "accepted_pixels": 0, "rejected": {REASONS[1]: "all"}})
        return result

    thin = st.build_wall_mask(ink, k2)
    extra = cv2.bitwise_and(thin, cv2.bitwise_not(near_primary))
    runs = _long_runs(extra, min_run)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(extra, connectivity=8)
    has_run = np.zeros(count, bool)
    has_run[np.unique(labels[runs > 0])] = True
    has_run[0] = False

    # Per straight run (component of the run mask): collinear fill, and door-leaf hinge test.
    widths = st._ridge_widths(primary)
    thick_max = float(np.percentile(widths, 95)) if widths.size else 2.0 * kernel
    faces = st._axis_end_faces(primary, kernel, thick_max)
    face_pts = np.array([f[0] for f in faces]) if faces else np.zeros((0, 2))
    face_len = np.array([f[2] for f in faces]) if faces else np.zeros(0)
    rcount, rlabels, rstats, _ = cv2.connectedComponentsWithStats(runs, connectivity=8)
    fill = np.zeros(count, bool)
    leaf = np.zeros(count, bool)
    for r in range(1, rcount):
        x, y, w, h = (int(v) for v in rstats[r, :4])
        comps = np.unique(labels[y:y + h, x:x + w][rlabels[y:y + h, x:x + w] == r])
        if _collinear_fill(primary, x, y, w, h, own_t):
            fill[comps] = True
        if face_pts.size:
            if w >= h:
                ends = np.array([[x - 1, y + h / 2], [x + w, y + h / 2]])
            else:
                ends = np.array([[x + w / 2, y - 1], [x + w / 2, y + h]])
            for e in ends:
                dist = np.hypot(*(face_pts - e).T)
                if (dist <= np.maximum(6.0, 1.2 * face_len)).any():
                    leaf[comps] = True
    fill[0] = leaf[0] = False

    # Connectivity to the primary structure, propagated through accepted candidates.
    candidate = has_run & ~fill & ~leaf
    touch_primary = np.zeros(count, bool)
    touch_primary[np.unique(labels[(cv2.dilate(near_primary, np.ones((3, 3), np.uint8)) > 0) & (labels > 0)])] = True
    # Component adjacency (components within ~2 px of each other).
    neighbours: dict[int, set] = {}
    dil = cv2.dilate(extra, np.ones((5, 5), np.uint8))
    _, near_map = cv2.connectedComponents(dil, connectivity=8)
    inside = labels > 0
    pairs = np.unique(np.stack([near_map[inside], labels[inside]]), axis=1)   # (group, component)
    members_of: dict[int, list[int]] = {}
    for group, comp in pairs.T:
        members_of.setdefault(int(group), []).append(int(comp))
    for members in members_of.values():
        for m in members:
            neighbours.setdefault(m, set()).update(v for v in members if v != m)
    accepted = np.zeros(count, bool)
    frontier = [i for i in range(1, count) if candidate[i] and touch_primary[i]]
    for i in frontier:
        accepted[i] = True
    while frontier:
        i = frontier.pop()
        for j in neighbours.get(i, ()):
            if candidate[j] and not accepted[j]:
                accepted[j] = True
                frontier.append(j)

    reason = np.zeros(count, np.uint8)
    reason[~has_run] = 1
    reason[has_run & fill] = 3
    reason[has_run & ~fill & leaf] = 4
    reason[candidate & ~accepted] = 2
    reason[0] = 0
    reason[accepted] = 0

    acc_mask = np.where(accepted[labels], 255, 0).astype(np.uint8)
    # Re-attach the accepted strokes to the primary walls (the pixels removed by the
    # dilation above are ink of the same strokes, so nothing is fabricated).
    joined = cv2.bitwise_and(thin, cv2.dilate(acc_mask, np.ones((5, 5), np.uint8)))
    mask = cv2.bitwise_or(primary, joined)

    result.mask = mask
    result.secondary_kernel = k2
    result.min_kernel = k2 if accepted.any() else kernel
    if accepted.any():
        # The thinnest real wall class is now the accepted secondary one (downstream rules
        # that treat pieces "much thinner than the plan's walls" as symbols use this).
        widths = st._ridge_widths(acc_mask)
        if widths.size:
            result.thinnest = float(min(thinnest, max(float(k2 + 1), float(np.median(widths)))))
    result.accepted = acc_mask
    result.rejected = np.where(labels > 0, reason[labels], 0).astype(np.uint8)
    result.classes.append({"kernel": k2, "role": "secondary", "candidates": int(count - 1),
                           "accepted": int(accepted.sum()), "accepted_pixels": int((acc_mask > 0).sum()),
                           "rejected": {REASONS[c]: int((reason[1:] == c).sum()) for c in REASONS}})
    for i in range(1, count):
        x, y, w, h, a = (int(v) for v in stats[i])
        result.components.append({"bbox": [x, y, w, h], "area": a, "accepted": bool(accepted[i]),
                                  "reason": REASONS.get(int(reason[i])) if not accepted[i] else "wall"})
    return result


# ---------------------------------------------------------------------------
# Line-structure recovery (walls drawn with ordinary line weight)
# ---------------------------------------------------------------------------

def interior_collapsed(spaces: list[dict], min_share: float = 0.6) -> bool:
    """One space holds most of the enclosed floor area: the wall structure is incomplete."""
    if not spaces:
        return False
    areas = np.array([sp["area_pixels"] for sp in spaces], float)
    return float(areas.max() / areas.sum()) >= min_share


def _segments(mask: np.ndarray, horizontal: bool) -> list[tuple[int, int, int, int]]:
    """Straight runs of a run mask as (x, y, w, h) boxes (one orientation)."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, count):
        x, y, w, h, _ = (int(v) for v in stats[i])
        if (w >= h) == horizontal:
            out.append((x, y, w, h))
    return out


def _anchor(structure: np.ndarray, seg, horizontal: bool, end: int, reach: int) -> tuple[bool, bool]:
    """(anchored, collinear) for one end of a segment.

    anchored: structure (walls or other lines) lies in a small window around the end — a
    T-junction, an L-corner or contact with a wall — excluding the segment's own pixels.
    collinear: that structure continues straight along the segment's axis beyond the end (the
    segment then sits inside a gap of that wall — glazing or a threshold).
    """
    H, W = structure.shape
    x, y, w, h = seg
    if horizontal:
        cy = y + h // 2
        if end == 0:
            x0, x1 = max(0, x - reach), min(W, x + 3)
            axis = structure[max(0, cy - 1):cy + 2, max(0, x - 4 * reach):max(0, x - 1)]
            axis = axis[:, ::-1]
        else:
            x0, x1 = max(0, x + w - 3), min(W, x + w + reach)
            axis = structure[max(0, cy - 1):cy + 2, min(W, x + w + 1):min(W, x + w + 4 * reach)]
        win = structure[max(0, cy - reach):min(H, cy + reach + 1), x0:x1].copy()
        oy = max(0, cy - reach)
        own_rows = slice(max(0, y - oy), max(0, y + h - oy))
        own_cols = slice(max(0, x - x0), max(0, x + w - x0))
        win[own_rows, own_cols] = 0
        col = axis.any(axis=0)
    else:
        cx = x + w // 2
        if end == 0:
            y0, y1 = max(0, y - reach), min(H, y + 3)
            axis = structure[max(0, y - 4 * reach):max(0, y - 1), max(0, cx - 1):cx + 2]
            axis = axis[::-1, :]
        else:
            y0, y1 = max(0, y + h - 3), min(H, y + h + reach)
            axis = structure[min(H, y + h + 1):min(H, y + h + 4 * reach), max(0, cx - 1):cx + 2]
        win = structure[y0:y1, max(0, cx - reach):min(W, cx + reach + 1)].copy()
        ox = max(0, cx - reach)
        own_rows = slice(max(0, y - y0), max(0, y + h - y0))
        own_cols = slice(max(0, x - ox), max(0, x + w - ox))
        win[own_rows, own_cols] = 0
        col = axis.any(axis=1)
    anchored = bool(win.any())
    run = 0
    for v in col:
        if not v:
            break
        run += 1
    return anchored, anchored and run >= 2 * reach


def _axis_hit(structure: np.ndarray, seg, horizontal: bool, end: int, distance: int) -> bool:
    """Structure straight ahead of the end, along the axis (±2 px), within `distance`."""
    H, W = structure.shape
    x, y, w, h = seg
    if horizontal:
        cy = y + h // 2
        rows = slice(max(0, cy - 2), cy + 3)
        cols = slice(max(0, x - distance), max(0, x - 1)) if end == 0 else slice(min(W, x + w + 1), min(W, x + w + distance))
    else:
        cx = x + w // 2
        cols = slice(max(0, cx - 2), cx + 3)
        rows = slice(max(0, y - distance), max(0, y - 1)) if end == 0 else slice(min(H, y + h + 1), min(H, y + h + distance))
    return bool(structure[rows, cols].any())


def _end_kind(structure: np.ndarray, seg, horizontal: bool, end: int, reach: int, jamb_reach: int) -> str:
    """'junction' (meets structure at the end), 'collinear' (that structure continues the
    segment's axis), 'jamb' (structure straight ahead after a door-width gap: the end of a wall
    interrupted by an opening) or 'free'."""
    anchored, collinear = _anchor(structure, seg, horizontal, end, reach)
    if anchored:
        return "collinear" if collinear else "junction"
    return "jamb" if _axis_hit(structure, seg, horizontal, end, jamb_reach) else "free"


def recover_line_walls(ink: np.ndarray, walls: np.ndarray, typical_t: float) -> tuple[np.ndarray, list[dict]]:
    """Straight drawn lines that behave like walls, for plans whose walls use line weight.

    Returns (mask of accepted line walls, evidence records). Candidate lines are long straight
    runs that are not part of a repeated set of closely spaced parallel lines (stairs,
    shelving, hatching). Each end is classified against the walls and the other candidates:
    a junction (meets structure), a jamb (structure continues along the axis after a
    door-width gap), collinear (the line sits inside a wall's own gap: glazing, threshold) or
    free. A line is a wall when neither end is free, at least one is a junction, and it is
    not collinear fill at both ends; the accepted network must connect to the existing walls.
    """
    ys, xs = np.nonzero(walls)
    if xs.size == 0:
        return np.zeros_like(walls), []
    extent = min(xs.max() - xs.min(), ys.max() - ys.min())
    min_len = int(max(16, 0.05 * extent))
    lines = cv2.bitwise_and(ink, cv2.bitwise_not(cv2.dilate(walls, np.ones((3, 3), np.uint8))))
    hmask = cv2.morphologyEx(lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1)))
    vmask = cv2.morphologyEx(lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len)))
    segs = [(s, True) for s in _segments(hmask, True)] + [(s, False) for s in _segments(vmask, False)]
    spacing = max(6.0, 2.5 * typical_t)

    def parallel_neighbours(i):
        (x, y, w, h), hor = segs[i]
        n = 0
        for j, ((x2, y2, w2, h2), hor2) in enumerate(segs):
            if j == i or hor2 != hor:
                continue
            if hor:
                overlap = min(x + w, x2 + w2) - max(x, x2)
                if overlap >= 0.5 * min(w, w2) and abs((y + h / 2) - (y2 + h2 / 2)) <= spacing:
                    n += 1
            else:
                overlap = min(y + h, y2 + h2) - max(y, y2)
                if overlap >= 0.5 * min(h, h2) and abs((x + w / 2) - (x2 + w2 / 2)) <= spacing:
                    n += 1
        return n

    status = ["repeated parallel lines (stairs/shelving)" if parallel_neighbours(i) >= 2 else "pending"
              for i in range(len(segs))]
    cand = np.zeros_like(walls)
    for i, ((x, y, w, h), _) in enumerate(segs):
        if status[i] == "pending":
            cand[y:y + h, x:x + w] = np.maximum(cand[y:y + h, x:x + w], lines[y:y + h, x:x + w])
    structure = cv2.bitwise_or(walls, cand)
    # Wall end faces (jambs): a line starting at one is a door leaf, not a partition.
    wwidths = st._ridge_widths(walls)
    thick_max = float(np.percentile(wwidths, 95)) if wwidths.size else 2.0 * typical_t
    faces = st._axis_end_faces(walls, max(3, int(round(typical_t))), thick_max)
    face_pts = np.array([f[0] for f in faces]) if faces else np.zeros((0, 2))
    face_len = np.array([f[2] for f in faces]) if faces else np.zeros(0)

    def hinged(seg, hor):
        if not face_pts.size:
            return False
        x, y, w, h = seg
        ends = ([[x - 1, y + h / 2], [x + w, y + h / 2]] if hor else [[x + w / 2, y - 1], [x + w / 2, y + h]])
        for e in np.array(ends, float):
            if (np.hypot(*(face_pts - e).T) <= np.maximum(6.0, 1.2 * face_len)).any():
                return True
        return False

    reach = int(max(4, 1.5 * typical_t))
    jamb_reach = int(max(reach + 4, 0.12 * extent))
    keep = np.zeros(len(segs), bool)
    for i, (seg, hor) in enumerate(segs):
        if status[i] != "pending":
            continue
        x, y, w, h = seg
        kinds = [_end_kind(structure, seg, hor, e, reach, jamb_reach) for e in (0, 1)]
        if hinged(seg, hor):
            status[i] = "hinged at a wall end (door leaf)"
        elif "free" in kinds:
            status[i] = "free end (not anchored)"
        elif kinds.count("collinear") == 2:
            status[i] = "fill inside a wall gap (glazing/threshold)"
        elif "junction" not in kinds and "collinear" not in kinds:
            status[i] = "no junction with structure"
        else:
            keep[i] = True
            status[i] = "wall"
    accepted = np.zeros_like(walls)
    for i, ((x, y, w, h), _) in enumerate(segs):
        if keep[i]:
            accepted[y:y + h, x:x + w] = np.maximum(accepted[y:y + h, x:x + w], lines[y:y + h, x:x + w])
    # The accepted network must connect to the existing walls (directly or through each other).
    if accepted.any():
        joined = cv2.bitwise_or(cv2.dilate(accepted, np.ones((5, 5), np.uint8)), walls)
        count, comp = cv2.connectedComponents(joined, connectivity=8)
        wall_ids = set(np.unique(comp[walls > 0])) - {0}
        connected = np.isin(comp, list(wall_ids)) & (accepted > 0)
        for i, ((x, y, w, h), _) in enumerate(segs):
            if keep[i] and not connected[y:y + h, x:x + w].any():
                status[i] = "not connected to the walls"
        accepted = np.where(connected, 255, 0).astype(np.uint8)
    # Represent accepted lines at a minimal structural width, inside the stroke's own
    # 1-px neighbourhood (no new walls: only the drawn lines themselves).
    widened = cv2.bitwise_and(cv2.dilate(accepted, np.ones((3, 3), np.uint8)), cv2.dilate(ink, np.ones((3, 3), np.uint8)))
    evidence = [{"bbox": list(seg), "horizontal": hor, "decision": st_} for (seg, hor), st_ in zip(segs, status)]
    return widened, evidence
