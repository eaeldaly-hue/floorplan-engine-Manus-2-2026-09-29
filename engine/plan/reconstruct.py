"""Architectural reconstruction: drawing -> Plan Model.

    strokes ─► wall hypotheses (filled band | face pair | single line)
            ─► re-drawing check: each wall drawn in its own convention must match the ink
            ─► wall network: collinear continuation, junctions, main components
            ─► openings: gaps between collinear walls / a wall end and a crossing wall,
               classified by door swing / window lines
            ─► spaces: the regions enclosed by walls + opening closures (exterior = outside)
            ─► topology: spaces connected through openings

Nothing is created without ink: every wall must pass the re-drawing check; openings only
close gaps between supported walls.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .model import Opening, PlanModel, Space, Stroke, Wall
from .primitives import extract_strokes, ink_mask
from .walls import dominant_separations, face_pairs, lattice_members, line_weight

HEAVY_FACES = True
SUPPORT_MIN = 0.7                 # share of a wall's re-drawing that must be ink


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _unit(p, q):
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy) or 1.0
    return dx / L, dy / L


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _point_segment(p, a, b) -> tuple[float, float]:
    """(distance, t in [0,1]) of point p to segment ab."""
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy or 1e-9
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy)), t


def _line_intersection(p, d, q, e):
    """Intersection of lines p + s d and q + u e (None if parallel)."""
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-9:
        return None
    s = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return (p[0] + s * d[0], p[1] + s * d[1]), s


# ---------------------------------------------------------------------------
# 1. wall hypotheses
# ---------------------------------------------------------------------------

def wall_hypotheses(strokes: list[Stroke], ink: np.ndarray) -> tuple[list[Wall], dict]:
    h, w = ink.shape
    diag = math.hypot(h, w)
    lw = line_weight(strokes)
    thick_min = max(2.2 * lw, lw + 4.0)
    thin = [i for i, s in enumerate(strokes) if s.width < thick_min]
    lattice = lattice_members(strokes, thin, max_spacing=max(30.0, 0.03 * diag))
    for i in lattice:
        strokes[i].roles["lattice"] = 1.0
    max_sep = min(90.0, max(25.0, 0.05 * diag))
    # wall faces can be drawn with a heavy pen (or thickened by the binarisation): a stroke above
    # the thin threshold can still be a face, if it is narrow next to the band it bounds
    candidates = [i for i, s in enumerate(strokes) if i not in lattice and s.length >= 15
                  and s.width < (max(thick_min, 0.3 * max_sep) if HEAVY_FACES else thick_min)]
    pairs = [p for p in face_pairs(strokes, candidates, ink, max_sep, lw)
             if p.w < thick_min or p.w <= 0.5 * p.d]
    classes = dominant_separations(pairs)

    def closeness(d):
        if not classes:
            return 0.5
        return max(0.0, 1.0 - min(abs(d - c) / (0.25 * c + 1.0) for c in classes))

    walls: list[Wall] = []
    used: set[int] = set()
    occupied: dict[int, list] = {}                 # a face may face several walls along its length
    for p in sorted(pairs, key=lambda p: -(p.b - p.a) * (0.3 + closeness(p.d))):
        if closeness(p.d) <= 0.0:
            continue
        ov = p.b - p.a
        if any(min(p.b, b) - max(p.a, a) > 0.3 * ov for k in (p.i, p.j) for a, b in occupied.get(k, [])):
            continue
        for k in (p.i, p.j):
            occupied.setdefault(k, []).append((p.a, p.b))
        used.update((p.i, p.j))
        t = math.radians(p.theta)
        tx, ty, nx, ny = math.cos(t), math.sin(t), -math.sin(t), math.cos(t)
        a0 = (p.a * tx + p.rho * nx, p.a * ty + p.rho * ny)
        a1 = (p.b * tx + p.rho * nx, p.b * ty + p.rho * ny)
        style = "hollow" if p.inside_ink < 0.25 else "hatched"
        walls.append(Wall(id="", p0=a0, p1=a1, thickness=p.d + p.w, style=style, confidence=0.5 + 0.4 * closeness(p.d),
                          provenance=[f"face pair (spacing {p.d:.1f}px, interior ink {p.inside_ink:.2f})"],
                          strokes=[p.i, p.j]))
        walls[-1]._face = (p.d, p.w)
    # single-stroke walls. Bands at or above the thick threshold are filled walls. A lighter
    # single line is a wall only in the plan's wall pen family: close to the filled walls' width
    # when the plan has filled walls (lines drawn just under the threshold), otherwise the
    # dominant pen of long lines. Door leaves, frames, furniture and symbols are lighter. In a
    # plan whose walls are face pairs, single lines are not walls (unpaired faces, frames).
    min_len = max(30.0, 0.02 * diag)
    filled_ws = []
    paired_len = sum(Wl.length for Wl in walls)
    bands = [s.width for i, s in enumerate(strokes) if i not in used and s.width >= thick_min and s.length >= 3 * s.width]
    # bars far thinner than every wall class: door leaves and frames in a plan drawn with face
    # pairs, unless they carry more length than the pairs (the plan's walls are heavy lines)
    bar_len = sum(s.length for i, s in enumerate(strokes) if i not in used and s.width >= thick_min
                  and classes and s.width < 0.5 * min(classes) and s.length >= min_len)
    bars_are_walls = bar_len > paired_len
    band_w = float(np.median(bands)) if bands else None
    for i, s in enumerate(strokes):
        if i in used or s.width < thick_min or s.length < 1.2 * s.width:
            continue
        # compact blocks much wider than the plan's wall bands: solid furniture, fixtures
        if band_w and s.width > 1.8 * band_w and s.length < 4 * s.width:
            s.roles["block"] = 1.0
            continue
        # in a plan whose walls are face pairs, a solid bar far thinner than every wall class is a
        # door leaf, a frame or a symbol, not a wall
        if classes and paired_len > 0 and not bars_are_walls and s.width < 0.5 * min(classes):
            s.roles["thin_bar"] = 1.0
            continue
        if True:
            walls.append(Wall(id="", p0=s.p0, p1=s.p1, thickness=s.width, style="filled", confidence=0.8,
                              provenance=[f"filled band {s.width:.1f}px"], strokes=[i]))
            filled_ws.append((s.width, s.length))
    long_ = [s for s in strokes if s.length >= 0.05 * diag and not s.roles.get("lattice")]
    if long_:                      # the pen carrying most of the long-stroke length (not one outlier)
        ws = np.array([s.width for s in long_]); ls = np.array([s.length for s in long_])
        order = np.argsort(ws)
        pen = float(ws[order][np.searchsorted(np.cumsum(ls[order]) / ls.sum(), 0.5)])
    else:
        pen = lw
    paired = sum(Wl.length for Wl in walls if Wl.style in ("hollow", "hatched"))
    singles = [i for i, s in enumerate(strokes) if i not in used and not s.roles.get("lattice")
               and s.width < thick_min and s.length >= min_len and s.width >= 0.75 * pen]
    if sum(strokes[i].length for i in singles) + sum(l for _, l in filled_ws) >= paired:
        for i in singles:
            s = strokes[i]
            walls.append(Wall(id="", p0=s.p0, p1=s.p1, thickness=max(s.width, 0.6 * (classes[0] if classes else 3 * s.width)),
                              style="thin", confidence=0.4, thickness_estimated=True,
                              provenance=[f"single line {s.width:.1f}px (plan wall pen {pen:.1f}px)"], strokes=[i]))
            walls[-1]._drawn = s.width
    # glazing: two thin lines much closer than any wall class, with paper between - a window or
    # glazed wall drawn in the wall line. Kept as opening evidence, never as a wall.
    glazing = []
    if classes:
        for p in pairs:
            if (p.i in used or p.j in used or p.d >= 0.7 * min(classes) or p.inside_ink >= 0.25
                    or p.b - p.a < max(40.0, 2 * min(classes)) or strokes[p.i].width >= thick_min):
                continue
            t = math.radians(p.theta)
            tx, ty, nx, ny = math.cos(t), math.sin(t), -math.sin(t), math.cos(t)
            glazing.append(((p.a * tx + p.rho * nx, p.a * ty + p.rho * ny), (p.b * tx + p.rho * nx, p.b * ty + p.rho * ny), p.d))
        # a window can also be one thin line (or several merged by the extractor) across the gap
        for i, s in enumerate(strokes):
            if (i not in used and not s.roles.get("lattice") and s.width < thick_min
                    and s.length >= max(40.0, 2 * min(classes))):
                glazing.append((s.p0, s.p1, 0.0))
    info = {"line_weight": lw, "thick_min": thick_min, "pen": pen, "pairs": len(pairs), "lattice": len(lattice),
            "classes": classes, "glazing": glazing}
    return walls, info


# ---------------------------------------------------------------------------
# 2. re-drawing check (analysis by synthesis)
# ---------------------------------------------------------------------------

def redraw_support(wall: Wall, near: np.ndarray, skip=None) -> float:
    """Share of the wall's expected drawing (in its own convention) found in the ink."""
    h, w = near.shape
    L = wall.length
    if L < 2:
        return 0.0
    tx, ty = _unit(wall.p0, wall.p1)
    nx, ny = -ty, tx
    us = np.arange(1.0, L - 1.0, 2.0)
    if wall.style in ("hollow", "hatched"):
        d, _ = getattr(wall, "_face", (wall.thickness * 0.7, 2.0))
        offs = [-d / 2, d / 2]
    elif wall.style == "thin":
        offs = [0.0]
    else:
        half = max(0.0, wall.thickness / 2 - 1.5)
        offs = [-half, 0.0, half]
    vals = []
    for o in offs:
        xs = np.clip(np.round(wall.p0[0] + us * tx + o * nx).astype(int), 0, w - 1)
        ys = np.clip(np.round(wall.p0[1] + us * ty + o * ny).astype(int), 0, h - 1)
        v = near[ys, xs]
        if skip is not None:
            keep = ~skip[ys, xs]
            v = v[keep] if keep.any() else v
        vals.append(v.mean() if len(v) else 0.0)
    return float(min(vals)) if wall.style in ("hollow", "hatched") else float(np.mean(vals))


# ---------------------------------------------------------------------------
# 3. network
# ---------------------------------------------------------------------------

def _seg_dist(M, P0, P1) -> np.ndarray:
    """Distances of point M to many segments (P0, P1: (n, 2))."""
    d = P1 - P0
    L2 = np.maximum((d * d).sum(axis=1), 1e-9)
    t = np.clip(((M[0] - P0[:, 0]) * d[:, 0] + (M[1] - P0[:, 1]) * d[:, 1]) / L2, 0.0, 1.0)
    return np.hypot(M[0] - (P0[:, 0] + t * d[:, 0]), M[1] - (P0[:, 1] + t * d[:, 1]))


def merge_collinear(walls: list[Wall]) -> list[Wall]:
    """Pieces of one wall split at junctions, by labels drawn over the wall or by the extractor:
    same line, similar thickness, and a break shorter than the wall is thick (a door or window
    is wider than its wall) or filled by a crossing wall (T / X junction)."""
    n = len(walls)
    if n < 2:
        return list(walls)
    P0 = np.array([W.p0 for W in walls], float)
    P1 = np.array([W.p1 for W in walls], float)
    ang = np.array([W.angle for W in walls])
    th = np.array([W.thickness for W in walls])
    filled = np.array([W.style == "filled" for W in walls])
    thin = np.array([W.style == "thin" for W in walls])
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        da = np.abs(ang - ang[i]) % 180.0
        da = np.minimum(da, 180.0 - da)
        js = np.nonzero((da <= 3.0) & (np.arange(n) > i))[0]
        if not len(js):
            continue
        js = js[~((filled[i] & thin[js]) | (thin[i] & filled[js]))]
        t = np.maximum(th[i], th[js])
        js, t = js[np.abs(th[i] - th[js]) <= 0.35 * t + 2], t[np.abs(th[i] - th[js]) <= 0.35 * t + 2]
        if not len(js):
            continue
        tx, ty = _unit(walls[i].p0, walls[i].p1)
        nx, ny = -ty, tx
        mid_i = (P0[i] + P1[i]) / 2
        mid = (P0[js] + P1[js]) / 2
        rho_i = mid_i[0] * nx + mid_i[1] * ny
        off = np.abs(mid[:, 0] * nx + mid[:, 1] * ny - rho_i)
        ok = off <= 0.3 * t + 1.5
        js, t = js[ok], t[ok]
        if not len(js):
            continue
        ai = sorted([P0[i, 0] * tx + P0[i, 1] * ty, P1[i, 0] * tx + P1[i, 1] * ty])
        a0 = P0[js, 0] * tx + P0[js, 1] * ty
        a1 = P1[js, 0] * tx + P1[js, 1] * ty
        lo, hi = np.minimum(a0, a1), np.maximum(a0, a1)
        gap = np.maximum(ai[0], lo) - np.minimum(ai[1], hi)
        for j, g, tj, lj, hj in zip(js, gap, t, lo, hi):
            if g <= max(3.0, tj):
                parent[find(int(j))] = find(i)
                continue
            # a crossing wall in the gap: a junction interrupts the face lines
            u = min(ai[1], hj) + g / 2
            M = (u * tx + rho_i * nx, u * ty + rho_i * ny)
            dc = np.abs(ang - ang[i]) % 180.0
            cross = (np.minimum(dc, 180.0 - dc) >= 60.0) & (g <= th + 6.0)
            cross[[i, int(j)]] = False
            ks = np.nonzero(cross)[0]
            if len(ks) and (_seg_dist(M, P0[ks], P1[ks]) <= 0.5 * th[ks] + 2.0).any():
                parent[find(int(j))] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        if len(members) == 1:
            out.append(walls[members[0]])
            continue
        keep = max(members, key=lambda k: walls[k].length)
        K = walls[keep]
        tx, ty = _unit(K.p0, K.p1)
        nx, ny = -ty, tx
        L = np.array([walls[k].length for k in members])
        proj = [v for k in members for v in (walls[k].p0[0] * tx + walls[k].p0[1] * ty, walls[k].p1[0] * tx + walls[k].p1[1] * ty)]
        lo, hi = min(proj), max(proj)
        rho = float(np.average([((walls[k].p0[0] + walls[k].p1[0]) / 2 * nx + (walls[k].p0[1] + walls[k].p1[1]) / 2 * ny)
                                for k in members], weights=L))
        merged = Wall(id="", p0=(lo * tx + rho * nx, lo * ty + rho * ny), p1=(hi * tx + rho * nx, hi * ty + rho * ny),
                      thickness=float(np.average([walls[k].thickness for k in members], weights=L)),
                      style=K.style, confidence=max(walls[k].confidence for k in members),
                      thickness_estimated=K.thickness_estimated,
                      provenance=[p for k in members for p in walls[k].provenance],
                      strokes=[s_ for k in members for s_ in walls[k].strokes])
        merged.support = K.support
        for attr in ("_face", "_drawn"):
            if hasattr(K, attr):
                setattr(merged, attr, getattr(K, attr))
        out.append(merged)
    return out


def connect(walls: list[Wall], line_w: float) -> tuple[list[Wall], list[tuple[int, int]]]:
    """Snap wall ends onto crossing walls (L / T junctions). Returns walls and junction pairs."""
    junctions: list[tuple[int, int]] = []
    for i, A in enumerate(walls):
        da = _unit(A.p0, A.p1)
        for end in (0, 1):
            P = A.p0 if end == 0 else A.p1
            outward = (-da[0], -da[1]) if end == 0 else da
            best = None
            for j, B in enumerate(walls):
                if j == i or _angle_diff(A.angle, B.angle) < 25.0:
                    continue
                db = _unit(B.p0, B.p1)
                hit = _line_intersection(P, outward, B.p0, db)
                if hit is None:
                    continue
                Q, s = hit
                tol = 0.75 * B.thickness + 0.75 * A.thickness + 2.0 * line_w + 3.0
                if not (-0.5 * B.thickness - 2 <= s <= tol):        # the junction lies at or just beyond the end
                    continue
                dist, tB = _point_segment(Q, B.p0, B.p1)
                # B may stop at A's face line rather than its axis: allow half A plus a pen width
                if dist > 0.5 * A.thickness + line_w + 3.0:
                    continue
                if best is None or abs(s) < abs(best[1]):
                    best = (j, s, Q)
            if best is not None:
                j, s, Q = best
                if end == 0:
                    A.p0 = Q
                else:
                    A.p1 = Q
                junctions.append((i, j))
    return walls, junctions


def components(n: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, set[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), set()).add(i)
    return list(groups.values())


def drop_door_leaves(walls, junctions, near):
    """A door leaf drawn like a wall: short, hinged at one end on another wall, the other end
    free, with a swing arc of its own length from the free tip. Not structure."""
    attached: dict[int, list] = {}
    for a, b in junctions:
        attached.setdefault(a, []).append(b)
    drop = set()
    for i, W in enumerate(walls):
        if W.length < 8:
            continue
        ends = []
        for e, P in ((0, W.p0), (1, W.p1)):
            on = any(_point_segment(P, walls[j].p0, walls[j].p1)[0] <= 0.5 * walls[j].thickness + 3 for j in attached.get(i, []))
            ends.append(on)
        if ends.count(True) != 1:
            continue
        # hinged at the attached end, the other end free
        options = [(W.p0, W.p1) if ends[0] else (W.p1, W.p0)]
        need = 0.6
        host = max((walls[j].thickness for j in attached.get(i, [])), default=W.thickness)
        best = 0.0
        for hinge, tip in options:
            toward = _unit(hinge, tip)
            for back in (0.0, host / 2, host):      # leaves are hinged on the wall face, not the axis
                c = (hinge[0] + toward[0] * back, hinge[1] + toward[1] * back)
                r = W.length - back
                if r > 4:
                    best = max(best, max(_arc_ink(near, c, toward, r, side) for side in (1, -1)))
        if best >= need:
            drop.add(i)
    if not drop:
        return walls, junctions, []
    index = {old: new for new, old in enumerate(i for i in range(len(walls)) if i not in drop)}
    leaves = [walls[i] for i in drop]
    walls = [W for i, W in enumerate(walls) if i not in drop]
    junctions = [(index[a], index[b]) for a, b in junctions if a in index and b in index]
    return walls, junctions, leaves


# ---------------------------------------------------------------------------
# 4. openings
# ---------------------------------------------------------------------------

def _arc_ink(near, center, toward, r, side) -> float:
    """Ink on the quarter circle of radius r around `center`, from `toward` turning to the
    swing side. `side` is +1 / -1 (relative to `toward`) or an explicit unit normal."""
    ux, uy = toward
    if isinstance(side, tuple):
        nx, ny = side
    else:
        nx, ny = -uy * side, ux * side
    phis = np.radians(np.linspace(12, 78, 18))
    xs = center[0] + r * (np.cos(phis) * ux + np.sin(phis) * nx)
    ys = center[1] + r * (np.cos(phis) * uy + np.sin(phis) * ny)
    h, w = near.shape
    xi, yi = np.round(xs).astype(int), np.round(ys).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if ok.sum() < 10:
        return 0.0
    return float(near[yi[ok], xi[ok]].mean())


def _line_ink(near, a, b, n=20) -> float:
    h, w = near.shape
    xs = np.clip(np.round(np.linspace(a[0], b[0], n)).astype(int), 0, w - 1)
    ys = np.clip(np.round(np.linspace(a[1], b[1], n)).astype(int), 0, h - 1)
    return float(near[ys, xs].mean())


def classify_gap(near, a, b, thickness) -> tuple[str, list]:
    """What closes a gap in a wall, from the drawing conventions for openings:
    door - a swing arc (hinged on the wall face at either jamb; two half arcs for a double
           door), a leaf standing at a jamb, or sliding panels each covering part of the gap;
    window - lines running across the whole gap inside the wall band;
    opening - none of these (a passage, or unknown)."""
    u = _unit(a, b)
    nx, ny = -u[1], u[0]
    r = math.hypot(b[0] - a[0], b[1] - a[1])
    half = thickness / 2
    best = 0.0
    for side in (1, -1):
        normal = (side * nx, side * ny)                 # the swing side, as a direction
        for off in (side * half, 0.0):                  # hinge on the face (or the axis)
            for center, toward in ((a, u), (b, (-u[0], -u[1]))):
                c = (center[0] + off * nx, center[1] + off * ny)
                best = max(best, _arc_ink(near, c, toward, r, normal))
            ca = (a[0] + side * half * nx, a[1] + side * half * ny)       # double door: two half arcs
            cb = (b[0] + side * half * nx, b[1] + side * half * ny)
            best = max(best, min(_arc_ink(near, ca, u, r / 2, normal), _arc_ink(near, cb, (-u[0], -u[1]), r / 2, normal)))
    if best >= 0.6:
        return "door", [f"swing arc ({best:.2f})"]
    for side in (1, -1):                               # leaf standing at a jamb (no arc drawn)
        for jamb in (a, b):
            f = (jamb[0] + side * (half + 2) * nx, jamb[1] + side * (half + 2) * ny)
            tip = (jamb[0] + side * (half + 0.85 * r) * nx, jamb[1] + side * (half + 0.85 * r) * ny)
            if _line_ink(near, f, tip) >= 0.85:
                return "door", ["leaf at a jamb"]
    full = partial = 0
    for o in np.linspace(-half, half, 9):
        p0 = (a[0] + 0.08 * r * u[0] + o * nx, a[1] + 0.08 * r * u[1] + o * ny)
        pm = (a[0] + 0.5 * r * u[0] + o * nx, a[1] + 0.5 * r * u[1] + o * ny)
        p1 = (a[0] + 0.92 * r * u[0] + o * nx, a[1] + 0.92 * r * u[1] + o * ny)
        first, second = _line_ink(near, p0, pm, 12), _line_ink(near, pm, p1, 12)
        if first >= 0.85 and second >= 0.85:
            full += 1
        elif max(first, second) >= 0.85 and min(first, second) <= 0.4:
            partial += 1
    if partial >= 2 and full <= 1:
        return "door", ["sliding panels"]
    # pocket / bi-parting panels left open: ink running continuously from a jamb along the wall line
    def run_from(start, direction):
        n_s = max(8, int(r / 3))
        best_run = 0.0
        for o in np.linspace(-half, half, 7):
            k = 0
            for i in range(1, n_s + 1):
                t_ = r * i / n_s
                x = start[0] + direction[0] * t_ + o * nx
                y = start[1] + direction[1] * t_ + o * ny
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= yi < near.shape[0] and 0 <= xi < near.shape[1] and near[yi, xi]):
                    break
                k = i
            best_run = max(best_run, k / n_s)
        return best_run
    ra, rb = run_from(a, u), run_from(b, (-u[0], -u[1]))
    if (min(ra, rb) >= 0.25 and ra + rb >= 0.6) and ra + rb <= 1.05:
        return "door", [f"sliding panels from the jambs ({ra:.2f} + {rb:.2f})"]
    if full >= 2:
        return "window", [f"{full} lines across the gap"]
    return "opening", ["gap between walls"]


def _crosses_wall(a, b, walls, skip) -> bool:
    """Does the closure a-b cut through another wall (not at its own jambs)?"""
    d = _unit(a, b)
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    for j, W in enumerate(walls):
        if j in skip or _angle_diff(W.angle, math.degrees(math.atan2(d[1], d[0])) % 180) < 20:
            continue
        hit = _line_intersection(a, d, W.p0, _unit(W.p0, W.p1))
        if hit is None:
            continue
        Q, s = hit
        if 0.15 * L < s < 0.85 * L and _point_segment(Q, W.p0, W.p1)[0] <= 0.5 * W.thickness:
            return True
    return False


REJECTED: list = []          # diagnostics: gaps considered and not closed, with the reason


def find_openings(walls: list[Wall], near: np.ndarray, connected_ends: set, t_int: float, diag: float,
                  trusted: bool = False) -> list[Opening]:
    """Close gaps in the wall network, only with evidence.

    Pass 1 collects, for every free wall end, its nearest collinear partner (same wall line,
    similar thickness) and the nearest crossing wall ahead, and classifies each gap from the
    drawing (door swing / leaf / sliding panels, window lines, or nothing). Pass 2 measures the
    plan's own door width from the symbol-backed doors and accepts gaps relative to it: doors up
    to 2.5 door widths (double doors), windows up to 3.5, unsupported collinear gaps (passages,
    the wall line continues) up to 2.2, unsupported corner gaps up to 1.2. A closure never cuts
    through another wall."""
    # a short wall attached to nothing (a pair of glyph strokes, a fixture edge) cannot carry an
    # opening: it would set the plan's door width from text
    attached = {i for i, _ in connected_ends}
    floating = {i for i, Wl in enumerate(walls) if i not in attached and Wl.length < 4 * Wl.thickness}
    ends = [(i, e) for i in range(len(walls)) for e in (0, 1) if (i, e) not in connected_ends and i not in floating]
    cands = []
    for i, e in ends:
        A = walls[i]
        P = A.p0 if e == 0 else A.p1
        da = _unit(A.p0, A.p1)
        out = (-da[0], -da[1]) if e == 0 else da
        best = None
        for j, f in ends:
            if j == i:
                continue
            B = walls[j]
            if _angle_diff(A.angle, B.angle) > 4.0 or not 0.6 <= A.thickness / B.thickness <= 1.7:
                continue
            Q = B.p0 if f == 0 else B.p1
            v = (Q[0] - P[0], Q[1] - P[1])
            along = v[0] * out[0] + v[1] * out[1]
            perp = abs(-v[0] * out[1] + v[1] * out[0])
            t = max(A.thickness, B.thickness)
            if along < 0.5 * t or along > 0.12 * diag or perp > 0.5 * t + 2:
                continue
            if best is None or along < best[0]:
                best = (along, Q, (j, f), [j], "collinear")
        if best is None:
            for j, B in enumerate(walls):
                if j == i or _angle_diff(A.angle, B.angle) < 60.0:
                    continue
                hit = _line_intersection(P, out, B.p0, _unit(B.p0, B.p1))
                if hit is None:
                    continue
                Q, s_ = hit
                if s_ < 0.5 * A.thickness or s_ > 0.06 * diag:
                    continue
                if _point_segment(Q, B.p0, B.p1)[0] > 0.5 * B.thickness + 2:
                    continue
                s_face = s_ - 0.5 * B.thickness
                if s_face < 0.5 * A.thickness:
                    continue
                Qf = (P[0] + out[0] * s_face, P[1] + out[1] * s_face)
                if best is None or s_face < best[0]:
                    best = (s_face, Qf, None, [j], "crossing")
        if best is not None:
            gap, Q, partner, others, how = best
            kind, why = classify_gap(near, P, Q, A.thickness)
            cands.append((gap, i, e, P, Q, partner, others, how, kind, why))
    doors = [c[0] for c in cands if c[8] == "door"]
    door_w = float(np.median(doors)) if len(doors) >= 2 else None
    limits = ({"door": 2.5 * door_w, "window": 3.5 * door_w, "collinear": 2.2 * door_w, "crossing": 1.2 * door_w}
              if door_w else {"door": 0.12 * diag, "window": 0.12 * diag, "collinear": 10.0 * t_int, "crossing": 3.0 * t_int})
    if trusted:
        # walls from a structural layer: their ends are real jambs, so a gap with a door or window
        # symbol is an opening whatever its width (glazed walls, door + side lights, bifold banks);
        # gaps without a symbol keep the plan's door-width limits
        limits["door"] = limits["window"] = 0.12 * diag
    openings: list[Opening] = []
    taken: set = set()
    REJECTED.clear()
    for gap, i, e, P, Q, partner, others, how, kind, why in sorted(cands, key=lambda c: c[0]):
        if (i, e) in taken or (partner and partner in taken):
            continue
        limit = limits[kind] if kind in ("door", "window") else limits[how]
        if gap > limit:
            REJECTED.append((P, Q, kind, how, f"gap {gap:.0f} > limit {limit:.0f}"))
            continue
        if _crosses_wall(P, Q, walls, {i, *others}):
            REJECTED.append((P, Q, kind, how, "crosses a wall"))
            continue
        taken.add((i, e))
        if partner:
            taken.add(partner)
        openings.append(Opening(id="", p0=P, p1=Q, kind=kind, wall_ids=tuple([i] + others), thickness=walls[i].thickness,
                                confidence=0.75 if kind != "opening" else 0.5,
                                provenance=why + [f"{how} gap {gap:.0f}px" + (f", plan door width {door_w:.0f}px" if door_w else "")]))
    return openings


def glazing_openings(glazing, walls: list[Wall], openings: list[Opening], t_class: float) -> list[Opening]:
    """Glazing spanning between walls is a window in the wall line: both ends on a wall band,
    no wall running alongside it (a counter or a shelf runs along a wall, a window replaces one),
    not already covered by an opening."""
    if not walls:
        return []
    P0 = np.array([W.p0 for W in walls], float)
    P1 = np.array([W.p1 for W in walls], float)
    th = np.array([W.thickness for W in walls])
    ang = np.array([W.angle for W in walls])
    out: list[Opening] = []
    for a, b, d in glazing:
        u = _unit(a, b)
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        hosts = []
        for P in (a, b):
            dist = _seg_dist(P, P0, P1)
            k = int(np.argmin(dist - 0.5 * th))
            hosts.append(k if dist[k] <= 0.5 * th[k] + max(6.0, 0.5 * t_class) else None)
        if None in hosts or hosts[0] == hosts[1]:
            continue
        g_ang = math.degrees(math.atan2(u[1], u[0])) % 180.0
        da = np.abs(ang - g_ang) % 180.0
        par = np.minimum(da, 180.0 - da) < 10.0
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        side = _seg_dist(mid, P0[par], P1[par]) if par.any() else np.array([])
        # single lines need more room: dimension strings and counters run parallel to a wall
        if len(side) and (side < 0.5 * th[par] + (3.0 if d == 0.0 else 2.0) * t_class).any():
            continue
        if _crosses_wall(a, b, walls, set(hosts)):
            continue
        if any(_point_segment(mid, o.p0, o.p1)[0] <= o.thickness for o in openings + out):
            continue
        t = max(th[hosts[0]], th[hosts[1]])
        out.append(Opening(id="", p0=a, p1=b, kind="window", wall_ids=(hosts[0], hosts[1]), thickness=float(t),
                           confidence=0.6, provenance=[(f"glazing lines {d:.0f}px apart" if d else "window line") + f" spanning {L:.0f}px between walls"]))
    return out


# ---------------------------------------------------------------------------
# 5. spaces
# ---------------------------------------------------------------------------

def _band(p0, p1, thickness, extend):
    """Rectangle of the wall band: the axis extended by `extend` at both ends."""
    u = _unit(p0, p1)
    n = (-u[1], u[0])
    h = thickness / 2.0
    a = (p0[0] - u[0] * extend, p0[1] - u[1] * extend)
    b = (p1[0] + u[0] * extend, p1[1] + u[1] * extend)
    return np.round(np.array([(a[0] + n[0] * h, a[1] + n[1] * h), (b[0] + n[0] * h, b[1] + n[1] * h),
                              (b[0] - n[0] * h, b[1] - n[1] * h), (a[0] - n[0] * h, a[1] - n[1] * h)])).astype(np.int32)


def render_walls(walls: list[Wall], shape, openings: list[Opening] | None = None) -> np.ndarray:
    """Walls as bands; each band runs half its thickness past its axis ends, so walls overlap at
    junctions as built walls do (no holes where several walls meet)."""
    mask = np.zeros(shape[:2], np.uint8)
    for W in walls:
        cv2.fillPoly(mask, [_band(W.p0, W.p1, max(1.0, W.thickness), W.thickness / 2.0)], 255)
    for o in openings or []:
        cv2.fillPoly(mask, [_band(o.p0, o.p1, max(1.0, o.thickness), 0.0)], 255)
    return mask


def spaces_from(walls, openings, shape, min_area: float, join: float = 0.0) -> tuple[list[Space], np.ndarray]:
    barrier = render_walls(walls, shape, openings)
    if join >= 2:          # model elements closer than the junction tolerance are joined
        k = int(round(join)) | 1
        barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    free = (barrier == 0).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(free, connectivity=4)
    H, W = shape[:2]
    spaces: list[Space] = []
    labels = np.zeros(shape[:2], np.int32)
    cand = [c for c in range(1, n)
            if not (stats[c][0] == 0 or stats[c][1] == 0 or stats[c][0] + stats[c][2] >= W or stats[c][1] + stats[c][3] >= H)
            and stats[c][4] >= min_area]
    # a face that holds other faces inside its holes is not a room (rooms do not contain rooms):
    # it is the outside of the building closed by a sheet frame, a border or a title block
    outside = set()
    for c in cand:
        x, y, w, h, area = stats[c]
        inner = [d for d in cand if d != c and x < cent[d][0] < x + w and y < cent[d][1] < y + h]
        if len(inner) < 2:
            continue
        region = (lab[y:y + h, x:x + w] == c).astype(np.uint8)
        cs, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(region)
        cv2.drawContours(filled, cs, -1, 1, -1)
        holes = (filled > 0) & (region == 0)
        held = sum(1 for d in inner if holes[int(cent[d][1]) - y, int(cent[d][0]) - x] and lab[int(cent[d][1]), int(cent[d][0])] == d)
        if held >= 2:
            outside.add(c)
    k = 0
    for c in cand:
        if c in outside:
            continue
        x, y, w, h, area = stats[c]
        k += 1
        region = (lab == c)
        labels[region] = k
        cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.5, True).reshape(-1, 2)
        spaces.append(Space(id=f"space-{k:02d}", polygon=[(int(px), int(py)) for px, py in poly], area_px=int(area),
                            centroid=(float(cent[c][0]), float(cent[c][1])), confidence=0.6,
                            provenance=["enclosed by reconstructed walls and opening closures"]))
    return spaces, labels


def topology(openings, labels) -> list:
    h, w = labels.shape
    out = []
    for o in openings:
        u = _unit(o.p0, o.p1)
        nx, ny = -u[1], u[0]
        mx, my = (o.p0[0] + o.p1[0]) / 2, (o.p0[1] + o.p1[1]) / 2
        r = o.thickness / 2 + 4
        sides = []
        for sg in (1, -1):
            x, y = int(round(mx + sg * r * nx)), int(round(my + sg * r * ny))
            v = int(labels[y, x]) if 0 <= x < w and 0 <= y < h else 0
            sides.append(f"space-{v:02d}" if v else "exterior")
        o.connects = tuple(sides)
        if sides[0] != sides[1] and o.kind != "window":
            out.append((sides[0], sides[1], o.id))
    return out


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def attach_labels(spaces, labels, room_labels) -> None:
    """Room labels name the space they fall in (they never create one)."""
    H, W = labels.shape
    for name, (x, y) in room_labels or ():
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H and labels[yi, xi] > 0:
            spaces[labels[yi, xi] - 1].labels.append(name)


def reconstruct(image: np.ndarray, room_labels: list | None = None, evidence_image: np.ndarray | None = None) -> PlanModel:
    """`evidence_image` (optional, same frame): the full drawing when `image` is a structural
    rendering (engine.semantic). Walls come from `image`; opening evidence (door leaves, swing
    arcs, sliding panels, glazing lines) is read from both, so symbols left out of the structural
    rendering still explain the gaps between its walls."""
    gray, ink = ink_mask(image)
    H, W = ink.shape
    diag = math.hypot(H, W)
    strokes = extract_strokes(ink)
    hyps, info = wall_hypotheses(strokes, ink)
    near = cv2.dilate((ink > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    evidence = near
    if evidence_image is not None:
        _, ink_ev = ink_mask(evidence_image)
        evidence = near | (cv2.dilate((ink_ev > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        classes = info["classes"] or [3.0]
        lw = info["line_weight"]
        info["glazing"] = info["glazing"] + [
            (s.p0, s.p1, 0.0) for s in extract_strokes(ink_ev)
            if s.width < info["thick_min"] and s.length >= max(40.0, 2 * min(classes)) and s.width <= 3 * lw + 2]
    # re-drawing check before anything is built on a hypothesis
    supported = []
    rejected = 0
    for Wl in hyps:
        Wl.support = redraw_support(Wl, near)
        if Wl.support >= SUPPORT_MIN:
            supported.append(Wl)
        else:
            rejected += 1
    walls = merge_collinear(supported)
    walls, junctions = connect(walls, info["line_weight"])
    walls, junctions, leaves = drop_door_leaves(walls, junctions, evidence)
    connected_ends = set()
    for a, b in junctions:
        Wa = walls[a]
        for e, P in ((0, Wa.p0), (1, Wa.p1)):
            if _point_segment(P, walls[b].p0, walls[b].p1)[0] <= 0.5 * walls[b].thickness + 2:
                connected_ends.add((a, e))
    t_class = float(np.median([w.thickness for w in walls])) if walls else 10.0
    t_int = float(np.percentile([w.thickness for w in walls], 25)) if walls else 10.0
    openings = find_openings(walls, evidence, connected_ends, t_int, diag, trusted=evidence_image is not None)
    openings += glazing_openings(info["glazing"], walls, openings, t_class)
    # the network: walls joined by junctions or by an opening (a gap in one wall line is a
    # structural connection too); small isolated groups are furniture, symbols or annotation
    links = list(junctions) + [(o.wall_ids[0], o.wall_ids[1]) for o in openings if len(o.wall_ids) > 1]
    comps = components(len(walls), links)
    sizes = [sum(walls[i].length for i in c) for c in comps]
    keep: set[int] = set()
    if sizes:
        top = max(sizes)
        for c, sz in zip(comps, sizes):
            if sz >= 0.2 * top and len(c) >= 3:
                keep |= c
    index = {old: new for new, old in enumerate(sorted(keep))}
    walls = [walls[i] for i in sorted(keep)]
    openings = [o for o in openings if all(i in index for i in o.wall_ids)]
    for o in openings:
        o.wall_ids = tuple(index[i] for i in o.wall_ids)
    for k, Wl in enumerate(walls, 1):
        Wl.id = f"wall-{k:03d}"
        Wl.confidence = min(0.99, Wl.confidence + 0.2 * (Wl.support or 0))
    for k, o in enumerate(openings, 1):
        o.id = f"opening-{k:03d}"
        o.wall_ids = tuple(walls[i].id for i in o.wall_ids)
    min_area = max(16.0 * t_class * t_class, 0.0008 * H * W)
    spaces, labels = spaces_from(walls, openings, (H, W), min_area, join=0.6 * t_class)
    adjacency = topology(openings, labels)
    attach_labels(spaces, labels, room_labels)
    status = "ok" if spaces else ("partial" if walls else "no_structure")
    model = PlanModel(width=W, height=H, walls=walls, openings=openings, spaces=spaces, adjacency=adjacency,
                      annotations={"lattice_strokes": info["lattice"]}, wall_classes=info["classes"], status=status,
                      diagnostics={"strokes": len(strokes), "hypotheses": len(hyps), "rejected_by_redraw": rejected,
                                   "door_leaves": len(leaves),
                                   **{k: v for k, v in info.items() if k != "classes"}})
    if not spaces:
        model.warnings.append("No enclosed space could be reconstructed from the drawing.")
    model._labels = labels
    model._wall_mask = render_walls(walls, (H, W))
    return model
