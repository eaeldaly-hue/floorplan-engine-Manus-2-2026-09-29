"""Typed objects from the vector drawing: fixtures and furniture as architectural evidence.

The Architectural Cleaner types walls, doors and windows and keeps every other vector element as
'OTHER'. That geometry is not noise: counters, a range, a sink, a dining set or a sofa say what a
part of a space is for. This module turns it into typed objects, measured in real units (scale
from the plan's own doors), so functional zones can be inferred without relying on labels.

Objects come from three kinds of evidence, combined (never a single rule):

  primitives    circles fitted to drawn arcs (burners, lamps, drains), rounded-rectangle basins
                (sinks), long lines parallel to a wall at counter depth (counter fronts)
  components    connected non-architectural ink (cut away from walls, so furniture against a wall
                stays its own object): size, aspect and enclosed sub-loops identify dining sets,
                sofas, loveseats, armchairs, coffee / end tables, media units, beds, washers
  words         OCR / text-layer words on or next to an object (REFRIGERATOR, WASHER, DW, RANGE...)

Dimensions are standard furniture and fixture sizes, not drawing-specific thresholds. Every object
carries its evidence and a confidence, and only proposes function weights; zones decide
(engine.arch.zones).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

DOOR_CM = 82.0          # typical interior door leaf (2'8")
_PPC = [1.0]            # scale of the current call (component loops are measured in px)


@dataclass
class Obj:
    id: str
    kind: str
    bbox: tuple                      # x0, y0, x1, y1 (px)
    functions: dict                  # function -> weight (evidence strength)
    confidence: float
    evidence: list = field(default_factory=list)

    @property
    def center(self) -> tuple:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "bbox": [int(v) for v in self.bbox],
                "functions": {k: round(v, 2) for k, v in self.functions.items()},
                "confidence": round(self.confidence, 2), "evidence": self.evidence}


# ------------------------------------------------------------------------------------------ scale
def px_per_cm(cleaned=None, pixel_scale: dict | None = None) -> tuple[float | None, str]:
    """Drawing scale: printed dimensions when the analysis calibrated them, else the plan's own
    door widths (a door is ~82 cm), else unknown."""
    if pixel_scale and pixel_scale.get("pixels_per_unit"):
        unit = pixel_scale.get("unit")
        ppu = float(pixel_scale["pixels_per_unit"])
        if unit == "ft":
            return ppu / 30.48, "printed dimensions"
        if unit == "m":
            return ppu / 100.0, "printed dimensions"
    if cleaned is not None:
        doors = [math.dist(o["p0"], o["p1"]) for o in getattr(cleaned, "openings", []) if o.get("kind") == "door"]
        if len(doors) >= 3:
            return float(np.median(doors)) / DOOR_CM, f"median of {len(doors)} door widths"
    return None, "unknown"


# -------------------------------------------------------------------------------------- primitives
def _fit_circle(pts: np.ndarray):
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x * x + y * y
    try:
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = c[0] / 2, c[1] / 2
    r2 = c[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    res = float(np.abs(np.hypot(x - cx, y - cy) - r).max())
    a = np.unwrap(np.arctan2(y - cy, x - cx))
    return cx, cy, r, res, float(abs(a[-1] - a[0]))


def arcs(elements) -> list:
    """Circular arcs among the OTHER elements: (cx, cy, r, sweep, element index, mid x, mid y)."""
    out = []
    for i, e in enumerate(elements):
        g = np.asarray(e.geometry, float)
        if len(g) < 6:
            continue
        chord = float(np.linalg.norm(g[-1] - g[0]))
        dv = g[-1] - g[0]
        rel = g - g[0]
        span = float(np.abs(dv[0] * rel[:, 1] - dv[1] * rel[:, 0]).max()) / max(chord, 1e-6) if chord > 1e-6 else 1.0
        if chord > 1e-6 and span < 0.05 * chord:
            continue                                   # a straight Bezier
        f = _fit_circle(g)
        if f is None:
            continue
        cx, cy, r, res, sweep = f
        if res <= max(0.6, 0.06 * r):
            mx, my = g[len(g) // 2]
            out.append((cx, cy, r, sweep, i, float(mx), float(my)))
    return out


def circles(arc_list, ppc: float, r_cm=(2.0, 30.0)) -> list:
    """Closed circles (several arcs on one centre covering most of the turn): (cx, cy, r_px, arc element ids)."""
    out = []
    used = set()
    for k, (cx, cy, r, sw, *_rest) in enumerate(arc_list):
        if k in used or not (r_cm[0] * ppc <= r <= r_cm[1] * ppc):
            continue
        group = [k]
        total = sw
        for j in range(k + 1, len(arc_list)):
            if j in used:
                continue
            x2, y2, r2, s2 = arc_list[j][:4]
            if math.hypot(x2 - cx, y2 - cy) <= 0.25 * r and 0.8 <= r2 / r <= 1.25:
                group.append(j)
                total += s2
        if total >= math.radians(250):
            used.update(group)
            out.append((cx, cy, r, tuple(arc_list[g][4] for g in group)))
    return out


def basins(arc_list, ppc: float, circle_arcs: set = frozenset()) -> list:
    """Rounded rectangles 10-60 cm (sink basins, lavatories): four outward-bulging corner arcs -
    top-left, top-right, bottom-left, bottom-right - aligned on two axes. Arcs of full circles
    (burners, drains, lamps) are not corners. Returns boxes (x0, y0, x1, y1)."""
    corners = {"tl": [], "tr": [], "bl": [], "br": []}
    for cx, cy, r, sw, i, mx, my in arc_list:
        if i in circle_arcs or r > 9 * ppc or not math.radians(55) <= sw <= math.radians(125):
            continue
        q = ("t" if my < cy else "b") + ("l" if mx < cx else "r")
        corners[q].append((cx, cy, r))
    tol = max(2.0, 1.5 * ppc)
    out = []
    for x0, y0, r in corners["tl"]:
        for x1, y1, _ in corners["tr"]:
            if abs(y1 - y0) > tol or not 6 * ppc <= x1 - x0 <= 60 * ppc:
                continue
            for x2, y2, _ in corners["bl"]:
                if abs(x2 - x0) > tol or not 6 * ppc <= y2 - y0 <= 60 * ppc:
                    continue
                if any(abs(x3 - x1) <= tol and abs(y3 - y2) <= tol for x3, y3, _ in corners["br"]):
                    box = (x0 - r, y0 - r, x1 + r, y2 + r)
                    if not any(abs(box[0] - o[0]) <= tol and abs(box[1] - o[1]) <= tol for o in out):
                        out.append(box)
    return out


# --------------------------------------------------------------------------------------- objects
WORDS = [
    (r"REFRIG|FRIDGE|\bREF\b", "refrigerator", {"kitchen": 1.0}),
    (r"DISH|\bDW\b|D\.W\.", "dishwasher", {"kitchen": 1.0}),
    (r"RANGE|OVEN|COOK|STOVE", "range", {"kitchen": 1.0}),
    (r"\bSINK\b", "sink", {"kitchen": 0.5, "bath": 0.3}),
    (r"PANTRY", "pantry", {"kitchen": 0.6}),
    (r"WASHER|\bW/D\b|\bWD\b", "washer", {"laundry": 0.8, "kitchen": 0.2}),
    (r"DRYER", "dryer", {"laundry": 1.0}),
    (r"\bTV\b|MEDIA", "media", {"living": 0.8}),
]


def _component_loops(comp: np.ndarray, ppc: float, min_cm: float = 15) -> list:
    """Enclosed sub-regions (holes) of one component: (w_cm, h_cm, box)."""
    cs, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is None:
        return out
    for i, c in enumerate(cs):
        if hier[0][i][3] < 0:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if max(w, h) >= min_cm * ppc:
            out.append((w / ppc, h / ppc, (x, y, x + w, y + h)))
    return out


def _classify_component(w: float, h: float, loops: list, near_wall: bool, curved: int = 0):
    """Furniture by standard dimensions (cm) and sub-structure. Returns (kind, functions, conf, why)."""
    long_, short = max(w, h), min(w, h)
    big_loops = [l for l in loops if max(l[0], l[1]) >= 70 and min(l[0], l[1]) >= 60 and max(l[0], l[1]) / min(l[0], l[1]) <= 3.0]
    chairish = [l for l in loops if 25 <= max(l[0], l[1]) <= 65 and min(l[0], l[1]) >= 12]          # seat outline
    for table in big_loops:
        tx0, ty0, tx1, ty1 = table[2]
        around = set()
        n_around = 0
        for c in chairish:
            cx0, cy0, cx1, cy1 = c[2]
            if cx0 >= tx0 and cx1 <= tx1 and cy0 >= ty0 and cy1 <= ty1:
                continue                                   # inside the table (or a sofa's cushions)
            gap_x = max(0, max(tx0, cx0) - min(tx1, cx1))
            gap_y = max(0, max(ty0, cy0) - min(ty1, cy1))
            if max(gap_x, gap_y) / _PPC[0] > 40:
                continue
            side = ("left" if cx1 <= tx0 + 2 else "right" if cx0 >= tx1 - 2 else
                    "top" if cy1 <= ty0 + 2 else "bottom" if cy0 >= ty1 - 2 else "edge")
            around.add(side)
            n_around += 1
        if n_around >= 3 and len(around - {"edge"}) >= 2 and 140 <= long_ <= 450 and short >= 120:
            return "dining set", {"dining": 1.0}, 0.9, f"table with {n_around} chairs on {len(around)} sides"
    covered = sum(l[0] * l[1] for l in loops) / max(1.0, w * h)
    large = [l for l in loops if max(l[0], l[1]) >= 100 and min(l[0], l[1]) >= 60]
    if near_wall and 180 <= long_ <= 290 and 90 <= short <= 280 and len(loops) >= 2 and large and covered >= 0.45:
        return "bed", {"bedroom": 1.0}, 0.8, f"{long_:.0f}x{short:.0f} cm, mattress / blanket / pillows ({covered:.0%} enclosed)"
    if near_wall and 140 <= long_ <= 195 and 65 <= short <= 100 and 1 <= len(loops) <= 2:
        return "bathtub", {"bath": 1.0}, 0.7, f"{long_:.0f}x{short:.0f} cm basin against a wall"
    if near_wall and curved >= 2 and 50 <= long_ <= 85 and 35 <= short <= 60 and long_ / short >= 1.25 and len(loops) >= 1:
        return "toilet", {"bath": 0.8}, 0.6, f"{long_:.0f}x{short:.0f} cm against a wall"
    if 150 <= long_ <= 320 and 65 <= short <= 115 and len(loops) >= 3:
        return "sofa", {"living": 1.0}, 0.85, f"{long_:.0f}x{short:.0f} cm, {len(loops)} cushions / arms"
    if 110 <= long_ < 190 and 65 <= short <= 115 and len(loops) >= 2:
        return "loveseat", {"living": 0.8}, 0.75, f"{long_:.0f}x{short:.0f} cm, {len(loops)} cushions"
    if 70 <= long_ <= 110 and 65 <= short <= 110 and len(loops) >= 1:
        return "armchair", {"living": 0.4}, 0.5, f"{long_:.0f}x{short:.0f} cm seat"
    if 80 <= long_ <= 150 and 40 <= short <= 85 and len(loops) <= 2 and not near_wall:
        return "table", {"living": 0.3, "dining": 0.1}, 0.4, f"{long_:.0f}x{short:.0f} cm free-standing"
    if 35 <= long_ <= 75 and 35 <= short <= 75 and len(loops) >= 1:
        return "side table", {"living": 0.15, "bedroom": 0.1}, 0.3, f"{long_:.0f}x{short:.0f} cm"
    if 100 <= long_ <= 260 and 20 <= short <= 60 and near_wall:
        return "media / cabinet", {"living": 0.3, "bedroom": 0.2}, 0.3, f"{long_:.0f}x{short:.0f} cm against a wall"
    return None


def vector_objects(cleaned, ocr_boxes=(), ppc: float | None = None) -> list[Obj]:
    """Typed objects from the cleaner's non-architectural vector elements (+ words)."""
    layer = getattr(cleaned, "layer", None)
    if layer is None or not ppc:
        return []
    h, w = layer.shape[:2]
    t = float(layer.wall_thickness or 10.0)
    others = [e for e in layer.elements if e.type == "OTHER" and not e.fill]
    arch = np.zeros((h, w), np.uint8)
    ink = np.zeros((h, w), np.uint8)
    for e in layer.elements:
        pts = np.round(np.asarray(e.geometry, float)).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) == 0:
            continue
        if e.type in ("WALL", "DOOR", "WINDOW", "COLUMN"):
            cv2.polylines(arch, [pts], False, 255, 2)
        elif e.type == "OTHER" and not e.fill:
            cv2.polylines(ink, [pts], False, 255, 2)
    k = max(3, int(round(t)) | 1)
    near_arch = cv2.dilate(arch, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    wall_dist = cv2.distanceTransform((arch == 0).astype(np.uint8), cv2.DIST_L2, 3)
    ink[near_arch] = 0                                  # furniture against a wall stays its own object
    ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))

    objs: list[Obj] = []
    _PPC[0] = ppc

    def add(kind, bbox, functions, conf, why):
        objs.append(Obj(f"OB{len(objs) + 1:03d}", kind, tuple(float(v) for v in bbox), dict(functions), conf, [why]))

    # --- fixtures from primitives -----------------------------------------------------------------
    al = arcs(others)
    circ = circles(al, ppc)
    circle_arcs = {i for c in circ for i in c[3]}
    bs = basins(al, ppc, circle_arcs)
    inside_basin = lambda c: any(b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3] for b in bs)  # noqa: E731
    burners = [c for c in circ if 3 * ppc <= c[2] <= 13 * ppc and not inside_basin(c)]
    used = set()
    for i, (cx, cy, r, _) in enumerate(burners):
        if i in used:
            continue
        grp = [j for j, (x, y, rr, _) in enumerate(burners) if j not in used and abs(x - cx) <= 90 * ppc
               and abs(y - cy) <= 90 * ppc and 0.75 <= rr / r <= 1.33]
        pts = np.array([burners[j][:2] for j in grp])
        if not 4 <= len(grp) <= 6:
            continue
        xs, ys = pts[:, 0], pts[:, 1]
        cols = len(np.unique(np.round(xs / (12 * ppc))))
        rows = len(np.unique(np.round(ys / (12 * ppc))))
        if cols >= 2 and rows >= 2 and (xs.max() - xs.min()) <= 80 * ppc and (ys.max() - ys.min()) <= 80 * ppc:
            used.update(grp)
            pad = 15 * ppc
            add("range", (xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad), {"kitchen": 1.0}, 0.9,
                f"{len(grp)} burners in a {cols}x{rows} grid")
    merged = []
    for b in sorted(bs):
        if merged and b[0] - merged[-1][2] <= 15 * ppc and abs(b[1] - merged[-1][1]) <= 20 * ppc:
            m = merged[-1]
            merged[-1] = (min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3]), m[4] + 1)
        else:
            merged.append((*b, 1))
    for x0, y0, x1, y1, n in merged:
        add("sink", (x0, y0, x1, y1), {"kitchen": 0.5, "bath": 0.4}, 0.7, f"{n} rounded basin(s)")

    # --- counter fronts: long lines parallel to a wall at counter depth (50-80 cm off its face) ----
    fronts = []
    for e in others:
        g = np.asarray(e.geometry, float)
        if len(g) != 2:
            continue
        L = float(np.linalg.norm(g[1] - g[0]))
        if L < 90 * ppc:
            continue
        samples = np.linspace(g[0], g[1], 9)
        d = np.array([wall_dist[min(h - 1, max(0, int(y))), min(w - 1, max(0, int(x)))] for x, y in samples]) / ppc
        if np.all((d >= 45) & (d <= 85)) and float(np.ptp(d)) <= 12:
            fronts.append(g)
    for g in fronts:
        x0, y0 = g.min(0)
        x1, y1 = g.max(0)
        pad = 35 * ppc
        add("counter", (x0 - pad if x1 - x0 < y1 - y0 else x0, y0 - pad if y1 - y0 <= x1 - x0 else y0,
                        x1 + pad if x1 - x0 < y1 - y0 else x1, y1 + pad if y1 - y0 <= x1 - x0 else y1),
            {"kitchen": 0.4}, 0.5, f"{np.linalg.norm(g[1] - g[0]) / ppc:.0f} cm line parallel to a wall at counter depth")

    # --- furniture from components ------------------------------------------------------------------
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    assemblies = []
    tables = []
    for k in range(1, n):
        x, y, bw, bh, area = st[k]
        wc, hc = bw / ppc, bh / ppc
        if max(wc, hc) < 30 or max(wc, hc) > 600:
            continue
        comp = (lab[y:y + bh, x:x + bw] == k).astype(np.uint8)
        loops = _component_loops(comp, ppc)
        edge = np.concatenate([wall_dist[y, x:x + bw], wall_dist[y + bh - 1, x:x + bw], wall_dist[y:y + bh, x], wall_dist[y:y + bh, x + bw - 1]])
        near_wall = bool(edge.size and edge.min() <= 1.5 * t + 6 * ppc)
        curved = sum(1 for ax, ay, ar, *_ in al if 8 * ppc <= ar <= 25 * ppc and x <= ax <= x + bw and y <= ay <= y + bh)
        c = _classify_component(wc, hc, loops, near_wall, curved)
        if c:
            add(c[0], (x, y, x + bw, y + bh), c[1], c[2], c[3])
        else:
            rows = _seat_rows(loops)
            for n_seats, (rx0, ry0, rx1, ry1) in rows:
                add("sofa" if n_seats >= 3 else "loveseat", (x + rx0, y + ry0, x + rx1, y + ry1),
                    {"living": 1.0 if n_seats >= 3 else 0.8}, 0.75, f"{n_seats} cushions side by side in a furniture group")
            if max(wc, hc) >= 120:
                assemblies.append(((x, y, x + bw, y + bh), wc, hc))
        if not c and 60 <= max(wc, hc) <= 260 and 60 <= min(wc, hc) <= 140 and len(loops) <= 2 and not near_wall:
            tables.append((x, y, x + bw, y + bh))
        # bar stools / dining chairs drawn as separate small squares are counted by the zone layer
        elif 30 <= max(wc, hc) <= 60 and 25 <= min(wc, hc) <= 60:
            add("seat", (x, y, x + bw, y + bh), {"dining": 0.15, "kitchen": 0.1}, 0.3, f"{wc:.0f}x{hc:.0f} cm")

    # --- words: label the object they sit on (or a fixture-sized box around them) -------------------
    for b in ocr_boxes:
        text = str(getattr(b, "text", "")).upper()
        for pattern, kind, functions in WORDS:
            if re.search(pattern, text):
                cx, cy = b.x + b.width / 2, b.y + b.height / 2
                half = 40 * ppc
                add(kind, (cx - half, cy - half, cx + half, cy + half), functions,
                    0.6 + 0.3 * min(1.0, float(getattr(b, "confidence", 60)) / 100), f"word '{b.text}'")
                break
    _dining_from_pieces(objs, tables, ppc, add)
    _laundry(objs, ppc)
    _context(objs, ppc)
    # a large drawn assembly holding several kitchen fixtures is the counter run / island
    strong = [o for o in objs if o.kind in STRONG_KITCHEN or (o.kind == "sink" and o.functions.get("kitchen", 0) >= 0.9)]
    for (x0, y0, x1, y1), wc, hc in assemblies:
        inside = [o for o in strong if x0 <= o.center[0] <= x1 and y0 <= o.center[1] <= y1]
        if len(inside) >= 2:
            add("counter run", (x0, y0, x1, y1), {"kitchen": 0.6}, 0.7,
                f"{max(wc, hc):.0f}x{min(wc, hc):.0f} cm drawn assembly holding {len(inside)} kitchen fixtures")
    return objs


STRONG_KITCHEN = ("range", "refrigerator", "dishwasher")


def _seat_rows(loops: list) -> list:
    """Rows of equal cushion loops (40-75 cm) touching edge to edge: a sofa / loveseat drawn inside a
    furniture group. Returns [(n_seats, box)] in component pixels."""
    ppc = _PPC[0]
    seats = [l for l in loops if 40 <= max(l[0], l[1]) <= 80 and 35 <= min(l[0], l[1]) <= 75]
    used = set()
    out = []
    for i, a in enumerate(seats):
        if i in used:
            continue
        row = [i]
        for j, b in enumerate(seats):
            if j == i or j in used:
                continue
            ax0, ay0, ax1, ay1 = a[2]
            bx0, by0, bx1, by1 = b[2]
            same_h = abs(ay0 - by0) <= 5 * ppc and abs(ay1 - by1) <= 5 * ppc
            same_v = abs(ax0 - bx0) <= 5 * ppc and abs(ax1 - bx1) <= 5 * ppc
            if (same_h or same_v) and abs(max(a[0], a[1]) - max(b[0], b[1])) <= 12:
                row.append(j)
        if len(row) < 2:
            continue
        boxes = [seats[k][2] for k in row]
        xs0 = min(b[0] for b in boxes); ys0 = min(b[1] for b in boxes)
        xs1 = max(b[2] for b in boxes); ys1 = max(b[3] for b in boxes)
        total = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
        if total >= 0.75 * (xs1 - xs0) * (ys1 - ys0):               # contiguous: cushions share edges
            used.update(row)
            out.append((len(row), (xs0, ys0, xs1, ys1)))
    return out


def _dining_from_pieces(objs: list[Obj], tables: list, ppc: float, add) -> None:
    """A free-standing table with >= 3 chair-sized pieces within 40 cm of its edges, on >= 2 sides."""
    seats = [o for o in objs if o.kind in ("seat", "side table")]
    for tx0, ty0, tx1, ty1 in tables:
        sides, n = set(), 0
        for o in seats:
            cx0, cy0, cx1, cy1 = o.bbox
            gx = max(0.0, max(tx0, cx0) - min(tx1, cx1))
            gy = max(0.0, max(ty0, cy0) - min(ty1, cy1))
            if max(gx, gy) > 40 * ppc:
                continue
            sides.add("l" if cx1 <= tx0 + 3 else "r" if cx0 >= tx1 - 3 else "t" if cy1 <= ty0 + 3 else "b" if cy0 >= ty1 - 3 else "x")
            n += 1
        if n >= 3 and len(sides - {"x"}) >= 2:
            pad = 45 * ppc
            add("dining set", (tx0 - pad, ty0 - pad, tx1 + pad, ty1 + pad), {"dining": 1.0}, 0.8,
                f"table with {n} chairs on {len(sides)} sides")
            for o in seats:
                cx, cy = o.center
                if tx0 - pad <= cx <= tx1 + pad and ty0 - pad <= cy <= ty1 + pad:
                    o.functions = {"dining": 0.15}


def _laundry(objs: list[Obj], ppc: float) -> None:
    """Appliance-sized boxes beside a laundry word are washers / dryers, not tables."""
    words = [o for o in objs if o.kind in ("washer", "dryer") and any("word" in e for e in o.evidence)]
    for o in objs:
        if o.kind in ("side table", "seat", "table") and any(math.dist(w.center, o.center) <= 120 * ppc for w in words):
            o.kind, o.functions = "laundry appliance", {"laundry": 0.8}
            o.evidence.append("appliance-sized, beside a laundry word")


def _context(objs: list[Obj], ppc: float) -> None:
    """Resolve ambiguous fixtures by what is around them: a sink or a 'washer' among kitchen fixtures
    is a kitchen sink / dishwasher; a sink beside a toilet or tub belongs to a bath."""
    kitchen = [o for o in objs if o.kind in STRONG_KITCHEN]          # decisive fixtures only, never counters
    for o in objs:
        near_k = [k for k in kitchen if k is not o and math.dist(k.center, o.center) <= 300 * ppc]
        if o.kind == "sink" and near_k:
            o.functions = {"kitchen": 0.9}
            o.evidence.append(f"next to {len(near_k)} kitchen fixture(s)")
        if o.kind == "washer" and (any(k.kind == "range" and math.dist(k.center, o.center) <= 300 * ppc for k in near_k)
                                   or any(s.kind == "sink" and math.dist(s.center, o.center) <= 200 * ppc for s in objs)):
            o.kind, o.functions = "dishwasher", {"kitchen": 1.0}
            o.evidence.append("among kitchen fixtures: a dishwasher")
