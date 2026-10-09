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
    """Closed circles (several arcs on one centre covering most of the turn): (cx, cy, r_px, arc element ids).
    Candidate arcs are pre-selected with a generous box test; the tests and sums are the original ones."""
    out = []
    n = len(arc_list)
    if not n:
        return out
    X = np.array([a[0] for a in arc_list], float)
    Y = np.array([a[1] for a in arc_list], float)
    used = np.zeros(n, bool)
    idx = np.arange(n)
    for k, (cx, cy, r, sw, *_rest) in enumerate(arc_list):
        if used[k] or not (r_cm[0] * ppc <= r <= r_cm[1] * ppc):
            continue
        group = [k]
        total = sw
        near = idx[(idx > k) & ~used & (np.abs(X - cx) <= 0.25 * r + 1.0) & (np.abs(Y - cy) <= 0.25 * r + 1.0)]
        for j in near:
            x2, y2, r2, s2 = arc_list[j][:4]
            if math.hypot(x2 - cx, y2 - cy) <= 0.25 * r and 0.8 <= r2 / r <= 1.25:
                group.append(int(j))
                total += s2
        if total >= math.radians(250):
            used[group] = True
            out.append((cx, cy, r, tuple(arc_list[g][4] for g in group)))
    return out


def _circles_reference(arc_list, ppc: float, r_cm=(2.0, 30.0)) -> list:
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
    (r"SHOWER", "shower", {"bath": 1.0}),
    (r"\bTUB\b|BATHTUB", "bathtub", {"bath": 1.0}),
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


def vector_objects(cleaned, ocr_boxes=(), ppc: float | None = None, wall_mask: np.ndarray | None = None,
                   wall_thickness: float | None = None, _rescaled: bool = False) -> list[Obj]:
    """Typed objects from the cleaner's non-architectural vector elements (+ words). When the
    cleaner could not type the page's walls (its layer has no WALL element), the walls come from
    `wall_mask` (the structural analysis, page frame) and the vectors lying on it are walls."""
    layer = getattr(cleaned, "layer", None)
    if layer is None or not ppc:
        return []
    f = 1.0 if _rescaled else _read_scale(layer.shape, ppc)
    if f > 1.0:
        # Small-scale sheets draw fixtures a few pixels wide; the vectors have no resolution, so they
        # are read at a canonical scale (up to READ_PPC px/cm, at most READ_PIXELS) and the boxes
        # mapped back to the page.
        from types import SimpleNamespace
        sl = SimpleNamespace(shape=(int(layer.shape[0] * f), int(layer.shape[1] * f)) + tuple(layer.shape[2:]),
                             wall_thickness=(layer.wall_thickness or 0) * f,
                             elements=[SimpleNamespace(type=e.type, fill=e.fill, evidence=getattr(e, "evidence", ""),
                                                       geometry=[(x * f, y * f) for x, y in e.geometry]) for e in layer.elements])
        wm = None if wall_mask is None else cv2.resize(wall_mask, (sl.shape[1], sl.shape[0]), interpolation=cv2.INTER_NEAREST)
        boxes = [SimpleNamespace(text=b.text, x=b.x * f, y=b.y * f, width=b.width * f, height=b.height * f,
                                 confidence=getattr(b, "confidence", 60)) for b in ocr_boxes]
        objs = vector_objects(SimpleNamespace(layer=sl), boxes, ppc * f, wm,
                              None if wall_thickness is None else wall_thickness * f, _rescaled=True)
        for o in objs:
            o.bbox = tuple(v / f for v in o.bbox)
            o.evidence.append(f"read at {f:.1f}x the page scale")
        return objs
    h, w = layer.shape[:2]
    typed = any(e.type == "WALL" for e in layer.elements)
    if not typed and (wall_mask is None or wall_mask.shape[:2] != (h, w)):
        return []
    t = float(layer.wall_thickness or wall_thickness or 10.0)
    others = [e for e in layer.elements if e.type == "OTHER" and not e.fill]
    arch = np.zeros((h, w), np.uint8)
    ink = np.zeros((h, w), np.uint8)
    if not typed:
        on_wall = cv2.dilate((wall_mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        arch = cv2.morphologyEx((wall_mask > 0).astype(np.uint8) * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

        def walled(e):
            g = np.asarray(e.geometry, float)
            pts = np.linspace(g[0], g[-1], 6) if len(g) == 2 else g
            hits = [on_wall[min(h - 1, max(0, int(y))), min(w - 1, max(0, int(x)))] for x, y in pts]
            return np.mean(hits) >= 0.8
        others = [e for e in others if len(e.geometry) and not walled(e)]
    for e in layer.elements:
        pts = np.round(np.asarray(e.geometry, float)).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) == 0:
            continue
        if e.type in ("WALL", "DOOR", "WINDOW", "COLUMN"):
            cv2.polylines(arch, [pts], False, 255, 2)
    for e in others:
        pts = np.round(np.asarray(e.geometry, float)).astype(np.int32).reshape(-1, 1, 2)
        if len(pts):
            cv2.polylines(ink, [pts], False, 255, 2)
    k = max(3, int(round(t)) | 1)
    near_arch = cv2.dilate(arch, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    wall_dist = cv2.distanceTransform((arch == 0).astype(np.uint8), cv2.DIST_L2, 3)
    full_ink = ink.copy()                                # every non-wall line, nothing erased (fixture outlines)
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
        grp = [j for j, (x, y, rr, _) in enumerate(burners) if j not in used and abs(x - cx) <= 60 * ppc
               and abs(y - cy) <= 60 * ppc and 0.75 <= rr / r <= 1.33]
        pts = np.array([burners[j][:2] for j in grp])
        if not 4 <= len(grp) <= 6:
            continue
        xs, ys = pts[:, 0], pts[:, 1]
        cols = len(np.unique(np.round(xs / (12 * ppc))))
        rows = len(np.unique(np.round(ys / (12 * ppc))))
        if cols >= 2 and rows >= 2 and (xs.max() - xs.min()) <= 60 * ppc and (ys.max() - ys.min()) <= 60 * ppc:
            used.update(grp)
            pad = 15 * ppc
            add("range", (xs.min() - pad, ys.min() - pad, xs.max() + pad, ys.max() + pad), {"kitchen": 1.0}, 0.9,
                f"{len(grp)} burners in a {cols}x{rows} grid")
    # a rounded rectangle is a basin only with its drain (a small circle inside): pillows, appliance
    # fronts and shelf ends are rounded rectangles too
    drains = circles(al, ppc, r_cm=(0.8, 6.0))
    bs = [b for b in bs if any(b[0] < c[0] < b[2] and b[1] < c[1] < b[3] for c in drains)
          and min(b[2] - b[0], b[3] - b[1]) >= 22 * ppc]
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

    # --- showers: the standard symbol, two diagonals crossing a rectangle ----------------------------
    segs = _segments(others)
    for box in _crossed_boxes(segs, ppc):
        add("shower", box, {"bath": 1.0}, 0.75, f"{(box[2] - box[0]) / ppc:.0f}x{(box[3] - box[1]) / ppc:.0f} cm crossed rectangle")

    # --- fixtures from closed outlines (independent of what they touch) --------------------------------
    for kind, box, fn, conf, why in _fixture_outlines(full_ink, drains, ppc, wall_dist):
        add(kind, box, fn, conf, why)

    # --- furniture from components ------------------------------------------------------------------
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    AX = np.array([a[0] for a in al], float)
    AY = np.array([a[1] for a in al], float)
    AR = np.array([a[2] for a in al], float)
    curved_ok = (8 * ppc <= AR) & (AR <= 25 * ppc)
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
        curved = int(np.count_nonzero(curved_ok & (AX >= x) & (AX <= x + bw) & (AY >= y) & (AY <= y + bh)))
        c = _bath_fixture(comp, (x, y, bw, bh), wall_dist, t, ppc) if near_wall else None
        c = c or _oval_basin(comp, (x, y), drains, ppc)
        c = c or _classify_component(wc, hc, loops, near_wall, curved)
        if c and c[0] == "media / cabinet" and _hanger_strokes(segs, (x, y, x + bw, y + bh)) >= 6:
            # many cross strokes: hangers on a rod when the box is closet-deep, else the ticks of a
            # dimension string or a hatch, not furniture
            c = ("wardrobe", {"bedroom": 0.4}, 0.6, f"{max(wc, hc):.0f}x{min(wc, hc):.0f} cm, rod with hangers") \
                if min(wc, hc) >= 45 else None
            if c is None:
                continue
        if c:
            add(c[0], (x, y, x + bw, y + bh), c[1], c[2], c[3])
        else:
            rows = _seat_rows(loops)
            for n_seats, (rx0, ry0, rx1, ry1) in rows:
                add("sofa" if n_seats >= 3 else "loveseat", (x + rx0, y + ry0, x + rx1, y + ry1),
                    {"living": 1.0 if n_seats >= 3 else 0.8}, 0.75, f"{n_seats} cushions side by side in a furniture group")
            if max(wc, hc) >= 120:
                assemblies.append(((x, y, x + bw, y + bh), wc, hc))
                for kind, (lx0, ly0, lx1, ly1), fn, why in _group_pieces(loops, circ, (x, y)):
                    add(kind, (x + lx0, y + ly0, x + lx1, y + ly1), fn, 0.6, why)
        if not c and 60 <= max(wc, hc) <= 260 and 60 <= min(wc, hc) <= 140 and len(loops) <= 2 and not near_wall:
            tables.append((x, y, x + bw, y + bh))
        # bar stools / dining chairs drawn as separate small squares are counted by the zone layer
        elif not c and 30 <= max(wc, hc) <= 60 and 25 <= min(wc, hc) <= 60:
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
    _tags(objs, ocr_boxes)
    _dining_from_pieces(objs, tables, ppc, add)
    _laundry(objs, ppc)
    _on_appliances(objs)
    _context(objs, ppc)
    _beds(objs, ppc)
    _within_fixtures(objs)
    _seating(objs, ppc)
    _lone_seats(objs, ppc)
    _dedupe(objs)
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
        if o.kind == "washbasin" and near_k and not any(
                b.kind in ("toilet", "shower", "bathtub") and math.dist(b.center, o.center) <= 250 * ppc for b in objs):
            o.kind, o.functions = "sink", {"kitchen": 0.9}           # a bowl in the kitchen counter
            o.evidence.append(f"among {len(near_k)} kitchen fixture(s), no toilet / shower near: a kitchen sink")
        if o.kind == "sink" and not near_k and any(
                b.kind in ("toilet", "shower", "bathtub") and math.dist(b.center, o.center) <= 250 * ppc for b in objs):
            o.kind, o.functions = "washbasin", {"bath": 0.6}
            o.evidence.append("beside a toilet / shower / tub")
        elif o.kind == "sink" and not near_k and any("oval bowl" in e for e in o.evidence):
            o.kind, o.functions = "washbasin", {"bath": 0.5}
            o.evidence.append("a single oval bowl away from kitchen fixtures")
        elif o.kind == "sink" and not near_k and any("bowl with a drain" in e for e in o.evidence) and not any(
                c.kind in ("counter", "counter run") and _gap(c.bbox, o.bbox) <= 100 * ppc for c in objs):
            o.kind, o.functions = "washbasin", {"bath": 0.5}
            o.evidence.append("no kitchen fixture within 3 m and no counter beside it")
        if o.kind == "washer" and (any(k.kind == "range" and math.dist(k.center, o.center) <= 300 * ppc for k in near_k)
                                   or any(s.kind == "sink" and math.dist(s.center, o.center) <= 200 * ppc for s in objs)):
            o.kind, o.functions = "dishwasher", {"kitchen": 1.0}
            o.evidence.append("among kitchen fixtures: a dishwasher")


READ_PPC = 1.0           # px per cm at which fixtures are read (a toilet bowl ~40 px)
SMALL_PPC = 0.6          # sheets drawn below this scale are re-read at READ_PPC
READ_PIXELS = 30e6       # at most this many pixels per raster of the reading


def _read_scale(shape, ppc: float) -> float:
    """Factor at which to rasterize the vectors for reading furniture (1 = the page as rendered)."""
    if ppc >= SMALL_PPC:
        return 1.0
    cap = math.sqrt(READ_PIXELS / max(1.0, float(shape[0]) * float(shape[1])))
    return max(1.0, min(READ_PPC / ppc, cap))


# ------------------------------------------------------------------------------ geometric helpers
def _segments(elements) -> np.ndarray:
    """Straight two-point elements as rows (x0, y0, x1, y1, length, angle 0-180)."""
    rows = []
    for e in elements:
        g = np.asarray(e.geometry, float)
        if len(g) == 2:
            (x0, y0), (x1, y1) = g
            rows.append((x0, y0, x1, y1, math.hypot(x1 - x0, y1 - y0), math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180))
    return np.asarray(rows, float).reshape(-1, 6)


def _crossed_boxes(segs: np.ndarray, ppc: float) -> list:
    """Rectangles 70-230 cm crossed corner to corner by two diagonals (shower / tub symbol)."""
    diag = segs[(segs[:, 4] >= 70 * ppc) & (((segs[:, 5] > 15) & (segs[:, 5] < 75)) | ((segs[:, 5] > 105) & (segs[:, 5] < 165)))]
    tol = 6 * ppc
    out = []
    for i in range(len(diag)):
        a = diag[i]
        ba = (min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3]))
        for j in range(i + 1, len(diag)):
            b = diag[j]
            if (a[5] < 90) == (b[5] < 90):
                continue                                   # same slope: not a cross
            bb = (min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3]))
            if all(abs(u - v) <= tol for u, v in zip(ba, bb)):
                w, h = (ba[2] - ba[0]) / ppc, (ba[3] - ba[1]) / ppc
                if 70 <= max(w, h) <= 230 and 60 <= min(w, h) <= 160 and not any(
                        abs(ba[0] - o[0]) <= tol and abs(ba[1] - o[1]) <= tol for o in out):
                    out.append(ba)
    return out


def _hanger_strokes(segs: np.ndarray, box) -> int:
    """Short strokes across the long axis of a wardrobe box (hangers on a rod)."""
    x0, y0, x1, y1 = box
    if not len(segs):
        return 0
    mx, my = (segs[:, 0] + segs[:, 2]) / 2, (segs[:, 1] + segs[:, 3]) / 2
    inside = (mx > x0) & (mx < x1) & (my > y0) & (my < y1)
    along = 0.0 if (x1 - x0) >= (y1 - y0) else 90.0
    off = np.abs(((segs[:, 5] - along) + 90) % 180 - 90)
    short = segs[:, 4] <= 0.9 * min(x1 - x0, y1 - y0)
    return int((inside & short & (off >= 45)).sum())


def _wall_side(box, wall_dist: np.ndarray) -> str:
    """Which edge of the box (x, y, w, h) lies against a wall."""
    x, y, w, h = box
    H, W = wall_dist.shape
    edges = {"top": wall_dist[max(0, y), x:x + w], "bottom": wall_dist[min(H - 1, y + h - 1), x:x + w],
             "left": wall_dist[y:y + h, max(0, x)], "right": wall_dist[y:y + h, min(W - 1, x + w - 1)]}
    return min(edges, key=lambda k: float(np.median(edges[k])) if edges[k].size else 1e9)


def _bath_fixture(comp: np.ndarray, box, wall_dist: np.ndarray, t: float, ppc: float):
    """Toilet or wall basin from a rounded bowl against a wall. A toilet's bowl is elongated away
    from the wall (tank at the wall); a basin's bowl is round or wider along the wall."""
    x, y, bw, bh = box
    side = _wall_side(box, wall_dist)
    along, away = ((bw, bh) if side in ("top", "bottom") else (bh, bw))
    along, away = along / ppc, away / ppc
    if not (28 <= min(along, away) and max(along, away) <= 90):
        return None
    cs, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return None
    bowls = []
    for i, c in enumerate(cs):
        if hier[0][i][3] < 0 or len(c) < 8:
            continue
        cx, cy, cw, ch = cv2.boundingRect(c)
        if max(cw, ch) < 18 * ppc:
            continue
        fill = cv2.contourArea(c) / max(1.0, cw * ch)
        if 0.62 <= fill <= 0.9:                      # an ellipse fills ~0.79 of its box, a rectangle ~1
            bowls.append((cw * ch, cw, ch))
    if not bowls:
        return None
    _, cw, ch = max(bowls)
    b_along, b_away = ((cw, ch) if side in ("top", "bottom") else (ch, cw))
    if 45 <= away <= 85 and 33 <= along <= 75 and b_away >= 1.1 * b_along and 25 * ppc <= b_away <= 55 * ppc:
        return "toilet", {"bath": 0.9}, 0.75, f"{along:.0f}x{away:.0f} cm, bowl elongated away from the wall"
    if 35 <= along <= 75 and 28 <= away <= 62 and b_along >= 0.95 * b_away:
        return "washbasin", {"bath": 0.6}, 0.65, f"{along:.0f}x{away:.0f} cm, rounded bowl against a wall"
    return None


def _gap(a, b) -> float:
    return max(0.0, max(a[0], b[0]) - min(a[2], b[2]), max(a[1], b[1]) - min(a[3], b[3]))


def _beds(objs: list[Obj], ppc: float) -> None:
    """Small pieces at a bed's head corners are nightstands; pieces inside a bed outline (pillows,
    blanket folds read as tables, counters or basins) are part of the bed."""
    beds = [o for o in objs if o.kind == "bed"]
    drop = set()
    for bed in beds:
        x0, y0, x1, y1 = bed.bbox
        for o in objs:
            if o is bed or o.kind not in ("seat", "side table", "armchair", "table", "counter", "sink", "washbasin"):
                continue
            lamp = o.kind in ("sink", "washbasin") and any("with a drain" in e for e in o.evidence)
            ox0, oy0, ox1, oy1 = o.bbox
            small = max(ox1 - ox0, oy1 - oy0) <= 80 * ppc
            if _gap(bed.bbox, o.bbox) > 12 * ppc:
                continue
            tol = 12 * ppc
            at_x = min(abs(ox0 - x0), abs(ox1 - x1), abs(ox1 - x0), abs(ox0 - x1)) <= tol
            at_y = min(abs(oy0 - y0), abs(oy1 - y1), abs(oy1 - y0), abs(oy0 - y1)) <= tol
            if small and at_x and at_y and (o.kind in ("seat", "side table", "armchair") or lamp):
                # a drained outline at a bed's head corner is a nightstand with its lamp, not a basin
                o.kind, o.functions, o.confidence = "nightstand", {"bedroom": 0.3}, 0.5
                o.evidence.append("at a bed's corner")
            elif x0 - 2 <= o.center[0] <= x1 + 2 and y0 - 2 <= o.center[1] <= y1 + 2:
                drop.add(id(o))
    objs[:] = [o for o in objs if id(o) not in drop]


def _lone_seats(objs: list[Obj], ppc: float) -> None:
    """A seat-sized square is a seat only beside a table, counter or another seat (bar stools,
    dining chairs); alone it is a symbol, a hamper or a lamp."""
    company = ("seat", "table", "dining set", "counter", "counter run", "sofa", "loveseat")
    keep = []
    for o in objs:
        if o.kind == "seat" and not any(p is not o and p.kind in company and _gap(o.bbox, p.bbox) <= 50 * ppc for p in objs):
            continue
        keep.append(o)
    objs[:] = keep


def _dedupe(objs: list[Obj]) -> None:
    """One object per drawn thing: of near-identical boxes keep the most confident reading."""
    def iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ar = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
        return inter / max(1e-6, ar(a) + ar(b) - inter)
    keep: list[Obj] = []
    for o in sorted(objs, key=lambda o: -o.confidence):
        if not any(iou(o.bbox, k.bbox) >= 0.85 or (k.kind == o.kind and iou(o.bbox, k.bbox) >= 0.5) for k in keep):
            keep.append(o)
    order = {id(o): i for i, o in enumerate(objs)}
    objs[:] = sorted(keep, key=lambda o: order[id(o)])


def within_walls(objs: list[Obj], building: dict) -> list[Obj]:
    """Objects inside the walls of a building: the convex hull of each building's walls (walls are
    assigned to the nearest footprint). Legend symbols, title-block words and equipment drawn
    outside the exterior walls are not furniture of the plan."""
    walls = building.get("walls") or []
    if not walls:
        return objs
    feet = [np.asarray(b["polygon"], np.float32).reshape(-1, 1, 2) for b in building.get("buildings", []) if b.get("polygon")]
    groups: dict[int, list] = {}
    for w in walls:
        mid = ((w["p0"][0] + w["p1"][0]) / 2, (w["p0"][1] + w["p1"][1]) / 2)
        k = max(range(len(feet)), key=lambda i: cv2.pointPolygonTest(feet[i], mid, True)) if feet else 0
        groups.setdefault(k, []).extend([w["p0"], w["p1"]])
    hulls = [cv2.convexHull(np.asarray(pts, np.float32)) for pts in groups.values() if len(pts) >= 3]
    kept = [o for o in objs if any(cv2.pointPolygonTest(h, tuple(map(float, o.center)), False) >= 0 for h in hulls)]
    for i, o in enumerate(kept, 1):
        o.id = f"OB{i:03d}"
    return kept


def _oval_basin(comp: np.ndarray, origin, drains: list, ppc: float):
    """An oval or round bowl 25-65 cm (its outline fills an ellipse's share of its box) with a
    drain inside: a basin. Bath or kitchen is decided by the fixtures around it (_context)."""
    h, w = comp.shape
    if not (25 * ppc <= max(h, w) <= 65 * ppc and min(h, w) >= 18 * ppc):
        return None
    cs, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None or not any(hier[0][i][3] >= 0 for i in range(len(cs))):
        return None
    outer = max((c for i, c in enumerate(cs) if hier[0][i][3] < 0), key=cv2.contourArea)
    ox, oy = origin
    x, y, cw, ch = cv2.boundingRect(outer)
    fill = cv2.contourArea(outer) / max(1.0, cw * ch)
    mx, my = 0.1 * cw, 0.1 * ch
    if 0.68 <= fill <= 0.88 and any(ox + x + mx < d[0] < ox + x + cw - mx and oy + y + my < d[1] < oy + y + ch - my for d in drains):
        return "sink", {"kitchen": 0.4, "bath": 0.5}, 0.65, f"{w / ppc:.0f}x{h / ppc:.0f} cm oval bowl with a drain"
    return None


CODED = re.compile(r"\d|[%#]")


def _tags(objs: list[Obj], ocr_boxes) -> None:
    """A box-like 'table' / 'chair' enclosing a coded annotation (KIT-1, A-3.7, 2% SLOPE, a detail
    number) is an annotation tag, not furniture. Room names (LIVING) are not codes: furniture
    under a room label stays."""
    words = [b for b in ocr_boxes if CODED.search(str(getattr(b, "text", ""))) and float(getattr(b, "confidence", 0)) >= 60]
    keep = []
    for o in objs:
        if o.kind in ("table", "armchair", "side table", "seat") and any(
                o.bbox[0] <= b.x + b.width / 2 <= o.bbox[2] and o.bbox[1] <= b.y + b.height / 2 <= o.bbox[3] for b in words):
            continue
        keep.append(o)
    objs[:] = keep


def _group_pieces(loops: list, circ: list, origin) -> list:
    """Pieces drawn inside a furniture group (a living set on a rug): a 60-100 cm square frame
    holding a seat cushion is an armchair; a 35-60 cm square holding a lamp circle is an end table."""
    ox, oy = origin
    ppc = _PPC[0]
    out = []

    def inside(b, a):
        return b[0] >= a[0] - 2 and b[1] >= a[1] - 2 and b[2] <= a[2] + 2 and b[3] <= a[3] + 2 and b != a
    for w, h, box in loops:
        if not (max(w, h) <= 1.3 * min(w, h)):
            continue
        if 60 <= min(w, h) and max(w, h) <= 100 and any(
                inside(b, box) and 40 <= max(lw, lh) <= 75 and min(lw, lh) >= 35 for lw, lh, b in loops):
            out.append(("armchair", box, {"living": 0.5}, f"{w:.0f}x{h:.0f} cm frame around a seat, in a furniture group"))
        elif 35 <= min(w, h) and max(w, h) <= 60 and any(
                box[0] <= cx - ox <= box[2] and box[1] <= cy - oy <= box[3] and 5 * ppc <= r <= 0.45 * min(w, h) * ppc
                for cx, cy, r, _ in circ):
            out.append(("side table", box, {"living": 0.2}, f"{w:.0f}x{h:.0f} cm with a lamp, in a furniture group"))
    return out


def _seating(objs: list[Obj], ppc: float) -> None:
    """Small tables flanking or fronting a sofa / loveseat (within 30 cm) are end / coffee tables of
    a living arrangement: the arrangement, not each piece, is the evidence."""
    seats = [o for o in objs if o.kind in ("sofa", "loveseat")]
    for o in objs:
        if o.kind == "side table" and any(_gap(o.bbox, s.bbox) <= 30 * ppc for s in seats):
            o.kind, o.functions, o.confidence = "end table", {"living": 0.35}, 0.5
            o.evidence.append("beside a sofa: part of a seating arrangement")


def _fixture_outlines(ink: np.ndarray, drains: list, ppc: float, wall_dist: np.ndarray | None = None) -> list:
    """Sanitary fixtures read from the closed outlines they are drawn with. Every fixture symbol is
    a closed shape; its interior is a hole in the ink whatever the symbol touches (walls, counters,
    neighbouring fixtures), so this does not depend on how the ink splits into components.

      bathtub      an elongated rounded basin 120-195 x 45-95 cm with a drain
      shower       a square-ish tray 65-200 cm (rectangular interior) with a drain
      basin        a rounded / oval bowl 25-65 cm with a drain (bath or kitchen: decided by context)
      toilet       an elliptical bowl 28-55 cm, no drain, with a narrow tank 6-30 cm deep beside it

    Drains are the circles found in the vectors plus small round holes in the ink (polygonised circles).

    Nested outlines (a tub's rim and basin) give one fixture: the outermost qualifying outline."""
    cs, hier = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    holes = []
    drains = list(drains)
    for i, c in enumerate(cs):
        if hier[0][i][3] < 0:
            continue
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if max(w, h) < 8 * ppc:
            # a drain drawn as a small ring (polygonised circles are not arcs): a round hole 1.5-9 cm
            if 1.5 * ppc <= max(w, h) <= 9 * ppc and max(w, h) <= 1.4 * max(1, min(w, h)) and area >= 0.3 * w * h:
                drains.append((x + w / 2, y + h / 2, max(w, h) / 2, ()))
            continue
        holes.append((x, y, x + w, y + h, w / ppc, h / ppc, area / max(1.0, w * h)))

    def drain_in(b, margin=0.15):
        # a drain is small beside its bowl (a lamp's circle on a nightstand is not)
        x0, y0, x1, y1 = b[:4]
        mx, my = margin * (x1 - x0), margin * (y1 - y0)
        r_max = 0.15 * min(x1 - x0, y1 - y0)
        return any(x0 + mx < d[0] < x1 - mx and y0 + my < d[1] < y1 - my and d[2] <= r_max for d in drains)
    out = []
    for hb in holes:
        x0, y0, x1, y1, wc, hc, fill = hb
        lng, sht = max(wc, hc), min(wc, hc)
        if 120 <= lng <= 195 and 45 <= sht <= 95 and 0.72 <= fill <= 0.97 and drain_in(hb):
            out.append(("bathtub", hb[:4], {"bath": 1.0}, 0.75, f"{lng:.0f}x{sht:.0f} cm rounded basin with a drain"))
        elif 65 <= sht and lng <= 200 and lng <= 1.8 * sht and fill >= 0.9 and drain_in(hb, 0.05):
            out.append(("shower", hb[:4], {"bath": 1.0}, 0.7, f"{lng:.0f}x{sht:.0f} cm tray with a drain"))
        elif 25 <= lng <= 65 and sht >= 18 and 0.62 <= fill <= 0.96 and drain_in(hb, 0.1):
            out.append(("sink", hb[:4], {"kitchen": 0.4, "bath": 0.5}, 0.65, f"{lng:.0f}x{sht:.0f} cm bowl with a drain"))
        elif 28 <= lng <= 55 and sht >= 22 and 0.68 <= fill <= 0.86 and not drain_in(hb, 0.0):
            # a toilet bowl: elliptical, beside a narrow tank on its long axis
            vertical = hc >= wc
            for tb in holes:
                tx0, ty0, tx1, ty1, tw, th, tfill = tb
                tl, ts = max(tw, th), min(tw, th)
                if tb is hb or not (6 <= ts <= 30 and 28 <= tl <= 65 and tfill >= 0.5):
                    continue
                if vertical and tw > th and min(abs(ty1 - y0), abs(y1 - ty0)) <= 15 * ppc and tx0 - 5 * ppc <= (x0 + x1) / 2 <= tx1 + 5 * ppc:
                    break
                if not vertical and th > tw and min(abs(tx1 - x0), abs(x1 - tx0)) <= 15 * ppc and ty0 - 5 * ppc <= (y0 + y1) / 2 <= ty1 + 5 * ppc:
                    break
            else:
                continue
            if wall_dist is not None:
                # the tank stands against a wall (a chair's back at a table does not)
                ty0i, ty1i, tx0i, tx1i = int(tb[1]), int(tb[3]) + 1, int(tb[0]), int(tb[2]) + 1
                near = wall_dist[max(0, ty0i - int(30 * ppc)):ty1i + int(30 * ppc), max(0, tx0i - int(30 * ppc)):tx1i + int(30 * ppc)]
                if not near.size or float(near.min()) > 2.0:
                    continue
            bx = (min(x0, tb[0]), min(y0, tb[1]), max(x1, tb[2]), max(y1, tb[3]))
            out.append(("toilet", bx, {"bath": 0.9}, 0.75, f"{lng:.0f}x{sht:.0f} cm bowl with its tank"))
    # nested outlines: keep the outermost fixture of each kind
    keep = []
    for k, o in enumerate(out):
        if any(j != k and p[0] == o[0] and p[1][0] <= o[1][0] and p[1][1] <= o[1][1] and p[1][2] >= o[1][2]
               and p[1][3] >= o[1][3] and p[1] != o[1] for j, p in enumerate(out)):
            continue
        keep.append(o)
    return keep


APPLIANCES = ("range", "refrigerator", "dishwasher", "washer", "dryer", "laundry appliance")


def _on_appliances(objs: list[Obj]) -> None:
    """A basin, tray or tub read from an outline that lies on an appliance (a burner in a range, the
    drum or knob of a washer) is that appliance's detail, not a fixture."""
    apps = [o for o in objs if o.kind in APPLIANCES]
    keep = []
    for o in objs:
        if o.kind in ("sink", "shower", "bathtub") and any("with a drain" in e for e in o.evidence) and any(
                a.bbox[0] <= o.center[0] <= a.bbox[2] and a.bbox[1] <= o.center[1] <= a.bbox[3] for a in apps):
            continue
        keep.append(o)
    objs[:] = keep


SANITARY = ("toilet", "shower", "bathtub", "sink", "washbasin")


def _within_fixtures(objs: list[Obj]) -> None:
    """A generic piece (table, seat, cabinet) lying mostly inside a sanitary fixture's outline is part
    of that fixture (a toilet's bowl read as a side table, a tray's corners read as seats)."""
    fixtures = [o for o in objs if o.kind in SANITARY]

    def inside(o, f):
        ix = max(0.0, min(o.bbox[2], f.bbox[2]) - max(o.bbox[0], f.bbox[0]))
        iy = max(0.0, min(o.bbox[3], f.bbox[3]) - max(o.bbox[1], f.bbox[1]))
        ao = max(1e-6, (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]))
        af = max(1e-6, (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return ix * iy >= 0.7 * ao or ix * iy >= 0.4 * (ao + af - ix * iy)
    objs[:] = [o for o in objs if not (o.kind in ("side table", "seat", "armchair", "table", "media / cabinet")
                                       and any(inside(o, f) for f in fixtures))]
