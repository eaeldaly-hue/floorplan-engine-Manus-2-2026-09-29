"""The reconstructed plan as a DXF drawing.

Source: the analyzer's result (the structured plan `building` from engine.arch.building, the room
records, the openings' evidence, the image size and the scale). The exporter only serialises what
the model holds; it detects nothing, corrects nothing and fills in nothing:

  WALLS                       each wall piece: its centreline offset by half its measured thickness
  WALL_CENTERLINES            the wall pieces' axes
  DOORS                       door span (threshold line) and, where the hinge and the swing side were
                              read from the drawing, the leaf and its swing arc; nothing otherwise
  WINDOWS                     window span: the frame across the wall's measured thickness + the glazing axis
  OPENINGS_UNCLASSIFIED       openings the classifier left untyped (passages, open sides, unknown)
  ROOM_BOUNDARIES             named rooms bounded by walls (boundary method 'wall-region')
  ROOM_BOUNDARIES_INFERRED    named rooms whose extent is inferred (open-plan zones, dimension estimates)
  SPACE_BOUNDARIES            structural spaces not already drawn as one wall-bounded room
  ROOM_LABELS                 room names (and areas when calibrated) at the label's position
  NOTES                       source, units and scale of the drawing

Coordinates: the model is in pixels of the analysed image (origin top-left, y down). The drawing
puts X = x * s, Y = (H - y) * s (y up, the plan not mirrored), H the image height. s converts pixels
to millimetres when the plan's printed dimensions calibrated it (result['pixel_scale']); otherwise
s = 1, the drawing is unitless ($INSUNITS 0) and 1 drawing unit = 1 source pixel. The door-width
estimate (doors taken as 82 cm) and a PDF page's paper size are reported in NOTES, never applied.

Every element carries XDATA (application FLOORPLAN_ENGINE): its model id, type and confidence.
Elements that cannot be drawn (non-finite or degenerate geometry) are skipped and listed in the
export report; they are never repaired.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import ezdxf
from ezdxf.document import Drawing

APPID = "FLOORPLAN_ENGINE"
WALL_BOUNDED = {"wall-region"}
# layer: (ACI colour, linetype, description)
LAYERS = {
    "WALLS": (7, "Continuous", "Wall pieces: centreline +/- half the measured wall thickness"),
    "WALL_CENTERLINES": (8, "CENTER", "Wall piece axes"),
    "DOORS": (3, "Continuous", "Door spans; leaves and swing arcs where read from the drawing"),
    "WINDOWS": (5, "Continuous", "Window frames across the wall and glazing axes"),
    "OPENINGS_UNCLASSIFIED": (4, "DASHED", "Openings left untyped by the classifier"),
    "ROOM_BOUNDARIES": (6, "Continuous", "Named rooms bounded by walls"),
    "ROOM_BOUNDARIES_INFERRED": (6, "DASHED", "Named rooms with an inferred extent (zones, estimates)"),
    "SPACE_BOUNDARIES": (8, "DASHED", "Structural spaces not drawn as one wall-bounded room"),
    "ROOM_LABELS": (2, "Continuous", "Room names"),
    "NOTES": (1, "Continuous", "Source, units and scale"),
}


class ExportError(ValueError):
    """The result holds no model that can be exported."""


@dataclass
class ExportReport:
    units: str                      # "mm" or "unitless (1 unit = 1 source pixel)"
    scale_source: str
    mm_per_px: float | None
    counts: dict = field(default_factory=dict)       # layer -> entities written
    skipped: list = field(default_factory=list)      # (element id, reason)
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"units": self.units, "scale_source": self.scale_source, "mm_per_px": self.mm_per_px,
                "counts": dict(self.counts), "skipped": [{"id": i, "reason": r} for i, r in self.skipped],
                "notes": list(self.notes)}


def scale_policy(result: dict) -> tuple[float | None, str]:
    """Millimetres per source pixel from the plan's printed dimensions, else (None, why)."""
    ps = result.get("pixel_scale") or {}
    ppu, unit = ps.get("pixels_per_unit"), ps.get("unit")
    if ppu and unit in ("m", "ft") and math.isfinite(float(ppu)) and float(ppu) > 0:
        mm_per_unit = 1000.0 if unit == "m" else 304.8
        return mm_per_unit / float(ppu), f"printed room dimensions ({float(ppu):.2f} px per {unit})"
    return None, "uncalibrated: no printed dimensions were read"


class _Writer:
    def __init__(self, doc: Drawing, height: float, s: float, report: ExportReport):
        self.msp = doc.modelspace()
        self.h, self.s, self.report = float(height), float(s), report

    def pt(self, p) -> tuple[float, float] | None:
        try:
            x, y = (float(p["x"]), float(p["y"])) if isinstance(p, dict) else (float(p[0]), float(p[1]))
        except (TypeError, ValueError, IndexError, KeyError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        return x * self.s, (self.h - y) * self.s

    def _tag(self, e, layer: str, eid: str, kind: str, confidence=None) -> None:
        tags = [(1000, f"id={eid}"), (1000, f"type={kind}")]
        if isinstance(confidence, (int, float)) and math.isfinite(confidence):
            tags.append((1040, float(confidence)))
        e.set_xdata(APPID, tags)
        self.report.counts[layer] = self.report.counts.get(layer, 0) + 1

    def skip(self, eid: str, reason: str) -> None:
        self.report.skipped.append((eid, reason))

    def line(self, a, b, layer, eid, kind, confidence=None) -> bool:
        if a is None or b is None or math.dist(a, b) < 1e-6:
            return False
        self._tag(self.msp.add_line(a, b, dxfattribs={"layer": layer}), layer, eid, kind, confidence)
        return True

    def polygon(self, pts, layer, eid, kind, confidence=None) -> bool:
        clean = []
        for p in pts:
            if p is None:
                return False
            if not clean or math.dist(clean[-1], p) > 1e-6:
                clean.append(p)
        if len(clean) > 1 and math.dist(clean[0], clean[-1]) <= 1e-6:
            clean.pop()
        if len(clean) < 3 or abs(_area(clean)) < 1e-9:
            return False
        e = self.msp.add_lwpolyline(clean, close=True, dxfattribs={"layer": layer})
        self._tag(e, layer, eid, kind, confidence)
        return True

    def arc(self, centre, radius, a0, a1, layer, eid, kind, confidence=None) -> bool:
        if centre is None or not (radius > 1e-6 and math.isfinite(radius)):
            return False
        e = self.msp.add_arc(centre, radius, a0, a1, dxfattribs={"layer": layer})
        self._tag(e, layer, eid, kind, confidence)
        return True

    def text(self, s, at, height, layer, eid, kind) -> bool:
        if at is None or not s:
            return False
        e = self.msp.add_mtext(s, dxfattribs={"layer": layer, "char_height": height})
        e.set_location(at, attachment_point=5)          # middle centre on the label point
        self._tag(e, layer, eid, kind)
        return True


def _area(pts) -> float:
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))


def _offset_rect(a, b, half: float):
    """The rectangle around segment a-b, half its width on each side (model frame)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    if L < 1e-9 or not (half > 0 and math.isfinite(half)):
        return None
    nx, ny = -dy / L * half, dx / L * half
    return [(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny), (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny)]


def _new_doc(result: dict, report: ExportReport) -> Drawing:
    # the full standard setup: LibreDWG (engine.export.dwg) cannot read back DWG made from a reduced one
    doc = ezdxf.new("R2010", setup=True)
    doc.appids.add(APPID)
    for name, (color, linetype, desc) in LAYERS.items():
        layer = doc.layers.add(name, color=color, linetype=linetype if linetype in doc.linetypes else "Continuous")
        layer.description = desc
    calibrated = report.mm_per_px is not None
    doc.header["$INSUNITS"] = 4 if calibrated else 0        # millimetres / unitless
    doc.header["$MEASUREMENT"] = 1 if calibrated else 0
    for tag, value in (("FPE_SOURCE", str(result.get("source_name") or "")), ("FPE_UNITS", report.units),
                       ("FPE_SCALE_SOURCE", report.scale_source),
                       ("FPE_MM_PER_PX", f"{report.mm_per_px:.6f}" if calibrated else "none")):
        doc.header.custom_vars.append(tag, value)
    return doc


def build_dxf(result: dict) -> tuple[Drawing, ExportReport]:
    """The analysis `result` (FloorPlanAnalyzer.analyze output, or its JSON) as a DXF drawing."""
    b = result.get("building")
    image = result.get("image") or {}
    if not isinstance(b, dict) or not image.get("height"):
        raise ExportError("No structured plan to export (the analysis produced no building model).")
    mm_per_px, how = scale_policy(result)
    report = ExportReport(units="mm" if mm_per_px else "unitless (1 unit = 1 source pixel)",
                          scale_source=how, mm_per_px=round(mm_per_px, 6) if mm_per_px else None)
    doc = _new_doc(result, report)
    w = _Writer(doc, image["height"], mm_per_px or 1.0, report)
    s = w.s

    # --- walls ------------------------------------------------------------------------------------
    for wl in b.get("walls", []):
        eid, t = wl.get("id", "?"), wl.get("thickness_px")
        a, c = w.pt(wl.get("p0")), w.pt(wl.get("p1"))
        if a is None or c is None or math.dist(a, c) < 1e-6:
            w.skip(eid, "wall: missing, non-finite or zero-length centreline")
            continue
        kind = "exterior wall" if wl.get("exterior") else "interior wall"
        w.line(a, c, "WALL_CENTERLINES", eid, kind)
        rect = _offset_rect(a, c, 0.5 * float(t) * s) if isinstance(t, (int, float)) else None
        if rect is None or not w.polygon(rect, "WALLS", eid, kind):
            w.skip(eid, "wall outline: no valid measured thickness (centreline only)")

    # --- openings ---------------------------------------------------------------------------------
    evidence = {o.get("id"): (o.get("evidence") or {}) for o in result.get("openings", [])}
    for o in b.get("openings", []):
        eid, typ, conf = o.get("id", "?"), o.get("type"), o.get("confidence")
        a, c = w.pt(o.get("p0")), w.pt(o.get("p1"))
        if a is None or c is None or math.dist(a, c) < 1e-6:
            w.skip(eid, f"{typ}: missing, non-finite or zero-length span")
            continue
        ev = evidence.get(eid, {})
        t = ev.get("wall_thickness_px")
        if typ == "door":
            w.line(a, c, "DOORS", eid, "door span", conf)
            _door_swing(w, o, ev, a, c, eid, conf)
        elif typ == "window":
            rect = _offset_rect(a, c, 0.5 * float(t) * s) if isinstance(t, (int, float)) else None
            if rect is None or not w.polygon(rect, "WINDOWS", eid, "window frame", conf):
                report.notes.append(f"{eid}: window frame not drawn (no measured wall thickness); glazing axis only")
            w.line(a, c, "WINDOWS", eid, "window glazing axis", conf)
        else:
            w.line(a, c, "OPENINGS_UNCLASSIFIED", eid, "unclassified opening", conf)

    # --- rooms and spaces -------------------------------------------------------------------------
    drawn_spaces = set()
    for room in result.get("rooms", []):
        rid, bd = room.get("id", "?"), room.get("boundary") or {}
        poly = bd.get("polygon") or []
        method = bd.get("method")
        layer = "ROOM_BOUNDARIES" if method in WALL_BOUNDED else "ROOM_BOUNDARIES_INFERRED"
        if poly and w.polygon([w.pt(p) for p in poly], layer, rid, f"room ({method})", bd.get("confidence")):
            if method in WALL_BOUNDED and room.get("space_id"):
                drawn_spaces.add(room["space_id"])
        elif poly:
            w.skip(rid, "room boundary: non-finite or degenerate polygon")
        lc = room.get("label_center") or {}
        name = room.get("name")
        area = room.get("area") or {}
        if name and area.get("value") is not None and not str(area.get("unit", "")).startswith("px"):
            name = f"{name}\\P{area['value']} {area.get('unit', '')}".strip()
        if name and not w.text(name, w.pt(lc), _text_height(image, s), "ROOM_LABELS", rid, "room label"):
            w.skip(rid, "room label: no label position")
    for sp in b.get("spaces", []):
        if sp.get("id") in drawn_spaces or not sp.get("polygon"):
            continue
        if not w.polygon([w.pt(p) for p in sp["polygon"]], "SPACE_BOUNDARIES", sp.get("id", "?"), f"space ({sp.get('kind')})"):
            w.skip(sp.get("id", "?"), "space boundary: non-finite or degenerate polygon")

    _notes(w, result, report, image)
    return doc, report


def _door_swing(w: _Writer, o: dict, ev: dict, a, c, eid: str, conf) -> None:
    """Leaf + 90 degree swing arc from each hinge read from the drawing, towards the swing side
    read from the drawing. Without both, the door is its span only."""
    door = o.get("door") or {}
    hinges = door.get("hinges") or []
    side = ev.get("swing_side")
    if not hinges or side not in (-1, 1):
        if door.get("operation", "").endswith("hinged"):
            w.report.notes.append(f"{eid}: hinge or swing side not read; door drawn as its span only")
        return
    # the swing side is measured along the span's normal in the image frame (start -> end, normal
    # (-dy, dx)); the drawing flips y, so the same side is the opposite normal there
    p0, p1 = o["p0"], o["p1"]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    nx, ny = -dy / L * side, dx / L * side                        # image frame, towards the swing
    n_draw = (nx, -ny)                                            # y flipped
    for hp in hinges:
        h = w.pt(hp)
        if h is None:
            continue
        free = c if math.dist(h, a) < math.dist(h, c) else a
        if len(hinges) == 2:                                      # double door: two half leaves
            free = ((a[0] + c[0]) / 2, (a[1] + c[1]) / 2)
        r = math.dist(h, free)
        if r < 1e-6:
            continue
        leaf_end = (h[0] + n_draw[0] * r, h[1] + n_draw[1] * r)
        w.line(h, leaf_end, "DOORS", eid, "door leaf", conf)
        a_free = math.degrees(math.atan2(free[1] - h[1], free[0] - h[0]))
        a_leaf = math.degrees(math.atan2(leaf_end[1] - h[1], leaf_end[0] - h[0]))
        ccw = (free[0] - h[0]) * (leaf_end[1] - h[1]) - (free[1] - h[1]) * (leaf_end[0] - h[0]) > 0
        start, end = (a_free, a_leaf) if ccw else (a_leaf, a_free)
        w.arc(h, r, start, end, "DOORS", eid, "door swing", conf)


def _text_height(image: dict, s: float) -> float:
    return max(1.0, 0.018 * min(float(image.get("width") or 0) or 1000.0, float(image["height"]))) * s


def _notes(w: _Writer, result: dict, report: ExportReport, image: dict) -> None:
    lines = [f"Floor Plan Engine export: {result.get('source_name') or 'plan'}",
             f"Units: {report.units}. Scale: {report.scale_source}."]
    est = (result.get("scale_estimate") or {})
    if report.mm_per_px is None and est.get("px_per_cm"):
        lines.append(f"Not applied: estimated {est['px_per_cm']} px/cm from {est.get('source')} (an assumption, not a measurement).")
    render = ((result.get("source") or {}).get("render") or {})
    if render.get("dpi"):
        lines.append(f"PDF page rendered at {render['dpi']} DPI (paper size, not the drawing scale).")
    if report.skipped:
        lines.append(f"{len(report.skipped)} element(s) not drawn: see the export report.")
    report.notes[:0] = lines
    s = w.s
    top = (0.0, float(image["height"]) * s + 4 * _text_height(image, s))
    w.text("\\P".join(lines), (top[0] + 0.5 * float(image.get("width") or 0) * s, top[1]),
           0.6 * _text_height(image, s), "NOTES", "notes", "notes")


def dxf_bytes(result: dict) -> tuple[bytes, ExportReport]:
    doc, report = build_dxf(result)
    stream = io.StringIO()
    doc.write(stream)
    return stream.getvalue().encode(doc.output_encoding), report


def save_dxf(result: dict, path) -> ExportReport:
    doc, report = build_dxf(result)
    doc.saveas(path)
    return report
