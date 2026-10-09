"""Architectural Cleaner: original drawing -> clean architectural representation, before recognition.

The cleaner does not recognise rooms. It decides, element by element, what is architecture:

  KEEP      walls (faces and solid wall bodies), columns, doors (swing arcs, leaves), windows and
            glazing, and the openings they close (a door or window seals its room as built)
  SUPPRESS  text and room names, dimensions and dimension lines, furniture, fixtures, symbols,
            tags, notes, title blocks, grids - everything typed OTHER / TEXT / DIMENSION

Nothing is destroyed: the original page stays the source and evidence layer (labels, dimensions
and scale are read from it), and every suppressed element keeps its type in the representation.

Sources, tried in this order (the first that applies is used):
  vector-wall-pen  PDF vectors; the walls are the pen(s) that reconstruct as a wall network on their
                   own (engine.semantic.layer)
  vector-poche     PDF vectors; walls drawn as outline + hatch (poché) in a shared pen
  raster           any image; walls and openings from the Plan Model (engine.plan) - only in the
                   explicit "raster" mode: on the development images it made recognition worse
  none             nothing reliable: the original is used unchanged (recognition is not affected)

Outputs: `recognition_image` (input for the existing recognition engine: walls, wall bodies, door
and window symbols, openings sealed), `skeleton_image` (the same, coloured for inspection),
`elements_image` (every element of the original coloured by its type) and `representation()`
(structured walls / openings / suppressed counts).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# on / vector: vector sources only (measured: large gains on vector CAD pages)
# raster: vector sources, then the raster source - experimental, measured WORSE on the development
#         images (separated room points 157 -> 108); for investigation only
MODES = ("off", "on", "vector", "raster")


def enabled(request_body: dict | None = None) -> str:
    """Cleaning mode for a request: request {"cleaning": "on"|"off"|"vector"|"raster"|true|false},
    else FLOORPLAN_CLEANING (default off). "on" == "vector"."""
    value = None
    if request_body:
        value = request_body.get("cleaning")
        if value is None and request_body.get("structural_layer") is not None:   # earlier experiment flag
            value = "vector" if request_body.get("structural_layer") else "off"
    if value is None:
        value = os.environ.get("FLOORPLAN_CLEANING")
    if value is None and os.environ.get("FLOORPLAN_STRUCTURAL_LAYER", "").lower() in ("1", "on", "true", "yes"):
        value = "vector"
    if value is True:
        value = "on"
    if value in (False, None):
        value = "off"
    value = str(value).strip().lower()
    return value if value in MODES else "off"


@dataclass
class CleanResult:
    source: str = "none"
    applicable: bool = False
    reason: str = ""
    recognition_image: np.ndarray | None = None
    skeleton_image: np.ndarray | None = None
    walls: list = field(default_factory=list)          # {"polyline", "width"} or {"p0","p1","thickness"}
    openings: list = field(default_factory=list)       # {"p0","p1","kind","evidence"}
    symbols: dict = field(default_factory=dict)        # type -> count kept (DOOR, WINDOW, COLUMN)
    suppressed: dict = field(default_factory=dict)     # type -> count removed (TEXT, DIMENSION, OTHER)
    wall_thickness: float = 0.0
    seconds: float = 0.0
    layer: object = None
    model: object = None

    def representation(self) -> dict:
        def pt(p):
            return [round(float(p[0]), 1), round(float(p[1]), 1)]
        return {
            "source": self.source, "applicable": self.applicable, "reason": self.reason,
            "wall_thickness_px": round(self.wall_thickness, 1),
            "walls": len(self.walls), "symbols_kept": self.symbols, "suppressed": self.suppressed,
            "openings": [{"p0": pt(o["p0"]), "p1": pt(o["p1"]), "kind": o["kind"], "evidence": o.get("evidence", "")}
                         for o in self.openings],
            "opening_counts": {k: sum(o["kind"] == k for o in self.openings)
                               for k in sorted({o["kind"] for o in self.openings})},
            "seconds": round(self.seconds, 2),
        }

    def elements_image(self, original: np.ndarray) -> np.ndarray:
        if self.layer is not None:
            return self.layer.debug_image(original)
        vis = (0.3 * original + 0.7 * 255).astype(np.uint8)
        if self.recognition_image is not None:
            vis[self.recognition_image[..., 0] < 128] = (30, 30, 30)
        return vis


# ---------------------------------------------------------------------------

KEEP_COLORS = {"WALL": (25, 25, 25), "COLUMN": (90, 0, 90), "DOOR": (0, 150, 0), "WINDOW": (220, 110, 0)}
OPENING_COLOR = (0, 0, 220)


def _skeleton_from_layer(layer) -> np.ndarray:
    from engine.semantic.layer import _draw

    h, w = layer.shape[:2]
    img = np.full((h, w, 3), 255, np.uint8)
    if layer.wall_body is not None:
        img[layer.wall_body] = KEEP_COLORS["WALL"]
    for kind in ("WALL", "COLUMN", "WINDOW", "DOOR"):
        for e in layer.elements:
            if e.type == kind:
                _draw(img, e, KEEP_COLORS[kind], minimum=2 if kind in ("DOOR", "WINDOW") else 1)
    for c in layer.closures:                      # sealed openings: thin red line across the gap
        cv2.line(img, tuple(int(round(v)) for v in c["p0"]), tuple(int(round(v)) for v in c["p1"]), OPENING_COLOR,
                 2, cv2.LINE_AA)
    return img


def _from_layer(layer, source: str, t0: float) -> CleanResult:
    counts = layer.counts()
    res = CleanResult(source=source, applicable=True, reason=layer.reason, layer=layer,
                      recognition_image=layer.structural_image(closures=True),
                      skeleton_image=_skeleton_from_layer(layer),
                      walls=[{"polyline": e.geometry, "width": e.width} for e in layer.elements if e.type == "WALL"],
                      openings=list(layer.closures),
                      symbols={k: counts.get(k, 0) for k in ("DOOR", "WINDOW", "COLUMN")},
                      suppressed={k: counts.get(k, 0) for k in ("TEXT", "DIMENSION", "OTHER")},
                      wall_thickness=layer.wall_thickness)
    res.seconds = time.perf_counter() - t0
    return res


def _from_raster(image: np.ndarray, t0: float) -> CleanResult:
    """Walls and openings of the Plan Model. Door / window symbols cannot be separated from the
    other ink on a raster yet: openings are kept as their closures."""
    from engine.plan.reconstruct import reconstruct, render_walls

    model = reconstruct(image)
    if not model.walls or not model.spaces:
        return CleanResult(source="none", reason="raster: the Plan Model found no enclosed wall network",
                           seconds=time.perf_counter() - t0, model=model)
    h, w = image.shape[:2]
    walls = render_walls(model.walls, (h, w))
    closed = walls.copy()
    for o in model.openings:
        cv2.line(closed, tuple(int(round(v)) for v in o.p0), tuple(int(round(v)) for v in o.p1), 255,
                 max(2, int(round(o.thickness))))
    rec = np.full((h, w, 3), 255, np.uint8)
    rec[closed > 0] = 0
    sk = np.full((h, w, 3), 255, np.uint8)
    sk[walls > 0] = KEEP_COLORS["WALL"]
    for o in model.openings:
        cv2.line(sk, tuple(int(round(v)) for v in o.p0), tuple(int(round(v)) for v in o.p1),
                 {"door": KEEP_COLORS["DOOR"], "window": KEEP_COLORS["WINDOW"]}.get(o.kind, OPENING_COLOR), 3)
    t_med = float(np.median([wl.thickness for wl in model.walls]))
    return CleanResult(source="raster", applicable=True,
                       reason=f"raster: {len(model.walls)} walls and {len(model.openings)} openings from the Plan Model",
                       recognition_image=rec, skeleton_image=sk, model=model,
                       walls=[{"p0": wl.p0, "p1": wl.p1, "thickness": wl.thickness} for wl in model.walls],
                       openings=[{"p0": o.p0, "p1": o.p1, "kind": o.kind, "evidence": "; ".join(o.provenance)}
                                 for o in model.openings],
                       symbols={}, suppressed={}, wall_thickness=t_med, seconds=time.perf_counter() - t0)


def clean(image: np.ndarray, pdf_path=None, page: int | None = None, text_boxes=(), mode: str = "on") -> CleanResult:
    """Clean one page. `pdf_path`/`page`: the PDF the image was rendered from (vector sources)."""
    t0 = time.perf_counter()
    reasons = []
    if pdf_path is not None and page:
        from engine.semantic.layer import build_layer
        from ingest.pdf_vectors import page_paths

        layer = build_layer(page_paths(pdf_path, page, image.shape), image.shape, text_boxes, image=image)
        if layer.applicable:
            return _from_layer(layer, "vector-wall-pen" if layer.wall_pens else "vector-poche", t0)
        reasons.append(f"vector: {layer.reason}")
        untyped = layer if layer.elements else None      # the page's vectors, walls not typed
    if mode == "raster":
        res = _from_raster(image, t0)
        if res.applicable:
            return res
        reasons.append(res.reason)
    res = CleanResult(source="none", reason="; ".join(reasons) or "no source", seconds=time.perf_counter() - t0)
    if pdf_path is not None and page:
        res.layer = untyped                              # still read for furniture (engine.arch.objects)
    return res


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------

def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    s = max(0.6, img.shape[1] / 1400)
    cv2.rectangle(out, (0, 0), (int(len(text) * 22 * s), int(46 * s)), (255, 255, 255), -1)
    cv2.putText(out, text, (int(10 * s), int(34 * s)), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), max(1, int(2 * s)), cv2.LINE_AA)
    return out


def comparison_sheet(original: np.ndarray, result: CleanResult, recognized: np.ndarray | None, width: int = 2400) -> np.ndarray:
    """2 x 2: Original | Cleaned skeleton / Structural elements | Recognised rooms."""
    tiles = [_label(original, "1  Original"),
             _label(result.skeleton_image if result.skeleton_image is not None else original, "2  Cleaned architectural skeleton"),
             _label(result.elements_image(original), "3  Structural elements (kept coloured, suppressed grey)"),
             _label(recognized if recognized is not None else original, "4  Recognised rooms")]
    h, w = original.shape[:2]
    scale = (width / 2) / w
    small = [cv2.resize(t, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) for t in tiles]
    pad = lambda t: cv2.copyMakeBorder(t, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(160, 160, 160))
    small = [pad(t) for t in small]
    return np.vstack([np.hstack(small[:2]), np.hstack(small[2:])])


def save_artifacts(directory, original: np.ndarray, result: CleanResult, recognized: np.ndarray | None = None) -> dict:
    """Write original / cleaned / elements / recognition images and the representation."""
    import json
    from pathlib import Path

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    images = {"cleaned.png": result.skeleton_image, "recognition-input.png": result.recognition_image,
              "elements.png": result.elements_image(original) if result.applicable or result.layer is not None else None,
              "comparison.png": comparison_sheet(original, result, recognized) if result.applicable else None}
    for name, im in images.items():
        if im is not None:
            ok, enc = cv2.imencode(".png", im)
            if ok:
                (d / name).write_bytes(enc.tobytes())
                files[name] = str(d / name)
    (d / "cleaning.json").write_text(json.dumps(result.representation(), indent=1))
    files["cleaning.json"] = str(d / "cleaning.json")
    return files
