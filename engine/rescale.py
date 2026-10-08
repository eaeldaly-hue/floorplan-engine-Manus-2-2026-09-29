"""Bring an analysis made on a rescaled working image back to the original image's pixel frame.

Keys are interpreted by meaning: pixel lengths and coordinates are divided by the scale, pixel
areas by its square; real-unit values (printed dimensions, real areas, counts, scores, ratios)
are left alone.
"""

from __future__ import annotations

import cv2
import numpy as np

LINEAR = {"x", "y", "width", "height", "start", "end", "center", "p0", "p1", "polygon", "width_pixels", "width_px",
          "thickness_px", "wall_thickness_px", "pixels_per_unit"}
AREA = {"area_pixels", "area_px"}
KEEP = {"dimensions", "summary", "image", "graph", "envelope", "plan_model", "reconstruction", "scale", "cleaning"}


def _num(v, k: float):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    out = v / k
    return int(round(out)) if isinstance(v, int) else round(out, 3)


def _all(v, k: float):
    if isinstance(v, (list, tuple)):
        return [_all(x, k) for x in v]
    if isinstance(v, dict):
        return {key: _all(x, k) for key, x in v.items()}
    return _num(v, k)


def _walk(v, f: float):
    if isinstance(v, list):
        return [_walk(x, f) for x in v]
    if not isinstance(v, dict):
        return v
    out = {}
    for key, x in v.items():
        if key in KEEP:
            out[key] = x
        elif key in AREA:
            out[key] = _num(x, f * f)
        elif key == "area" and isinstance(x, dict) and str(x.get("unit", "")).startswith("px"):
            out[key] = {**x, "value": _num(x.get("value"), f * f)}
        elif key in LINEAR:
            out[key] = _all(x, f) if isinstance(x, (list, tuple, dict)) else _num(x, f)
        else:
            out[key] = _walk(x, f)
    return out


def _png(data: bytes, size: tuple[int, int]) -> bytes:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return data
    ok, enc = cv2.imencode(".png", cv2.resize(img, size, interpolation=cv2.INTER_AREA))
    return enc.tobytes() if ok else data


def to_original(result: dict, f: float, width: int, height: int) -> dict:
    """`result` was computed on the image scaled by `f`; return it in the original frame."""
    if abs(f - 1.0) < 1e-6:
        return result
    out = _walk(result, f)
    out["image"] = {"width": width, "height": height}
    for key in ("overlay_png", "openings_overlay_png"):
        if isinstance(result.get(key), (bytes, bytearray)):
            out[key] = _png(result[key], (width, height))
    for key in ("_clean_result",):
        if key in result:
            out[key] = result[key]
    return out
