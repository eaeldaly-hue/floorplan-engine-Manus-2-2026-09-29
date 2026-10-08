"""The structured plan drawn over the plan: what the engine believes the building is.

  building footprint        green outline
  wall runs                 exterior red, interior blue (thickness as measured)
  doors                     magenta span; hinged doors show the leaf and swing arc toward the
                            space they open into; double doors both leaves
  windows                   orange span (exterior), light orange onto an outdoor space
  passages                  grey span
  spaces                    id and name(s) at the centre; unnamed spaces as an id only
  graph                     thin purple lines between spaces joined by an opening
  header                    the chosen reading (engine.reconstruction) and summary counts
"""

from __future__ import annotations

import math

import cv2
import numpy as np

COL = {"footprint": (0, 160, 0), "ext": (40, 40, 220), "int": (210, 120, 20), "door": (200, 0, 200),
       "window": (0, 140, 255), "window_out": (120, 200, 255), "passage": (140, 140, 140), "graph": (150, 60, 150),
       "text": (20, 90, 20), "unnamed": (30, 150, 200)}


def _pt(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _door(vis, o, lw):
    d = o.get("door") or {}
    p0, p1 = np.array(o["p0"], float), np.array(o["p1"], float)
    cv2.line(vis, _pt(p0), _pt(p1), COL["door"], 2 * lw)
    hinges = [np.array(h, float) for h in d.get("hinges") or []]
    if not hinges or not str(d.get("operation", "")).endswith("hinged"):
        return
    width = float(np.linalg.norm(p1 - p0))
    sp = o.get("_swing_point")
    for h in hinges:
        other = p1 if np.allclose(h, p0) else p0
        r = width / len(hinges)
        u = (other - h) / max(1e-6, np.linalg.norm(other - h))
        n = np.array([-u[1], u[0]])
        if sp is not None and float(np.dot(np.asarray(sp, float) - h, n)) < 0:
            n = -n
        ang = math.radians(float(d.get("leaf_angle") or 90))
        tip = h + r * (math.cos(ang) * u + math.sin(ang) * n)
        cv2.line(vis, _pt(h), _pt(tip), COL["door"], lw)
        arc = [h + r * (math.cos(a) * u + math.sin(a) * n) for a in np.linspace(0, ang, 16)]
        cv2.polylines(vis, [np.array([_pt(q) for q in arc], np.int32).reshape(-1, 1, 2)], False, COL["door"], max(1, lw // 2))


ZONE_COL = {"kitchen": (0, 140, 255), "dining": (0, 170, 60), "living": (210, 90, 0), "bedroom": (170, 60, 170),
            "bath": (200, 170, 0), "laundry": (120, 120, 120)}


def _dashed(vis, pts, col, lw, dash=14):
    pts = [np.asarray(p, float) for p in pts]
    for a, b in zip(pts, pts[1:] + pts[:1]):
        L = float(np.linalg.norm(b - a))
        for s in np.arange(0, L, 2 * dash):
            p0 = a + (b - a) * (s / max(L, 1e-6))
            p1 = a + (b - a) * (min(L, s + dash) / max(L, 1e-6))
            cv2.line(vis, _pt(p0), _pt(p1), col, lw, cv2.LINE_AA)


def _zones(vis, b, lw, fs):
    """Functional zones: tinted area, dashed implied boundary, function and confidence; the typed
    objects that support them as thin boxes."""
    overlay = vis.copy()
    for sp in b.get("spaces", []):
        for z in sp.get("zones", []):
            col = ZONE_COL.get(z["function"], (0, 0, 0))
            cv2.fillPoly(overlay, [np.array(z["polygon"], np.int32).reshape(-1, 1, 2)], col)
    cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, dst=vis)
    for o in b.get("objects", []):
        f = max(o["functions"], key=o["functions"].get) if o["functions"] else ""
        x0, y0, x1, y1 = o["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), ZONE_COL.get(f, (90, 90, 90)), max(1, lw // 2))
        cv2.putText(vis, o["kind"], (x0 + 2, y0 + int(12 * fs) + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * fs,
                    ZONE_COL.get(f, (90, 90, 90)), 1, cv2.LINE_AA)
    for sp in b.get("spaces", []):
        for z in sp.get("zones", []):
            col = ZONE_COL.get(z["function"], (0, 0, 0))
            _dashed(vis, z["polygon"], col, 2 * lw)
            label = f"{z['function'].upper()} zone {z['confidence']:.2f}"
            c = _pt(z["center"])
            cv2.putText(vis, label, c, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * fs, (255, 255, 255), 4 * lw, cv2.LINE_AA)
            cv2.putText(vis, label, c, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * fs, col, 2 * lw, cv2.LINE_AA)


def render(image: np.ndarray, b: dict, reconstruction: dict | None = None) -> np.ndarray:
    vis = cv2.addWeighted(image, 0.4, np.full_like(image, 255), 0.6, 0)
    lw = max(1, int(round(max(image.shape[:2]) / 1100)))
    for bd in b.get("buildings", []):
        cv2.polylines(vis, [np.array(bd["polygon"], np.int32).reshape(-1, 1, 2)], True, COL["footprint"], 2 * lw)
    for w in b.get("walls", []):
        t = max(lw, int(round(w["thickness_px"] * 0.6)))
        cv2.line(vis, _pt(w["p0"]), _pt(w["p1"]), COL["ext"] if w["exterior"] else COL["int"], t)
    centre = {s["id"]: s["center"] for s in b.get("spaces", [])}
    for o in b.get("openings", []):
        if o["type"] == "door":
            d = o.get("door") or {}
            into = d.get("swing_into")
            o = {**o, "_swing_point": centre.get(into)}
            _door(vis, o, lw)
        elif o["type"] == "window":
            w = o.get("window") or {}
            col = COL["window_out"] if w.get("outdoor_space") else COL["window"]
            cv2.line(vis, _pt(o["p0"]), _pt(o["p1"]), col, 3 * lw)
        else:
            cv2.line(vis, _pt(o["p0"]), _pt(o["p1"]), COL["passage"], 2 * lw)
    for e in b.get("graph", {}).get("edges", []):
        if e["kind"] != "wall" and e["a"] in centre and e["b"] in centre:
            cv2.line(vis, _pt(centre[e["a"]]), _pt(centre[e["b"]]), COL["graph"], 1, cv2.LINE_AA)
    fs = max(0.35, max(image.shape[:2]) / 2600)
    _zones(vis, b, lw, fs)
    for s in b.get("spaces", []):
        c = _pt(s["center"])
        sid = s["id"].replace("space_", "S")
        inferred = (s.get("function") or {}).get("function")
        label = (f"{sid} {' / '.join(s['names'])}" if s["names"] else
                 f"{sid} ({inferred}?)" if inferred else sid)
        col = COL["text"] if s["names"] else COL["unnamed"]
        cv2.putText(vis, label, c, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 3 * lw, cv2.LINE_AA)
        cv2.putText(vis, label, c, cv2.FONT_HERSHEY_SIMPLEX, fs, col, lw, cv2.LINE_AA)
    sm = b.get("summary", {})
    head = (f"reading: {(reconstruction or {}).get('chosen', 'default')}   spaces {sm.get('spaces')}   "
            f"doors {sm.get('doors')}   windows {sm.get('windows')}   wall runs {sm.get('wall_runs')}")
    cv2.putText(vis, head, (10, int(30 * fs) + 10), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 3 * lw, cv2.LINE_AA)
    cv2.putText(vis, head, (10, int(30 * fs) + 10), cv2.FONT_HERSHEY_SIMPLEX, fs, (40, 40, 40), lw, cv2.LINE_AA)
    return vis


def render_png(image: np.ndarray, b: dict, reconstruction: dict | None = None) -> bytes:
    ok, enc = cv2.imencode(".png", render(image, b, reconstruction))
    return enc.tobytes() if ok else b""
