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
    for s in b.get("spaces", []):
        c = _pt(s["center"])
        sid = s["id"].replace("space_", "S")
        label = f"{sid} {' / '.join(s['names'])}" if s["names"] else sid
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
