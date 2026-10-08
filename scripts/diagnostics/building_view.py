"""Visual check of the structured plan (engine.arch.building) on canonical test cases.

    python -m scripts.diagnostics.building_view [--cases 12.png,22.pdf:6] [--cleaning] [--out output/building]

Draws, over the plan: building footprints (green outline), exterior walls (red) and interior walls
(blue), openings by role (exterior: orange, interior: magenta, inside one space: grey), space names
at their centres, unnamed spaces (yellow dot), spaces outside every building (black cross) and the
space graph (thin lines between connected space centres; dashed = wall neighbours are not drawn).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASES = Path(os.environ.get("FLOORPLAN_REAL_PLANS", ROOT.parent / "Test Cases"))


def render(image: np.ndarray, b: dict) -> np.ndarray:
    vis = cv2.addWeighted(image, 0.45, np.full_like(image, 255), 0.55, 0)
    lw = max(2, int(round(max(image.shape[:2]) / 900)))
    for bd in b["buildings"]:
        cv2.polylines(vis, [np.array(bd["polygon"], np.int32).reshape(-1, 1, 2)], True, (0, 170, 0), 2 * lw)
    for w in b["walls"]:
        cv2.line(vis, tuple(w["p0"]), tuple(w["p1"]), (0, 0, 220) if w["exterior"] else (220, 120, 0), lw)
    colours = {"exterior": (0, 140, 255), "interior": (200, 0, 200), "inside one space": (150, 150, 150)}
    for o in b["openings"]:
        cv2.line(vis, tuple(o["p0"]), tuple(o["p1"]), colours.get(o["role"], (0, 0, 0)), 2 * lw)
    centre = {s["id"]: tuple(s["center"]) for s in b["spaces"]}
    for e in b["graph"]["edges"]:
        if e["kind"] != "wall" and e["a"] in centre and e["b"] in centre:
            cv2.line(vis, centre[e["a"]], centre[e["b"]], (120, 60, 160), 1)
    fs = max(0.4, max(image.shape[:2]) / 2200)
    for s in b["spaces"]:
        c = tuple(s["center"])
        if s["building"] is None and b["envelope"]["available"]:
            cv2.drawMarker(vis, c, (0, 0, 0), cv2.MARKER_TILTED_CROSS, 6 * lw, lw)
        elif s["names"]:
            cv2.putText(vis, " / ".join(s["names"]), c, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 90, 0), max(1, lw // 2), cv2.LINE_AA)
        else:
            cv2.circle(vis, c, 3 * lw, (0, 200, 230), -1)
    return vis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="12.png,6.png,10.png,19.jpg,22.pdf:6,3.pdf:4")
    ap.add_argument("--cleaning", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "output" / "building"))
    args = ap.parse_args()
    from app import _decode_upload
    from engine.analyzer import FloorPlanAnalyzer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for case in args.cases.split(","):
        name, _, page = case.partition(":")
        if page:
            from benchmark.holdout_inputs import load
            image, evidence, _ = load(CASES / name, int(page))
            cleaner = None
            if args.cleaning:
                from engine.cleaning.cleaner import clean
                cleaner = lambda: clean(image, CASES / name, int(page), evidence, mode="on")   # noqa: E731
            result = FloorPlanAnalyzer().analyze(image, name, text_evidence=evidence or None, cleaner=cleaner)
        else:
            image, _ = _decode_upload((CASES / name).read_bytes(), name)
            result = FloorPlanAnalyzer().analyze(image, name)
        b = result["building"]
        stem = case.replace(":", "_p").replace(".", "_")
        vis = render(image, b)
        scale = min(1.0, 2400 / max(vis.shape[:2]))
        cv2.imwrite(str(out / f"{stem}.png"), cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        (out / f"{stem}.json").write_text(json.dumps(b, indent=1))
        print(case, json.dumps(b["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
