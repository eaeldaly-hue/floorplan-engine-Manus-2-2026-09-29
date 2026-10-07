"""Benchmark N.1 overlays: ground truth against the current engine for each real plan.

    python scripts/diagnostics/n1_overlays.py [--only 5.jpeg,16.png] [--out output/realplans/n1]

Each image shows the wall mask (gray tint), labelled room points (dot, coloured by open-plan
group, ring when the point's group shares a space with another group), GT openings (green door,
blue window, orange ambiguous), nonstructural points (red X if covered by wall, pink if not),
protected points (blue square, cyan if absent from the wall mask), and the engine's false
opening candidates coloured by probable cause (magenta furniture jamb, yellow pattern, brown
small isolated jamb, red other), and gaps rejected by N.2 jamb verification (gray line with an
X at each end; the reason is in the candidate's `jambs`). Measurement only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.real_plan_probes import structural_probe  # noqa: E402
from benchmark.real_plans import DEFAULT_DIR, FIXTURE, load_plan  # noqa: E402
from engine.opening_detection import classify_openings  # noqa: E402
from engine.structure import analyze_structure  # noqa: E402

CAUSE = {"furniture_jamb": (255, 0, 255), "pattern": (0, 210, 255), "small_isolated_jamb": (30, 90, 160), "other": (0, 0, 255)}
KIND = {"door": (0, 160, 0), "window": (220, 120, 0), "opening": (0, 140, 255)}


def render(image, s, spec, probe) -> np.ndarray:
    out = image.copy()
    wall = s.wall_mask > 0
    out[wall] = (0.45 * out[wall] + 0.55 * np.array([90, 90, 90])).astype(np.uint8)
    sc = max(1, round(max(image.shape[:2]) / 900))
    for g in spec.get("openings", []):
        cv2.line(out, tuple(map(int, g["p0"])), tuple(map(int, g["p1"])), KIND[g["kind"]], 3 * sc)
    for c in getattr(s, "rejected_candidates", []):          # N.2: gaps rejected by jamb verification
        a_, b_ = tuple(int(v) for v in c.start), tuple(int(v) for v in c.end)
        cv2.line(out, a_, b_, (120, 120, 120), 2 * sc)
        for q in (a_, b_):
            cv2.drawMarker(out, q, (120, 120, 120), cv2.MARKER_TILTED_CROSS, 8 * sc, 1 * sc)
    for f in probe.get("openings_detail", {}).get("false", []):
        cv2.circle(out, tuple(f["center"]), 7 * sc, CAUSE[f["cause"]], 2 * sc)
    merged = set(probe["under_segmentation"]["merged_groups"])
    groups = sorted({r.get("group") or f"_{i}" for i, r in enumerate(spec.get("rooms", []))})
    rng = np.random.default_rng(3)
    colours = {g: tuple(int(c) for c in rng.integers(40, 220, 3)) for g in groups}
    for i, r in enumerate(spec.get("rooms", [])):
        g = r.get("group") or f"_{i}"
        p = tuple(map(int, r["point"]))
        cv2.circle(out, p, 5 * sc, colours[g], -1)
        if g in merged:
            cv2.circle(out, p, 10 * sc, (0, 0, 255), 2 * sc)
    for p, f in zip(spec.get("nonstructural", []), probe["false_walls"]["points"]):
        x, y = map(int, p["point"])
        c = (0, 0, 255) if f["wall"] else (180, 150, 255)
        cv2.drawMarker(out, (x, y), c, cv2.MARKER_TILTED_CROSS, 12 * sc, 2 * sc)
    for p, f in zip(spec.get("protected", []), probe["protected"]["pieces"]):
        x, y = map(int, p["point"])
        c = (255, 120, 0) if f["present"] else (255, 255, 0)
        cv2.drawMarker(out, (x, y), c, cv2.MARKER_SQUARE, 12 * sc, 2 * sc)
        if f.get("at_risk"):
            cv2.drawMarker(out, (x, y), (0, 0, 255), cv2.MARKER_SQUARE, 18 * sc, 1 * sc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", default="2.jpg,5.jpeg,8.png,11.png,14.png,16.png,19.jpg,21.png")
    ap.add_argument("--plans", default=str(DEFAULT_DIR))
    ap.add_argument("--out", default=str(ROOT / "output" / "realplans" / "n1"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = json.loads(FIXTURE.read_text())["plans"]
    for name in args.only.split(","):
        spec = plans[name]
        image = load_plan(name, spec, Path(args.plans))
        s = analyze_structure(image)
        openings = classify_openings(image, s)
        probe = structural_probe(s, spec, openings)
        cv2.imwrite(str(out_dir / f"{Path(name).stem}.png"), render(image, s, spec, probe))
        print(name, "->", out_dir / f"{Path(name).stem}.png")
        for c in getattr(s, "rejected_candidates", []):
            print("   rejected", tuple(int(v) for v in c.center), int(c.width), "|", " / ".join(c.jambs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
