"""Draw hold-out ground truth on the plan for review (no engine output is used or shown).

    python scripts/diagnostics/holdout_review.py LABELS.json [...] [--out output/holdout_review]

Spaces: numbered dots (one colour per physical space). Rooms / zones: name next to their point.
Openings: door = green, window = blue, opening = orange, with ids. Non-structural elements:
grey boxes. Labelled region (roi): dashed purple frame. The plan is loaded exactly as the
engine sees it (benchmark.holdout_inputs).
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

from benchmark.holdout_inputs import load  # noqa: E402
from benchmark.real_plans import HOLDOUT_DIR  # noqa: E402

COLOURS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
           (128, 128, 0), (0, 128, 128), (170, 110, 40), (128, 0, 0), (0, 0, 128), (128, 128, 128), (210, 245, 60)]


def draw(spec: dict, image: np.ndarray) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    s = max(1.0, max(h, w) / 2500)                     # text / mark size follows the frame
    th = max(1, int(round(2 * s)))
    font = 0.6 * s
    for x0, y0, x1, y1 in spec.get("roi", []):
        for k in range(int(x0), int(x1), int(24 * s)):
            cv2.line(out, (k, int(y0)), (min(int(x1), k + int(12 * s)), int(y0)), (200, 0, 160), th)
            cv2.line(out, (k, int(y1)), (min(int(x1), k + int(12 * s)), int(y1)), (200, 0, 160), th)
        for k in range(int(y0), int(y1), int(24 * s)):
            cv2.line(out, (int(x0), k), (int(x0), min(int(y1), k + int(12 * s))), (200, 0, 160), th)
            cv2.line(out, (int(x1), k), (int(x1), min(int(y1), k + int(12 * s))), (200, 0, 160), th)
    for el in spec.get("nonstructural", []):
        if el.get("bbox"):
            x0, y0, x1, y1 = map(int, el["bbox"])
            cv2.rectangle(out, (x0, y0), (x1, y1), (150, 150, 150), th)
            cv2.putText(out, el.get("what", el["kind"])[:18], (x0 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (110, 110, 110), max(1, th - 1), cv2.LINE_AA)
    col = {sp["id"]: COLOURS[i % len(COLOURS)] for i, sp in enumerate(spec.get("spaces", []))}
    for o in spec.get("openings", []):
        c = {"door": (0, 160, 0), "window": (220, 90, 0), "opening": (0, 140, 255)}[o["kind"]]
        p0, p1 = tuple(map(int, o["p0"])), tuple(map(int, o["p1"]))
        cv2.line(out, p0, p1, c, max(th + 2, int(o.get("wall_thickness_px", 8) * 0.5)))
        cv2.putText(out, o["id"], (p1[0] + 4, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, font, c, th, cv2.LINE_AA)
    for sp in spec.get("spaces", []):
        for k, (x, y) in enumerate(sp["points"]):
            cv2.circle(out, (int(x), int(y)), int(9 * s), col[sp["id"]], -1)
            cv2.circle(out, (int(x), int(y)), int(9 * s), (0, 0, 0), max(1, th - 1))
            cv2.putText(out, sp["id"], (int(x) + int(11 * s), int(y) + int(5 * s)), cv2.FONT_HERSHEY_SIMPLEX, font, col[sp["id"]], th, cv2.LINE_AA)
    for r in spec.get("rooms", []):
        x, y = map(int, r["point"])
        c = col.get(r["space"], (0, 0, 0))
        cv2.drawMarker(out, (x, y), c, cv2.MARKER_TILTED_CROSS, int(16 * s), th)
        label = f'{r["name"]}' + (f' "{r["printed"]}"' if r.get("printed") and r["printed"] != r["name"] else "") + f' [{r["space"]}]'
        cv2.putText(out, label, (x + int(10 * s), y - int(10 * s)), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), th + 3, cv2.LINE_AA)
        cv2.putText(out, label, (x + int(10 * s), y - int(10 * s)), cv2.FONT_HERSHEY_SIMPLEX, font, c, th, cv2.LINE_AA)
    for p in spec.get("protected", []):
        cv2.drawMarker(out, tuple(map(int, p["point"])), (0, 0, 255), cv2.MARKER_DIAMOND, int(14 * s), th)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--plans", default=str(HOLDOUT_DIR))
    ap.add_argument("--out", default=str(ROOT / "output" / "holdout_review"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in args.labels:
        spec = json.loads(Path(path).read_text())
        image, _, info = load(Path(args.plans) / spec["file"], spec.get("page"), spec.get("analysis_scale"))
        name = Path(spec["file"]).stem
        cv2.imwrite(str(out / f"{name}_review.png"), draw(spec, image))
        print(f"{spec['file']}: {len(spec.get('spaces', []))} spaces, {len(spec.get('rooms', []))} rooms, "
              f"{len(spec.get('openings', []))} openings, {len(spec.get('nonstructural', []))} non-structural -> {out / (name + '_review.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
