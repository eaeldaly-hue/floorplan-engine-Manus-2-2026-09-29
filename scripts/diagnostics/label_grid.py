"""Labelling aid for the hold-out set: the raw plan with a coordinate grid, nothing else.

    python scripts/diagnostics/label_grid.py PLAN [--step 50] [--tile 600] [--out DIR]

Writes the whole plan with a grid (grid.png) and enlarged overlapping tiles (tile_X_Y.png) whose
grid labels are native image pixels. No engine output is drawn, so ground truth labelled from
these images is independent of the engine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))



def gridded(img: np.ndarray, step: int, x0: int = 0, y0: int = 0, scale: float = 1.0) -> np.ndarray:
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1.0 else img.copy()
    h, w = out.shape[:2]
    first_x = (-x0) % step
    first_y = (-y0) % step
    for gx in range(first_x, int(w / scale) + 1, step):
        X = int(gx * scale)
        major = (x0 + gx) % (2 * step) == 0
        cv2.line(out, (X, 0), (X, h), (255, 120, 0) if major else (255, 200, 150), 1)
        cv2.putText(out, str(x0 + gx), (X + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 60, 0), 1, cv2.LINE_AA)
    for gy in range(first_y, int(h / scale) + 1, step):
        Y = int(gy * scale)
        major = (y0 + gy) % (2 * step) == 0
        cv2.line(out, (0, Y), (w, Y), (255, 120, 0) if major else (255, 200, 150), 1)
        cv2.putText(out, str(y0 + gy), (2, Y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 60, 0), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan")
    ap.add_argument("--step", type=int, default=0, help="grid step in native pixels (default: ~1/12 of the longest side)")
    ap.add_argument("--tile", type=int, default=0, help="tile size in native pixels (default: ~1/3 of the longest side)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--page", type=int, default=None, help="PDF page (hold-out plans are rendered as the engine sees them)")
    args = ap.parse_args()
    path = Path(args.plan)
    # The engine's own view of the plan (benchmark.holdout_inputs): a PDF page at its analysis
    # DPI, an image above 24 MP reduced. Labels are made in this frame.
    from benchmark.holdout_inputs import load
    img, _, info = load(path, args.page)
    print("analysis frame:", info)
    h, w = img.shape[:2]
    step = args.step or max(10, int(round(max(h, w) / 12 / 10)) * 10)
    tile = args.tile or max(200, int(max(h, w) / 3))
    out = Path(args.out) if args.out else ROOT / "output" / "holdout_label" / path.stem
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "grid.png"), gridded(img, step))
    scale = max(1.0, 900 / tile)
    fine = max(5, step // 2)
    for y in range(0, h, int(tile * 0.8)):
        for x in range(0, w, int(tile * 0.8)):
            crop = img[y:y + tile, x:x + tile]
            cv2.imwrite(str(out / f"tile_{x}_{y}.png"), gridded(crop, fine, x, y, scale))
    print(f"{path.name}: {w}x{h}, grid {step}px, tiles {tile}px -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
