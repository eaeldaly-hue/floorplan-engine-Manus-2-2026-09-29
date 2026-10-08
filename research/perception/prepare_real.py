"""Write the canonical real test cases for the perception experiment (engine venv).

Each case: grey PNG, the engine's measured wall thickness and the working scale that brings walls
to ~12 px (the engine's normalised reading). Also writes x0.5 / x2 scale variants of that scale
to measure scale sensitivity.

    .venv/bin/python research/perception/prepare_real.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "output/perception/real"
PDF_PAGES = [("22.pdf", 6), ("3.pdf", 4), ("3.pdf", 5)]
TARGET_T = 12.0


def cases():
    from benchmark.holdout_inputs import load
    from benchmark.real_plans import DEFAULT_DIR, FIXTURE, load_plan

    doc = json.loads(FIXTURE.read_text())["plans"]
    for name, spec in doc.items():
        if (DEFAULT_DIR / name).exists() and spec.get("rooms"):
            yield name, load_plan(name, spec, DEFAULT_DIR)
    for pdf, page in PDF_PAGES:
        img, _, _ = load(DEFAULT_DIR / pdf, page)
        yield f"{pdf}:{page}", img


def main() -> int:
    from engine.structure import analyze_structure

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, img in cases():
        stem = name.replace(":", "_p").replace(".", "_")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        path = OUT / f"{stem}.png"
        cv2.imwrite(str(path), gray)
        t = float(analyze_structure(img).wall_thickness or 0) or 8.0
        base = float(np.clip(TARGET_T / t, 0.2, 3.5))
        base = min(base, (16e6 / (gray.shape[0] * gray.shape[1])) ** 0.5)
        for tag, f in (("", 1.0), ("_s05", 0.5), ("_s2", 2.0)):
            sc = base * f
            if gray.shape[0] * gray.shape[1] * sc * sc > 30e6:
                continue
            manifest.append({"name": name + tag, "case": name, "variant": tag or "norm", "image": str(path),
                             "scale": round(sc, 3), "wall_t": round(t, 1), "out": str(OUT / f"{stem}{tag}.npz")})
        print(name, gray.shape, "t", t, "scale", round(base, 2), flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
