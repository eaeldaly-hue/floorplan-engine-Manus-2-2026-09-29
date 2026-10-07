"""Hold-out novelty check: is each candidate plan genuinely new w.r.t. the development set?

    python scripts/diagnostics/holdout_novelty.py [--holdout DIR] [--dev DIR]

For every hold-out file: identical file hash with any development plan, and perceptual
similarity (white margins trimmed, downscaled, normalised) to every development plan and to the
other hold-out plans. Similarity ≥ 0.90 is reported as a likely duplicate (same plan at another
size, crop or format) and must be inspected before the plan is admitted. No engine code runs.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import _decode_upload  # noqa: E402
from benchmark.real_plans import DEFAULT_DIR, HOLDOUT_DIR  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp", ".avif", ".pdf"}


def signature(path: Path) -> np.ndarray | None:
    try:
        img, _ = _decode_upload(path.read_bytes(), path.name)
    except Exception:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ys, xs = np.nonzero(g < 235)                       # trim the paper margin
    if len(xs):
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    v = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n else v


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--holdout", default=str(HOLDOUT_DIR))
    ap.add_argument("--dev", default=str(DEFAULT_DIR))
    args = ap.parse_args()
    files = lambda d: sorted(p for p in Path(d).iterdir() if p.suffix.lower() in EXTS) if Path(d).is_dir() else []
    dev, hold = files(args.dev), files(args.holdout)
    dev_sig = {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), signature(p)) for p in dev}
    hold_sig = {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), signature(p)) for p in hold}
    print(f"development plans: {len(dev)}   hold-out candidates: {len(hold)}")
    for name, (digest, sig) in hold_sig.items():
        same_file = [d for d, (dd, _) in dev_sig.items() if dd == digest]
        if sig is None:
            print(f"{name:32s} UNREADABLE by the app (format) {'IDENTICAL FILE: ' + ', '.join(same_file) if same_file else ''}")
            continue
        sims = sorted(((similarity(sig, s), d) for d, (_, s) in dev_sig.items() if s is not None), reverse=True)
        peers = sorted(((similarity(sig, s), d) for d, (_, s) in hold_sig.items() if d != name and s is not None), reverse=True)
        best, best_name = sims[0] if sims else (0.0, "-")
        peer, peer_name = peers[0] if peers else (0.0, "-")
        verdict = ("DUPLICATE (identical file)" if same_file else
                   "LIKELY DUPLICATE of dev plan" if best >= 0.90 else
                   "LIKELY DUPLICATE within hold-out" if peer >= 0.90 else "new")
        print(f"{name:32s} {verdict:32s} closest dev {best_name} ({best:.2f}), closest hold-out {peer_name} ({peer:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
