"""Synthetic training data for learned floor-plan perception (research; not used by the engine).

Every sample is a plan from benchmark.generator with randomised drawing conventions, rendered with
exact per-pixel ground truth:

    0 exterior / paper      1 room interior     2 wall      3 door (span in the wall band)
    4 window (span)         5 passage (bare opening span)   6 other ink (furniture, fixtures,
    text, dimensions, hatch, stairs, markers)              255 ignore (door-symbol ink: arcs /
    leaves belong to the door, not to 'other')

Commercially clean: only our own generator. Plans are drawn at ~0.35-0.75 px/cm, i.e. exterior
walls of ~6-25 px, around the engine's normalised reading (walls ~12 px).

    python research/perception/synth_data.py --count 2000 --out output/perception/synth
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.generator import DOOR_STYLES, PlanSpec, Style, generate  # noqa: E402

CLASSES = ["exterior", "interior", "wall", "door", "window", "passage", "other"]
WINDOW_STYLES = ("triple", "double", "sample", "weak")


def _subset(rng, items):
    k = rng.randint(1, len(items))
    return tuple(rng.sample(list(items), k))


def random_spec(seed: int) -> PlanSpec:
    rng = random.Random(seed)
    s = rng.uniform(0.35, 0.75)
    lw = rng.choice((1, 1, 2, 2, 3))
    render = rng.choices(("filled", "hollow", "hatched", "thin"), weights=(0.45, 0.2, 0.15, 0.2))[0]
    int_cm = rng.uniform(8, 18)
    if render != "thin":
        int_cm = max(int_cm, (3.2 * lw + 2) / s)
    style = Style(
        px_per_cm=s, line_px=lw, ext_wall_cm=rng.uniform(15, 35), int_wall_cm=int_cm,
        wall_gray=rng.choice((0, 0, 0, 40, 80)), symbol_gray=rng.choice((0, 0, 60, 110)),
        door_styles=_subset(rng, DOOR_STYLES), window_styles=_subset(rng, WINDOW_STYLES),
        antialias=rng.random() < 0.8, text=rng.random() < 0.8, furniture=rng.random() < 0.7,
        dimension_lines=rng.random() < 0.5,
        speckle=rng.uniform(0, 0.004) if rng.random() < 0.3 else 0.0,
        blur=rng.uniform(0.3, 1.2) if rng.random() < 0.4 else 0.0,
        jpeg=rng.randint(40, 90) if rng.random() < 0.4 else 0,
        background=rng.randint(225, 255) if rng.random() < 0.3 else 255,
        rotate_deg=rng.uniform(-3, 3) if rng.random() < 0.2 else 0.0,
        dark_furniture=rng.uniform(0.2, 0.7) if rng.random() < 0.35 else 0.0,
        furniture_tone=rng.choice((30, 60, 100, 140)), furniture_touch=rng.uniform(0.2, 0.8),
        markers=rng.randint(2, 8) if rng.random() < 0.2 else 0,
        hatch=rng.uniform(0.2, 0.5) if rng.random() < 0.3 else 0.0,
        stairs=rng.randint(1, 2) if rng.random() < 0.3 else 0,
        wall_render=render)
    w, h = rng.uniform(800, 2000), rng.uniform(650, 1500)
    return PlanSpec(seed=seed, footprint_cm=(w, h), min_room_cm=rng.uniform(240, 340),
                    corridor_prob=rng.uniform(0.1, 0.4), drop_corner_prob=rng.uniform(0.2, 0.5),
                    merge_room_prob=rng.uniform(0.1, 0.4), windows_per_room=(1, rng.choice((1, 2, 3))),
                    exterior_doors=rng.randint(1, 3), passage_prob=rng.uniform(0.05, 0.35),
                    adjacent_window_prob=rng.uniform(0.1, 0.4), corner_door_prob=rng.uniform(0.1, 0.4),
                    style=style)


def _band(p0, p1, half, extend=1.0):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    u = (p1 - p0) / max(1e-6, np.linalg.norm(p1 - p0))
    n = np.array([-u[1], u[0]])
    a, b = p0 - u * extend, p1 + u * extend
    return np.round([a + n * half, b + n * half, b - n * half, a - n * half]).astype(np.int32)


def render_sample(seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    spec = random_spec(seed)
    plan = generate(spec)
    gray = cv2.cvtColor(plan.image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)                                   # exterior
    mask[plan.room_labels > 0] = 1
    mask[plan.wall_mask > 0] = 2
    spans = np.zeros((h, w), np.uint8)
    door_reach = np.zeros((h, w), np.uint8)
    cls = {"door": 3, "window": 4, "opening": 5}
    for o in plan.openings:
        half = o.wall_thickness_px / 2 + 2
        poly = _band(o.p0, o.p1, half)
        cv2.fillPoly(mask, [poly], cls.get(o.kind, 5))
        cv2.fillPoly(spans, [poly], 255)
        if o.kind == "door":                                            # the swing / leaf region
            r = int(round(1.08 * o.width_px))
            for p in (o.p0, o.p1):
                cv2.circle(door_reach, (int(round(p[0])), int(round(p[1]))), r, 255, -1)
    bg = float(np.percentile(gray, 90))
    ink = gray < min(bg - 40, 200)
    near_arch = cv2.dilate(((plan.wall_mask > 0) | (spans > 0)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    other = ink & ~near_arch
    mask[other & (door_reach > 0)] = 255
    mask[other & (door_reach == 0)] = 6
    meta = {"seed": seed, "px_per_cm": spec.style.px_per_cm, "wall_render": spec.style.wall_render,
            "ext_wall_px": round(spec.style.ext_wall_cm * spec.style.px_per_cm, 1),
            "int_wall_px": round(spec.style.int_wall_cm * spec.style.px_per_cm, 1)}
    return gray, mask, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "output/perception/synth"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metas = []
    for i in range(args.start, args.start + args.count):
        seed = 900_000 + i
        try:
            gray, mask, meta = render_sample(seed)
        except Exception as exc:                                        # an occasional degenerate layout
            print("skip", seed, type(exc).__name__, flush=True)
            continue
        cv2.imwrite(str(out / f"{i:05d}.png"), gray)
        cv2.imwrite(str(out / f"{i:05d}_m.png"), mask)
        metas.append(meta)
        if i % 200 == 0:
            print(i, gray.shape, meta, flush=True)
    (out / f"meta_{args.start}.json").write_text(json.dumps(metas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
