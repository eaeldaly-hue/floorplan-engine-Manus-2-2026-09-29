"""Plan-adaptive wall tone (supporting evidence only).

Walls of one plan are drawn in one or a few tones: black, gray, a colour (e.g. teal), and
sometimes exterior and interior walls in different tones. The tones are learned from the
plan's own clearly structural wall pixels — the inside of long, thin, straight wall runs (solid
blocks thicker than a wall are excluded, so large furniture does not define a wall tone) — so
the model adapts to black-and-white, gray and coloured plans alike.

The tone never decides on its own whether something is a wall. It is consulted only for a
piece of the wall mask whose structural role is already in doubt (see structure._jamb_support):
a matching tone keeps the piece, a clearly different tone lets the structural doubt stand.

If the plan's walls do not have a consistent tone (too few run pixels, or a wide spread), the
model reports itself unreliable and gives no evidence either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MIN_TOLERANCE = 12.0        # Lab units (OpenCV 8-bit scale): smallest difference treated as "different"
MAX_CLUSTERS = 3
MIN_SHARE = 0.15            # a further wall tone must hold this share of the wall-run pixels


@dataclass
class WallTone:
    tones: list = field(default_factory=list)     # [(median Lab, tolerance)]
    reliable: bool = False
    pixels: int = 0

    def distance(self, lab_pixels: np.ndarray) -> float | None:
        """Distance of a set of Lab pixels (median) from the nearest wall tone, in units of
        that tone's tolerance (≥ 1 means outside every wall tone). None: no evidence."""
        px = np.asarray(lab_pixels).reshape(-1, 3)
        if not self.reliable or len(px) < 6:
            return None
        med = np.median(px.astype(np.float32), axis=0)
        return min(float(np.linalg.norm(med - lab)) / tol for lab, tol in self.tones)


def _wall_runs(wall_mask: np.ndarray, typical_t: float) -> np.ndarray:
    m = (wall_mask > 0).astype(np.uint8)
    L = max(9, int(round(4 * typical_t)))
    runs = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)) | cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    blob = max(5, int(round(2.5 * typical_t)))
    solid = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((blob, blob), np.uint8))   # thicker than any wall band
    runs = runs & (1 - cv2.dilate(solid, np.ones((3, 3), np.uint8)))
    return runs & _core(m, typical_t)


def _core(mask: np.ndarray, typical_t: float) -> np.ndarray:
    """Pixels near the middle of a stroke: anti-aliased edges carry the paper tone."""
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    return (dist >= max(1.0, 0.4 * typical_t)).astype(np.uint8)


def learn_wall_tone(image: np.ndarray, wall_mask: np.ndarray, typical_t: float) -> WallTone:
    lab = cv2.cvtColor(image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
    px = lab[_wall_runs(wall_mask, typical_t) > 0]
    if len(px) < 200:
        return WallTone([], False, int(len(px)))
    if len(px) > 200_000:
        px = px[np.random.default_rng(0).choice(len(px), 200_000, replace=False)]
    px = px.astype(np.float32)
    total = len(px)
    tones = []
    remaining = px
    while len(remaining) >= MIN_SHARE * total and len(tones) < MAX_CLUSTERS:
        med = np.median(remaining, axis=0)
        dist = np.linalg.norm(remaining - med, axis=1)
        near = dist[dist <= 2 * MIN_TOLERANCE]
        spread = float(np.median(near) * 1.4826) if near.size else 0.0
        tol = max(MIN_TOLERANCE, 3.0 * spread)
        member = dist <= tol
        if member.sum() < MIN_SHARE * total:
            break
        tones.append((med, tol))
        remaining = remaining[~member]
    # a reference is usable when the learned tones explain most wall-run pixels
    explained = 1.0 - len(remaining) / total
    reliable = bool(tones) and explained >= 0.75 and all(tol <= 60.0 for _, tol in tones)
    return WallTone(tones, reliable, total)


def piece_lab(image_lab: np.ndarray, region: np.ndarray, typical_t: float) -> np.ndarray:
    """Lab pixels of a wall piece, from its core when it has one."""
    m = region.astype(np.uint8)
    core = m & _core(m, min(typical_t, 0.5 * max(1.0, float(cv2.distanceTransform(m, cv2.DIST_L2, 3).max()) * 2)))
    if core.sum() >= 6:
        m = core
    return image_lab[m > 0]


__all__ = ["WallTone", "learn_wall_tone", "piece_lab"]
