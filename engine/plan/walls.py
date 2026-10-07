"""Wall reconstruction: strokes -> walls (centerline + thickness) -> a connected wall network.

A wall can be drawn as a wide stroke (filled band), as two parallel thin strokes (hollow or
hatched band) or as one line (thin wall). All three become the same structural element. What
decides is not one stroke's appearance but structure: walls share a few thicknesses across the
plan, run in few directions, meet other walls at junctions and form one network. Regular lattices
(floor tiles, hatching, stair treads) and isolated shapes (furniture) are negative evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import Stroke, Wall

ANGLE_TOL = 2.0          # degrees: parallel


@dataclass
class _S:                # stroke in a local frame
    i: int
    a: float             # interval along the direction
    b: float
    rho: float           # offset along the normal
    w: float
    L: float


def _frame(theta_deg: float):
    t = math.radians(theta_deg)
    return math.cos(t), math.sin(t), -math.sin(t), math.cos(t)


def line_weight(strokes: list[Stroke]) -> float:
    """The finest common pen width (length-weighted lower mode of stroke widths)."""
    ws = np.array([s.width for s in strokes if s.length >= 15])
    ls = np.array([s.length for s in strokes if s.length >= 15])
    if not len(ws):
        return 2.0
    order = np.argsort(ws)
    cum = np.cumsum(ls[order]) / ls.sum()
    return float(ws[order][np.searchsorted(cum, 0.2)])


def _buckets(strokes: list[Stroke], idx: list[int]) -> dict[int, list[int]]:
    """Group stroke indices by angle (2-degree buckets, wrapping at 180)."""
    out: dict[int, list[int]] = {}
    for i in idx:
        out.setdefault(int(round(strokes[i].angle / ANGLE_TOL)) % int(180 / ANGLE_TOL), []).append(i)
    return out


def _local(strokes, ids, theta) -> list[_S]:
    tx, ty, nx, ny = _frame(theta)
    out = []
    for i in ids:
        s = strokes[i]
        a, b = s.p0[0] * tx + s.p0[1] * ty, s.p1[0] * tx + s.p1[1] * ty
        rho = ((s.p0[0] + s.p1[0]) * nx + (s.p0[1] + s.p1[1]) * ny) / 2
        out.append(_S(i, min(a, b), max(a, b), rho, s.width, s.length))
    return out


def lattice_members(strokes: list[Stroke], thin: list[int], max_spacing: float) -> set[int]:
    """Strokes in a run of >= 4 parallel, overlapping strokes at near-equal spacing: tiles,
    hatching, stair treads, mullion grids. Not walls."""
    found: set[int] = set()
    for key, ids in _buckets(strokes, thin).items():
        if len(ids) < 4:
            continue
        loc = sorted(_local(strokes, ids, key * ANGLE_TOL), key=lambda s: s.rho)
        for k, s in enumerate(loc):
            chain = [s]
            for t in loc[k + 1:]:
                gap = t.rho - chain[-1].rho
                if gap > max_spacing:
                    break
                ov = min(t.b, chain[-1].b) - max(t.a, chain[-1].a)
                if ov <= 0.6 * max(t.L, chain[-1].L) or gap < 1.0:     # repeated strokes of similar extent
                    continue
                if len(chain) >= 2:
                    ref = chain[1].rho - chain[0].rho
                    if abs(gap - ref) > 0.2 * ref + 1.0:
                        continue
                chain.append(t)
            if len(chain) >= 4:
                found.update(c.i for c in chain)
    return found


@dataclass
class Pair:
    i: int
    j: int
    theta: float
    a: float
    b: float
    rho: float           # midline offset
    d: float             # center-to-center separation
    w: float             # mean stroke width
    inside_ink: float    # ink coverage on the midline (0 hollow, >0 hatched)


def face_pairs(strokes, thin: list[int], ink: np.ndarray, max_sep: float, line_w: float) -> list[Pair]:
    """Parallel thin strokes facing each other across a narrow band: the two faces of a wall.
    Each stroke pairs with its nearest overlapping parallel neighbour on each side."""
    h, w = ink.shape
    on = ink > 0
    pairs: list[Pair] = []
    for key, ids in _buckets(strokes, thin).items():
        theta = key * ANGLE_TOL
        loc = sorted(_local(strokes, ids, theta), key=lambda s: s.rho)
        tx, ty, nx, ny = _frame(theta)
        for k, s in enumerate(loc):
            for t in loc[k + 1:]:
                d = t.rho - s.rho
                if d > max_sep:
                    break
                if d < 0.5 * (s.w + t.w) + 1.5:          # the faces must leave a paper gap between them
                    continue
                a, b = max(s.a, t.a), min(s.b, t.b)
                ov = b - a
                if ov < max(1.5 * d, 15):              # the wall exists where its faces overlap
                    continue
                if not 0.5 <= s.w / t.w <= 2.0:
                    continue
                # every overlapping partner is a candidate: lines inside a band (window frames,
                # hatching) sit between the faces; the selection keeps the pairs that match the
                # plan's wall thicknesses and are longest
                mid = (s.rho + t.rho) / 2
                us = np.linspace(a, b, 24)
                xs = np.clip(np.round(us * tx + mid * nx).astype(int), 0, w - 1)
                ys = np.clip(np.round(us * ty + mid * ny).astype(int), 0, h - 1)
                pairs.append(Pair(s.i, t.i, theta, a, b, mid, d, (s.w + t.w) / 2, float(on[ys, xs].mean())))
    return pairs


def dominant_separations(pairs: list[Pair], tol: float = 0.18) -> list[float]:
    """Peaks of the overlap-weighted separation histogram: the plan's wall thicknesses."""
    if not pairs:
        return []
    ds = np.array([p.d for p in pairs])
    wt = np.array([p.b - p.a for p in pairs])
    classes = []
    remaining = np.ones(len(ds), bool)
    total = wt.sum()
    for _ in range(3):
        if not remaining.any():
            break
        best, best_w = None, 0.0
        for d in np.unique(np.round(ds[remaining])):
            m = remaining & (np.abs(ds - d) <= tol * d + 1)
            if wt[m].sum() > best_w:
                best, best_w = d, wt[m].sum()
        if best is None or best_w < 0.12 * total:
            break
        m = remaining & (np.abs(ds - best) <= tol * best + 1)
        classes.append(float(np.average(ds[m], weights=wt[m])))
        remaining &= ~m
    return classes
