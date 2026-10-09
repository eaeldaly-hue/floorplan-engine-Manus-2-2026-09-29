"""Drawing primitives: every straight line of the plan as a stroke with its width.

The edges of the ink are found with OpenCV's line segment detector (any orientation). From
each edge the stroke width is measured by walking perpendicular into the ink, and the edge is
moved half a width inward to the stroke's centerline. The two edges of one stroke, and the
pieces of one stroke split at junctions, are merged. A filled wall band becomes one wide
stroke, a hollow wall two thin parallel strokes, a thin wall one stroke - the same
representation for every drawing style.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .model import Stroke

MAX_WIDTH = 120          # px; wider "strokes" are filled areas, not lines
SAMPLES = 24             # width samples per edge
PAPER_RUN = 3            # paper pixels that end a stroke
WIDTH_Q = 30             # width quantile over the samples
START_STEPS = 3          # px the ink may start past the detected edge


def ink_mask(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(gray, ink 0/255): the engine's own binarisation (dark lines + lighter symbol lines)."""
    from engine.structure import _binarize

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, ink = _binarize(gray)
    return gray, ink


def _edges(ink: np.ndarray, min_length: float) -> np.ndarray:
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    img = cv2.GaussianBlur(255 - ink, (3, 3), 0)
    lines = lsd.detect(img)[0]
    if lines is None:
        return np.zeros((0, 4), np.float32)
    lines = lines.reshape(-1, 4)
    length = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    return lines[length >= min_length]


def _measure(ink: np.ndarray, edges: np.ndarray, batch: int = 512) -> list[Stroke]:
    """Edge -> stroke: ink side, width (median over samples), centerline. All edges are measured
    together in batches; every value is computed as _measure_reference computes it, edge by edge."""
    h, w = ink.shape
    on = ink > 0
    out: list[Stroke] = []
    if not len(edges):
        return out
    p0, p1 = edges[:, :2], edges[:, 2:]
    d = p1 - p0
    L = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
    t = d / L[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    s = np.linspace(0.15, 0.85, SAMPLES)                                  # avoid the ends (junctions)
    pts = p0[:, None, :] + s[None, :, None] * d[:, None, :]               # (N, S, 2)

    def ink_at(xy):
        x = np.clip(np.round(xy[..., 0]).astype(int), 0, w - 1)
        y = np.clip(np.round(xy[..., 1]).astype(int), 0, h - 1)
        return on[y, x]

    plus = ink_at(pts + 1.2 * n[:, None, :]).mean(axis=1)
    minus = ink_at(pts - 1.2 * n[:, None, :]).mean(axis=1)
    side = np.where(plus >= minus, 1.0, -1.0)
    darkness = np.maximum(plus, minus)
    keep = np.nonzero(darkness >= 0.6)[0]
    steps = np.arange(0.6, MAX_WIDTH + 0.6, 1.0)
    span = len(steps) - START_STEPS
    for b0 in range(0, len(keep), batch):
        idx = keep[b0:b0 + batch]
        direction = side[idx, None] * n[idx]                                              # (B, 2)
        ray = pts[idx][:, :, None, :] + steps[None, None, :, None] * direction[:, None, None, :]   # (B, S, K, 2)
        raw = ink_at(ray)                                                                 # (B, S, K)
        valid = raw[:, :, :START_STEPS].any(axis=2)                                       # (B, S)
        ok = valid.sum(axis=1) >= 0.5 * SAMPLES
        s0 = np.argmax(raw[:, :, :START_STEPS], axis=2)
        vals = np.take_along_axis(raw, s0[:, :, None] + np.arange(span)[None, None, :], axis=2)   # (B, S, span)
        plain = np.where(vals.all(axis=2), span, np.argmin(vals, axis=2)).astype(float)
        plain_v = np.where(valid, plain, np.nan)
        s0_v = np.where(valid, s0.astype(float), np.nan)
        paper = ~vals
        n_run = span - PAPER_RUN + 1
        run = paper[:, :, :n_run].copy()
        for k in range(1, PAPER_RUN):
            run &= paper[:, :, k:n_run + k]
        bridged = np.where(run.any(axis=2), np.argmax(run, axis=2), span).astype(float)
        bridged_v = np.where(valid, bridged, np.nan)
        for r in np.nonzero(ok)[0]:
            i = idx[r]
            pv = plain_v[r][valid[r]]
            start = float(np.median(s0_v[r][valid[r]]))
            width = float(np.percentile(pv, WIDTH_Q))
            if np.mean(np.abs(pv - np.median(pv)) <= 1.0) < 0.6:
                # the first paper pixel wanders along the stroke: slits of a dense hatch or a scanned
                # band, not the clean gap to a parallel line; the band ends at the first paper run
                width = float(np.percentile(bridged_v[r][valid[r]], WIDTH_Q))
            if width >= MAX_WIDTH - START_STEPS:
                continue
            off = direction[r] * (start + width / 2.0 + 0.1)
            out.append(Stroke(p0=(float(p0[i, 0] + off[0]), float(p0[i, 1] + off[1])),
                              p1=(float(p1[i, 0] + off[0]), float(p1[i, 1] + off[1])),
                              width=max(1.0, width), darkness=float(darkness[i])))
    return out


def _measure_reference(ink: np.ndarray, edges: np.ndarray) -> list[Stroke]:
    """The per-edge implementation of _measure (kept as the reference the batched one must equal)."""
    h, w = ink.shape
    on = ink > 0
    out: list[Stroke] = []
    if not len(edges):
        return out
    p0, p1 = edges[:, :2], edges[:, 2:]
    d = p1 - p0
    L = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
    t = d / L[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    s = np.linspace(0.15, 0.85, SAMPLES)                                  # avoid the ends (junctions)
    pts = p0[:, None, :] + s[None, :, None] * d[:, None, :]               # (N, S, 2)

    def ink_at(xy):
        x = np.clip(np.round(xy[..., 0]).astype(int), 0, w - 1)
        y = np.clip(np.round(xy[..., 1]).astype(int), 0, h - 1)
        return on[y, x]

    plus = ink_at(pts + 1.2 * n[:, None, :]).mean(axis=1)
    minus = ink_at(pts - 1.2 * n[:, None, :]).mean(axis=1)
    side = np.where(plus >= minus, 1.0, -1.0)
    darkness = np.maximum(plus, minus)
    keep = darkness >= 0.6
    steps = np.arange(0.6, MAX_WIDTH + 0.6, 1.0)
    for i in np.nonzero(keep)[0]:
        direction = side[i] * n[i]
        ray = pts[i][:, None, :] + steps[None, :, None] * direction[None, None, :]   # (S, K, 2)
        raw = ink_at(ray)
        # the edge of a thin line can sit a pixel off the ink: the stroke starts at the first
        # ink pixel within the first steps
        valid = raw[:, :START_STEPS].any(axis=1)
        if valid.sum() < 0.5 * SAMPLES:
            continue
        s0 = np.argmax(raw[:, :START_STEPS], axis=1)
        vals = np.take_along_axis(raw, s0[:, None] + np.arange(len(steps) - START_STEPS)[None, :], axis=1)
        start = float(np.median(s0[valid]))
        plain = np.where(vals.all(axis=1), vals.shape[1], np.argmin(vals, axis=1))[valid]
        # lower quantile: lines touching the stroke (hatching, junctions) only make it look wider
        width = float(np.percentile(plain, WIDTH_Q))
        if np.mean(np.abs(plain - np.median(plain)) <= 1.0) < 0.6:
            # the first paper pixel wanders along the stroke: slits of a dense hatch or a scanned
            # band, not the clean gap to a parallel line. The band ends at the first run of
            # PAPER_RUN paper pixels.
            paper = ~vals
            n_run = vals.shape[1] - PAPER_RUN + 1
            run = paper[:, :n_run].copy()
            for k in range(1, PAPER_RUN):
                run &= paper[:, k:n_run + k]
            bridged = np.where(run.any(axis=1), np.argmax(run, axis=1), vals.shape[1])[valid]
            width = float(np.percentile(bridged, WIDTH_Q))
        if width >= MAX_WIDTH - START_STEPS:
            continue
        off = direction * (start + width / 2.0 + 0.1)
        out.append(Stroke(p0=(float(p0[i, 0] + off[0]), float(p0[i, 1] + off[1])),
                          p1=(float(p1[i, 0] + off[0]), float(p1[i, 1] + off[1])),
                          width=max(1.0, width), darkness=float(darkness[i])))
    return out


def merge_strokes(strokes: list[Stroke], angle_tol: float = 1.5) -> list[Stroke]:
    """Merge the two edges of one stroke and its collinear pieces (split at junctions)."""
    if not strokes:
        return []
    ang = np.array([s.angle for s in strokes])
    order = np.argsort(ang)
    groups: list[list[int]] = []
    for i in order:                                                       # cluster by angle (wrap at 180)
        if groups and abs(ang[i] - ang[groups[-1][0]]) <= angle_tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    if len(groups) > 1 and ang[groups[0][0]] + 180.0 - ang[groups[-1][0]] <= angle_tol:
        groups[0] = groups[-1] + groups[0]
        groups.pop()
    merged: list[Stroke] = []
    for g in groups:
        doubled = np.radians([2.0 * strokes[i].angle for i in g])         # undirected: circular mean of 2*angle
        theta = 0.5 * math.atan2(float(np.mean(np.sin(doubled))), float(np.mean(np.cos(doubled))))
        tx, ty = math.cos(theta), math.sin(theta)
        nx, ny = -ty, tx
        items = []
        for i in g:
            s = strokes[i]
            a = s.p0[0] * tx + s.p0[1] * ty
            b = s.p1[0] * tx + s.p1[1] * ty
            rho = ((s.p0[0] + s.p1[0]) / 2) * nx + ((s.p0[1] + s.p1[1]) / 2) * ny
            items.append([min(a, b), max(a, b), rho, s.width, s.darkness, s.length])
        items.sort(key=lambda r: (r[2], r[0]))
        # cluster by offset (rho), then sweep along the line within each cluster
        clusters: list[list] = []
        for it in items:
            if clusters and abs(it[2] - clusters[-1][-1][2]) <= max(1.5, 0.35 * min(it[3], clusters[-1][-1][3])):
                clusters[-1].append(it)
            else:
                clusters.append([it])
        for cl in clusters:
            cl.sort(key=lambda r: r[0])
            cur = None
            for a, b, rho, wd, dk, ln in cl:
                if cur is not None and a <= cur[1] + max(2.0, 0.6 * max(cur[3], wd)) and 0.6 <= wd / cur[3] <= 1.7:
                    tot = cur[5] + ln
                    cur = [cur[0], max(cur[1], b), (cur[2] * cur[5] + rho * ln) / tot, (cur[3] * cur[5] + wd * ln) / tot,
                           (cur[4] * cur[5] + dk * ln) / tot, tot]
                else:
                    if cur is not None:
                        merged.append(_stroke(cur, tx, ty, nx, ny))
                    cur = [a, b, rho, wd, dk, ln]
            merged.append(_stroke(cur, tx, ty, nx, ny))
    return merged


def _stroke(c, tx, ty, nx, ny) -> Stroke:
    a, b, rho, wd, dk, _ = c
    return Stroke(p0=(a * tx + rho * nx, a * ty + rho * ny), p1=(b * tx + rho * nx, b * ty + rho * ny),
                  width=float(wd), darkness=float(dk))


def extract_strokes(ink: np.ndarray, min_length: float = 8.0) -> list[Stroke]:
    strokes = _measure(ink, _edges(ink, min_length))
    return [s for s in merge_strokes(strokes) if s.length >= min_length]
