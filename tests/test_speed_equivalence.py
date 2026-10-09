"""The 2026-10-09 speed-ups must give exactly what the original algorithms gave."""

from __future__ import annotations

import cv2
import numpy as np

from engine import structure as S


def _scalar_trace(thin, wall_mask, point, d, reach, limit):
    """The original per-line trace (before engine.structure._trace_many)."""
    n = np.array([-d[1], d[0]])
    s = np.arange(1.0, limit, 1.0)
    ink = S._cross_occupancy(thin, point, d, n, s, [-1.5, -0.5, 0.5, 1.5]) > 0
    misses, last = 0, 0.0
    for step, has_ink in zip(s, ink):
        if has_ink:
            last, misses = step, 0
        else:
            misses += 1
            if misses > 5:
                break
    end = np.asarray(point, float) + d * last
    ss = np.arange(0.0, reach + 1.0, 1.0)
    touch = bool(S._cross_occupancy(wall_mask, end, d, np.array([-d[1], d[0]]), ss, [-1.0, 0.0, 1.0]).max() > 0)
    return end, touch


def test_batched_line_tracing_equals_tracing_one_by_one():
    rng = np.random.default_rng(3)
    thin = np.zeros((300, 400), np.uint8)
    wall = np.zeros((300, 400), np.uint8)
    for _ in range(40):
        p, q = rng.integers(0, 400, 2), rng.integers(0, 300, 2)
        cv2.line(thin, (int(p[0]), int(q[0])), (int(p[1]), int(q[1])), 255, 1)
    cv2.rectangle(wall, (20, 20), (380, 280), 255, 6)
    pts = rng.uniform(0, 300, (200, 2))
    ang = rng.uniform(0, 2 * np.pi, 200)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    ends, touch = S._trace_many(thin, wall, pts, dirs, 8.0, 300.0, batch=37)
    for k in range(len(pts)):
        e, t = _scalar_trace(thin, wall, pts[k], dirs[k], 8.0, 300.0)
        assert np.array_equal(e, ends[k]) and t == touch[k]


def test_parallel_hough_tiles_equal_sequential(monkeypatch):
    img = np.zeros((1400, 1500), np.uint8)
    for k in range(0, 1400, 37):
        cv2.line(img, (0, k), (1499, (k * 7) % 1400), 255, 1)
    monkeypatch.setenv("FLOORPLAN_HOUGH_WORKERS", "1")
    seq = S.hough_tiles(img, threshold=20, minLineLength=20, maxLineGap=3)
    monkeypatch.setenv("FLOORPLAN_HOUGH_WORKERS", "4")
    par = S.hough_tiles(img, threshold=20, minLineLength=20, maxLineGap=3)
    assert np.array_equal(seq, par)


def test_reach_by_distance_and_local_dilation_equals_page_dilation():
    """engine.semantic.layer._drop_isolated_walls: 'a small cluster within reach of a kept network'."""
    rng = np.random.default_rng(5)
    for k2 in (21, 63, 201):
        big = np.zeros((500, 600), np.uint8)
        cv2.rectangle(big, (250, 200), (350, 300), 1, -1)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
        reach = cv2.dilate(big, se) > 0
        r = k2 // 2
        dist = cv2.distanceTransform((big == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        for _ in range(60):
            x, y = int(rng.integers(0, 590)), int(rng.integers(0, 490))
            own = np.zeros_like(big)
            own[y:y + 8, x:x + 8] = 1
            truth = bool(reach[own > 0].any())
            nearest = float(dist[own > 0].min())
            if nearest <= r - 1.5:
                fast = True
            elif nearest > r + 1.5:
                fast = False
            else:
                fast = bool((cv2.dilate(own, se) > 0)[big > 0].any())
            assert fast == truth


def test_batched_stroke_measurement_equals_per_edge():
    from engine.plan import primitives as P
    rng = np.random.default_rng(1)
    for _ in range(4):
        ink = np.zeros((500, 700), np.uint8)
        for _ in range(50):
            a, b = rng.integers(0, 700, 2), rng.integers(0, 500, 2)
            cv2.line(ink, (int(a[0]), int(b[0])), (int(a[1]), int(b[1])), 255, int(rng.integers(1, 12)))
        for _ in range(30):
            x, y = int(rng.integers(0, 680)), int(rng.integers(0, 480))
            cv2.line(ink, (x, y), (x + 15, y + 15), 255, 1)
        edges = P._edges(ink, 8.0)
        assert P._measure(ink, edges, batch=53) == P._measure_reference(ink, edges)
