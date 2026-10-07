"""Room recovery inside a space that holds several room labels.

When the wall structure leaves two or more labelled rooms in one sealed space (a door gap
that was not found, partitions too faint to keep), the labels themselves say the space is
several rooms. This module splits such a space only where the drawing supports a boundary:

* Free floor is the space minus every drawn stroke (walls, partitions, furniture outlines),
  with OCR text removed so labels do not wall themselves in.
* A marker-controlled watershed on the distance transform of the free floor, seeded at the
  labels, places boundaries along the narrowest passages between them — doorways and gaps
  between partitions — because the distance to the nearest stroke is smallest there.
* A boundary is accepted only if it is a real constriction: much shorter than the rooms it
  separates. A long boundary means the rooms are open to each other (open-plan living and
  kitchen, a hall opening onto a room); those labels stay together in one region.

No boundary is invented in open floor, and no box is drawn around text: every accepted cut
runs across a narrow passage between drawn strokes.
"""

from __future__ import annotations

import cv2
import numpy as np


def _flood(elevation: np.ndarray, markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Meyer priority flood: grows marker labels over `mask` from low to high elevation;
    pixels reached by two different labels become boundary (-1)."""
    import heapq
    h, w = mask.shape
    labels = markers.copy()
    heap = []
    order = 0
    nbrs = ((0, 1), (1, 0), (0, -1), (-1, 0))
    ys, xs = np.nonzero(labels > 0)
    for y, x in zip(ys, xs):
        for dy, dx in nbrs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and labels[yy, xx] == 0:
                heapq.heappush(heap, (float(elevation[yy, xx]), order, yy, xx))
                order += 1
    while heap:
        _, _, y, x = heapq.heappop(heap)
        if labels[y, x] != 0:
            continue
        seen = set()
        for dy, dx in nbrs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and labels[yy, xx] > 0:
                seen.add(int(labels[yy, xx]))
        if len(seen) > 1:
            labels[y, x] = -1
            continue
        labels[y, x] = seen.pop() if seen else 0
        if labels[y, x] <= 0:
            continue
        for dy, dx in nbrs:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and labels[yy, xx] == 0:
                heapq.heappush(heap, (float(elevation[yy, xx]), order, yy, xx))
                order += 1
    return labels


def _climb(dist: np.ndarray, y: int, x: int) -> tuple[int, int]:
    """Steepest ascent on the distance map to a local maximum (8-neighbourhood)."""
    h, w = dist.shape
    while True:
        best, by, bx = dist[y, x], y, x
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and dist[yy, xx] > best:
                    best, by, bx = dist[yy, xx], yy, xx
        if (by, bx) == (y, x):
            return y, x
        y, x = by, bx


def _boundary_lengths(labels: np.ndarray, n: int) -> np.ndarray:
    """Boundary pixels (-1) shared by each pair of regions, as an (n+1) x (n+1) matrix."""
    lengths = np.zeros((n + 1, n + 1), int)
    h, w = labels.shape
    ys, xs = np.nonzero(labels == -1)
    for y, x in zip(ys, xs):
        neigh = set()
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and labels[yy, xx] > 0:
                neigh.add(int(labels[yy, xx]))
        neigh = sorted(neigh)
        for a in range(len(neigh)):
            for b in range(a + 1, len(neigh)):
                lengths[neigh[a], neigh[b]] += 1
                lengths[neigh[b], neigh[a]] += 1
    return lengths


def partition_space(region: np.ndarray, strokes: np.ndarray, points: list[tuple[int, int]],
                    min_area: float) -> list[tuple[np.ndarray, list[int]]] | None:
    """Split a space between several labels where the drawing has a constriction.

    region: bool mask of the space; strokes: uint8 mask of drawn ink (text removed);
    points: label positions (one per room label); min_area: smallest room accepted.
    Returns [(mask, label indices)] for each recovered room, or None when no split is
    supported (the labels then remain one region).
    """
    if len(points) < 2:
        return None
    ys, xs = np.nonzero(region)
    if xs.size == 0:
        return None
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    pad = 2
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(region.shape[1], x1 + pad), min(region.shape[0], y1 + pad)
    reg = region[y0:y1, x0:x1]
    free = reg & ~(cv2.dilate(strokes[y0:y1, x0:x1], np.ones((3, 3), np.uint8)) > 0)
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)

    # Work on a reduced grid (longest side <= 256 px): the flood is a pure-Python heap.
    scale = min(1.0, 256.0 / max(reg.shape))
    small = (max(1, int(round(reg.shape[1] * scale))), max(1, int(round(reg.shape[0] * scale))))
    reg_s = cv2.resize(reg.astype(np.uint8), small, interpolation=cv2.INTER_NEAREST) > 0
    free_s = cv2.resize(free.astype(np.uint8), small, interpolation=cv2.INTER_NEAREST) > 0
    dist_s = cv2.distanceTransform(free_s.astype(np.uint8), cv2.DIST_L2, 5)
    markers = np.zeros(reg_s.shape, np.int32)
    seeds = []
    for k, (px, py) in enumerate(points, 1):
        lx, ly = int((int(px) - x0) * scale), int((int(py) - y0) * scale)
        if not (0 <= lx < reg_s.shape[1] and 0 <= ly < reg_s.shape[0]):
            return None
        r = max(3, int(15 * scale))
        win = free_s[max(0, ly - r):ly + r + 1, max(0, lx - r):lx + r + 1]
        wy, wx = np.nonzero(win)
        if wx.size == 0:
            return None
        j = int(np.argmin((wx + max(0, lx - r) - lx) ** 2 + (wy + max(0, ly - r) - ly) ** 2))
        seeds.append(_climb(dist_s, int(wy[j] + max(0, ly - r)), int(wx[j] + max(0, lx - r))))
    # Each label climbs to the interior peak of the room around it: the label position is
    # evidence of which room it names, not the place to grow that room from. Labels that reach
    # the same peak are in one open area and share a basin.
    peaks: dict[tuple[int, int], int] = {}
    owner = []
    for (py, px) in seeds:
        key = next((q for q in peaks if abs(q[0] - py) <= 2 and abs(q[1] - px) <= 2), None)
        if key is None:
            key = (py, px)
            peaks[key] = len(peaks) + 1
            markers[py, px] = peaks[key]
        owner.append(peaks[key])
    if len(peaks) < 2:
        return None
    flooded = _flood(-dist_s, markers, reg_s)
    markers = cv2.resize(flooded.astype(np.float32), (reg.shape[1], reg.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    lengths_scale = 1.0 / scale

    n = len(peaks)
    areas = np.array([float(((markers == k) & reg).sum()) for k in range(1, n + 1)])
    if (areas == 0).any():
        return None
    lengths = _boundary_lengths(flooded, n) * lengths_scale
    # Union labels whose regions meet along a long boundary (open to each other).
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            shared = lengths[a, b]
            if shared == 0:
                continue
            size = min(np.sqrt(areas[a - 1]), np.sqrt(areas[b - 1]))
            if shared > 0.45 * size:                      # not a constriction: open plan
                parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for k in range(1, n + 1):
        groups.setdefault(find(k), []).append(k)
    if len(groups) < 2:
        return None
    out = []
    for members in groups.values():
        mask = np.zeros(region.shape, bool)
        sub = np.isin(markers, members) & reg
        # watershed lines between merged members belong to the room
        line = (markers == -1) & reg
        if len(members) > 1:
            grown = cv2.dilate(sub.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            sub |= line & grown
        mask[y0:y1, x0:x1] = sub
        if mask.sum() < min_area:
            return None
        out.append((mask, [i for i, o in enumerate(owner) if o in members]))
    return out


def functional_zones(region: np.ndarray, points: list[tuple[int, int]]) -> list[np.ndarray] | None:
    """Divide one physical space among its functional zones (labels inside an open plan, or rooms
    of a space the structure did not separate): every pixel goes to the label it is geodesically
    nearest to inside the space. This is an estimate of where each function is; it never
    creates or removes a physical boundary. Returns one bool mask per point (None if impossible)."""
    ys, xs = np.nonzero(region)
    if xs.size == 0 or len(points) < 2:
        return None
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    reg = region[y0:y1, x0:x1].astype(np.uint8)
    scale = min(1.0, 256.0 / max(reg.shape))
    small = cv2.resize(reg, (max(1, int(round(reg.shape[1] * scale))), max(1, int(round(reg.shape[0] * scale)))),
                       interpolation=cv2.INTER_NEAREST) > 0
    sy, sx = np.nonzero(small)
    if sx.size == 0:
        return None
    markers = np.zeros(small.shape, np.int32)
    for k, (px, py) in enumerate(points, 1):
        lx, ly = (px - x0) * scale, (py - y0) * scale
        j = int(np.argmin((sy - ly) ** 2 + (sx - lx) ** 2))      # nearest pixel of the space to the label
        if markers[sy[j], sx[j]]:
            return None                                          # two labels at one place
        markers[sy[j], sx[j]] = k
    flooded = _flood(np.zeros(small.shape, np.float32), markers, small)   # flat: breadth-first, i.e. geodesic
    full = cv2.resize(flooded.astype(np.float32), (reg.shape[1], reg.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    out = []
    for k in range(1, len(points) + 1):
        m = np.zeros(region.shape, bool)
        m[y0:y1, x0:x1] = (full == k) & (reg > 0)
        out.append(m)
    return out
