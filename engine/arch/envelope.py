"""Building envelope: which part of the sheet is the building.

A drawing sheet holds more than the building: a frame, a title block, legends, schedules, key
plans, dimension boxes. Their lines are drawn like walls and enclose regions like rooms, so the
structural analysis finds 'spaces' in them. The envelope is the prior that decides, before any
room reasoning, what belongs to the building.

Evidence (no drawing-specific thresholds):
  - A building's wall network is interrupted by openings. With the openings sealed (the
    structure's `sealed_mask`), each building is one connected network that carries doors and
    windows; frames, title-block grids and legend boxes carry (almost) none.
  - A network that encloses a building with more openings than its own is a frame around it,
    not a building (hierarchy: sheet > building > spaces).
The footprint of a building is the region inside its sealed network's outer outline. Sheets with
several buildings (floors side by side) keep every network with substantial opening evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


MIN_BUILDING_SHARE = 0.05     # of the largest footprint: smaller wall clusters are details, not buildings


@dataclass
class Building:
    id: str
    mask: np.ndarray                      # footprint (bool)
    polygon: list                         # outer outline, image px
    area_px: int
    openings: int                         # opening candidates on its wall network
    wall_px: int


@dataclass
class Envelope:
    buildings: list = field(default_factory=list)
    mask: np.ndarray | None = None        # union of the footprints; None = no reliable prior
    rejected: list = field(default_factory=list)   # (reason, bbox) of networks that are not buildings

    @property
    def available(self) -> bool:
        return self.mask is not None and bool(self.buildings)

    def inside_share(self, region: np.ndarray) -> float:
        if not self.available:
            return 1.0
        n = int(region.sum())
        return float((self.mask & region).sum()) / n if n else 1.0


def _fill(component: np.ndarray) -> np.ndarray:
    """Region enclosed by the component's outer outline (holes filled)."""
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(component.shape, np.uint8)
    cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out > 0


def estimate_envelope(structure, min_openings: int = 2, network: np.ndarray | None = None,
                      opening_points: list | None = None) -> Envelope:
    """`network`/`opening_points` override the structure's sealed walls and gap candidates (e.g. the
    Plan Model's walls with its openings sealed, when its spaces replaced the legacy ones)."""
    sealed = network if network is not None else getattr(structure, "sealed_mask", None)
    if sealed is None or not np.any(sealed):
        return Envelope()
    points = opening_points if opening_points is not None else [c.center for c in structure.candidates]
    net = cv2.dilate((sealed > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    count, comp, stats, _ = cv2.connectedComponentsWithStats(net, connectivity=8)
    if count <= 1:
        return Envelope()
    h, w = comp.shape
    votes = np.zeros(count, int)
    for p in points:
        x, y = (int(round(v)) for v in p)
        if 0 <= x < w and 0 <= y < h and comp[y, x] > 0:
            votes[comp[y, x]] += 1
    wall = sealed > 0
    wall_px = np.bincount(comp[wall].ravel(), minlength=count)
    order = [k for k in range(1, count) if votes[k] >= min_openings]
    if not order:
        return Envelope()
    best = max(votes[k] for k in order)
    t = float(getattr(structure, "wall_thickness", 0) or 4.0)
    env = Envelope()
    cands = []
    for k in sorted(order, key=lambda k: -votes[k]):
        if votes[k] < max(min_openings, 0.15 * best):
            x, y, bw, bh, _ = stats[k]
            env.rejected.append(("few openings", [int(x), int(y), int(bw), int(bh)]))
            continue
        foot = _fill(comp == k)
        # line-thin appendages (leader lines, dimension strings tied to the walls) are not building:
        # a building's body is at least a few wall thicknesses wide everywhere
        r = max(2, int(round(1.5 * t)))
        body = cv2.morphologyEx(foot.astype(np.uint8), cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        if body.any():
            n_parts, parts, pstats, _ = cv2.connectedComponentsWithStats(body.astype(np.uint8), connectivity=8)
            areas = pstats[1:, cv2.CC_STAT_AREA]
            core = np.isin(parts, 1 + np.flatnonzero(areas >= 0.1 * areas.max()))
            # keep everything wider than a drawn line that is connected to the body (narrow wings,
            # bays); what hangs on the body only by a line (a leader to a schedule) is cut off
            k_line = max(5, int(round(t)) | 1)
            solid = cv2.morphologyEx(foot.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k_line, k_line), np.uint8))
            n_s, pieces = cv2.connectedComponents(solid, connectivity=4)
            keep = np.unique(pieces[core & (solid > 0)])
            foot = np.isin(pieces, keep[keep > 0])
        cands.append((k, foot))
    kept = []
    for k, foot in cands:
        # a frame: encloses another network that carries more openings than its own
        inner = [j for j, f in cands if j != k and votes[j] > votes[k] and (f & foot).sum() >= 0.8 * f.sum()]
        if inner:
            x, y, bw, bh, _ = stats[k]
            env.rejected.append(("frame around a building", [int(x), int(y), int(bw), int(bh)]))
            continue
        kept.append((k, foot))
    # a network inside another building's footprint is part of that building (core, stair, column)
    buildings = []
    largest = max((int(f.sum()) for _, f in kept), default=0)
    for k, foot in kept:
        if any(j != k and (f & foot).sum() >= 0.8 * foot.sum() and f.sum() > foot.sum() for j, f in kept):
            continue
        if foot.sum() < MIN_BUILDING_SHARE * largest:
            x, y, bw, bh, _ = stats[k]
            env.rejected.append(("too small beside the main building", [int(x), int(y), int(bw), int(bh)]))
            continue
        contours, _ = cv2.findContours(foot.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        poly = cv2.approxPolyDP(cnt, max(2.0, 0.5 * float(structure.wall_thickness or 4.0)), True)
        buildings.append(Building(id=f"B{len(buildings) + 1}", mask=foot, polygon=[[int(p[0][0]), int(p[0][1])] for p in poly],
                                  area_px=int(foot.sum()), openings=int(votes[k]), wall_px=int(wall_px[k])))
    if not buildings:
        return Envelope(rejected=env.rejected)
    mask = np.zeros((h, w), bool)
    for b in buildings:
        mask |= b.mask
    env.buildings, env.mask = buildings, mask
    return env


def outside_spaces(structure, envelope: Envelope, min_inside: float = 0.5) -> set:
    """Space labels (1-based, as in structure.space_labels) lying outside every building."""
    if not envelope.available:
        return set()
    labels = structure.space_labels
    out = set()
    for k in range(1, len(structure.spaces) + 1):
        region = labels == k
        if region.any() and envelope.inside_share(region) < min_inside:
            out.add(k)
    return out


def plan_model_network(model_payload: dict, shape) -> tuple[np.ndarray, list]:
    """(sealed wall network, opening centres) from a Plan Model payload (engine.plan.adapter.payload)."""
    net = np.zeros(shape[:2], np.uint8)
    for w in model_payload.get("walls", []):
        t = max(1, int(round(w["thickness_px"])))
        cv2.line(net, tuple(int(v) for v in w["p0"]), tuple(int(v) for v in w["p1"]), 255, t)
    points = []
    for o in model_payload.get("openings", []):
        p0, p1 = o["p0"], o["p1"]
        cv2.line(net, tuple(int(v) for v in p0), tuple(int(v) for v in p1), 255, 3)
        points.append(((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2))
    return net, points
