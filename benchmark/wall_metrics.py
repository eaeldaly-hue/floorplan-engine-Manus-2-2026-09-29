"""Wall-level measurements against synthetic ground truth (benchmark.wall_styles).

Room metrics say *that* a plan failed; these say whether the walls were understood:

  pixel precision     share of predicted wall pixels that lie on a drawn wall (within TOL px)
  pixel recall        share of the drawn wall that is covered by predicted wall (within TOL px).
                      "Drawn wall" is the band for filled / hollow / hatched walls and the line
                      itself for thin single-line walls.
  centerline recall   share of the wall axes (opening gaps excluded) covered by predicted wall.
                      Style-independent: a hollow wall whose two outlines are found but whose band
                      is not understood as wall scores low, as it should.
  segment recall      share of wall pieces (one wall between two rooms / room and outside) whose
                      axis is covered over >= 80 % of its length; also length-weighted.
  fabricated          predicted wall pixels farther than TOL from any wall band (furniture,
                      text, symbols taken as wall), as a share of predicted wall; and phantom wall
                      components (< 10 % of their pixels on a wall band).
  thickness ratio     the engine's typical wall thickness / the GT interior wall thickness.
  fragmentation       predicted wall components / GT wall components.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

TOL = 2                    # px: anti-aliasing and rounding at wall faces
SEGMENT_FOUND = 0.8        # share of a piece's axis that must be covered


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def drawn_wall(plan, wall_render: str) -> np.ndarray:
    """Where the drawing designates wall, for the style: the band, or the line of a thin wall."""
    band = plan.wall_mask > 0
    if wall_render != "thin":
        return band
    lines = np.zeros(band.shape, np.uint8)
    width = max(2, plan.spec.style.line_px + 1)
    for p0, p1, _thick, _e0, _e1, _a, _b in plan.wall_pieces:
        cv2.line(lines, tuple(int(round(v)) for v in p0), tuple(int(round(v)) for v in p1), 1, width)
    return (lines > 0) & band                      # opening gaps excluded


def _axis_pixels(p0, p1, shape) -> tuple[np.ndarray, np.ndarray]:
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    xs = np.clip(np.round(np.linspace(p0[0], p1[0], n)).astype(int), 0, shape[1] - 1)
    ys = np.clip(np.round(np.linspace(p0[1], p1[1], n)).astype(int), 0, shape[0] - 1)
    return ys, xs


@dataclass
class WallStats:
    plans: int = 0
    pred_px: int = 0
    pred_on_wall: int = 0
    drawn_px: int = 0
    drawn_covered: int = 0
    axis_px: int = 0
    axis_covered: int = 0
    pieces: int = 0
    pieces_found: int = 0
    piece_len: float = 0.0
    piece_len_found: float = 0.0
    fabricated_px: int = 0
    phantom_components: int = 0
    thickness_ratios: list = field(default_factory=list)
    pred_components: int = 0
    gt_components: int = 0
    empty_masks: int = 0                      # plans where the engine found no wall at all

    def add(self, pred_wall: np.ndarray, plan, wall_render: str, typical_thickness: float | None = None) -> dict:
        pred = pred_wall > 0
        band = plan.wall_mask > 0
        drawn = drawn_wall(plan, wall_render)
        near_drawn = _dilate(drawn, TOL)
        near_band = _dilate(band, TOL)
        pred_near = _dilate(pred, TOL)
        row = {
            "pred_px": int(pred.sum()),
            "pred_on_wall": int((pred & near_drawn).sum()),
            "drawn_px": int(drawn.sum()),
            "drawn_covered": int((drawn & pred_near).sum()),
            "fabricated_px": int((pred & ~near_band).sum()),
        }
        axis_px = axis_cov = pieces_found = 0
        len_total = len_found = 0.0
        for p0, p1, thick, _e0, _e1, _a, _b in plan.wall_pieces:
            ys, xs = _axis_pixels(p0, p1, band.shape)
            keep = band[ys, xs]                                 # opening gaps are not wall
            if not keep.any():
                continue
            cov = pred_near[ys[keep], xs[keep]]
            axis_px += int(keep.sum()); axis_cov += int(cov.sum())
            share = float(cov.mean())
            length = float(keep.sum())
            len_total += length
            if share >= SEGMENT_FOUND:
                pieces_found += 1
                len_found += length
        row.update(axis_px=axis_px, axis_covered=axis_cov, pieces=sum(1 for _ in plan.wall_pieces),
                   pieces_found=pieces_found, piece_len=len_total, piece_len_found=len_found)
        n, comp = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
        phantom = 0
        if n > 1:
            on = np.bincount(comp[near_band].ravel(), minlength=n)
            size = np.bincount(comp.ravel(), minlength=n)
            phantom = int(sum(1 for k in range(1, n) if size[k] >= 20 and on[k] < 0.1 * size[k]))
        row["phantom_components"] = phantom
        row["pred_components"] = int(n - 1)
        row["gt_components"] = int(cv2.connectedComponents(band.astype(np.uint8), connectivity=8)[0] - 1)
        gt_int = min((p[2] for p in plan.wall_pieces), default=None)
        row["thickness_ratio"] = (round(typical_thickness / gt_int, 3)
                                  if typical_thickness and gt_int and pred.any() else None)
        self.plans += 1
        for k in ("pred_px", "pred_on_wall", "drawn_px", "drawn_covered", "fabricated_px", "axis_px", "axis_covered",
                  "pieces", "pieces_found", "piece_len", "piece_len_found", "phantom_components", "pred_components",
                  "gt_components"):
            setattr(self, k, getattr(self, k) + row[k])
        if row["thickness_ratio"] is not None:
            self.thickness_ratios.append(row["thickness_ratio"])
        self.empty_masks += int(not pred.any())
        row.update(self.row_metrics(row))
        return row

    @staticmethod
    def row_metrics(r: dict) -> dict:
        div = lambda a, b: round(a / b, 3) if b else None   # noqa: E731
        return {
            "pixel_precision": div(r["pred_on_wall"], r["pred_px"]),
            "pixel_recall": div(r["drawn_covered"], r["drawn_px"]),
            "centerline_recall": div(r["axis_covered"], r["axis_px"]),
            "segment_recall": div(r["pieces_found"], r["pieces"]),
            "segment_recall_length": div(r["piece_len_found"], r["piece_len"]),
            "fabricated_share": div(r["fabricated_px"], r["pred_px"]),
        }

    def merge(self, other: "WallStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def metrics(self) -> dict:
        out = self.row_metrics(self.__dict__)
        p, r = out["pixel_precision"], out["pixel_recall"]
        out["pixel_f1"] = None if p is None or r is None else (round(2 * p * r / (p + r), 3) if p + r else 0.0)
        out["phantom_components_per_plan"] = round(self.phantom_components / self.plans, 2) if self.plans else None
        out["fragmentation"] = round(self.pred_components / self.gt_components, 2) if self.gt_components else None
        out["thickness_ratio_median"] = float(np.median(self.thickness_ratios)) if self.thickness_ratios else None
        out["plans"] = self.plans
        out["plans_without_walls"] = self.empty_masks
        return out
