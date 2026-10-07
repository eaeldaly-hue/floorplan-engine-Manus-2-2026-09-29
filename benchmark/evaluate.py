"""Metrics for opening detection/classification and structural geometry."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

CLASSES = ("door", "window")


def _gt(o) -> dict:
    if isinstance(o, dict):
        return o
    return o.to_dict() | {"kind": o.kind}


def _match_score(pred: dict, gt: dict) -> float:
    """1-D IoU of a predicted opening against a GT opening on the GT wall axis (0 if not compatible)."""
    p0, p1 = np.asarray(gt["p0"], float), np.asarray(gt["p1"], float)
    width = float(np.linalg.norm(p1 - p0))
    if width < 1:
        return 0.0
    t = (p1 - p0) / width
    n = np.array([-t[1], t[0]])
    a, b = np.asarray(pred["start"], float), np.asarray(pred["end"], float)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1:
        return 0.0
    if abs(float(np.dot(d / length, t))) < math.cos(math.radians(25)):
        return 0.0
    thickness = float(gt.get("wall_thickness_px") or 20.0)
    mid = (a + b) / 2
    if abs(float(np.dot(mid - p0, n))) > max(0.9 * thickness, 8.0):
        return 0.0
    u = sorted((float(np.dot(a - p0, t)), float(np.dot(b - p0, t))))
    overlap = max(0.0, min(u[1], width) - max(u[0], 0.0))
    union = max(u[1], width) - min(u[0], 0.0)
    if overlap < 0.4 * min(width, length):
        return 0.0
    return overlap / union if union > 0 else 0.0


def match(preds: list[dict], gts: list) -> list[tuple[int, int, float]]:
    gts = [_gt(g) for g in gts]
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            score = _match_score(p, g)
            if score >= 0.2:
                pairs.append((score, i, j))
    pairs.sort(reverse=True)
    used_p, used_g, out = set(), set(), []
    for score, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        out.append((i, j, score))
    return out


@dataclass
class OpeningStats:
    gt: Counter = field(default_factory=Counter)          # GT counts by kind
    pred: Counter = field(default_factory=Counter)        # predictions by type
    found: Counter = field(default_factory=Counter)       # GT kind matched by any prediction
    confusion: Counter = field(default_factory=Counter)   # (gt kind, pred type)
    unmatched_pred: Counter = field(default_factory=Counter)  # false candidates by pred type
    iou_sum: float = 0.0
    errors: list = field(default_factory=list)

    def add(self, preds: list[dict], gts: list, label: str = "") -> None:
        gts = [_gt(g) for g in gts]
        pairs = match(preds, gts)
        matched_p = {i for i, _, _ in pairs}
        for g in gts:
            self.gt[g["kind"]] += 1
        for p in preds:
            self.pred[p["type"]] += 1
        for i, j, score in pairs:
            g, p = gts[j], preds[i]
            self.found[g["kind"]] += 1
            self.confusion[(g["kind"], p["type"])] += 1
            self.iou_sum += score
            if g["kind"] in CLASSES and p["type"] != g["kind"]:
                self.errors.append({"plan": label, "gt": g.get("id"), "gt_kind": g["kind"], "style": g.get("style"), "pred": p["type"]})
        for i, p in enumerate(preds):
            if i not in matched_p:
                self.unmatched_pred[p["type"]] += 1
                self.errors.append({"plan": label, "false_candidate": True, "pred": p["type"], "center": p.get("center")})
        matched_g = {j for _, j, _ in pairs}
        for j, g in enumerate(gts):
            if j not in matched_g and g["kind"] in CLASSES:
                self.errors.append({"plan": label, "missed": g.get("id"), "gt_kind": g["kind"], "style": g.get("style")})

    def merge(self, other: "OpeningStats") -> None:
        for name in ("gt", "pred", "found", "confusion", "unmatched_pred"):
            getattr(self, name).update(getattr(other, name))
        self.iou_sum += other.iou_sum
        self.errors.extend(other.errors)

    def metrics(self) -> dict:
        out = {}
        dw = self.gt["door"] + self.gt["window"]
        out["candidate_recall"] = (self.found["door"] + self.found["window"]) / dw if dw else None
        for c in CLASSES:
            tp = self.confusion[(c, c)]
            fp = self.pred[c] - tp - self.confusion[("opening", c)]  # predictions of c on 'opening' GT are not counted
            out[f"{c}_precision"] = tp / (tp + fp) if (tp + fp) else None
            out[f"{c}_recall"] = tp / self.gt[c] if self.gt[c] else None
            out[f"{c}_tp"], out[f"{c}_fp"], out[f"{c}_fn"] = tp, fp, self.gt[c] - tp
        matched_dw = sum(self.confusion[(g, p)] for g in CLASSES for p in ("door", "window", "opening"))
        correct = self.confusion[("door", "door")] + self.confusion[("window", "window")]
        out["classification_accuracy"] = correct / matched_dw if matched_dw else None
        out["end_to_end_accuracy"] = correct / dw if dw else None
        unknown = self.confusion[("door", "opening")] + self.confusion[("window", "opening")]
        out["unknown_rate"] = unknown / matched_dw if matched_dw else None
        out["door_as_window"] = self.confusion[("door", "window")]
        out["window_as_door"] = self.confusion[("window", "door")]
        out["false_candidates"] = sum(self.unmatched_pred.values())
        out["false_door_or_window"] = self.unmatched_pred["door"] + self.unmatched_pred["window"]
        out["missed_candidates"] = dw - (self.found["door"] + self.found["window"])
        out["passages"] = self.gt["opening"]
        out["passages_found"] = self.found["opening"]
        out["passages_called_door"] = self.confusion[("opening", "door")]
        out["passages_called_window"] = self.confusion[("opening", "window")]
        out["gt_doors"], out["gt_windows"] = self.gt["door"], self.gt["window"]
        return out


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def polygons_to_labels(shape, spaces: list[dict]) -> np.ndarray:
    labels = np.zeros(shape[:2], np.int32)
    for k, space in enumerate(spaces, 1):
        poly = space["polygon"]
        if poly and isinstance(poly[0], dict):
            poly = [(p["x"], p["y"]) for p in poly]
        cv2.fillPoly(labels, [np.asarray(poly, np.int32)], k)
    return labels


@dataclass
class StructureStats:
    rooms: int = 0
    detected: int = 0           # GT rooms matched by a space with IoU >= 0.7
    iou_sum: float = 0.0
    merged: int = 0             # predicted spaces covering >= 2 GT rooms
    split: int = 0              # GT rooms covered by >= 2 predicted spaces
    false_spaces: int = 0       # predicted spaces not corresponding to any GT room
    spaces: int = 0
    wall_iou_sum: float = 0.0
    fabricated_wall: float = 0.0
    plans: int = 0
    adj_tp: int = 0
    adj_fp: int = 0
    adj_fn: int = 0

    def add(self, pred_labels: np.ndarray, gt_labels: np.ndarray, pred_wall: np.ndarray | None = None,
            gt_wall: np.ndarray | None = None, pred_adjacency: set | None = None, gt_adjacency: set | None = None):
        self.plans += 1
        gt_ids = [r for r in np.unique(gt_labels) if r > 0]
        pred_ids = [s for s in np.unique(pred_labels) if s > 0]
        self.rooms += len(gt_ids)
        self.spaces += len(pred_ids)
        inter = Counter(zip(gt_labels[(gt_labels > 0) & (pred_labels > 0)].tolist(), pred_labels[(gt_labels > 0) & (pred_labels > 0)].tolist()))
        gt_area = {r: int((gt_labels == r).sum()) for r in gt_ids}
        pred_area = {s: int((pred_labels == s).sum()) for s in pred_ids}
        best_for_gt = {}
        for r in gt_ids:
            ious = [(inter[(r, s)] / (gt_area[r] + pred_area[s] - inter[(r, s)]), s) for s in pred_ids if inter[(r, s)]]
            best = max(ious, default=(0.0, None))
            best_for_gt[r] = best
            self.iou_sum += best[0]
            if best[0] >= 0.7:
                self.detected += 1
            if sum(1 for s in pred_ids if inter[(r, s)] >= 0.2 * gt_area[r]) >= 2:
                self.split += 1
        for s in pred_ids:
            covering = [r for r in gt_ids if inter[(r, s)] >= 0.3 * gt_area[r]]
            if len(covering) >= 2:
                self.merged += 1
            if sum(inter[(r, s)] for r in gt_ids) < 0.5 * pred_area[s]:
                self.false_spaces += 1
        if pred_wall is not None and gt_wall is not None:
            p, g = pred_wall > 0, gt_wall > 0
            self.wall_iou_sum += (p & g).sum() / max(1, (p | g).sum())
            near = cv2.dilate(gt_wall, np.ones((5, 5), np.uint8)) > 0
            self.fabricated_wall += (p & ~near).sum() / max(1, p.sum())
        if pred_adjacency is not None and gt_adjacency is not None:
            # map predicted space ids to GT room ids via best overlap
            to_gt = {}
            for s in pred_ids:
                cand = max(((inter[(r, s)], r) for r in gt_ids), default=(0, None))
                if cand[0] >= 0.5 * pred_area[s]:
                    to_gt[s] = cand[1]
            mapped = set()
            for a, b in pred_adjacency:
                ga = 0 if a == 0 else to_gt.get(a)
                gb = 0 if b == 0 else to_gt.get(b)
                if ga is None or gb is None or ga == gb:
                    continue
                mapped.add(tuple(sorted((ga, gb))))
            gt_adj = {tuple(sorted(p)) for p in gt_adjacency}
            self.adj_tp += len(mapped & gt_adj)
            self.adj_fp += len(mapped - gt_adj)
            self.adj_fn += len(gt_adj - mapped)

    def merge(self, other: "StructureStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def metrics(self) -> dict:
        return {
            "rooms": self.rooms, "spaces": self.spaces,
            "rooms_detected_iou70": self.detected,
            "room_recall_iou70": self.detected / self.rooms if self.rooms else None,
            "mean_best_room_iou": self.iou_sum / self.rooms if self.rooms else None,
            "merged_spaces": self.merged, "split_rooms": self.split, "false_spaces": self.false_spaces,
            "wall_iou": self.wall_iou_sum / self.plans if self.plans else None,
            "fabricated_wall_fraction": self.fabricated_wall / self.plans if self.plans else None,
            "adjacency_precision": self.adj_tp / (self.adj_tp + self.adj_fp) if (self.adj_tp + self.adj_fp) else None,
            "adjacency_recall": self.adj_tp / (self.adj_tp + self.adj_fn) if (self.adj_tp + self.adj_fn) else None,
        }
