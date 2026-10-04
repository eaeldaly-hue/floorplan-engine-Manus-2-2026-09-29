"""Detect and visually classify likely doors and windows from wall gaps."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from engine.geometry_v2.local_opening_analyzer import LocalOpeningAnalyzer
from engine.geometry_v2.segments import WallSegmentBuilder
from engine.walls.mask import WallMaskBuilder
from engine.walls.segments import WallSegmentExtractor


TYPE_LABELS = {
    "door": "باب محتمل",
    "window": "شباك محتمل",
    "opening": "فتحة تحتاج مراجعة",
}


def _opening_frame_marks(
    gray: np.ndarray,
    opening: Any,
) -> tuple[int, float]:
    """Count repeated parallel frame rails inside the wall thickness."""
    x1, y1 = opening.start
    x2, y2 = opening.end
    half_band = max(5, int(round(opening.wall_thickness * 1.35)))
    if opening.orientation == "horizontal":
        lo, hi = sorted((int(round(x1)), int(round(x2))))
        axis = int(round((y1 + y2) / 2))
        crop = gray[max(0, axis - half_band):min(gray.shape[0], axis + half_band + 1), max(0, lo):min(gray.shape[1], hi + 1)]
        if crop.size == 0:
            return 0, 0.0
        support = np.mean(crop < 165, axis=1)
    else:
        lo, hi = sorted((int(round(y1)), int(round(y2))))
        axis = int(round((x1 + x2) / 2))
        crop = gray[max(0, lo):min(gray.shape[0], hi + 1), max(0, axis - half_band):min(gray.shape[1], axis + half_band + 1)]
        if crop.size == 0:
            return 0, 0.0
        support = np.mean(crop < 165, axis=0)

    marked = support >= 0.55
    changes = np.diff(np.pad(marked.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    positions = []
    for start, end in zip(starts, ends):
        if end - start < 2:
            continue
        center = (start + end - 1) / 2
        if 0.08 * len(marked) <= center <= 0.92 * len(marked):
            positions.append(center)

    count = len(positions)
    if count < 2 or count > 5:
        return count, 0.0
    intervals = np.diff(positions)
    if len(intervals) < 1 or float(np.mean(intervals)) <= 0:
        return count, 0.0
    regularity = max(0.0, 1.0 - float(np.std(intervals)) / float(np.mean(intervals)))
    return count, regularity


def _door_swing_score(
    edge_distance: np.ndarray,
    opening: Any,
) -> float:
    """Look for a quarter-circle door swing near either gap endpoint."""
    best = 0.0
    radius = float(opening.width)
    if radius < 35:
        return best
    for endpoint_index, endpoint in enumerate((opening.start, opening.end)):
        hx, hy = endpoint
        direction = 1 if endpoint_index == 0 else -1
        for side in (-1, 1):
            supported = 0
            total = 0
            for theta in np.linspace(0.18, math.pi / 2 - 0.18, 40):
                if opening.orientation == "horizontal":
                    x = hx + direction * radius * math.cos(theta)
                    y = hy + side * radius * math.sin(theta)
                else:
                    y = hy + direction * radius * math.cos(theta)
                    x = hx + side * radius * math.sin(theta)
                ix, iy = int(round(x)), int(round(y))
                if 0 <= ix < edge_distance.shape[1] and 0 <= iy < edge_distance.shape[0]:
                    total += 1
                    if edge_distance[iy, ix] <= 2.5:
                        supported += 1
            if total:
                best = max(best, supported / total)
    return float(best)


def _deduplicate(candidates: list[Any]) -> list[Any]:
    """Remove long gap candidates that duplicate shorter nearby gaps."""
    kept = []
    for candidate in sorted(candidates, key=lambda item: (item.confidence, -item.width), reverse=True):
        a1, a2 = (candidate.start[0], candidate.end[0]) if candidate.orientation == "horizontal" else (candidate.start[1], candidate.end[1])
        low, high = sorted((a1, a2))
        axis = (candidate.start[1] + candidate.end[1]) / 2 if candidate.orientation == "horizontal" else (candidate.start[0] + candidate.end[0]) / 2
        duplicate = False
        for existing in kept:
            if existing.orientation != candidate.orientation:
                continue
            other_axis = (existing.start[1] + existing.end[1]) / 2 if existing.orientation == "horizontal" else (existing.start[0] + existing.end[0]) / 2
            if abs(axis - other_axis) > max(2 * candidate.wall_thickness, 2 * existing.wall_thickness, 18):
                continue
            b1, b2 = (existing.start[0], existing.end[0]) if existing.orientation == "horizontal" else (existing.start[1], existing.end[1])
            overlap = max(0.0, min(high, max(b1, b2)) - max(low, min(b1, b2)))
            if overlap / max(1.0, min(candidate.width, existing.width)) >= 0.72:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def detect_openings(image: np.ndarray, pixel_scale: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return plausible opening locations, rough types, and visible evidence."""
    height, width = image.shape[:2]
    diagonal = math.hypot(width, height)
    wall_mask = WallMaskBuilder(image).build()
    raw_segments = WallSegmentExtractor(wall_mask).detect()
    segments = WallSegmentBuilder().build_many(raw_segments)
    minimum_gap = max(30, int(round(diagonal * 0.018)))
    candidates = LocalOpeningAnalyzer(
        min_gap_px=minimum_gap,
        max_gap_factor=14.0,
    ).detect(segments, wall_mask=wall_mask)
    candidates = [
        candidate for candidate in candidates
        if candidate.confidence >= 0.52
        and candidate.width <= min(520, int(min(height, width) * 0.22))
    ]
    candidates = _deduplicate(candidates)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edge_distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    wall_points = np.column_stack(np.where(wall_mask > 0))[:, ::-1].astype(np.int32)
    hull = cv2.convexHull(wall_points) if len(wall_points) >= 3 else None

    openings = []
    for candidate in candidates:
        center = (
            (candidate.start[0] + candidate.end[0]) / 2,
            (candidate.start[1] + candidate.end[1]) / 2,
        )
        hull_distance = (
            abs(cv2.pointPolygonTest(hull, center, True))
            if hull is not None else diagonal
        )
        frame_count, frame_regularity = _opening_frame_marks(gray, candidate)
        swing_score = _door_swing_score(edge_distance, candidate)
        thickness = max(1.0, float(candidate.wall_thickness))
        near_outer_wall = hull_distance <= max(2.4 * thickness, diagonal * 0.035)

        if frame_count >= 2 and frame_regularity >= 0.35 and (
            near_outer_wall or frame_count >= 3 or frame_regularity >= 0.75
        ):
            opening_type = "window"
            type_confidence = min(0.86, 0.66 + 0.05 * min(frame_count - 1, 3) + 0.08 * frame_regularity + 0.06 * candidate.confidence)
            reason = f"إطارات متقاطعة {frame_count} · انتظام {frame_regularity:.2f} · قرب من المحيط {hull_distance:.0f}px"
        elif not near_outer_wall and swing_score >= 0.28 and (
            (1.8 <= candidate.width / pixel_scale["pixels_per_unit"] <= 4.6)
            if pixel_scale and pixel_scale.get("pixels_per_unit")
            else (2.2 <= candidate.width / thickness <= 7.5)
        ):
            opening_type = "door"
            type_confidence = min(0.82, 0.57 + 0.12 * candidate.confidence + 0.24 * min(1.0, swing_score))
            reason = f"فتحة داخلية ضمن عرض باب معتاد مع دليل قوس/ضلفة {swing_score:.2f}"
        else:
            opening_type = "opening"
            type_confidence = min(0.62, 0.38 + 0.25 * candidate.confidence)
            reason = f"فجوة جدار محتملة؛ لم تكفِ العلامات لتمييز باب من شباك · إطار {frame_count} · قوس {swing_score:.2f}"

        if opening_type == "window":
            prefix = "W"
        elif opening_type == "door":
            prefix = "D"
        else:
            prefix = "O"
        item = {
            "id": "",
            "type": opening_type,
            "type_label": TYPE_LABELS[opening_type],
            "confidence": round(type_confidence, 2),
            "geometry_confidence": round(float(candidate.confidence), 2),
            "start": [int(round(candidate.start[0])), int(round(candidate.start[1]))],
            "end": [int(round(candidate.end[0])), int(round(candidate.end[1]))],
            "center": [int(round(center[0])), int(round(center[1]))],
            "orientation": candidate.orientation,
            "width_pixels": int(round(candidate.width)),
            "width_display": (
                f"{candidate.width / pixel_scale['pixels_per_unit']:.1f} {pixel_scale['unit']}"
                if pixel_scale and pixel_scale.get("pixels_per_unit") else None
            ),
            "evidence": {
                "frame_marks": frame_count,
                "frame_regularity": round(frame_regularity, 2),
                "door_swing_score": round(swing_score, 2),
                "distance_to_outer_footprint_px": round(float(hull_distance), 1),
                "reason": reason,
            },
        }
        openings.append(item)

    openings.sort(key=lambda item: (item["type"], item["center"][1], item["center"][0]))
    type_index = {"window": 0, "door": 0, "opening": 0}
    for item in openings:
        type_index[item["type"]] += 1
        item["id"] = f"{item['type'][0].upper()}{type_index[item['type']]:02d}"
    return openings


def draw_openings_overlay(image: np.ndarray, openings: list[dict[str, Any]]) -> bytes:
    """Draw a separate, readable openings layer over the original plan."""
    overlay = image.copy()
    colors = {
        "door": (50, 160, 65),
        "window": (220, 100, 25),
        "opening": (0, 145, 225),
    }
    thickness = max(4, min(8, image.shape[1] // 360))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    placed_labels: list[tuple[int, int, int, int]] = []
    for item in openings:
        color = colors[item["type"]]
        start = tuple(item["start"])
        end = tuple(item["end"])
        cv2.line(overlay, start, end, color, thickness, cv2.LINE_AA)
        radius = max(5, thickness // 2 + 2)
        cv2.circle(overlay, start, radius, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, end, radius, color, -1, cv2.LINE_AA)
        cx, cy = item["center"]
        label = item["id"]
        font_scale = 0.62
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        label_width = text_width + 10
        label_height = text_height + baseline + 8
        offsets = [
            (-label_width // 2, -34), (-label_width // 2, 48),
            (-label_width // 2 - 55, -24), (25, -24),
            (-label_width // 2 - 90, 42), (60, 42),
            (-label_width // 2, -68), (-label_width // 2, 78),
        ]
        placements = []
        for dx, dy in offsets:
            x = min(max(3, cx + dx), image.shape[1] - label_width - 3)
            y = min(max(label_height + 3, cy + dy), image.shape[0] - 3)
            rect = (x - 2, y - label_height, x + label_width + 2, y + 3)
            patch = gray[max(0, rect[1]):min(gray.shape[0], rect[3]), max(0, rect[0]):min(gray.shape[1], rect[2])]
            if patch.size == 0:
                continue
            collisions = sum(
                not (rect[2] < old[0] or rect[0] > old[2] or rect[3] < old[1] or rect[1] > old[3])
                for old in placed_labels
            )
            placements.append((float(np.mean(patch)) - collisions * 45, x, y, rect))
        if not placements:
            continue
        _, x, y, rect = max(placements, key=lambda entry: entry[0])
        placed_labels.append(rect)
        cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 255), -1)
        cv2.putText(overlay, label, (x + 5, y - baseline - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)

    success, encoded = cv2.imencode(".png", overlay)
    if not success:
        raise RuntimeError("Could not encode openings overlay")
    return encoded.tobytes()
