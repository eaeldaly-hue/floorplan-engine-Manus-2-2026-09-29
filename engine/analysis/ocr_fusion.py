"""OCR-to-geometry fusion helpers for floor-plan analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from .ocr import OCRBox


@dataclass(frozen=True)
class OCRTextGroup:
    """A logical text label assembled from one or more OCR boxes."""

    text: str
    boxes: tuple[OCRBox, ...]
    confidence: float

    @property
    def x(self) -> int:
        return min(box.x for box in self.boxes)

    @property
    def y(self) -> int:
        return min(box.y for box in self.boxes)

    @property
    def right(self) -> int:
        return max(box.x + box.width for box in self.boxes)

    @property
    def bottom(self) -> int:
        return max(box.y + box.height for box in self.boxes)

    @property
    def width(self) -> int:
        return self.right - self.x

    @property
    def height(self) -> int:
        return self.bottom - self.y

    @property
    def center(self) -> tuple[int, int]:
        return (
            round((self.x + self.right) / 2),
            round((self.y + self.bottom) / 2),
        )


@dataclass(frozen=True)
class RoomLabelEvidence:
    """A room label associated with a geometric space."""

    text: str
    confidence: float
    center: tuple[int, int]
    box: dict[str, int]
    space_id: str | None
    association_score: float


_NOISE_EXACT = {
    "",
    "x",
    "xx",
    "|",
    "_",
    "-",
    "~",
    "aa",
    "aaa",
}


def normalize_ocr_text(text: str) -> str:
    """Normalize OCR text without changing its meaning."""
    text = " ".join(text.strip().split())
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("′", "'").replace("″", '"')
    return text


def is_ocr_noise(text: str) -> bool:
    """Reject obvious OCR artifacts."""
    normalized = normalize_ocr_text(text)
    if not normalized:
        return True

    if normalized.lower() in _NOISE_EXACT:
        return True

    compact = re.sub(r"[^A-Za-z0-9\u0600-\u06FF]", "", normalized)

    if len(compact) <= 1:
        return True

    if re.fullmatch(r"(.)\1{2,}", compact):
        return True

    return False


def _horizontal_compatible(a: OCRBox, b: OCRBox) -> bool:
    """Return whether two words plausibly belong to the same text line."""
    a_cy = a.y + a.height / 2
    b_cy = b.y + b.height / 2

    vertical_delta = abs(a_cy - b_cy)
    allowed_vertical = max(a.height, b.height) * 0.75

    gap = max(b.x, a.x) - min(a.x + a.width, b.x + b.width)

    return (
        vertical_delta <= allowed_vertical
        and gap <= max(45, max(a.height, b.height) * 2.5)
    )


def group_room_labels(boxes: tuple[OCRBox, ...] | list[OCRBox]) -> list[OCRTextGroup]:
    """
    Group adjacent room-label OCR words.

    Examples:
        MASTER + BEDROOM -> MASTER BEDROOM
        LIVING + ROOM    -> LIVING ROOM
        LAUNDRY + ROOM   -> LAUNDRY ROOM
    """
    candidates = [
        box
        for box in boxes
        if box.kind == "ROOM_LABEL"
        and not is_ocr_noise(box.text)
    ]

    candidates.sort(key=lambda box: (box.y, box.x))

    groups: list[list[OCRBox]] = []

    for box in candidates:
        placed = False

        for group in groups:
            if any(_horizontal_compatible(box, existing) for existing in group):
                group.append(box)
                placed = True
                break

        if not placed:
            groups.append([box])

    result: list[OCRTextGroup] = []

    for group in groups:
        group.sort(key=lambda box: box.x)

        text = normalize_ocr_text(
            " ".join(box.text for box in group)
        )

        confidence = sum(box.confidence for box in group) / len(group)

        result.append(
            OCRTextGroup(
                text=text,
                boxes=tuple(group),
                confidence=round(confidence, 1),
            )
        )

    result.sort(key=lambda group: (group.y, group.x))
    return result


def group_dimensions(
    boxes: tuple[OCRBox, ...] | list[OCRBox],
) -> list[OCRTextGroup]:
    """Group nearby dimension tokens while filtering obvious OCR noise."""
    candidates = [
        box
        for box in boxes
        if box.kind == "DIMENSION"
        and not is_ocr_noise(box.text)
    ]

    candidates.sort(key=lambda box: (box.y, box.x))

    groups: list[list[OCRBox]] = []

    for box in candidates:
        placed = False

        for group in groups:
            if any(_horizontal_compatible(box, existing) for existing in group):
                group.append(box)
                placed = True
                break

        if not placed:
            groups.append([box])

    result: list[OCRTextGroup] = []

    for group in groups:
        group.sort(key=lambda box: box.x)
        text = normalize_ocr_text(" ".join(box.text for box in group))
        confidence = sum(box.confidence for box in group) / len(group)

        result.append(
            OCRTextGroup(
                text=text,
                boxes=tuple(group),
                confidence=round(confidence, 1),
            )
        )

    return result


def _point_in_polygon(
    point: tuple[int, int],
    polygon: list[tuple[int, int]],
) -> bool:
    if len(polygon) < 3:
        return False

    contour = np.asarray(polygon, dtype=np.int32)

    return (
        cv2.pointPolygonTest(
            contour,
            (float(point[0]), float(point[1])),
            False,
        )
        >= 0
    )


def associate_room_labels(
    groups: list[OCRTextGroup],
    spaces: list[dict],
) -> list[RoomLabelEvidence]:
    """
    Associate grouped OCR labels with detected geometric spaces.

    OCR never creates a room. It only supplies evidence for an existing space.
    """
    evidence: list[RoomLabelEvidence] = []

    for group in groups:
        center = group.center

        containing = [
            space
            for space in spaces
            if _point_in_polygon(center, space["polygon"])
        ]

        if containing:
            chosen = min(
                containing,
                key=lambda space: space["area_pixels"],
            )

            evidence.append(
                RoomLabelEvidence(
                    text=group.text,
                    confidence=group.confidence,
                    center=center,
                    box={
                        "x": group.x,
                        "y": group.y,
                        "width": group.width,
                        "height": group.height,
                    },
                    space_id=chosen["id"],
                    association_score=1.0,
                )
            )
            continue

        # Allow a small proximity fallback for labels sitting directly
        # on a wall boundary or just outside a segmented polygon.
        best_space = None
        best_distance = float("inf")

        for space in spaces:
            x, y, width, height = (
                space["bbox"]["x"],
                space["bbox"]["y"],
                space["bbox"]["width"],
                space["bbox"]["height"],
            )

            closest_x = min(max(center[0], x), x + width)
            closest_y = min(max(center[1], y), y + height)

            distance = float(
                np.hypot(
                    center[0] - closest_x,
                    center[1] - closest_y,
                )
            )

            if distance < best_distance:
                best_distance = distance
                best_space = space

        if best_space is not None and best_distance <= 35:
            score = max(0.0, 1.0 - best_distance / 35.0)

            evidence.append(
                RoomLabelEvidence(
                    text=group.text,
                    confidence=group.confidence,
                    center=center,
                    box={
                        "x": group.x,
                        "y": group.y,
                        "width": group.width,
                        "height": group.height,
                    },
                    space_id=best_space["id"],
                    association_score=round(score, 3),
                )
            )
        else:
            evidence.append(
                RoomLabelEvidence(
                    text=group.text,
                    confidence=group.confidence,
                    center=center,
                    box={
                        "x": group.x,
                        "y": group.y,
                        "width": group.width,
                        "height": group.height,
                    },
                    space_id=None,
                    association_score=0.0,
                )
            )

    return evidence


__all__ = [
    "OCRTextGroup",
    "RoomLabelEvidence",
    "associate_room_labels",
    "group_dimensions",
    "group_room_labels",
    "is_ocr_noise",
    "normalize_ocr_text",
]
