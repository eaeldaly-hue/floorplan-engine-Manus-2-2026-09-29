from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from engine.geometry_v2.orientation import Orientation
from engine.geometry_v2.segments import WallSegment


Point = tuple[float, float]


@dataclass
class RoomPolygon:
    id: str
    polygon: list[Point]
    area: float
    bbox: tuple[int, int, int, int]
    center: Point

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "polygon": [[float(x), float(y)] for x, y in self.polygon],
            "area": float(self.area),
            "bbox": list(self.bbox),
            "center": [float(self.center[0]), float(self.center[1])],
        }


class RoomReconstructor:
    """
    Room reconstruction V2.

    Pipeline:
        wall segments
          -> endpoint snapping
          -> collinear merging
          -> small-gap repair
          -> raster wall network
          -> flood-fill enclosed regions
          -> room polygons

    This is deliberately independent from the existing analyzer.
    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        snap_tolerance: float = 18.0,
        bridge_gap: float = 24.0,
        min_area: float = 3000.0,
        max_area_ratio: float = 0.90,
    ):
        self.height = int(image_shape[0])
        self.width = int(image_shape[1])

        self.snap_tolerance = float(snap_tolerance)
        self.bridge_gap = float(bridge_gap)
        self.min_area = float(min_area)
        self.max_area_ratio = float(max_area_ratio)

    # ---------------------------------------------------------
    # Segment normalization
    # ---------------------------------------------------------

    @staticmethod
    def _normalize(
        segment: WallSegment,
    ) -> tuple[str, float, float, float, float, float] | None:

        orientation = segment.orientation

        if orientation == Orientation.HORIZONTAL:
            x1 = float(min(segment.start[0], segment.end[0]))
            x2 = float(max(segment.start[0], segment.end[0]))
            y = float((segment.start[1] + segment.end[1]) / 2.0)

            thickness = max(
                2.0,
                float(segment.thickness),
            )

            return (
                "h",
                x1,
                y,
                x2,
                y,
                thickness,
            )

        if orientation == Orientation.VERTICAL:
            y1 = float(min(segment.start[1], segment.end[1]))
            y2 = float(max(segment.start[1], segment.end[1]))
            x = float((segment.start[0] + segment.end[0]) / 2.0)

            thickness = max(
                2.0,
                float(segment.thickness),
            )

            return (
                "v",
                x,
                y1,
                x,
                y2,
                thickness,
            )

        return None

    # ---------------------------------------------------------
    # Endpoint snapping
    # ---------------------------------------------------------

    def _snap_segments(
        self,
        segments: list[tuple[str, float, float, float, float, float]],
    ) -> list[tuple[str, float, float, float, float, float]]:

        result = []

        horizontal = [
            s for s in segments if s[0] == "h"
        ]

        vertical = [
            s for s in segments if s[0] == "v"
        ]

        # Horizontal segments snap to nearby horizontal wall levels.
        for segment in horizontal:
            _, x1, y, x2, _, thickness = segment

            nearby_y = [
                other[2]
                for other in horizontal
                if abs(other[2] - y) <= self.snap_tolerance
            ]

            if nearby_y:
                y = float(
                    sum(nearby_y) / len(nearby_y)
                )

            result.append(
                ("h", x1, y, x2, y, thickness)
            )

        # Vertical segments snap to nearby vertical wall levels.
        for segment in vertical:
            _, x, y1, _, y2, thickness = segment

            nearby_x = [
                other[1]
                for other in vertical
                if abs(other[1] - x) <= self.snap_tolerance
            ]

            if nearby_x:
                x = float(
                    sum(nearby_x) / len(nearby_x)
                )

            result.append(
                ("v", x, y1, x, y2, thickness)
            )

        return result

    # ---------------------------------------------------------
    # Collinear merge
    # ---------------------------------------------------------

    def _merge_horizontal(
        self,
        segments,
    ):
        if not segments:
            return []

        segments = sorted(
            segments,
            key=lambda s: (s[2], s[1]),
        )

        merged = []

        current = list(segments[0])

        for segment in segments[1:]:

            _, x1, y, x2, _, thickness = segment

            _, cx1, cy, cx2, _, cthickness = current

            same_level = (
                abs(y - cy)
                <= self.snap_tolerance
            )

            touching = (
                x1
                <= cx2 + self.bridge_gap
            )

            if same_level and touching:
                current[1] = min(cx1, x1)
                current[3] = max(cx2, x2)
                current[2] = (
                    cy + y
                ) / 2.0
                current[4] = current[2]
                current[5] = max(
                    cthickness,
                    thickness,
                )
            else:
                merged.append(
                    tuple(current)
                )
                current = list(segment)

        merged.append(tuple(current))

        return merged

    def _merge_vertical(
        self,
        segments,
    ):
        if not segments:
            return []

        segments = sorted(
            segments,
            key=lambda s: (s[1], s[2]),
        )

        merged = []

        current = list(segments[0])

        for segment in segments[1:]:

            _, x, y1, _, y2, thickness = segment

            _, cx, cy1, _, cy2, cthickness = current

            same_level = (
                abs(x - cx)
                <= self.snap_tolerance
            )

            touching = (
                y1
                <= cy2 + self.bridge_gap
            )

            if same_level and touching:
                current[1] = (
                    cx + x
                ) / 2.0
                current[3] = current[1]
                current[2] = min(
                    cy1,
                    y1,
                )
                current[4] = max(
                    cy2,
                    y2,
                )
                current[5] = max(
                    cthickness,
                    thickness,
                )
            else:
                merged.append(
                    tuple(current)
                )
                current = list(segment)

        merged.append(tuple(current))

        return merged

    def _merge_collinear(
        self,
        segments,
    ):
        horizontal = [
            s for s in segments
            if s[0] == "h"
        ]

        vertical = [
            s for s in segments
            if s[0] == "v"
        ]

        return (
            self._merge_horizontal(horizontal)
            + self._merge_vertical(vertical)
        )

    # ---------------------------------------------------------
    # Wall mask
    # ---------------------------------------------------------

    def _build_wall_mask(
        self,
        segments,
    ):
        mask = np.zeros(
            (self.height, self.width),
            dtype=np.uint8,
        )

        for segment in segments:

            orientation, x1, y1, x2, y2, thickness = segment

            # Slightly thicker rasterization improves topological closure.
            draw_thickness = max(
                3,
                int(round(thickness)),
            )

            cv2.line(
                mask,
                (
                    int(round(x1)),
                    int(round(y1)),
                ),
                (
                    int(round(x2)),
                    int(round(y2)),
                ),
                255,
                draw_thickness,
                cv2.LINE_AA,
            )

        # Bridge tiny diagonal/corner cracks.
        kernel_size = max(
            3,
            int(round(self.bridge_gap / 6)) * 2 + 1,
        )

        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        return mask

    # ---------------------------------------------------------
    # Region extraction
    # ---------------------------------------------------------

    def _extract_regions(
        self,
        wall_mask,
    ):

        flood = wall_mask.copy()

        flood_mask = np.zeros(
            (
                self.height + 2,
                self.width + 2,
            ),
            dtype=np.uint8,
        )

        # Flood the exterior.
        cv2.floodFill(
            flood,
            flood_mask,
            (0, 0),
            128,
        )

        enclosed = np.where(
            flood == 0,
            255,
            0,
        ).astype(np.uint8)

        # Clean tiny artifacts.
        kernel = np.ones(
            (5, 5),
            dtype=np.uint8,
        )

        enclosed = cv2.morphologyEx(
            enclosed,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        contours, _ = cv2.findContours(
            enclosed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        rooms = []

        max_area = (
            self.width
            * self.height
            * self.max_area_ratio
        )

        for contour in contours:

            area = float(
                cv2.contourArea(contour)
            )

            if area < self.min_area:
                continue

            if area > max_area:
                continue

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            epsilon = max(
                3.0,
                perimeter * 0.004,
            )

            polygon = cv2.approxPolyDP(
                contour,
                epsilon,
                True,
            )

            points = [
                (
                    float(point[0][0]),
                    float(point[0][1]),
                )
                for point in polygon
            ]

            if len(points) < 3:
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            moments = cv2.moments(contour)

            if moments["m00"]:

                cx = (
                    moments["m10"]
                    / moments["m00"]
                )

                cy = (
                    moments["m01"]
                    / moments["m00"]
                )

            else:

                cx = x + w / 2.0
                cy = y + h / 2.0

            rooms.append(
                RoomPolygon(
                    id="",
                    polygon=points,
                    area=area,
                    bbox=(x, y, w, h),
                    center=(cx, cy),
                )
            )

        # Largest enclosed areas first.
        rooms.sort(
            key=lambda room: room.area,
            reverse=True,
        )

        for index, room in enumerate(
            rooms,
            start=1,
        ):
            room.id = f"room_{index}"

        return rooms

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def reconstruct(
        self,
        segments: Iterable[WallSegment],
    ):

        normalized = []

        for segment in segments:

            item = self._normalize(segment)

            if item is not None:
                normalized.append(item)

        if not normalized:
            return []

        snapped = self._snap_segments(
            normalized
        )

        merged = self._merge_collinear(
            snapped
        )

        wall_mask = self._build_wall_mask(
            merged
        )

        return self._extract_regions(
            wall_mask
        )


# -------------------------------------------------------------
# Visualization
# -------------------------------------------------------------

def draw_room_overlay(
    image: np.ndarray,
    rooms: list[RoomPolygon],
) -> np.ndarray:

    overlay = image.copy()

    for index, room in enumerate(
        rooms,
        start=1,
    ):

        points = np.array(
            [
                [
                    int(round(x)),
                    int(round(y)),
                ]
                for x, y in room.polygon
            ],
            dtype=np.int32,
        )

        if len(points) < 3:
            continue

        color = (
            int((index * 67) % 255),
            int((index * 113) % 255),
            int((index * 173) % 255),
        )

        layer = overlay.copy()

        cv2.fillPoly(
            layer,
            [points],
            color,
        )

        overlay = cv2.addWeighted(
            layer,
            0.25,
            overlay,
            0.75,
            0,
        )

        cv2.polylines(
            overlay,
            [points],
            True,
            color,
            5,
            cv2.LINE_AA,
        )

        cx, cy = room.center

        cv2.circle(
            overlay,
            (
                int(round(cx)),
                int(round(cy)),
            ),
            9,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            overlay,
            room.id.upper(),
            (
                int(round(cx)) + 14,
                int(round(cy)),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    return overlay


def draw_wall_debug(
    image: np.ndarray,
    segments: Iterable[WallSegment],
) -> np.ndarray:

    overlay = image.copy()

    for segment in segments:

        normalized = RoomReconstructor._normalize(
            segment
        )

        if normalized is None:
            continue

        _, x1, y1, x2, y2, thickness = normalized

        cv2.line(
            overlay,
            (
                int(round(x1)),
                int(round(y1)),
            ),
            (
                int(round(x2)),
                int(round(y2)),
            ),
            (255, 0, 255),
            max(
                2,
                int(round(thickness)),
            ),
        )

    return overlay
