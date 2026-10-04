from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from engine.geometry_v2.geometry import GeometryEngine
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
            "polygon": [
                [float(x), float(y)]
                for x, y in self.polygon
            ],
            "area": float(self.area),
            "bbox": list(self.bbox),
            "center": [
                float(self.center[0]),
                float(self.center[1]),
            ],
        }


class RoomReconstructor:
    """
    Experimental room reconstruction engine.

    Converts normalized wall segments into candidate room polygons.

    V1 intentionally focuses on:
        - horizontal walls
        - vertical walls
        - wall intersections
        - closed rectangular / orthogonal regions

    It does NOT modify the existing analyzer pipeline.
    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        snap_tolerance: float = 12.0,
        min_area: float = 2000.0,
        max_area_ratio: float = 0.95,
    ):
        self.height = int(image_shape[0])
        self.width = int(image_shape[1])

        self.snap_tolerance = float(snap_tolerance)
        self.min_area = float(min_area)
        self.max_area_ratio = float(max_area_ratio)

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalized_segment(
        segment: WallSegment,
    ) -> tuple[float, float, float, float] | None:

        if segment.orientation == Orientation.HORIZONTAL:
            x1 = min(segment.start[0], segment.end[0])
            x2 = max(segment.start[0], segment.end[0])
            y = (segment.start[1] + segment.end[1]) / 2.0

            return x1, y, x2, y

        if segment.orientation == Orientation.VERTICAL:
            y1 = min(segment.start[1], segment.end[1])
            y2 = max(segment.start[1], segment.end[1])
            x = (segment.start[0] + segment.end[0]) / 2.0

            return x, y1, x, y2

        return None

    @staticmethod
    def _point_key(
        point: Point,
        tolerance: float,
    ) -> tuple[int, int]:

        if tolerance <= 0:
            tolerance = 1.0

        return (
            round(point[0] / tolerance),
            round(point[1] / tolerance),
        )

    # ------------------------------------------------------------------
    # Intersection extraction
    # ------------------------------------------------------------------

    def _intersections(
        self,
        segments: list[WallSegment],
    ) -> list[Point]:

        horizontal = []
        vertical = []

        for segment in segments:

            normalized = self._normalized_segment(segment)

            if normalized is None:
                continue

            x1, y1, x2, y2 = normalized

            if segment.orientation == Orientation.HORIZONTAL:
                horizontal.append((x1, y1, x2))

            elif segment.orientation == Orientation.VERTICAL:
                vertical.append((x1, y1, y2))

        points: list[Point] = []

        for hx1, hy, hx2 in horizontal:

            for vx, vy1, vy2 in vertical:

                if (
                    vx >= hx1 - self.snap_tolerance
                    and vx <= hx2 + self.snap_tolerance
                    and hy >= vy1 - self.snap_tolerance
                    and hy <= vy2 + self.snap_tolerance
                ):
                    points.append((vx, hy))

        return points

    # ------------------------------------------------------------------
    # Raster wall graph
    # ------------------------------------------------------------------

    def _build_wall_mask(
        self,
        segments: list[WallSegment],
    ) -> np.ndarray:

        mask = np.zeros(
            (self.height, self.width),
            dtype=np.uint8,
        )

        for segment in segments:

            normalized = self._normalized_segment(segment)

            if normalized is None:
                continue

            x1, y1, x2, y2 = normalized

            thickness = max(
                2,
                int(round(segment.thickness))
                if segment.thickness > 0
                else 4,
            )

            cv2.line(
                mask,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                255,
                thickness=thickness,
            )

        return mask

    # ------------------------------------------------------------------
    # Closed-region extraction
    # ------------------------------------------------------------------

    def _extract_regions(
        self,
        wall_mask: np.ndarray,
    ) -> list[RoomPolygon]:

        kernel = np.ones((3, 3), dtype=np.uint8)

        closed_mask = cv2.morphologyEx(
            wall_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

        flood = closed_mask.copy()

        flood_mask = np.zeros(
            (
                self.height + 2,
                self.width + 2,
            ),
            dtype=np.uint8,
        )

        cv2.floodFill(
            flood,
            flood_mask,
            (0, 0),
            128,
        )

        # Pixels still zero are enclosed by walls.
        enclosed = np.where(
            flood == 0,
            255,
            0,
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            enclosed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        rooms: list[RoomPolygon] = []

        max_area = (
            self.width
            * self.height
            * self.max_area_ratio
        )

        for contour in contours:

            area = float(cv2.contourArea(contour))

            if area < self.min_area:
                continue

            if area > max_area:
                continue

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            epsilon = max(
                2.0,
                perimeter * 0.005,
            )

            polygon = cv2.approxPolyDP(
                contour,
                epsilon,
                True,
            )

            points: list[Point] = []

            for point in polygon:
                x, y = point[0]

                points.append(
                    (
                        float(x),
                        float(y),
                    )
                )

            if len(points) < 3:
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            moments = cv2.moments(contour)

            if moments["m00"] != 0:

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
                    id=f"room_{len(rooms) + 1}",
                    polygon=points,
                    area=area,
                    bbox=(x, y, w, h),
                    center=(cx, cy),
                )
            )

        rooms.sort(
            key=lambda room: room.area,
            reverse=True,
        )

        # Re-number after sorting.
        for index, room in enumerate(
            rooms,
            start=1,
        ):
            room.id = f"room_{index}"

        return rooms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        segments: Iterable[WallSegment],
    ) -> list[RoomPolygon]:

        segments = list(segments)

        if not segments:
            return []

        # Intersections are calculated explicitly so the
        # reconstruction layer has geometric junction information.
        intersections = self._intersections(
            segments
        )

        wall_mask = self._build_wall_mask(
            segments
        )

        rooms = self._extract_regions(
            wall_mask
        )

        return rooms


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
                [int(x), int(y)]
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
            0.22,
            overlay,
            0.78,
            0,
        )

        cv2.polylines(
            overlay,
            [points],
            True,
            color,
            5,
        )

        cx, cy = room.center

        cv2.circle(
            overlay,
            (int(cx), int(cy)),
            8,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            overlay,
            room.id.upper(),
            (
                int(cx) + 12,
                int(cy),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    return overlay
