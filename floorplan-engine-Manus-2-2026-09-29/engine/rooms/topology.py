from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from engine.geometry_v2.groups import WallGroupBuilder
from engine.geometry_v2.orientation import Orientation
from engine.geometry_v2.segments import WallSegment


@dataclass
class TopologyRegion:
    id: str
    polygon: list[tuple[float, float]]
    area: float
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]


class WallTopologyBuilder:
    """
    V5 experimental wall-topology builder.

    The important difference from V2/V3:
    wall groups already know where long gaps exist.
    For topology reconstruction, those gaps are bridged virtually.
    Openings remain separate semantic data and are not destroyed.
    """

    def __init__(
        self,
        image_shape: tuple[int, int],
        min_region_area: float = 5000.0,
        max_region_area_ratio: float = 0.90,
        bridge_padding: int = 2,
    ):
        self.height = int(image_shape[0])
        self.width = int(image_shape[1])
        self.min_region_area = float(min_region_area)
        self.max_region_area_ratio = float(max_region_area_ratio)
        self.bridge_padding = int(bridge_padding)

    @staticmethod
    def _line_from_segment(segment: WallSegment):
        if segment.orientation == Orientation.HORIZONTAL:
            x1 = float(min(segment.start[0], segment.end[0]))
            x2 = float(max(segment.start[0], segment.end[0]))
            y = float((segment.start[1] + segment.end[1]) / 2.0)
            return "h", x1, y, x2, y, max(2.0, float(segment.thickness))

        if segment.orientation == Orientation.VERTICAL:
            y1 = float(min(segment.start[1], segment.end[1]))
            y2 = float(max(segment.start[1], segment.end[1]))
            x = float((segment.start[0] + segment.end[0]) / 2.0)
            return "v", x, y1, x, y2, max(2.0, float(segment.thickness))

        return None

    def _draw_line(
        self,
        mask: np.ndarray,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        thickness: float,
    ):
        cv2.line(
            mask,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            255,
            max(3, int(round(thickness))),
            cv2.LINE_AA,
        )

    def build_mask(self, segments: Iterable[WallSegment]):
        segments = list(segments)

        groups = WallGroupBuilder().build(segments)

        mask = np.zeros(
            (self.height, self.width),
            dtype=np.uint8,
        )

        # Draw original wall geometry.
        for segment in segments:
            normalized = self._line_from_segment(segment)

            if normalized is None:
                continue

            _, x1, y1, x2, y2, thickness = normalized

            self._draw_line(
                mask,
                x1,
                y1,
                x2,
                y2,
                thickness,
            )

        # Virtually close every group gap.
        # This is topology-only; semantic openings are handled elsewhere.
        bridges = []

        for group in groups:
            for gap in group.gaps:
                x1, y1 = gap.start
                x2, y2 = gap.end

                bridges.append(
                    (
                        group.orientation,
                        x1,
                        y1,
                        x2,
                        y2,
                        max(2.0, float(group.thickness)),
                    )
                )

                self._draw_line(
                    mask,
                    x1,
                    y1,
                    x2,
                    y2,
                    float(group.thickness) + self.bridge_padding,
                )

        return mask, groups, bridges

    def extract_regions(self, topology_mask: np.ndarray):
        flood = topology_mask.copy()

        flood_mask = np.zeros(
            (self.height + 2, self.width + 2),
            dtype=np.uint8,
        )

        # Exterior is connected to the image corner.
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

        # Remove tiny topology noise.
        kernel = np.ones((5, 5), dtype=np.uint8)

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

        regions = []

        max_area = (
            self.width
            * self.height
            * self.max_region_area_ratio
        )

        for contour in contours:
            area = float(cv2.contourArea(contour))

            if area < self.min_region_area:
                continue

            if area > max_area:
                continue

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            epsilon = max(
                3.0,
                perimeter * 0.003,
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
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
            else:
                cx = x + w / 2.0
                cy = y + h / 2.0

            regions.append(
                TopologyRegion(
                    id="",
                    polygon=points,
                    area=area,
                    bbox=(x, y, w, h),
                    center=(cx, cy),
                )
            )

        regions.sort(
            key=lambda region: region.area,
            reverse=True,
        )

        for index, region in enumerate(
            regions,
            start=1,
        ):
            region.id = f"topology_room_{index}"

        return regions

    def build(self, segments):
        mask, groups, bridges = self.build_mask(
            segments
        )

        regions = self.extract_regions(mask)

        return {
            "regions": regions,
            "mask": mask,
            "groups": groups,
            "bridges": bridges,
        }


def draw_topology_overlay(
    image: np.ndarray,
    result: dict,
):
    overlay = image.copy()

    regions = result["regions"]
    bridges = result["bridges"]

    # Show virtual bridge segments.
    for bridge in bridges:
        _, x1, y1, x2, y2, thickness = bridge

        cv2.line(
            overlay,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (255, 0, 255),
            max(3, int(round(thickness))),
            cv2.LINE_AA,
        )

    # Show reconstructed regions.
    for index, region in enumerate(
        regions,
        start=1,
    ):
        points = np.array(
            [
                [
                    int(round(x)),
                    int(round(y)),
                ]
                for x, y in region.polygon
            ],
            dtype=np.int32,
        )

        if len(points) < 3:
            continue

        value = index * 53

        color = (
            int((value * 3) % 255),
            int((value * 5) % 255),
            int((value * 7) % 255),
        )

        layer = overlay.copy()

        cv2.fillPoly(
            layer,
            [points],
            color,
        )

        overlay = cv2.addWeighted(
            layer,
            0.20,
            overlay,
            0.80,
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

        cx, cy = region.center

        cv2.circle(
            overlay,
            (
                int(round(cx)),
                int(round(cy)),
            ),
            8,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            overlay,
            region.id.upper(),
            (
                int(round(cx)) + 12,
                int(round(cy)),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return overlay


# ---------------------------------------------------------------------------
# V5.1 — Internal subdivision
# ---------------------------------------------------------------------------

def subdivide_topology_regions(
    image: np.ndarray,
    topology_result: dict,
    min_room_area: float = 12000.0,
    large_region_area: float = 180000.0,
):
    """
    Split large topology regions using the original topology wall mask.

    This is intentionally experimental and does not modify the existing
    analyzer room detector.
    """

    topology_mask = topology_result["mask"]
    regions = topology_result["regions"]

    final_regions = []

    for region in regions:

        if region.area < min_room_area:
            continue

        # Small/normal regions are already useful room candidates.
        if region.area < large_region_area:
            final_regions.append(region)
            continue

        x, y, w, h = region.bbox

        if w < 100 or h < 100:
            final_regions.append(region)
            continue

        # Crop the topology region.
        crop_mask = topology_mask[
            y : y + h,
            x : x + w,
        ]

        crop_image = image[
            y : y + h,
            x : x + w,
        ]

        # Look for strong internal wall structures.
        #
        # A modest morphological opening removes isolated pixels while
        # preserving long horizontal/vertical wall structures.
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (31, 3),
        )

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 31),
        )

        horizontal = cv2.morphologyEx(
            crop_mask,
            cv2.MORPH_OPEN,
            horizontal_kernel,
        )

        vertical = cv2.morphologyEx(
            crop_mask,
            cv2.MORPH_OPEN,
            vertical_kernel,
        )

        internal_walls = cv2.bitwise_or(
            horizontal,
            vertical,
        )

        # Reconstruct the local region boundary and internal wall network.
        local = np.zeros_like(crop_mask)

        cv2.fillPoly(
            local,
            [
                np.array(
                    [
                        [
                            int(round(px - x)),
                            int(round(py - y)),
                        ]
                        for px, py in region.polygon
                    ],
                    dtype=np.int32,
                )
            ],
            255,
        )

        local_walls = cv2.bitwise_and(
            internal_walls,
            local,
        )

        # Remove the outer boundary influence.
        border = np.zeros_like(local)

        cv2.polylines(
            border,
            [
                np.array(
                    [
                        [
                            int(round(px - x)),
                            int(round(py - y)),
                        ]
                        for px, py in region.polygon
                    ],
                    dtype=np.int32,
                )
            ],
            True,
            255,
            15,
            cv2.LINE_AA,
        )

        local_walls = cv2.bitwise_and(
            local_walls,
            cv2.bitwise_not(border),
        )

        # Build a local enclosure mask.
        enclosure = cv2.bitwise_or(
            local_walls,
            cv2.bitwise_not(local),
        )

        flood = enclosure.copy()

        flood_mask = np.zeros(
            (
                enclosure.shape[0] + 2,
                enclosure.shape[1] + 2,
            ),
            dtype=np.uint8,
        )

        # Try exterior points around the crop.
        for seed in (
            (0, 0),
            (enclosure.shape[1] - 1, 0),
            (0, enclosure.shape[0] - 1),
            (
                enclosure.shape[1] - 1,
                enclosure.shape[0] - 1,
            ),
        ):
            if (
                0 <= seed[0] < enclosure.shape[1]
                and 0 <= seed[1] < enclosure.shape[0]
            ):
                cv2.floodFill(
                    flood,
                    flood_mask,
                    seed,
                    128,
                )

        enclosed = np.where(
            flood == 0,
            255,
            0,
        ).astype(np.uint8)

        enclosed = cv2.morphologyEx(
            enclosed,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )

        contours, _ = cv2.findContours(
            enclosed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        subregions = []

        for contour in contours:

            area = float(
                cv2.contourArea(contour)
            )

            if area < min_room_area:
                continue

            if area >= region.area * 0.98:
                continue

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            polygon = cv2.approxPolyDP(
                contour,
                max(3.0, perimeter * 0.003),
                True,
            )

            points = [
                (
                    float(p[0][0] + x),
                    float(p[0][1] + y),
                )
                for p in polygon
            ]

            if len(points) < 3:
                continue

            bx, by, bw, bh = cv2.boundingRect(
                contour
            )

            moments = cv2.moments(contour)

            if moments["m00"]:
                cx = (
                    moments["m10"] /
                    moments["m00"]
                ) + x

                cy = (
                    moments["m01"] /
                    moments["m00"]
                ) + y

            else:
                cx = bx + bw / 2 + x
                cy = by + bh / 2 + y

            subregions.append(
                TopologyRegion(
                    id="",
                    polygon=points,
                    area=area,
                    bbox=(
                        bx + x,
                        by + y,
                        bw,
                        bh,
                    ),
                    center=(cx, cy),
                )
            )

        # Only replace the original region if subdivision actually
        # produced multiple meaningful regions.
        if len(subregions) >= 2:
            final_regions.extend(subregions)
        else:
            final_regions.append(region)

    final_regions.sort(
        key=lambda item: item.area,
        reverse=True,
    )

    for index, region in enumerate(
        final_regions,
        start=1,
    ):
        region.id = f"room_{index}"

    return final_regions


def draw_subdivision_overlay(
    image: np.ndarray,
    regions: list[TopologyRegion],
):
    overlay = image.copy()

    for index, region in enumerate(
        regions,
        start=1,
    ):
        points = np.array(
            [
                [
                    int(round(x)),
                    int(round(y)),
                ]
                for x, y in region.polygon
            ],
            dtype=np.int32,
        )

        if len(points) < 3:
            continue

        value = index * 47

        color = (
            int((value * 3) % 255),
            int((value * 5) % 255),
            int((value * 7) % 255),
        )

        layer = overlay.copy()

        cv2.fillPoly(
            layer,
            [points],
            color,
        )

        overlay = cv2.addWeighted(
            layer,
            0.18,
            overlay,
            0.82,
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

        cx, cy = region.center

        cv2.circle(
            overlay,
            (
                int(round(cx)),
                int(round(cy)),
            ),
            8,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            overlay,
            region.id.upper(),
            (
                int(round(cx)) + 12,
                int(round(cy)),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

    return overlay
