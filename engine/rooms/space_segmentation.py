from pathlib import Path

import cv2
import numpy as np


class SpaceSegmenter:
    """Find enclosed floor-plan spaces from a binary wall mask.

    ``wall_mask`` uses non-zero pixels for walls. When omitted, the legacy
    line-rendering path is retained for callers that only have vector lines.
    """

    def __init__(
        self,
        image,
        lines=None,
        wall_mask=None,
        gap_close_ratio=0.0625,
        min_area=10000,
        max_image_area_ratio=0.80,
        debug_dir=None,
    ):
        self.image = image
        self.lines = lines or []
        self.wall_mask = wall_mask
        self.gap_close_ratio = float(gap_close_ratio)
        self.min_area = int(min_area)
        self.max_image_area_ratio = float(max_image_area_ratio)
        self.debug_dir = Path(debug_dir) if debug_dir is not None else None

        if not 0 <= self.gap_close_ratio < 0.5:
            raise ValueError("gap_close_ratio must be in [0, 0.5)")
        if self.min_area < 0:
            raise ValueError("min_area must be non-negative")
        if not 0 < self.max_image_area_ratio <= 1:
            raise ValueError("max_image_area_ratio must be in (0, 1]")

    def create_wall_mask(self):
        height, width = self.image.shape[:2]

        if self.wall_mask is not None:
            if self.wall_mask.shape[:2] != (height, width):
                raise ValueError("wall_mask dimensions must match image dimensions")
            wall_mask = np.where(self.wall_mask > 0, 255, 0).astype(np.uint8)
        else:
            wall_mask = np.zeros((height, width), dtype=np.uint8)
            for line in self.lines:
                x1, y1, x2, y2 = map(int, line)
                cv2.line(wall_mask, (x1, y1), (x2, y2), 255, 12)

        # Bridge likely door-sized interruptions. Scale the kernel to the plan
        # dimensions and pad before closing so the exterior border cannot be
        # accidentally sealed merely because the drawing touches the image edge.
        kernel_size = max(3, int(round(min(height, width) * self.gap_close_ratio)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, max(3, min(height, width) - 1))
        if kernel_size % 2 == 0:
            kernel_size -= 1

        padding = kernel_size // 2 + 2
        padded = cv2.copyMakeBorder(
            wall_mask,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size),
        )
        closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
        wall_mask = closed[padding : padding + height, padding : padding + width]

        return wall_mask

    def create_space_mask(self, wall_mask):
        height, width = wall_mask.shape

        # A small exterior frame guarantees that flood fill starts outside the
        # drawing, even when its perimeter walls touch the image boundary.
        padded = cv2.copyMakeBorder(
            wall_mask,
            2,
            2,
            2,
            2,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        flood_image = padded.copy()
        flood_mask = np.zeros((height + 6, width + 6), dtype=np.uint8)
        cv2.floodFill(flood_image, flood_mask, (0, 0), 128)

        spaces = np.where((flood_image == 0) & (padded == 0), 255, 0).astype(np.uint8)
        return spaces[2 : height + 2, 2 : width + 2]

    def detect_spaces(self):
        wall_mask = self.create_wall_mask()
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(self.debug_dir / "wall_mask.png"), wall_mask):
                raise OSError(f"Could not write {self.debug_dir / 'wall_mask.png'}")

        space_mask = self.create_space_mask(wall_mask)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            space_mask,
            connectivity=4,
        )

        spaces = []
        image_area = self.image.shape[0] * self.image.shape[1]

        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])

            if area < self.min_area or area > image_area * self.max_image_area_ratio:
                continue

            component = np.zeros_like(space_mask)
            component[labels == label] = 255
            contours, _ = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            points = [(int(point[0][0]), int(point[0][1])) for point in polygon]

            spaces.append(
                {
                    "id": f"space_{len(spaces) + 1}",
                    "bbox": {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                    },
                    "area_pixels": area,
                    "center": (
                        int(centroids[label][0]),
                        int(centroids[label][1]),
                    ),
                    "polygon": points,
                }
            )

        return spaces
