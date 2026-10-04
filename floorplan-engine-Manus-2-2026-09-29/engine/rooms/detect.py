import cv2
import numpy as np


class RoomDetector:
    """
    Detects enclosed rooms by flood-filling from the image border on a
    closed wall mask. Anything not reached by that flood fill, and not
    part of a wall, is an enclosed space.

    This replaces two things from the previous version:
    - A brute-force O(H^2 * V^2) scan over every combination of 4 walls,
      which reported a "room" for every rectangle-shaped combination of
      walls rather than only real enclosed spaces.
    - A flood fill over raw, unmerged lines with no door handling, which
      merged every room into its neighbours through every doorway.
    """

    def __init__(
        self,
        room_mask,
        min_area_ratio=0.003,
        max_area_ratio=0.85,
    ):
        self.room_mask = room_mask
        self.height, self.width = room_mask.shape
        self.image_area = self.height * self.width
        self.min_area = min_area_ratio * self.image_area
        self.max_area = max_area_ratio * self.image_area

    def _enclosed_mask(self):
        # Add a guaranteed background border: (0, 0) in the source image
        # may itself be occupied by a wall stroke or scan artifact.
        flood_source = cv2.copyMakeBorder(
            self.room_mask,
            1,
            1,
            1,
            1,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        flood_fill_mask = np.zeros(
            (flood_source.shape[0] + 2, flood_source.shape[1] + 2), np.uint8
        )

        cv2.floodFill(flood_source, flood_fill_mask, (0, 0), 128)

        reached_from_outside = flood_source[1:-1, 1:-1] == 128
        enclosed = np.zeros_like(self.room_mask)
        enclosed[~reached_from_outside] = 255
        enclosed[self.room_mask > 0] = 0

        return enclosed

    def detect(self):
        enclosed = self._enclosed_mask()

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(enclosed, connectivity=4)
        )

        rooms = []
        room_id = 1

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            if area < self.min_area or area > self.max_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            component_mask = np.zeros_like(enclosed)
            component_mask[labels == i] = 255

            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)

            points = [(int(p[0][0]), int(p[0][1])) for p in polygon]

            rooms.append({
                "id": f"room_{room_id}",
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "area_pixels": int(area),
                "center": (int(centroids[i][0]), int(centroids[i][1])),
                "polygon": points,
            })

            room_id += 1

        return rooms
