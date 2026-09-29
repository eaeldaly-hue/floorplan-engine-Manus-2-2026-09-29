import cv2
import numpy as np


class WallGeometryExtractor:

    def __init__(self, wall_mask):
        self.wall_mask = wall_mask

    def extract(self):

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                self.wall_mask,
                connectivity=8
            )
        )

        walls = []

        wall_id = 1

        for i in range(1, num_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Ignore small components
            if area < 1000:
                continue

            # Ignore very small objects
            if w < 40 and h < 40:
                continue

            # Determine dominant orientation
            if w >= h:
                orientation = "horizontal"
                length = w
                thickness = h
            else:
                orientation = "vertical"
                length = h
                thickness = w

            # A wall should be significantly longer than thick
            if length < 100:
                continue

            if thickness <= 0:
                continue

            aspect_ratio = length / thickness

            if aspect_ratio < 2.0:
                continue

            # Calculate solidity
            component_mask = np.zeros_like(
                self.wall_mask
            )

            component_mask[labels == i] = 255

            contours, _ = cv2.findContours(
                component_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            contour = max(
                contours,
                key=cv2.contourArea
            )

            contour_area = cv2.contourArea(
                contour
            )

            if contour_area <= 0:
                continue

            solidity = area / (
                contour_area + 1
            )

            walls.append({
                "id": f"wall_{wall_id}",

                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                },

                "orientation": orientation,

                "length_pixels": int(length),

                "thickness_pixels": int(thickness),

                "aspect_ratio": round(
                    aspect_ratio,
                    2
                ),

                "area_pixels": int(area),

                "solidity": round(
                    float(solidity),
                    3
                ),

                "center": (
                    int(centroids[i][0]),
                    int(centroids[i][1])
                )
            })

            wall_id += 1

        return walls