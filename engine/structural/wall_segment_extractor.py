import cv2
import numpy as np


class WallSegmentExtractor:

    def __init__(
        self,
        centerline,
        distance,
        min_segment_length=40
    ):
        self.centerline = centerline
        self.distance = distance
        self.min_segment_length = min_segment_length

    def extract(self):

        binary = (
            self.centerline > 0
        ).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5)
        )

        connected = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                connected,
                connectivity=8
            )
        )

        segments = []

        for i in range(1, num_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if area < 20:
                continue

            component_mask = (
                labels == i
            )

            ys, xs = np.where(
                component_mask
            )

            if len(xs) == 0:
                continue

            distances = self.distance[
                component_mask
            ]

            max_distance = float(
                distances.max()
            )

            median_distance = float(
                np.median(distances)
            )

            if (
                w >= self.min_segment_length
                and w >= h * 3
            ):

                center_y = int(
                    np.median(ys)
                )

                x_start = int(xs.min())
                x_end = int(xs.max())

                segments.append({
                    "orientation": "horizontal",
                    "start": (
                        x_start,
                        center_y
                    ),
                    "end": (
                        x_end,
                        center_y
                    ),
                    "length": (
                        x_end - x_start + 1
                    ),
                    "half_thickness": round(
                        max_distance,
                        2
                    ),
                    "wall_thickness": round(
                        max_distance * 2,
                        2
                    ),
                    "median_half_thickness": round(
                        median_distance,
                        2
                    ),
                    "area": int(area)
                })

            elif (
                h >= self.min_segment_length
                and h >= w * 3
            ):

                center_x = int(
                    np.median(xs)
                )

                y_start = int(ys.min())
                y_end = int(ys.max())

                segments.append({
                    "orientation": "vertical",
                    "start": (
                        center_x,
                        y_start
                    ),
                    "end": (
                        center_x,
                        y_end
                    ),
                    "length": (
                        y_end - y_start + 1
                    ),
                    "half_thickness": round(
                        max_distance,
                        2
                    ),
                    "wall_thickness": round(
                        max_distance * 2,
                        2
                    ),
                    "median_half_thickness": round(
                        median_distance,
                        2
                    ),
                    "area": int(area)
                })

        return segments
