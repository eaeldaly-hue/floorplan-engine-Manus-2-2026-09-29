import cv2
import numpy as np


class WallRegionDetector:

    def __init__(self, image):
        self.image = image

    def detect(self):

        gray = cv2.cvtColor(
            self.image,
            cv2.COLOR_BGR2GRAY
        )

        binary = cv2.threshold(
            gray,
            200,
            255,
            cv2.THRESH_BINARY_INV
        )[1]

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )

        cleaned = np.zeros_like(binary)

        for i in range(1, num_labels):

            area = stats[i, cv2.CC_STAT_AREA]

            if area < 20:
                continue

            h = stats[i, cv2.CC_STAT_HEIGHT]
            w = stats[i, cv2.CC_STAT_WIDTH]

            if h < 80 and w < 300 and area < 2000:
                continue

            cleaned[labels == i] = 255

        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (31, 3)
        )

        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 31)
        )

        horizontal = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            horizontal_kernel
        )

        vertical = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            vertical_kernel
        )

        combined = cv2.bitwise_or(
            horizontal,
            vertical
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (7, 7)
        )

        wall_mask = cv2.morphologyEx(
            combined,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        return wall_mask