import cv2
import numpy as np


class WallCenterlineExtractor:

    def __init__(
        self,
        wall_mask,
        min_distance=3
    ):
        self.mask = wall_mask
        self.min_distance = min_distance

    def extract(self):

        binary = (
            self.mask > 0
        ).astype(
            np.uint8
        )

        distance = cv2.distanceTransform(
            binary,
            cv2.DIST_L2,
            5
        )

        # Local maximum filter
        kernel = np.ones(
            (7, 7),
            np.uint8
        )

        local_max = cv2.dilate(
            distance,
            kernel
        )

        # Keep only pixels whose distance
        # is equal to the local maximum.
        centerline = np.zeros_like(
            binary
        )

        centerline[
            (
                distance >= self.min_distance
            ) &
            (
                distance >= local_max - 0.01
            )
        ] = 255

        return centerline, distance