import cv2
import numpy as np


class WallBandDetector:

    def __init__(
        self,
        wall_mask,
        min_length=100,
        min_thickness=5,
        max_thickness=80
    ):
        self.wall_mask = wall_mask
        self.min_length = min_length
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness

    def detect_horizontal(self):

        mask = self.wall_mask
        height, width = mask.shape

        row_counts = np.sum(
            mask > 0,
            axis=1
        )

        bands = []

        start = None

        for y in range(height):

            # A row belongs to a possible wall
            # if it contains a sufficiently long
            # continuous structural signal.
            if row_counts[y] >= self.min_length:

                if start is None:
                    start = y

            else:

                if start is not None:

                    end = y - 1
                    thickness = end - start + 1

                    if (
                        self.min_thickness
                        <= thickness
                        <= self.max_thickness
                    ):

                        segment_mask = mask[
                            start:end + 1,
                            :
                        ]

                        columns = np.where(
                            np.any(
                                segment_mask > 0,
                                axis=0
                            )
                        )[0]

                        if len(columns) > 0:

                            x1 = int(columns.min())
                            x2 = int(columns.max())

                            if x2 - x1 >= self.min_length:

                                bands.append({
                                    "orientation": "horizontal",
                                    "start": (
                                        x1,
                                        start
                                    ),
                                    "end": (
                                        x2,
                                        end
                                    ),
                                    "length": x2 - x1,
                                    "thickness": thickness
                                })

                    start = None

        return bands

    def detect_vertical(self):

        mask = self.wall_mask
        height, width = mask.shape

        column_counts = np.sum(
            mask > 0,
            axis=0
        )

        bands = []

        start = None

        for x in range(width):

            if column_counts[x] >= self.min_length:

                if start is None:
                    start = x

            else:

                if start is not None:

                    end = x - 1
                    thickness = end - start + 1

                    if (
                        self.min_thickness
                        <= thickness
                        <= self.max_thickness
                    ):

                        segment_mask = mask[
                            :,
                            start:end + 1
                        ]

                        rows = np.where(
                            np.any(
                                segment_mask > 0,
                                axis=1
                            )
                        )[0]

                        if len(rows) > 0:

                            y1 = int(rows.min())
                            y2 = int(rows.max())

                            if y2 - y1 >= self.min_length:

                                bands.append({
                                    "orientation": "vertical",
                                    "start": (
                                        start,
                                        y1
                                    ),
                                    "end": (
                                        end,
                                        y2
                                    ),
                                    "length": y2 - y1,
                                    "thickness": thickness
                                })

                    start = None

        return bands

    def detect(self):

        horizontal = (
            self.detect_horizontal()
        )

        vertical = (
            self.detect_vertical()
        )

        return horizontal + vertical