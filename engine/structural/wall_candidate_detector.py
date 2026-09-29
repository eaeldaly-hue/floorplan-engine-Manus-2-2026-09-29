import cv2
import numpy as np


class WallCandidateDetector:

    def __init__(
        self,
        wall_mask,
        min_run_length=60,
        min_wall_thickness=8,
        max_wall_thickness=80
    ):
        self.mask = wall_mask
        self.min_run_length = min_run_length
        self.min_wall_thickness = min_wall_thickness
        self.max_wall_thickness = max_wall_thickness

    def find_runs(self, values):

        runs = []

        start = None

        for i, value in enumerate(values):

            if value:

                if start is None:
                    start = i

            else:

                if start is not None:

                    end = i - 1

                    if (
                        end - start + 1
                        >= self.min_run_length
                    ):
                        runs.append(
                            (start, end)
                        )

                    start = None

        if start is not None:

            end = len(values) - 1

            if (
                end - start + 1
                >= self.min_run_length
            ):
                runs.append(
                    (start, end)
                )

        return runs

    def detect_horizontal(self):

        mask = self.mask
        height, width = mask.shape

        candidates = []

        for y in range(height):

            row = mask[y] > 0

            runs = self.find_runs(row)

            for x1, x2 in runs:

                candidates.append({
                    "orientation": "horizontal",
                    "start": (x1, y),
                    "end": (x2, y),
                    "length": x2 - x1 + 1
                })

        return candidates

    def detect_vertical(self):

        mask = self.mask
        height, width = mask.shape

        candidates = []

        for x in range(width):

            column = mask[:, x] > 0

            runs = self.find_runs(column)

            for y1, y2 in runs:

                candidates.append({
                    "orientation": "vertical",
                    "start": (x, y1),
                    "end": (x, y2),
                    "length": y2 - y1 + 1
                })

        return candidates

    def merge_horizontal(self, candidates):

        if not candidates:
            return []

        candidates = sorted(
            candidates,
            key=lambda c: (
                c["start"][1],
                c["start"][0]
            )
        )

        merged = []

        for candidate in candidates:

            x1, y = candidate["start"]
            x2, _ = candidate["end"]

            found = False

            for wall in merged:

                wx1, wy = wall["start"]
                wx2, _ = wall["end"]

                if abs(y - wy) <= 5:

                    overlap = min(
                        x2,
                        wx2
                    ) - max(
                        x1,
                        wx1
                    )

                    if overlap >= 20:

                        wall["start"] = (
                            min(x1, wx1),
                            round((y + wy) / 2)
                        )

                        wall["end"] = (
                            max(x2, wx2),
                            round((y + wy) / 2)
                        )

                        found = True
                        break

            if not found:

                merged.append({
                    "orientation": "horizontal",
                    "start": candidate["start"],
                    "end": candidate["end"]
                })

        return merged

    def merge_vertical(self, candidates):

        if not candidates:
            return []

        candidates = sorted(
            candidates,
            key=lambda c: (
                c["start"][0],
                c["start"][1]
            )
        )

        merged = []

        for candidate in candidates:

            x, y1 = candidate["start"]
            _, y2 = candidate["end"]

            found = False

            for wall in merged:

                wx, wy1 = wall["start"]
                _, wy2 = wall["end"]

                if abs(x - wx) <= 5:

                    overlap = min(
                        y2,
                        wy2
                    ) - max(
                        y1,
                        wy1
                    )

                    if overlap >= 20:

                        wall["start"] = (
                            round((x + wx) / 2),
                            min(y1, wy1)
                        )

                        wall["end"] = (
                            round((x + wx) / 2),
                            max(y2, wy2)
                        )

                        found = True
                        break

            if not found:

                merged.append({
                    "orientation": "vertical",
                    "start": candidate["start"],
                    "end": candidate["end"]
                })

        return merged

    def detect(self):

        horizontal_candidates = (
            self.detect_horizontal()
        )

        vertical_candidates = (
            self.detect_vertical()
        )

        horizontal_walls = (
            self.merge_horizontal(
                horizontal_candidates
            )
        )

        vertical_walls = (
            self.merge_vertical(
                vertical_candidates
            )
        )

        return (
            horizontal_walls
            + vertical_walls
        )