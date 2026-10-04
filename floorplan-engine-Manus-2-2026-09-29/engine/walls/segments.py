import numpy as np


class WallSegmentExtractor:
    """
    Turns a binary wall mask into a list of wall segments with a measured
    thickness, by scanning rows (for horizontal walls) and columns (for
    vertical walls) and grouping contiguous, overlapping runs into bands.

    Why this instead of connected-components or Hough lines
    ----------------------------------------------------------
    - Connected components merge at every T/L junction into one oddly
      shaped blob, which breaks orientation/thickness for every wall that
      touches another wall (i.e. almost every wall in a real floor plan).
    - Hough lines find edges, not the wall body, so two Hough lines have
      to be paired up afterwards to recover a thickness - fragile whenever
      a wall is partly occluded or a corner is drawn slightly unevenly.

    Scanning row/column runs and clustering them by (closeness in the
    perpendicular direction + overlap along the wall) sidesteps both
    problems: a wall keeps being "the same band" straight through any
    junction, and its thickness falls out of how many rows/columns the
    band spans.
    """

    def __init__(
        self,
        wall_mask,
        min_length_ratio=0.02,
        min_thickness=3,
        max_thickness_ratio=0.05,
    ):
        self.mask = wall_mask
        self.height, self.width = wall_mask.shape
        self.reference_size = float(np.hypot(self.width, self.height))

        self.min_length = max(10, int(self.reference_size * min_length_ratio))
        self.min_thickness = min_thickness
        self.max_thickness = max(
            self.min_thickness + 1,
            int(self.reference_size * max_thickness_ratio),
        )

    # --------------------------------------------------
    # Run finding
    # --------------------------------------------------

    @staticmethod
    def _find_runs(bool_row, min_run_length):
        runs = []
        start = None

        for i, value in enumerate(bool_row):
            if value:
                if start is None:
                    start = i
            elif start is not None:
                if i - start >= min_run_length:
                    runs.append((start, i - 1))
                start = None

        if start is not None and len(bool_row) - start >= min_run_length:
            runs.append((start, len(bool_row) - 1))

        return runs

    # --------------------------------------------------
    # Band clustering (shared by both orientations)
    # --------------------------------------------------

    def _cluster_into_bands(self, num_lines, get_runs, max_axis_gap=1):
        """
        `get_runs(i)` returns the runs found on row/column `i`.
        A band tracks a contiguous stretch of rows/columns whose runs
        overlap each other; closing a band yields one wall candidate.
        """

        open_bands = []  # each: dict(axis_start, axis_end, x1, x2)
        finished = []

        def close(band):
            thickness = band["axis_end"] - band["axis_start"] + 1
            length = band["x2"] - band["x1"]
            if (
                self.min_thickness <= thickness <= self.max_thickness
                and length >= self.min_length
            ):
                finished.append(band)

        for i in range(num_lines):
            runs = get_runs(i)

            for (r1, r2) in runs:
                best = None
                for band in open_bands:
                    if band["axis_end"] < i - max_axis_gap:
                        continue
                    overlap = min(r2, band["x2"]) - max(r1, band["x1"])
                    if overlap > 0:
                        best = band
                        break

                if best is not None:
                    best["axis_end"] = i
                    best["x1"] = min(best["x1"], r1)
                    best["x2"] = max(best["x2"], r2)
                else:
                    open_bands.append({
                        "axis_start": i,
                        "axis_end": i,
                        "x1": r1,
                        "x2": r2,
                    })

            still_open = []
            for band in open_bands:
                if band["axis_end"] < i - max_axis_gap:
                    close(band)
                else:
                    still_open.append(band)
            open_bands = still_open

        for band in open_bands:
            close(band)

        return finished

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def detect_horizontal(self):
        mask_bool = self.mask > 0

        def get_runs(y):
            return self._find_runs(mask_bool[y, :], self.min_length)

        bands = self._cluster_into_bands(self.height, get_runs)

        segments = []
        for band in bands:
            center_y = round((band["axis_start"] + band["axis_end"]) / 2)
            segments.append({
                "orientation": "horizontal",
                "start": (band["x1"], center_y),
                "end": (band["x2"], center_y),
                "length": band["x2"] - band["x1"],
                "thickness": band["axis_end"] - band["axis_start"] + 1,
            })

        return segments

    def detect_vertical(self):
        mask_bool = self.mask > 0

        def get_runs(x):
            return self._find_runs(mask_bool[:, x], self.min_length)

        bands = self._cluster_into_bands(self.width, get_runs)

        segments = []
        for band in bands:
            center_x = round((band["axis_start"] + band["axis_end"]) / 2)
            segments.append({
                "orientation": "vertical",
                "start": (center_x, band["x1"]),
                "end": (center_x, band["x2"]),
                "length": band["x2"] - band["x1"],
                "thickness": band["axis_end"] - band["axis_start"] + 1,
            })

        return segments

    def detect(self):
        return self.detect_horizontal() + self.detect_vertical()
