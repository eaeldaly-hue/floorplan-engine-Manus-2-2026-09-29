import cv2
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
        """[(start, end)] of the True runs at least min_run_length long (end inclusive)."""
        row = np.asarray(bool_row, dtype=bool)
        if not row.any():
            return []
        edges = np.diff(np.concatenate(([False], row, [False])).astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)                 # one past the run's last index
        keep = stops - starts >= min_run_length
        return [(int(a), int(b) - 1) for a, b in zip(starts[keep], stops[keep])]

    @staticmethod
    def _find_runs_reference(bool_row, min_run_length):
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

    def detect_diagonal(self):
        """Extract long oblique wall runs; axis scans cannot represent them."""
        minimum_line = max(24, int(round(self.min_length * 1.25)))
        lines = cv2.HoughLinesP(
            self.mask,
            rho=1,
            theta=np.pi / 180.0,
            threshold=max(12, int(round(minimum_line * 0.55))),
            minLineLength=minimum_line,
            maxLineGap=max(4, int(round(self.max_thickness * 1.25))),
        )
        if lines is None:
            return []

        distance = cv2.distanceTransform((self.mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
        candidates = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, line)
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            angle = abs(float(np.degrees(np.arctan2(dy, dx)))) % 180.0
            distance_from_axis = min(angle, 180.0 - angle, abs(angle - 90.0))
            if length < minimum_line or distance_from_axis <= 12.0:
                continue

            sample_count = max(3, int(length / 8))
            xs = np.linspace(x1, x2, sample_count).round().astype(int)
            ys = np.linspace(y1, y2, sample_count).round().astype(int)
            inside = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
            radii = distance[ys[inside], xs[inside]]
            thickness = float(np.clip(2.0 * np.median(radii), self.min_thickness, self.max_thickness))
            candidates.append({
                "orientation": "diagonal",
                "start": (x1, y1),
                "end": (x2, y2),
                "length": length,
                "thickness": thickness,
                "confidence": 0.72,
            })

        # Hough sees both edges of thick walls. Keep the longest member of
        # overlapping, nearly parallel detections in the same wall band.
        kept = []
        for candidate in sorted(candidates, key=lambda item: item["length"], reverse=True):
            ax, ay = candidate["start"]
            bx, by = candidate["end"]
            ux, uy = (bx - ax) / candidate["length"], (by - ay) / candidate["length"]
            midpoint = ((ax + bx) / 2.0, (ay + by) / 2.0)
            duplicate = False
            for existing in kept:
                ex, ey = existing["start"]
                fx, fy = existing["end"]
                elen = existing["length"]
                vx, vy = (fx - ex) / elen, (fy - ey) / elen
                if abs(ux * vx + uy * vy) < 0.985:
                    continue
                distance_between = abs((midpoint[0] - ex) * (-uy) + (midpoint[1] - ey) * ux)
                if distance_between <= max(candidate["thickness"], existing["thickness"]) * 1.4:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept

    def detect(self, include_diagonal=False):
        segments = self.detect_horizontal() + self.detect_vertical()
        if include_diagonal:
            segments.extend(self.detect_diagonal())
        return segments
