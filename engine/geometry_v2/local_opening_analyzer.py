from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from engine.geometry_v2.segments import WallSegment


Point = Tuple[float, float]


@dataclass
class LocalOpeningCandidate:
    id: str
    start: Point
    end: Point
    width: float
    orientation: str
    wall_thickness: float
    segment_a: int
    segment_b: int
    source: str
    confidence: float


class LocalOpeningAnalyzer:
    """
    Local opening candidate detector.

    Pipeline:

        Wall Segments
              ↓
        Local Gap Detection
              ↓
        Wall Mask Validation
              ↓
        Duplicate Removal
              ↓
        Valid Opening Candidates

    This class does NOT classify:
        - door
        - window
        - garage door

    It only detects geometrically plausible openings.

    Diagonal wall segments are intentionally ignored.
    """

    def __init__(
        self,
        axis_tolerance_factor: float = 2.0,
        min_gap_factor: float = 1.5,
        max_gap_factor: float = 30.0,
        min_gap_px: float = 30.0,

        duplicate_center_distance_px: float = 50.0,
        duplicate_width_tolerance: float = 0.35,
        duplicate_endpoint_distance_px: float = 45.0,

        # -----------------------------------------------------
        # Wall-mask validation parameters
        # -----------------------------------------------------

        validation_margin_px: float = 8.0,
        max_wall_occupancy: float = 0.55,
        min_side_support: float = 0.35,
    ):

        self.axis_tolerance_factor = (
            axis_tolerance_factor
        )

        self.min_gap_factor = (
            min_gap_factor
        )

        self.max_gap_factor = (
            max_gap_factor
        )

        self.min_gap_px = (
            min_gap_px
        )

        self.duplicate_center_distance_px = (
            duplicate_center_distance_px
        )

        self.duplicate_width_tolerance = (
            duplicate_width_tolerance
        )

        self.duplicate_endpoint_distance_px = (
            duplicate_endpoint_distance_px
        )

        self.validation_margin_px = (
            validation_margin_px
        )

        self.max_wall_occupancy = (
            max_wall_occupancy
        )

        self.min_side_support = (
            min_side_support
        )

    # =========================================================
    # Main detection
    # =========================================================

    def detect(
        self,
        segments: List[WallSegment],
        wall_mask: Optional[np.ndarray] = None,
    ) -> List[LocalOpeningCandidate]:

        candidates = []

        for i in range(
            len(segments)
        ):

            a = segments[i]

            # -------------------------------------------------
            # Ignore diagonal geometry.
            # -------------------------------------------------

            if a.orientation.value not in (
                "horizontal",
                "vertical",
            ):
                continue

            for j in range(
                i + 1,
                len(segments),
            ):

                b = segments[j]

                if (
                    b.orientation.value
                    != a.orientation.value
                ):
                    continue

                candidate = self._analyze_pair(
                    a,
                    b,
                    i,
                    j,
                )

                if candidate is None:
                    continue

                # -------------------------------------------------
                # Optional pixel-level validation.
                #
                # If no mask is supplied, geometry-only behavior
                # remains available.
                # -------------------------------------------------

                if wall_mask is not None:

                    validation = (
                        self._validate_with_wall_mask(
                            candidate,
                            wall_mask,
                        )
                    )

                    if validation is None:
                        continue

                    candidate.confidence = round(
                        min(
                            1.0,
                            candidate.confidence
                            * validation,
                        ),
                        3,
                    )

                    candidate.source = (
                        "local_segment_gap_mask_validated"
                    )

                candidates.append(
                    candidate
                )

        # -----------------------------------------------------
        # Remove duplicates.
        # -----------------------------------------------------

        candidates = self._deduplicate(
            candidates
        )

        # -----------------------------------------------------
        # Assign IDs after filtering.
        # -----------------------------------------------------

        for index, candidate in enumerate(
            candidates,
            start=1,
        ):

            candidate.id = (
                f"local_{index}"
            )

        return candidates

    # =========================================================
    # Pair analysis
    # =========================================================

    def _analyze_pair(
        self,
        a: WallSegment,
        b: WallSegment,
        index_a: int,
        index_b: int,
    ):

        thickness = (
            a.thickness
            + b.thickness
        ) / 2.0

        orientation = (
            a.orientation.value
        )

        if orientation == "horizontal":

            return self._horizontal_pair(
                a,
                b,
                index_a,
                index_b,
                thickness,
            )

        if orientation == "vertical":

            return self._vertical_pair(
                a,
                b,
                index_a,
                index_b,
                thickness,
            )

        return None

    # =========================================================
    # Horizontal
    # =========================================================

    def _horizontal_pair(
        self,
        a,
        b,
        index_a,
        index_b,
        thickness,
    ):

        ay = (
            a.start[1]
            + a.end[1]
        ) / 2.0

        by = (
            b.start[1]
            + b.end[1]
        ) / 2.0

        axis_distance = abs(
            ay - by
        )

        tolerance = (
            self.axis_tolerance_factor
            * thickness
        )

        if axis_distance > tolerance:
            return None

        a_min = min(
            a.start[0],
            a.end[0],
        )

        a_max = max(
            a.start[0],
            a.end[0],
        )

        b_min = min(
            b.start[0],
            b.end[0],
        )

        b_max = max(
            b.start[0],
            b.end[0],
        )

        if a_max < b_min:

            left_end = a_max
            right_start = b_min

        elif b_max < a_min:

            left_end = b_max
            right_start = a_min

        else:

            return None

        gap = (
            right_start
            - left_end
        )

        if not self._valid_gap(
            gap,
            thickness,
        ):

            return None

        y = (
            ay + by
        ) / 2.0

        start = (
            left_end,
            y,
        )

        end = (
            right_start,
            y,
        )

        confidence = self._confidence(
            gap,
            thickness,
            axis_distance,
        )

        return LocalOpeningCandidate(
            id="",
            start=start,
            end=end,
            width=gap,
            orientation="horizontal",
            wall_thickness=thickness,
            segment_a=index_a,
            segment_b=index_b,
            source="local_segment_gap",
            confidence=confidence,
        )

    # =========================================================
    # Vertical
    # =========================================================

    def _vertical_pair(
        self,
        a,
        b,
        index_a,
        index_b,
        thickness,
    ):

        ax = (
            a.start[0]
            + a.end[0]
        ) / 2.0

        bx = (
            b.start[0]
            + b.end[0]
        ) / 2.0

        axis_distance = abs(
            ax - bx
        )

        tolerance = (
            self.axis_tolerance_factor
            * thickness
        )

        if axis_distance > tolerance:
            return None

        a_min = min(
            a.start[1],
            a.end[1],
        )

        a_max = max(
            a.start[1],
            a.end[1],
        )

        b_min = min(
            b.start[1],
            b.end[1],
        )

        b_max = max(
            b.start[1],
            b.end[1],
        )

        if a_max < b_min:

            top_end = a_max
            bottom_start = b_min

        elif b_max < a_min:

            top_end = b_max
            bottom_start = a_min

        else:

            return None

        gap = (
            bottom_start
            - top_end
        )

        if not self._valid_gap(
            gap,
            thickness,
        ):

            return None

        x = (
            ax + bx
        ) / 2.0

        start = (
            x,
            top_end,
        )

        end = (
            x,
            bottom_start,
        )

        confidence = self._confidence(
            gap,
            thickness,
            axis_distance,
        )

        return LocalOpeningCandidate(
            id="",
            start=start,
            end=end,
            width=gap,
            orientation="vertical",
            wall_thickness=thickness,
            segment_a=index_a,
            segment_b=index_b,
            source="local_segment_gap",
            confidence=confidence,
        )

    # =========================================================
    # Gap validation
    # =========================================================

    def _valid_gap(
        self,
        gap: float,
        thickness: float,
    ):

        if gap < self.min_gap_px:
            return False

        if gap < (
            self.min_gap_factor
            * thickness
        ):
            return False

        # Keep large openings such as garage doors.
        if gap > (
            self.max_gap_factor
            * thickness
        ):
            return False

        return True

    # =========================================================
    # Confidence
    # =========================================================

    def _confidence(
        self,
        gap: float,
        thickness: float,
        axis_distance: float,
    ):

        if thickness <= 0:
            return 0.0

        ratio = (
            gap / thickness
        )

        axis_score = max(
            0.0,
            1.0
            - (
                axis_distance
                / (
                    self.axis_tolerance_factor
                    * thickness
                )
            ),
        )

        if 2.0 <= ratio <= 12.0:
            width_score = 1.0

        elif ratio < 2.0:
            width_score = 0.5

        else:
            width_score = 0.75

        confidence = (
            0.55 * axis_score
            + 0.45 * width_score
        )

        return float(
            confidence
        )

    # =========================================================
    # Duplicate removal
    # =========================================================

    def _deduplicate(
        self,
        candidates: List[LocalOpeningCandidate],
    ):

        if not candidates:
            return []

        result = []

        for candidate in sorted(
            candidates,
            key=lambda c: c.confidence,
            reverse=True,
        ):

            duplicate = False

            for existing in result:

                if (
                    candidate.orientation
                    != existing.orientation
                ):
                    continue

                candidate_center = (
                    (
                        candidate.start[0]
                        + candidate.end[0]
                    ) / 2.0,
                    (
                        candidate.start[1]
                        + candidate.end[1]
                    ) / 2.0,
                )

                existing_center = (
                    (
                        existing.start[0]
                        + existing.end[0]
                    ) / 2.0,
                    (
                        existing.start[1]
                        + existing.end[1]
                    ) / 2.0,
                )

                center_distance = (
                    (
                        (
                            candidate_center[0]
                            - existing_center[0]
                        ) ** 2
                    )
                    +
                    (
                        (
                            candidate_center[1]
                            - existing_center[1]
                        ) ** 2
                    )
                ) ** 0.5

                if (
                    center_distance
                    > self.duplicate_center_distance_px
                ):
                    continue

                max_width = max(
                    candidate.width,
                    existing.width,
                    1.0,
                )

                width_difference = abs(
                    candidate.width
                    - existing.width
                ) / max_width

                endpoint_distance = min(
                    self._point_distance(
                        candidate.start,
                        existing.start,
                    ),
                    self._point_distance(
                        candidate.start,
                        existing.end,
                    ),
                    self._point_distance(
                        candidate.end,
                        existing.start,
                    ),
                    self._point_distance(
                        candidate.end,
                        existing.end,
                    ),
                )

                if (
                    width_difference
                    <= self.duplicate_width_tolerance
                    or endpoint_distance
                    <= self.duplicate_endpoint_distance_px
                ):
                    duplicate = True
                    break

            if not duplicate:
                result.append(
                    candidate
                )

        return result

    # =========================================================
    # Wall-mask validation
    # =========================================================

    def _validate_with_wall_mask(
        self,
        candidate: LocalOpeningCandidate,
        wall_mask: np.ndarray,
    ):

        if wall_mask is None:
            return 1.0

        if wall_mask.ndim == 3:
            mask = wall_mask[:, :, 0]
        else:
            mask = wall_mask

        if mask.size == 0:
            return None

        x1, y1 = candidate.start
        x2, y2 = candidate.end

        margin = (
            self.validation_margin_px
        )

        if candidate.orientation == "horizontal":

            left = int(
                min(x1, x2)
                - margin
            )

            right = int(
                max(x1, x2)
                + margin
            )

            center_y = int(
                (y1 + y2) / 2.0
            )

            half_thickness = max(
                2,
                int(
                    candidate.wall_thickness
                    / 2.0
                ),
            )

            top = (
                center_y
                - half_thickness
            )

            bottom = (
                center_y
                + half_thickness
            )

        else:

            top = int(
                min(y1, y2)
                - margin
            )

            bottom = int(
                max(y1, y2)
                + margin
            )

            center_x = int(
                (x1 + x2) / 2.0
            )

            half_thickness = max(
                2,
                int(
                    candidate.wall_thickness
                    / 2.0
                ),
            )

            left = (
                center_x
                - half_thickness
            )

            right = (
                center_x
                + half_thickness
            )

        crop = self._safe_crop(
            mask,
            left,
            top,
            right,
            bottom,
        )

        if crop is None:
            return None

        if crop.size == 0:
            return None

        wall_pixels = (
            crop > 0
        )

        occupancy = float(
            np.mean(
                wall_pixels
            )
        )

        if occupancy > self.max_wall_occupancy:
            return None

        # -----------------------------------------------------
        # The opening corridor should not be mostly wall.
        # -----------------------------------------------------

        empty_score = max(
            0.0,
            1.0
            - (
                occupancy
                / max(
                    self.max_wall_occupancy,
                    0.001,
                )
            ),
        )

        # -----------------------------------------------------
        # Check that there is wall support around the opening.
        # -----------------------------------------------------

        support_score = (
            self._side_support(
                mask,
                candidate,
            )
        )

        if (
            support_score
            < self.min_side_support
        ):
            return None

        validation_score = (
            0.55 * empty_score
            + 0.45 * support_score
        )

        # Don't completely destroy a geometrically strong
        # candidate because the raster mask is imperfect.
        validation_score = max(
            0.60,
            validation_score,
        )

        return float(
            validation_score
        )

    # =========================================================
    # Side support
    # =========================================================

    def _side_support(
        self,
        mask: np.ndarray,
        candidate: LocalOpeningCandidate,
    ):

        h, w = mask.shape[:2]

        thickness = max(
            2.0,
            candidate.wall_thickness,
        )

        support = max(
            4,
            int(
                thickness
            ),
        )

        if candidate.orientation == "horizontal":

            y = int(
                (
                    candidate.start[1]
                    + candidate.end[1]
                ) / 2.0
            )

            x1 = int(
                min(
                    candidate.start[0],
                    candidate.end[0],
                )
            )

            x2 = int(
                max(
                    candidate.start[0],
                    candidate.end[0],
                )
            )

            left_crop = self._safe_crop(
                mask,
                x1 - support,
                y - support,
                x1,
                y + support,
            )

            right_crop = self._safe_crop(
                mask,
                x2,
                y - support,
                x2 + support,
                y + support,
            )

        else:

            x = int(
                (
                    candidate.start[0]
                    + candidate.end[0]
                ) / 2.0
            )

            y1 = int(
                min(
                    candidate.start[1],
                    candidate.end[1],
                )
            )

            y2 = int(
                max(
                    candidate.start[1],
                    candidate.end[1],
                )
            )

            left_crop = self._safe_crop(
                mask,
                x - support,
                y1 - support,
                x + support,
                y1,
            )

            right_crop = self._safe_crop(
                mask,
                x - support,
                y2,
                x + support,
                y2 + support,
            )

        if (
            left_crop is None
            or right_crop is None
        ):
            return 0.0

        left_score = float(
            np.mean(
                left_crop > 0
            )
        )

        right_score = float(
            np.mean(
                right_crop > 0
            )
        )

        return min(
            1.0,
            (
                left_score
                + right_score
            ) / 2.0,
        )

    # =========================================================
    # Helpers
    # =========================================================

    @staticmethod
    def _point_distance(
        a: Point,
        b: Point,
    ):

        return float(
            (
                (
                    a[0] - b[0]
                ) ** 2
                +
                (
                    a[1] - b[1]
                ) ** 2
            ) ** 0.5
        )

    @staticmethod
    def _safe_crop(
        image: np.ndarray,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ):

        h, w = image.shape[:2]

        left = max(
            0,
            min(
                left,
                w,
            ),
        )

        right = max(
            0,
            min(
                right,
                w,
            ),
        )

        top = max(
            0,
            min(
                top,
                h,
            ),
        )

        bottom = max(
            0,
            min(
                bottom,
                h,
            ),
        )

        if (
            right <= left
            or bottom <= top
        ):
            return None

        return image[
            top:bottom,
            left:right,
        ]
