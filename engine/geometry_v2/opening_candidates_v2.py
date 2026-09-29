from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


Point = Tuple[int, int]


@dataclass
class CandidateOpeningV2:
    id: str
    start: Point
    end: Point
    width: float
    orientation: str
    source: str
    confidence: float
    wall_thickness: float = 0.0


class OpeningCandidateGeneratorV2:
    """Generate broad, local opening candidates from a binary wall mask.

    This class does not classify doors, windows, or false positives. It
    reports image coordinates and local thickness evidence so downstream
    validation can decide which candidates to keep.
    """

    def __init__(
        self,
        min_width_factor: float = 1.0,
        max_width_factor: float = 15.0,
    ):
        if min_width_factor <= 0 or max_width_factor < min_width_factor:
            raise ValueError("Require 0 < min_width_factor <= max_width_factor.")
        self.min_width_factor = float(min_width_factor)
        self.max_width_factor = float(max_width_factor)

    def detect(
        self,
        image: np.ndarray,
        wall_mask: np.ndarray,
        wall_groups=None,
    ) -> List[CandidateOpeningV2]:
        if image is None or image.size == 0:
            raise ValueError("Image cannot be empty.")
        if wall_mask is None or wall_mask.size == 0:
            raise ValueError("Wall mask cannot be empty.")
        if wall_mask.ndim != 2:
            raise ValueError("Wall mask must be grayscale.")
        if image.shape[:2] != wall_mask.shape[:2]:
            raise ValueError("Image and wall mask dimensions must match.")

        candidates = self._detect_horizontal_openings(wall_mask)
        candidates.extend(self._detect_vertical_openings(wall_mask))
        if wall_groups:
            candidates.extend(self._from_wall_groups(wall_groups))

        candidates = self._deduplicate(candidates)
        for index, candidate in enumerate(candidates, start=1):
            candidate.id = f"candidate_{index}"
        return candidates

    def _detect_horizontal_openings(
        self,
        wall_mask: np.ndarray,
    ) -> List[CandidateOpeningV2]:
        return self._detect_scanline_gaps(wall_mask, orientation="horizontal")

    def _detect_vertical_openings(
        self,
        wall_mask: np.ndarray,
    ) -> List[CandidateOpeningV2]:
        return self._detect_scanline_gaps(wall_mask, orientation="vertical")

    def _detect_scanline_gaps(
        self,
        wall_mask: np.ndarray,
        orientation: str,
    ) -> List[CandidateOpeningV2]:
        """Find gaps between supported local wall runs, not empty full-image columns."""
        mask = wall_mask > 0
        scan_mask = mask if orientation == "horizontal" else mask.T
        height, width = mask.shape
        minimum_support = max(10, int(round(min(height, width) * 0.01)))
        candidates: List[CandidateOpeningV2] = []

        for axis_position, scanline in enumerate(scan_mask):
            runs = [
                (start, end)
                for start, end in self._runs(scanline)
                if end - start + 1 >= minimum_support
            ]
            for left_run, right_run in zip(runs, runs[1:]):
                gap_start = left_run[1] + 1
                gap_end = right_run[0] - 1
                gap_width = gap_end - gap_start + 1
                if gap_width <= 0:
                    continue

                thickness = self._estimate_thickness(
                    mask,
                    orientation,
                    axis_position,
                    left_run,
                    right_run,
                )
                if thickness <= 0:
                    continue
                max_plausible_thickness = max(
                    20.0,
                    0.05 * min(height, width),
                )
                if thickness > max_plausible_thickness:
                    continue
                if gap_width < self.min_width_factor * thickness:
                    continue
                if gap_width > self.max_width_factor * thickness:
                    continue

                if orientation == "horizontal":
                    start = (gap_start, axis_position)
                    end = (gap_end, axis_position)
                    source = "horizontal_scanline_gap"
                else:
                    start = (axis_position, gap_start)
                    end = (axis_position, gap_end)
                    source = "vertical_scanline_gap"

                candidates.append(
                    CandidateOpeningV2(
                        id="",
                        start=start,
                        end=end,
                        width=float(gap_width),
                        orientation=orientation,
                        source=source,
                        confidence=0.35,
                        wall_thickness=float(thickness),
                    )
                )
        return candidates

    @staticmethod
    def _runs(line: np.ndarray) -> List[Tuple[int, int]]:
        values = np.asarray(line, dtype=np.uint8) > 0
        if not np.any(values):
            return []
        padded = np.pad(values.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1
        return list(zip(starts.tolist(), ends.tolist()))

    @staticmethod
    def _run_extent(line: np.ndarray, index: int) -> int:
        if index < 0 or index >= len(line) or not line[index]:
            return 0
        start = index
        end = index
        while start > 0 and line[start - 1]:
            start -= 1
        while end + 1 < len(line) and line[end + 1]:
            end += 1
        return end - start + 1

    @staticmethod
    def _sample_positions(run: Tuple[int, int]) -> List[int]:
        start, end = run
        span = end - start
        return sorted({int(round(start + span * fraction)) for fraction in (0.2, 0.5, 0.8)})

    def _estimate_thickness(
        self,
        mask: np.ndarray,
        orientation: str,
        axis_position: int,
        left_run: Tuple[int, int],
        right_run: Tuple[int, int],
    ) -> float:
        if orientation == "horizontal":
            positions = self._sample_positions(left_run) + self._sample_positions(right_run)
            extents = [
                self._run_extent(mask[:, x], axis_position)
                for x in positions
            ]
        else:
            positions = self._sample_positions(left_run) + self._sample_positions(right_run)
            extents = [
                self._run_extent(mask[y, :], axis_position)
                for y in positions
            ]
        measured = [value for value in extents if value > 0]
        return float(min(measured)) if measured else 0.0

    def _from_wall_groups(self, groups) -> List[CandidateOpeningV2]:
        candidates = []
        for group in groups:
            orientation = getattr(group.orientation, "value", str(group.orientation))
            for gap in getattr(group, "gaps", []):
                candidates.append(
                    CandidateOpeningV2(
                        id="",
                        start=(int(gap.start[0]), int(gap.start[1])),
                        end=(int(gap.end[0]), int(gap.end[1])),
                        width=float(gap.width),
                        orientation=orientation,
                        source="wall_group_gap",
                        confidence=float(getattr(group, "confidence", 0.5)),
                        wall_thickness=float(getattr(group, "thickness", 0.0)),
                    )
                )
        return candidates

    def _deduplicate(
        self,
        candidates: List[CandidateOpeningV2],
    ) -> List[CandidateOpeningV2]:
        result: List[CandidateOpeningV2] = []
        for candidate in candidates:
            duplicate = None
            for existing in result:
                if candidate.orientation != existing.orientation:
                    continue

                if candidate.orientation == "horizontal":
                    cross_a = (candidate.start[1] + candidate.end[1]) / 2
                    cross_b = (existing.start[1] + existing.end[1]) / 2
                    along_a = (candidate.start[0] + candidate.end[0]) / 2
                    along_b = (existing.start[0] + existing.end[0]) / 2
                else:
                    cross_a = (candidate.start[0] + candidate.end[0]) / 2
                    cross_b = (existing.start[0] + existing.end[0]) / 2
                    along_a = (candidate.start[1] + candidate.end[1]) / 2
                    along_b = (existing.start[1] + existing.end[1]) / 2

                thickness = min(
                    value for value in (candidate.wall_thickness, existing.wall_thickness)
                    if value > 0
                ) if candidate.wall_thickness > 0 and existing.wall_thickness > 0 else 1.0
                cross_tolerance = max(2.0, 1.5 * thickness)
                along_tolerance = max(3.0, 0.15 * min(candidate.width, existing.width))
                width_tolerance = max(3.0, 0.15 * max(candidate.width, existing.width))
                if (
                    abs(cross_a - cross_b) <= cross_tolerance
                    and abs(along_a - along_b) <= along_tolerance
                    and abs(candidate.width - existing.width) <= width_tolerance
                ):
                    duplicate = existing
                    break

            if duplicate is None:
                result.append(candidate)
            elif candidate.confidence > duplicate.confidence:
                duplicate.start = candidate.start
                duplicate.end = candidate.end
                duplicate.width = candidate.width
                duplicate.source = candidate.source
                duplicate.confidence = candidate.confidence
                duplicate.wall_thickness = candidate.wall_thickness
        return result
