from dataclasses import dataclass, field
from typing import List

from engine.geometry_v2.segments import WallSegment
from engine.geometry_v2.geometry import GeometryEngine
from engine.geometry_v2.orientation import Orientation


@dataclass
class WallGap:
    """
    A gap between two wall segments.

    A gap is preserved as information.
    It may later become a door, window, or another opening.
    """

    start: tuple[float, float]
    end: tuple[float, float]
    width: float

    def as_dict(self) -> dict:
        return {
            "start": list(self.start),
            "end": list(self.end),
            "width": self.width,
        }


@dataclass
class WallGroup:
    """
    A logical wall made from one or more WallSegments.

    Important:
    WallGroup does NOT merge the underlying segments geometrically.
    Gaps remain explicit.
    """

    id: str
    orientation: Orientation
    thickness: float
    segments: List[WallSegment] = field(default_factory=list)
    gaps: List[WallGap] = field(default_factory=list)
    confidence: float = 0.0

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "orientation": self.orientation.value,
            "thickness": self.thickness,
            "segments": [
                segment.as_dict()
                for segment in self.segments
            ],
            "gaps": [
                gap.as_dict()
                for gap in self.gaps
            ],
            "confidence": self.confidence,
        }


class WallGroupBuilder:
    """
    Groups compatible WallSegments into logical walls.

    The builder uses relative tolerances based on wall thickness
    instead of relying only on fixed pixel distances.
    """

    def __init__(
        self,
        axis_tolerance_factor: float = 1.5,
        thickness_tolerance_factor: float = 0.75,
        max_gap_factor: float = 8.0,
    ):
        self.axis_tolerance_factor = axis_tolerance_factor
        self.thickness_tolerance_factor = thickness_tolerance_factor
        self.max_gap_factor = max_gap_factor

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------

    @staticmethod
    def _axis_position(segment: WallSegment) -> float:
        """
        Return the average position of the segment on its secondary axis.
        """

        if segment.orientation == Orientation.HORIZONTAL:
            return (
                segment.start[1]
                + segment.end[1]
            ) / 2.0

        if segment.orientation == Orientation.VERTICAL:
            return (
                segment.start[0]
                + segment.end[0]
            ) / 2.0

        # For diagonal walls we currently use the midpoint.
        midpoint = GeometryEngine.midpoint(
            (
                segment.start[0],
                segment.start[1],
                segment.end[0],
                segment.end[1],
            )
        )

        return midpoint[0] + midpoint[1]

    @staticmethod
    def _axis_interval(segment: WallSegment) -> tuple[float, float]:
        """
        Return the segment interval along its main axis.
        """

        if segment.orientation == Orientation.HORIZONTAL:
            return (
                min(segment.start[0], segment.end[0]),
                max(segment.start[0], segment.end[0]),
            )

        if segment.orientation == Orientation.VERTICAL:
            return (
                min(segment.start[1], segment.end[1]),
                max(segment.start[1], segment.end[1]),
            )

        # Diagonal segments are not grouped by interval in this V2 step.
        midpoint = GeometryEngine.midpoint(
            (
                segment.start[0],
                segment.start[1],
                segment.end[0],
                segment.end[1],
            )
        )

        return (
            midpoint[0],
            midpoint[0],
        )

    @staticmethod
    def _effective_thickness(
        segment_a: WallSegment,
        segment_b: WallSegment,
    ) -> float:
        """
        Estimate representative thickness between two segments.
        """

        values = [
            value
            for value in (
                segment_a.thickness,
                segment_b.thickness,
            )
            if value > 0
        ]

        if not values:
            return 0.0

        return sum(values) / len(values)

    def can_group(
        self,
        segment_a: WallSegment,
        segment_b: WallSegment,
    ) -> bool:
        """
        Determine whether two segments may belong to the same wall.
        """

        # Diagonal grouping requires a dedicated geometric strategy.
        # We deliberately avoid guessing here.
        if (
            segment_a.orientation == Orientation.DIAGONAL
            or segment_b.orientation == Orientation.DIAGONAL
        ):
            return False

        if segment_a.orientation != segment_b.orientation:
            return False

        reference_thickness = self._effective_thickness(
            segment_a,
            segment_b,
        )

        if reference_thickness <= 0:
            reference_thickness = max(
                segment_a.thickness,
                segment_b.thickness,
                1.0,
            )

        axis_tolerance = (
            reference_thickness
            * self.axis_tolerance_factor
        )

        axis_a = self._axis_position(segment_a)
        axis_b = self._axis_position(segment_b)

        if abs(axis_a - axis_b) > axis_tolerance:
            return False

        # If both thicknesses are known, reject obviously different walls.
        if segment_a.thickness > 0 and segment_b.thickness > 0:

            thickness_difference = abs(
                segment_a.thickness
                - segment_b.thickness
            )

            if (
                thickness_difference
                > reference_thickness
                * self.thickness_tolerance_factor
            ):
                return False

        interval_a = self._axis_interval(segment_a)
        interval_b = self._axis_interval(segment_b)

        if interval_a[1] < interval_b[0]:
            gap = interval_b[0] - interval_a[1]

        elif interval_b[1] < interval_a[0]:
            gap = interval_a[0] - interval_b[1]

        else:
            gap = 0.0

        max_gap = (
            reference_thickness
            * self.max_gap_factor
        )

        return gap <= max_gap

    # ------------------------------------------------------------------
    # Gap creation
    # ------------------------------------------------------------------

    @staticmethod
    def _make_gap(
        segment_a: WallSegment,
        segment_b: WallSegment,
    ) -> WallGap | None:
        """
        Create a WallGap when two compatible segments do not touch.
        """

        if segment_a.orientation != segment_b.orientation:
            return None

        if segment_a.orientation == Orientation.HORIZONTAL:

            a_start, a_end = (
                min(segment_a.start[0], segment_a.end[0]),
                max(segment_a.start[0], segment_a.end[0]),
            )

            b_start, b_end = (
                min(segment_b.start[0], segment_b.end[0]),
                max(segment_b.start[0], segment_b.end[0]),
            )

            axis = (
                segment_a.start[1]
                + segment_b.start[1]
            ) / 2.0

            if a_end < b_start:
                return WallGap(
                    start=(a_end, axis),
                    end=(b_start, axis),
                    width=b_start - a_end,
                )

            if b_end < a_start:
                return WallGap(
                    start=(b_end, axis),
                    end=(a_start, axis),
                    width=a_start - b_end,
                )

            return None

        if segment_a.orientation == Orientation.VERTICAL:

            a_start, a_end = (
                min(segment_a.start[1], segment_a.end[1]),
                max(segment_a.start[1], segment_a.end[1]),
            )

            b_start, b_end = (
                min(segment_b.start[1], segment_b.end[1]),
                max(segment_b.start[1], segment_b.end[1]),
            )

            axis = (
                segment_a.start[0]
                + segment_b.start[0]
            ) / 2.0

            if a_end < b_start:
                return WallGap(
                    start=(axis, a_end),
                    end=(axis, b_start),
                    width=b_start - a_end,
                )

            if b_end < a_start:
                return WallGap(
                    start=(axis, b_end),
                    end=(axis, a_start),
                    width=a_start - b_end,
                )

            return None

        return None

    # ------------------------------------------------------------------
    # Group building
    # ------------------------------------------------------------------

    def _build_group(
        self,
        group_id: str,
        segments: List[WallSegment],
    ) -> WallGroup:
        """
        Build one WallGroup from compatible segments.
        """

        segments = sorted(
            segments,
            key=self._axis_interval,
        )

        orientation = segments[0].orientation

        thicknesses = [
            segment.thickness
            for segment in segments
            if segment.thickness > 0
        ]

        if thicknesses:
            thickness = sum(thicknesses) / len(thicknesses)
        else:
            thickness = 0.0

        gaps = []

        for first, second in zip(
            segments,
            segments[1:],
        ):
            gap = self._make_gap(
                first,
                second,
            )

            if gap is not None:
                gaps.append(gap)

        confidence = sum(
            segment.confidence
            for segment in segments
        ) / len(segments)

        return WallGroup(
            id=group_id,
            orientation=orientation,
            thickness=thickness,
            segments=segments,
            gaps=gaps,
            confidence=confidence,
        )

    def build(
        self,
        segments: List[WallSegment],
    ) -> List[WallGroup]:
        """
        Group compatible segments into logical walls.
        """

        if not segments:
            return []

        groups: List[List[WallSegment]] = []

        for segment in segments:

            placed = False

            for group in groups:

                # Try the closest/last segment in the group first.
                reference = group[-1]

                if self.can_group(
                    reference,
                    segment,
                ):
                    group.append(segment)
                    placed = True
                    break

            if not placed:
                groups.append([segment])

        return [
            self._build_group(
                group_id=f"wall_{index}",
                segments=group,
            )
            for index, group in enumerate(groups, start=1)
        ]


if __name__ == "__main__":

    from engine.geometry_v2.segments import WallSegmentBuilder

    segment_builder = WallSegmentBuilder(
        default_thickness=18,
        default_confidence=0.90,
    )

    candidates = [
        (100, 200, 400, 200),

        (405, 201, 700, 201),

        (800, 200, 1100, 200),

        (100, 400, 500, 400),

        (100, 100, 100, 350),

        (100, 360, 100, 600),

        (100, 100, 300, 200),
    ]

    segments = segment_builder.build_many(
        candidates
    )

    group_builder = WallGroupBuilder()

    groups = group_builder.build(
        segments
    )

    print("Wall Group V2 Test")
    print("------------------")

    print(
        f"Segments: {len(segments)}"
    )

    print(
        f"Groups: {len(groups)}"
    )

    for group in groups:

        print(f"\n{group.id}")

        print(
            f"  orientation: "
            f"{group.orientation.value}"
        )

        print(
            f"  thickness: "
            f"{group.thickness:.2f}"
        )

        print(
            f"  segments: "
            f"{len(group.segments)}"
        )

        print(
            f"  gaps: "
            f"{len(group.gaps)}"
        )

        for gap in group.gaps:
            print(
                f"    gap: "
                f"{gap.start} → {gap.end} "
                f"width={gap.width:.2f}"
            )

        print(
            f"  confidence: "
            f"{group.confidence:.2f}"
        )