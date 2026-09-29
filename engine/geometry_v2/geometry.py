import math
from typing import Optional, Tuple


Point = Tuple[float, float]
Line = Tuple[float, float, float, float]


class GeometryEngine:
    """
    General-purpose 2D geometry utilities for the floor plan engine.

    This module intentionally knows nothing about:
    - walls
    - doors
    - windows
    - rooms
    - floor-plan images

    It only performs geometric calculations.
    """

    # ------------------------------------------------------------------
    # Basic measurements
    # ------------------------------------------------------------------

    @staticmethod
    def length(line: Line) -> float:
        """Return the Euclidean length of a line segment."""

        x1, y1, x2, y2 = line

        return math.hypot(
            x2 - x1,
            y2 - y1,
        )

    @staticmethod
    def midpoint(line: Line) -> Point:
        """Return the midpoint of a line segment."""

        x1, y1, x2, y2 = line

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    @staticmethod
    def angle(line: Line) -> float:
        """
        Return the line orientation in degrees.

        Orientation is normalized to [0, 180).

        Examples:
            horizontal → 0°
            vertical   → 90°
            diagonal   → e.g. 35°
        """

        x1, y1, x2, y2 = line

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return 0.0

        angle = math.degrees(
            math.atan2(dy, dx)
        )

        return angle % 180.0

    # ------------------------------------------------------------------
    # Angle relationships
    # ------------------------------------------------------------------

    @staticmethod
    def angle_difference(
        angle1: float,
        angle2: float,
    ) -> float:
        """
        Return the smallest orientation difference.

        Because floor-plan lines have no direction, 0° and 180°
        represent the same orientation.
        """

        difference = abs(
            (angle1 - angle2) % 180.0
        )

        return min(
            difference,
            180.0 - difference,
        )

    @classmethod
    def is_parallel(
        cls,
        line1: Line,
        line2: Line,
        tolerance: float = 5.0,
    ) -> bool:
        """Return True when two lines have approximately the same orientation."""

        return (
            cls.angle_difference(
                cls.angle(line1),
                cls.angle(line2),
            )
            <= tolerance
        )

    # ------------------------------------------------------------------
    # Distances
    # ------------------------------------------------------------------

    @staticmethod
    def point_to_line_distance(
        point: Point,
        line: Line,
    ) -> float:
        """
        Return the perpendicular distance from a point
        to the infinite line defined by a segment.
        """

        px, py = point
        x1, y1, x2, y2 = line

        dx = x2 - x1
        dy = y2 - y1

        denominator = math.hypot(dx, dy)

        if denominator == 0:
            return math.hypot(
                px - x1,
                py - y1,
            )

        return abs(
            dy * px
            - dx * py
            + x2 * y1
            - y2 * x1
        ) / denominator

    @staticmethod
    def point_to_segment_distance(
        point: Point,
        line: Line,
    ) -> float:
        """
        Return the shortest distance from a point
        to the actual finite line segment.
        """

        px, py = point
        x1, y1, x2, y2 = line

        dx = x2 - x1
        dy = y2 - y1

        length_squared = dx * dx + dy * dy

        if length_squared == 0:
            return math.hypot(
                px - x1,
                py - y1,
            )

        t = (
            (px - x1) * dx
            + (py - y1) * dy
        ) / length_squared

        t = max(
            0.0,
            min(1.0, t),
        )

        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return math.hypot(
            px - closest_x,
            py - closest_y,
        )

    # ------------------------------------------------------------------
    # Collinearity
    # ------------------------------------------------------------------

    @classmethod
    def are_collinear(
        cls,
        line1: Line,
        line2: Line,
        angle_tolerance: float = 5.0,
        distance_tolerance: float = 10.0,
    ) -> bool:
        """
        Return True when two line segments are approximately
        on the same infinite line.
        """

        if not cls.is_parallel(
            line1,
            line2,
            tolerance=angle_tolerance,
        ):
            return False

        midpoint2 = cls.midpoint(line2)

        distance = cls.point_to_line_distance(
            midpoint2,
            line1,
        )

        return distance <= distance_tolerance

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    @staticmethod
    def project_point_to_line(
        point: Point,
        line: Line,
    ) -> Point:
        """
        Project a point onto the infinite line defined by a segment.
        """

        px, py = point
        x1, y1, x2, y2 = line

        dx = x2 - x1
        dy = y2 - y1

        length_squared = dx * dx + dy * dy

        if length_squared == 0:
            return (x1, y1)

        t = (
            (px - x1) * dx
            + (py - y1) * dy
        ) / length_squared

        return (
            x1 + t * dx,
            y1 + t * dy,
        )

    # ------------------------------------------------------------------
    # Segment intervals
    # ------------------------------------------------------------------

    @staticmethod
    def interval_on_axis(
        line: Line,
        axis: str,
    ) -> Tuple[float, float]:
        """
        Return the min/max interval of a line on X or Y.

        Example:
            line = (100, 200, 500, 200)

            axis='x'
            → (100, 500)

            axis='y'
            → (200, 200)
        """

        x1, y1, x2, y2 = line

        if axis == "x":
            return (
                min(x1, x2),
                max(x1, x2),
            )

        if axis == "y":
            return (
                min(y1, y2),
                max(y1, y2),
            )

        raise ValueError(
            "axis must be 'x' or 'y'"
        )

    @classmethod
    def interval_gap(
        cls,
        line1: Line,
        line2: Line,
        axis: str,
    ) -> float:
        """
        Return the gap between two line intervals.

        Returns 0 when the intervals overlap or touch.
        """

        start1, end1 = cls.interval_on_axis(
            line1,
            axis,
        )

        start2, end2 = cls.interval_on_axis(
            line2,
            axis,
        )

        if end1 < start2:
            return start2 - end1

        if end2 < start1:
            return start1 - end2

        return 0.0

    @classmethod
    def interval_overlap(
        cls,
        line1: Line,
        line2: Line,
        axis: str,
    ) -> float:
        """
        Return the amount of overlap between two intervals.
        """

        start1, end1 = cls.interval_on_axis(
            line1,
            axis,
        )

        start2, end2 = cls.interval_on_axis(
            line2,
            axis,
        )

        return max(
            0.0,
            min(end1, end2)
            - max(start1, start2),
        )

    # ------------------------------------------------------------------
    # Intersection
    # ------------------------------------------------------------------

    @staticmethod
    def segment_intersection(
        line1: Line,
        line2: Line,
    ) -> Optional[Point]:
        """
        Return the intersection point of two finite line segments.

        Returns None when the segments do not intersect.
        """

        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denominator = (
            (x1 - x2) * (y3 - y4)
            - (y1 - y2) * (x3 - x4)
        )

        if denominator == 0:
            return None

        px = (
            (x1 * y2 - y1 * x2) * (x3 - x4)
            - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / denominator

        py = (
            (x1 * y2 - y1 * x2) * (y3 - y4)
            - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / denominator

        def within(
            value: float,
            a: float,
            b: float,
        ) -> bool:
            tolerance = 1e-9
            return (
                min(a, b) - tolerance
                <= value
                <= max(a, b) + tolerance
            )

        if (
            within(px, x1, x2)
            and within(py, y1, y2)
            and within(px, x3, x4)
            and within(py, y3, y4)
        ):
            return (px, py)

        return None


if __name__ == "__main__":

    horizontal = (100, 200, 500, 200)
    horizontal_2 = (450, 203, 800, 203)
    vertical = (300, 100, 300, 500)
    diagonal = (100, 100, 300, 200)

    print("Geometry V2 Test")
    print("-----------------")

    print(
        "Length:",
        GeometryEngine.length(horizontal),
    )

    print(
        "Angle:",
        GeometryEngine.angle(horizontal),
    )

    print(
        "Midpoint:",
        GeometryEngine.midpoint(horizontal),
    )

    print(
        "Parallel:",
        GeometryEngine.is_parallel(
            horizontal,
            horizontal_2,
        ),
    )

    print(
        "Collinear:",
        GeometryEngine.are_collinear(
            horizontal,
            horizontal_2,
        ),
    )

    print(
        "Point → line distance:",
        GeometryEngine.point_to_line_distance(
            (300, 250),
            horizontal,
        ),
    )

    print(
        "Interval gap:",
        GeometryEngine.interval_gap(
            horizontal,
            horizontal_2,
            "x",
        ),
    )

    print(
        "Interval overlap:",
        GeometryEngine.interval_overlap(
            horizontal,
            horizontal_2,
            "x",
        ),
    )

    print(
        "Intersection:",
        GeometryEngine.segment_intersection(
            horizontal,
            vertical,
        ),
    )

    print(
        "Diagonal angle:",
        GeometryEngine.angle(diagonal),
    )