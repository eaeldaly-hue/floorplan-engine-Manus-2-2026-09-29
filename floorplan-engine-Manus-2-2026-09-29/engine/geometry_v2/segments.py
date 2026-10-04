from dataclasses import dataclass
from typing import Any, Optional, Tuple

from engine.geometry_v2.geometry import GeometryEngine, Line
from engine.geometry_v2.orientation import (
    Orientation,
    OrientationAnalyzer,
)


Point = Tuple[float, float]


@dataclass
class WallSegment:
    """
    Normalized geometric representation of one wall piece.

    A WallSegment is intentionally independent from other segments.
    Gaps are preserved and handled later by the wall-grouping layer.
    """

    start: Point
    end: Point
    orientation: Orientation
    length: float
    thickness: float
    confidence: float

    def as_dict(self) -> dict:
        """Return a JSON-friendly representation."""

        return {
            "start": list(self.start),
            "end": list(self.end),
            "orientation": self.orientation.value,
            "length": self.length,
            "thickness": self.thickness,
            "confidence": self.confidence,
        }


class WallSegmentBuilder:
    """
    Convert different wall-candidate formats into normalized WallSegments.

    Supported input formats:

    1. Tuple/list:
        (x1, y1, x2, y2)

    2. Fixed Cloud style dictionary:
        {
            "orientation": "horizontal",
            "start": (x1, y1),
            "end": (x2, y2),
            "length": ...,
            "thickness": ...
        }

    3. Generic dictionary containing:
        {
            "start": ...,
            "end": ...
        }

    The builder does not merge segments.
    """

    def __init__(
        self,
        orientation_analyzer: Optional[OrientationAnalyzer] = None,
        default_thickness: float = 0.0,
        default_confidence: float = 0.5,
    ):
        self.orientation_analyzer = (
            orientation_analyzer
            or OrientationAnalyzer()
        )

        self.default_thickness = float(default_thickness)
        self.default_confidence = float(default_confidence)

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _point(value: Any) -> Point:
        """
        Convert a point-like value into (x, y).
        """

        if value is None:
            raise ValueError("Point cannot be None.")

        if len(value) != 2:
            raise ValueError(
                f"Point must contain exactly 2 values: {value}"
            )

        return (
            float(value[0]),
            float(value[1]),
        )

    @classmethod
    def _extract_line(cls, candidate: Any) -> Line:
        """
        Extract a normalized line tuple from a candidate.
        """

        if isinstance(candidate, dict):
            if "start" not in candidate or "end" not in candidate:
                raise ValueError(
                    "Candidate dictionary must contain 'start' and 'end'."
                )

            start = cls._point(candidate["start"])
            end = cls._point(candidate["end"])

            return (
                start[0],
                start[1],
                end[0],
                end[1],
            )

        if len(candidate) != 4:
            raise ValueError(
                f"Line must contain exactly 4 values: {candidate}"
            )

        return (
            float(candidate[0]),
            float(candidate[1]),
            float(candidate[2]),
            float(candidate[3]),
        )

    @staticmethod
    def _extract_thickness(
        candidate: Any,
        default: float,
    ) -> float:
        """
        Extract wall thickness if available.
        """

        if isinstance(candidate, dict):
            value = candidate.get("thickness")

            if value is not None:
                try:
                    return max(0.0, float(value))
                except (TypeError, ValueError):
                    pass

        return default

    @staticmethod
    def _extract_confidence(
        candidate: Any,
        default: float,
    ) -> float:
        """
        Extract confidence if available and clamp it to [0, 1].
        """

        if isinstance(candidate, dict):
            value = candidate.get("confidence")

            if value is not None:
                try:
                    return max(
                        0.0,
                        min(1.0, float(value)),
                    )
                except (TypeError, ValueError):
                    pass

        return max(
            0.0,
            min(1.0, default),
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        candidate: Any,
    ) -> WallSegment:
        """
        Convert one candidate into a normalized WallSegment.
        """

        line = self._extract_line(candidate)

        length = GeometryEngine.length(line)

        if length == 0:
            raise ValueError(
                "A wall segment cannot have zero length."
            )

        orientation = self.orientation_analyzer.classify(line)

        thickness = self._extract_thickness(
            candidate,
            self.default_thickness,
        )

        confidence = self._extract_confidence(
            candidate,
            self.default_confidence,
        )

        return WallSegment(
            start=(line[0], line[1]),
            end=(line[2], line[3]),
            orientation=orientation,
            length=length,
            thickness=thickness,
            confidence=confidence,
        )

    def build_many(
        self,
        candidates: list[Any],
    ) -> list[WallSegment]:
        """
        Convert multiple candidates.

        Invalid candidates are skipped rather than crashing the
        entire pipeline.
        """

        segments = []

        for candidate in candidates:
            try:
                segment = self.build(candidate)
            except (TypeError, ValueError):
                continue

            segments.append(segment)

        return segments


if __name__ == "__main__":

    builder = WallSegmentBuilder(
        default_thickness=18,
        default_confidence=0.80,
    )

    candidates = [
        (100, 200, 500, 200),

        {
            "start": (300, 100),
            "end": (300, 500),
            "thickness": 20,
            "confidence": 0.94,
        },

        {
            "start": (100, 100),
            "end": (300, 200),
            "thickness": 12,
            "confidence": 0.72,
        },

        {
            "start": (400, 400),
            "end": (700, 400),
            "thickness": 16,
        },
    ]

    segments = builder.build_many(candidates)

    print("Wall Segment V2 Test")
    print("--------------------")

    print(f"Input candidates: {len(candidates)}")
    print(f"Segments created: {len(segments)}")

    for index, segment in enumerate(segments, start=1):

        print(f"\nSegment {index}")

        for key, value in segment.as_dict().items():
            print(f"  {key}: {value}")