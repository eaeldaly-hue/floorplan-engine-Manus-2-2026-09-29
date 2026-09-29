from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from engine.geometry_v2.groups import WallGroup


Point = Tuple[float, float]


class OpeningType(str, Enum):
    UNKNOWN = "unknown"
    DOOR = "door"
    WINDOW = "window"
    OTHER = "other"


@dataclass
class OpeningCandidate:
    id: str
    wall_id: str
    start: Point
    end: Point
    width: float
    wall_thickness: float
    opening_type: OpeningType
    confidence: float

    def as_dict(self):
        return {
            "id": self.id,
            "wall_id": self.wall_id,
            "start": list(self.start),
            "end": list(self.end),
            "width": round(self.width, 2),
            "wall_thickness": round(self.wall_thickness, 2),
            "type": self.opening_type.value,
            "confidence": round(self.confidence, 3),
        }


class OpeningDetector:
    """
    Converts significant wall gaps into opening candidates.

    Important:
    - A gap is NOT automatically a door or window.
    - Small gaps are treated as detection imperfections.
    - Thresholds are relative to wall thickness, not fixed pixels.
    """

    def __init__(self, min_width_factor: float = 1.5):
        self.min_width_factor = min_width_factor

    def is_significant_gap(
        self,
        gap_width: float,
        wall_thickness: float,
    ) -> bool:

        if gap_width <= 0:
            return False

        if wall_thickness <= 0:
            return False

        minimum_width = wall_thickness * self.min_width_factor

        return gap_width >= minimum_width

    def detect(
        self,
        wall_groups: List[WallGroup],
    ) -> List[OpeningCandidate]:

        candidates = []
        opening_index = 1

        for group in wall_groups:

            wall_thickness = group.thickness

            if wall_thickness <= 0:
                continue

            for gap in group.gaps:

                if not self.is_significant_gap(
                    gap.width,
                    wall_thickness,
                ):
                    continue

                candidate = OpeningCandidate(
                    id=f"opening_{opening_index}",
                    wall_id=group.id,
                    start=gap.start,
                    end=gap.end,
                    width=gap.width,
                    wall_thickness=wall_thickness,
                    opening_type=OpeningType.UNKNOWN,

                    # This confidence represents the supporting
                    # wall geometry confidence, NOT door/window
                    # classification confidence.
                    confidence=group.confidence,
                )

                candidates.append(candidate)
                opening_index += 1

        return candidates


if __name__ == "__main__":

    from engine.geometry_v2.segments import WallSegmentBuilder
    from engine.geometry_v2.groups import WallGroupBuilder

    print("Opening Detection V2 Test")
    print("-------------------------")

    segment_builder = WallSegmentBuilder()

    candidates = [
        {
            "start": (100, 200),
            "end": (400, 200),
            "thickness": 18,
            "confidence": 0.90,
        },
        {
            "start": (405, 200),
            "end": (700, 200),
            "thickness": 18,
            "confidence": 0.90,
        },
        {
            "start": (800, 200),
            "end": (1100, 200),
            "thickness": 18,
            "confidence": 0.90,
        },
        {
            "start": (100, 300),
            "end": (500, 300),
            "thickness": 18,
            "confidence": 0.90,
        },
        {
            "start": (100, 100),
            "end": (100, 350),
            "thickness": 18,
            "confidence": 0.90,
        },
        {
            "start": (100, 360),
            "end": (100, 600),
            "thickness": 18,
            "confidence": 0.90,
        },
        {
            "start": (300, 100),
            "end": (500, 200),
            "thickness": 18,
            "confidence": 0.90,
        },
    ]

    segments = segment_builder.build_many(candidates)

    print(f"Wall segments: {len(segments)}")

    group_builder = WallGroupBuilder()

    groups = group_builder.build(segments)

    print(f"Wall groups: {len(groups)}")

    detector = OpeningDetector(
        min_width_factor=1.5
    )

    openings = detector.detect(groups)

    print(f"Opening candidates: {len(openings)}")
    print()

    for opening in openings:

        print(opening.id)
        print(f"  wall: {opening.wall_id}")
        print(f"  start: {opening.start}")
        print(f"  end: {opening.end}")
        print(f"  width: {opening.width:.2f}")
        print(f"  wall thickness: {opening.wall_thickness:.2f}")
        print(f"  type: {opening.opening_type.value}")
        print(f"  confidence: {opening.confidence:.2f}")