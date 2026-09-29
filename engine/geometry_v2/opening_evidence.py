from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from engine.geometry_v2.openings import OpeningCandidate
from engine.geometry_v2.orientation import Orientation


Point = Tuple[float, float]


@dataclass
class OpeningEvidence:
    """
    Evidence collected for an opening candidate.

    This class intentionally does NOT decide whether the opening
    is a door or a window.
    """

    opening_id: str

    wall_id: str

    width: float

    wall_thickness: float

    orientation: Orientation

    width_to_thickness_ratio: float

    geometry_score: float

    evidence: Dict[str, object] = field(default_factory=dict)

    classification: str = "unknown"

    def as_dict(self):
        return {
            "opening_id": self.opening_id,
            "wall_id": self.wall_id,
            "width": round(self.width, 2),
            "wall_thickness": round(self.wall_thickness, 2),
            "orientation": self.orientation.value,
            "width_to_thickness_ratio": round(
                self.width_to_thickness_ratio,
                2,
            ),
            "geometry_score": round(
                self.geometry_score,
                3,
            ),
            "evidence": self.evidence,
            "classification": self.classification,
        }


class OpeningEvidenceBuilder:
    """
    Builds geometry-based evidence for opening candidates.

    No image analysis is performed here yet.

    The purpose of this layer is to keep raw opening geometry
    separate from future pixel-level evidence.
    """

    def build(
        self,
        opening: OpeningCandidate,
        orientation: Orientation,
    ) -> OpeningEvidence:

        width = float(opening.width)
        thickness = float(opening.wall_thickness)

        if thickness > 0:
            ratio = width / thickness
        else:
            ratio = 0.0

        geometry_score = self._geometry_score(
            ratio=ratio,
            confidence=opening.confidence,
        )

        evidence = {
            "has_valid_width": width > 0,
            "has_valid_wall_thickness": thickness > 0,
            "width_to_thickness_ratio": round(ratio, 2),
            "source_confidence": round(
                opening.confidence,
                3,
            ),

            # Reserved for future image analysis.
            "pixel_evidence": None,

            # Reserved for future wall classification.
            "wall_context": None,

            # Reserved for future OCR/architectural symbols.
            "symbol_evidence": None,
        }

        return OpeningEvidence(
            opening_id=opening.id,
            wall_id=opening.wall_id,
            width=width,
            wall_thickness=thickness,
            orientation=orientation,
            width_to_thickness_ratio=ratio,
            geometry_score=geometry_score,
            evidence=evidence,
        )

    def build_many(
        self,
        openings: List[OpeningCandidate],
        orientations: Dict[str, Orientation],
    ) -> List[OpeningEvidence]:

        results = []

        for opening in openings:

            orientation = orientations.get(
                opening.wall_id,
                Orientation.DIAGONAL,
            )

            results.append(
                self.build(
                    opening,
                    orientation,
                )
            )

        return results

    @staticmethod
    def _geometry_score(
        ratio: float,
        confidence: float,
    ) -> float:
        """
        Produces a conservative geometry score.

        This is NOT a door/window probability.

        It only measures whether the opening has reasonable
        geometric evidence and supporting wall confidence.
        """

        if ratio <= 0:
            return 0.0

        # A gap approximately equal to wall thickness is weak.
        # Larger gaps provide stronger evidence of a real opening.
        size_score = min(
            1.0,
            ratio / 4.0,
        )

        confidence_score = max(
            0.0,
            min(1.0, confidence),
        )

        return (
            size_score * 0.6
            + confidence_score * 0.4
        )


if __name__ == "__main__":

    from engine.geometry_v2.groups import WallGroupBuilder
    from engine.geometry_v2.segments import WallSegmentBuilder

    print("Opening Evidence V2 Test")
    print("------------------------")

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

    group_builder = WallGroupBuilder()

    groups = group_builder.build(segments)

    from engine.geometry_v2.openings import OpeningDetector

    opening_detector = OpeningDetector(
        min_width_factor=1.5
    )

    openings = opening_detector.detect(groups)

    orientations = {
        group.id: group.orientation
        for group in groups
    }

    evidence_builder = OpeningEvidenceBuilder()

    evidence = evidence_builder.build_many(
        openings,
        orientations,
    )

    print(f"Opening candidates: {len(openings)}")
    print(f"Evidence records: {len(evidence)}")
    print()

    for item in evidence:

        print(item.opening_id)
        print(f"  wall: {item.wall_id}")
        print(f"  width: {item.width:.2f}")
        print(f"  wall thickness: {item.wall_thickness:.2f}")
        print(
            "  orientation:",
            item.orientation.value,
        )
        print(
            "  width/thickness:",
            f"{item.width_to_thickness_ratio:.2f}",
        )
        print(
            "  geometry score:",
            f"{item.geometry_score:.3f}",
        )
        print(
            "  classification:",
            item.classification,
        )