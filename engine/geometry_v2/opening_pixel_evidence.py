from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

from engine.geometry_v2.openings import OpeningCandidate


Point = Tuple[float, float]


@dataclass
class PixelEvidence:
    """
    Pixel-level evidence extracted around an opening.

    This class does NOT classify the opening as a door or window.
    """

    opening_id: str

    crop_bbox: Tuple[int, int, int, int]

    dark_pixel_ratio: float

    edge_density: float

    horizontal_edge_density: float

    vertical_edge_density: float

    evidence: Dict[str, object]

    def as_dict(self):
        return {
            "opening_id": self.opening_id,
            "crop_bbox": list(self.crop_bbox),
            "dark_pixel_ratio": round(
                self.dark_pixel_ratio,
                4,
            ),
            "edge_density": round(
                self.edge_density,
                4,
            ),
            "horizontal_edge_density": round(
                self.horizontal_edge_density,
                4,
            ),
            "vertical_edge_density": round(
                self.vertical_edge_density,
                4,
            ),
            "evidence": self.evidence,
        }


class OpeningPixelEvidenceAnalyzer:
    """
    Extracts explainable pixel-level features around an opening.

    No AI / Vision API is used.
    """

    def __init__(
        self,
        context_factor: float = 2.0,
        min_context: int = 20,
        max_context: int = 250,
    ):
        self.context_factor = context_factor
        self.min_context = min_context
        self.max_context = max_context

    def analyze(
        self,
        image: np.ndarray,
        opening: OpeningCandidate,
    ) -> PixelEvidence:

        if image is None:
            raise ValueError("Image cannot be None.")

        if image.size == 0:
            raise ValueError("Image is empty.")

        height, width = image.shape[:2]

        x1, y1 = opening.start
        x2, y2 = opening.end

        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)

        opening_width = max(
            1.0,
            opening.width,
        )

        wall_thickness = max(
            1.0,
            opening.wall_thickness,
        )

        context = int(
            round(
                wall_thickness
                * self.context_factor
            )
        )

        context = max(
            self.min_context,
            context,
        )

        context = min(
            self.max_context,
            context,
        )

        min_x = int(
            max(
                0,
                min(x1, x2) - context,
            )
        )

        max_x = int(
            min(
                width - 1,
                max(x1, x2) + context,
            )
        )

        min_y = int(
            max(
                0,
                min(y1, y2) - context,
            )
        )

        max_y = int(
            min(
                height - 1,
                max(y1, y2) + context,
            )
        )

        if max_x <= min_x:
            max_x = min(
                width - 1,
                min_x + 1,
            )

        if max_y <= min_y:
            max_y = min(
                height - 1,
                min_y + 1,
            )

        crop = image[
            min_y:max_y + 1,
            min_x:max_x + 1,
        ]

        gray = self._to_gray(crop)

        dark_pixel_ratio = self._dark_pixel_ratio(
            gray
        )

        edges = cv2.Canny(
            gray,
            50,
            150,
        )

        edge_density = float(
            np.count_nonzero(edges)
            / edges.size
        )

        horizontal_edges = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=3,
        )

        vertical_edges = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=3,
        )

        horizontal_edge_density = (
            self._normalized_edge_density(
                horizontal_edges
            )
        )

        vertical_edge_density = (
            self._normalized_edge_density(
                vertical_edges
            )
        )

        evidence = {
            "opening_width": round(
                opening_width,
                2,
            ),
            "wall_thickness": round(
                wall_thickness,
                2,
            ),
            "context_pixels": context,
            "crop_width": int(crop.shape[1]),
            "crop_height": int(crop.shape[0]),

            # Reserved for future detectors.
            "door_leaf_evidence": None,
            "door_swing_evidence": None,
            "window_frame_evidence": None,
            "parallel_frame_evidence": None,

            "note": (
                "These are raw pixel features only. "
                "No door/window classification is performed."
            ),
        }

        return PixelEvidence(
            opening_id=opening.id,
            crop_bbox=(
                min_x,
                min_y,
                max_x,
                max_y,
            ),
            dark_pixel_ratio=dark_pixel_ratio,
            edge_density=edge_density,
            horizontal_edge_density=horizontal_edge_density,
            vertical_edge_density=vertical_edge_density,
            evidence=evidence,
        )

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:

        if len(image.shape) == 2:
            return image

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    @staticmethod
    def _dark_pixel_ratio(
        gray: np.ndarray,
    ) -> float:

        threshold = 180

        dark_pixels = np.count_nonzero(
            gray < threshold
        )

        return float(
            dark_pixels / gray.size
        )

    @staticmethod
    def _normalized_edge_density(
        gradient: np.ndarray,
    ) -> float:

        magnitude = np.abs(gradient)

        if magnitude.size == 0:
            return 0.0

        threshold = np.percentile(
            magnitude,
            90,
        )

        if threshold <= 0:
            return 0.0

        strong_edges = (
            magnitude >= threshold
        )

        return float(
            np.count_nonzero(strong_edges)
            / strong_edges.size
        )


if __name__ == "__main__":

    from engine.geometry_v2.groups import (
        WallGroupBuilder,
    )
    from engine.geometry_v2.openings import (
        OpeningDetector,
    )
    from engine.geometry_v2.segments import (
        WallSegmentBuilder,
    )

    print("Opening Pixel Evidence V2 Test")
    print("--------------------------------")

    image_path = "test_floorplan.png"

    image = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load: {image_path}"
        )

    print(
        f"Image loaded: "
        f"{image.shape[1]}x{image.shape[0]}"
    )

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
    ]

    segments = (
        segment_builder.build_many(
            candidates
        )
    )

    group_builder = WallGroupBuilder()

    groups = group_builder.build(
        segments
    )

    opening_detector = OpeningDetector(
        min_width_factor=1.5
    )

    openings = opening_detector.detect(
        groups
    )

    print(
        f"Opening candidates: "
        f"{len(openings)}"
    )

    analyzer = OpeningPixelEvidenceAnalyzer()

    for opening in openings:

        result = analyzer.analyze(
            image,
            opening,
        )

        print()
        print(result.opening_id)

        print(
            "  crop:",
            result.crop_bbox,
        )

        print(
            "  dark pixel ratio:",
            f"{result.dark_pixel_ratio:.4f}",
        )

        print(
            "  edge density:",
            f"{result.edge_density:.4f}",
        )

        print(
            "  horizontal edge density:",
            f"{result.horizontal_edge_density:.4f}",
        )

        print(
            "  vertical edge density:",
            f"{result.vertical_edge_density:.4f}",
        )

        print(
            "  classification:",
            "NOT PERFORMED",
        )