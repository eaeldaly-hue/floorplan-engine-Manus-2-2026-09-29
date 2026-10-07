import cv2

from engine.walls.mask import WallMaskBuilder
from engine.walls.segments import WallSegmentExtractor

from engine.geometry_v2.segments import WallSegmentBuilder
from engine.geometry_v2.local_opening_analyzer import (
    LocalOpeningAnalyzer,
)


IMAGE_PATH = "test_floorplan.png"


def main():

    print("Local Opening Analyzer V2")
    print("=========================")

    image = cv2.imread(
        IMAGE_PATH,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(
            IMAGE_PATH
        )

    print(
        f"Image: "
        f"{image.shape[1]}x{image.shape[0]}"
    )

    # -----------------------------------------------------
    # Existing wall pipeline
    # -----------------------------------------------------

    wall_mask_builder = WallMaskBuilder(image)
    wall_mask = wall_mask_builder.build()

    extractor = WallSegmentExtractor(
        wall_mask
    )

    raw_segments = extractor.detect()

    print(
        f"Raw wall segments: "
        f"{len(raw_segments)}"
    )

    # -----------------------------------------------------
    # V2 wall segments
    # -----------------------------------------------------

    segment_builder = WallSegmentBuilder()

    segments = (
        segment_builder.build_many(
            raw_segments
        )
    )

    print(
        f"V2 wall segments: "
        f"{len(segments)}"
    )

    # -----------------------------------------------------
    # Local opening analysis
    # -----------------------------------------------------

    analyzer = LocalOpeningAnalyzer(
        axis_tolerance_factor=2.0,
        min_gap_factor=1.5,
        max_gap_factor=30.0,
        min_gap_px=30.0,
    )

    candidates = analyzer.detect(
        segments
    )

    print()
    print(
        f"Local candidates: "
        f"{len(candidates)}"
    )

    print()

    for candidate in candidates:

        print(
            f"{candidate.id} | "
            f"{candidate.orientation:10} | "
            f"width={candidate.width:7.1f} | "
            f"thickness={candidate.wall_thickness:5.1f} | "
            f"segments="
            f"{candidate.segment_a}-"
            f"{candidate.segment_b} | "
            f"confidence="
            f"{candidate.confidence:.3f}"
        )

    print()
    print("=========================")
    print("Test completed.")
    print("=========================")


if __name__ == "__main__":
    main()