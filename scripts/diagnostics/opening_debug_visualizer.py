
import cv2

from engine.walls.mask import WallMaskBuilder
from engine.walls.segments import WallSegmentExtractor

from engine.geometry_v2.segments import WallSegmentBuilder
from engine.geometry_v2.local_opening_analyzer import (
    LocalOpeningAnalyzer,
)

IMAGE_PATH = "test_floorplan.png"
OUTPUT_PATH = "output/opening_candidates_debug.png"


def main():

    print("Opening Debug Visualizer")
    print("========================")

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

    # ---------------------------------------------------------
    # Wall mask
    # ---------------------------------------------------------

    wall_mask_builder = WallMaskBuilder(
        image
    )

    wall_mask = wall_mask_builder.build()

    # ---------------------------------------------------------
    # Wall segments
    # ---------------------------------------------------------

    extractor = WallSegmentExtractor(
        wall_mask
    )

    raw_segments = extractor.detect()

    segment_builder = WallSegmentBuilder()

    segments = (
        segment_builder.build_many(
            raw_segments
        )
    )

    print(
        f"Wall segments: "
        f"{len(segments)}"
    )

    # ---------------------------------------------------------
    # Opening candidates
    # ---------------------------------------------------------

    analyzer = LocalOpeningAnalyzer(
        axis_tolerance_factor=2.0,
        min_gap_factor=1.5,
        max_gap_factor=30.0,
        min_gap_px=30.0,
    )

    candidates = analyzer.detect(
        segments,
        wall_mask=wall_mask,
    )

    print(
        f"Opening candidates: "
        f"{len(candidates)}"
    )

    # ---------------------------------------------------------
    # Debug image
    # ---------------------------------------------------------

    debug = image.copy()

    # ---------------------------------------------------------
    # Draw wall segments
    # ---------------------------------------------------------

    for index, segment in enumerate(
        segments
    ):

        x1 = int(
            segment.start[0]
        )

        y1 = int(
            segment.start[1]
        )

        x2 = int(
            segment.end[0]
        )

        y2 = int(
            segment.end[1]
        )

        cv2.line(
            debug,
            (x1, y1),
            (x2, y2),
            (180, 180, 180),
            2,
        )

        cx = int(
            (
                segment.start[0]
                + segment.end[0]
            ) / 2
        )

        cy = int(
            (
                segment.start[1]
                + segment.end[1]
            ) / 2
        )

        cv2.putText(
            debug,
            str(index),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )

    # ---------------------------------------------------------
    # Draw candidates
    # ---------------------------------------------------------

    for candidate in candidates:

        x1 = int(
            candidate.start[0]
        )

        y1 = int(
            candidate.start[1]
        )

        x2 = int(
            candidate.end[0]
        )

        y2 = int(
            candidate.end[1]
        )

        confidence = (
            candidate.confidence
        )

        if confidence >= 0.85:

            color = (
                0,
                255,
                0,
            )

        elif confidence >= 0.60:

            color = (
                0,
                255,
                255,
            )

        else:

            color = (
                0,
                0,
                255,
            )

        cv2.line(
            debug,
            (x1, y1),
            (x2, y2),
            color,
            8,
        )

        cx = int(
            (
                x1 + x2
            ) / 2
        )

        cy = int(
            (
                y1 + y2
            ) / 2
        )

        label = (
            f"{candidate.id} "
            f"{candidate.width:.0f}px "
            f"{candidate.orientation[0].upper()}"
        )

        cv2.putText(
            debug,
            label,
            (
                cx + 8,
                cy - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    cv2.imwrite(
        OUTPUT_PATH,
        debug,
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print()
    print("========================")
    print("Visualization completed.")
    print("========================")


if __name__ == "__main__":
    main()
