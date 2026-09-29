from pathlib import Path

import cv2

from engine.walls.mask import WallMaskBuilder
from engine.walls.segments import WallSegmentExtractor

from engine.geometry_v2.segments import WallSegmentBuilder
from engine.geometry_v2.local_opening_analyzer import (
    LocalOpeningAnalyzer,
)


IMAGE_PATH = "test_floorplan.png"
OUTPUT_DIR = Path("output_v2")


def draw_label(image, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2

    (w, h), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    # White background
    cv2.rectangle(
        image,
        (x, y - h - baseline - 5),
        (x + w + 8, y + 5),
        (255, 255, 255),
        -1,
    )

    # Red label
    cv2.putText(
        image,
        text,
        (x + 4, y - 4),
        font,
        scale,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA,
    )


def main():

    print("Local Opening Visual Overlay")
    print("============================")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(
        IMAGE_PATH,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(
            IMAGE_PATH
        )

    # ---------------------------------------------------------
    # Existing wall pipeline
    # ---------------------------------------------------------

    wall_mask_builder = WallMaskBuilder(image)
    wall_mask = wall_mask_builder.build()

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

    # ---------------------------------------------------------
    # Local candidates
    # ---------------------------------------------------------

    analyzer = LocalOpeningAnalyzer(
        axis_tolerance_factor=2.0,
        min_gap_factor=1.5,
        max_gap_factor=30.0,
        min_gap_px=30.0,
    )

    candidates = analyzer.detect(
        segments
    )

    print(
        f"Candidates: {len(candidates)}"
    )

    # ---------------------------------------------------------
    # Overlay
    # ---------------------------------------------------------

    overlay = image.copy()

    for candidate in candidates:

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

        # Expand the visualization box around
        # the candidate so it is easier to see.
        padding = 12

        x1 = max(
            0,
            x1 - padding,
        )

        y1 = max(
            0,
            y1 - padding,
        )

        x2 = min(
            image.shape[1] - 1,
            x2 + padding,
        )

        y2 = min(
            image.shape[0] - 1,
            y2 + padding,
        )

        # Draw candidate box.
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
        )

        # Center.
        cx = int(
            (
                candidate.start[0]
                + candidate.end[0]
            )
            / 2
        )

        cy = int(
            (
                candidate.start[1]
                + candidate.end[1]
            )
            / 2
        )

        label = (
            f"L{candidate.id.split('_')[-1]}"
        )

        draw_label(
            overlay,
            label,
            cx,
            cy,
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "local_opening_candidates.png"
    )

    cv2.imwrite(
        str(output_path),
        overlay,
    )

    print()
    print(
        "Overlay saved:"
    )
    print(output_path)

    print()
    print("============================")
    print("Overlay completed.")
    print("============================")


if __name__ == "__main__":
    main()