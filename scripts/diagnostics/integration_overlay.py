import json
from pathlib import Path

import cv2

from engine.walls.mask import WallMaskBuilder
from engine.walls.segments import WallSegmentExtractor

from engine.geometry_v2.segments import WallSegmentBuilder
from engine.geometry_v2.groups import WallGroupBuilder
from engine.geometry_v2.openings import OpeningDetector
from engine.geometry_v2.opening_pixel_evidence import (
    OpeningPixelEvidenceAnalyzer,
)


IMAGE_PATH = "test_floorplan.png"
OUTPUT_DIR = Path("output_v2")


def draw_label(image, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2

    (w, h), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    # Background rectangle
    cv2.rectangle(
        image,
        (x, y - h - baseline - 4),
        (x + w + 8, y + 4),
        (255, 255, 255),
        -1,
    )

    # Text
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
    print("Opening Overlay Test")
    print("====================")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {IMAGE_PATH}"
        )

    print(f"Image: {image.shape[1]}x{image.shape[0]}")

    # ---------------------------------------------------------
    # 1. Build wall mask
    # ---------------------------------------------------------

    print()
    print("Building wall mask...")

    wall_mask_builder = WallMaskBuilder(image)
    wall_mask = wall_mask_builder.build()

    # ---------------------------------------------------------
    # 2. Extract wall segments
    # ---------------------------------------------------------

    print("Extracting wall segments...")

    extractor = WallSegmentExtractor(wall_mask)
    raw_segments = extractor.detect()

    print(f"Raw wall segments: {len(raw_segments)}")

    # ---------------------------------------------------------
    # 3. Convert to V2 segments
    # ---------------------------------------------------------

    segment_builder = WallSegmentBuilder()
    segments = segment_builder.build_many(raw_segments)

    print(f"V2 wall segments: {len(segments)}")

    # ---------------------------------------------------------
    # 4. Build wall groups
    # ---------------------------------------------------------

    group_builder = WallGroupBuilder()
    groups = group_builder.build(segments)

    print(f"Wall groups: {len(groups)}")

    # ---------------------------------------------------------
    # 5. Detect opening candidates
    # ---------------------------------------------------------

    opening_detector = OpeningDetector(
        min_width_factor=1.5
    )

    openings = opening_detector.detect(groups)

    print(f"Opening candidates: {len(openings)}")

    # ---------------------------------------------------------
    # 6. Create overlay
    # ---------------------------------------------------------

    overlay = image.copy()

    pixel_analyzer = OpeningPixelEvidenceAnalyzer()

    results = []

    for index, opening in enumerate(openings, start=1):

        pixel_result = pixel_analyzer.analyze(
            image,
            opening,
        )

        x1, y1, x2, y2 = pixel_result.crop_bbox

        # Keep coordinates inside image
        x1 = max(0, min(x1, image.shape[1] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        y2 = max(0, min(y2, image.shape[0] - 1))

        # Draw candidate box
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
        )

        # Opening center
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Opening number
        label = f"O{index}"

        draw_label(
            overlay,
            label,
            cx,
            cy,
        )

        results.append(
            {
                "number": index,
                "opening_id": opening.id,
                "wall_id": opening.wall_id,
                "width": round(opening.width, 2),
                "wall_thickness": round(
                    opening.wall_thickness,
                    2,
                ),
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                "dark_pixel_ratio": round(
                    pixel_result.dark_pixel_ratio,
                    4,
                ),
                "edge_density": round(
                    pixel_result.edge_density,
                    4,
                ),
            }
        )

    # ---------------------------------------------------------
    # 7. Save overlay
    # ---------------------------------------------------------

    overlay_path = (
        OUTPUT_DIR / "opening_candidates_overlay.png"
    )

    cv2.imwrite(
        str(overlay_path),
        overlay,
    )

    print()
    print("Overlay saved:")
    print(overlay_path)

    # ---------------------------------------------------------
    # 8. Save metadata
    # ---------------------------------------------------------

    json_path = (
        OUTPUT_DIR / "opening_overlay_data.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Metadata saved:")
    print(json_path)

    print()
    print("====================")
    print("Overlay test completed.")
    print("====================")


if __name__ == "__main__":
    main()