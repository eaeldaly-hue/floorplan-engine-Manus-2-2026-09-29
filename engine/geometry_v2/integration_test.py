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


def save_opening_crop(
    image,
    opening,
    bbox,
    output_path,
):

    x1, y1, x2, y2 = bbox

    crop = image[
        y1:y2 + 1,
        x1:x2 + 1,
    ]

    if crop.size == 0:
        return False

    # Draw a visible border around the crop.
    crop = crop.copy()

    cv2.rectangle(
        crop,
        (0, 0),
        (
            crop.shape[1] - 1,
            crop.shape[0] - 1,
        ),
        (0, 0, 255),
        2,
    )

    cv2.imwrite(
        str(output_path),
        crop,
    )

    return True


def main():

    print("Floor Plan V2 Integration Test")
    print("==============================")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # 1. Load original image
    # --------------------------------------------------

    image = cv2.imread(
        IMAGE_PATH,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load: {IMAGE_PATH}"
        )

    print(
        f"Image: "
        f"{image.shape[1]}x{image.shape[0]}"
    )

    # --------------------------------------------------
    # 2. Existing wall mask
    # --------------------------------------------------

    print()
    print("Building wall mask...")

    wall_mask_builder = WallMaskBuilder(
        image
    )

    wall_mask = wall_mask_builder.build()

    wall_mask_path = (
        OUTPUT_DIR / "wall_mask_v2_source.png"
    )

    cv2.imwrite(
        str(wall_mask_path),
        wall_mask,
    )

    print(
        f"Wall mask saved: {wall_mask_path}"
    )

    # --------------------------------------------------
    # 3. Existing wall segment extractor
    # --------------------------------------------------

    print()
    print("Extracting wall segments...")

    extractor = WallSegmentExtractor(
        wall_mask
    )

    raw_segments = extractor.detect()

    print(
        f"Raw wall segments: "
        f"{len(raw_segments)}"
    )

    # --------------------------------------------------
    # 4. Convert into V2 WallSegment objects
    # --------------------------------------------------

    print()
    print("Building V2 wall segments...")

    v2_segment_builder = WallSegmentBuilder()

    v2_segments = (
        v2_segment_builder.build_many(
            raw_segments
        )
    )

    print(
        f"V2 wall segments: "
        f"{len(v2_segments)}"
    )

    # --------------------------------------------------
    # 5. Build V2 wall groups
    # --------------------------------------------------

    print()
    print("Building V2 wall groups...")

    group_builder = WallGroupBuilder()

    groups = group_builder.build(
        v2_segments
    )

    print(
        f"Wall groups: "
        f"{len(groups)}"
    )

    # --------------------------------------------------
    # 6. Detect opening candidates
    # --------------------------------------------------

    print()
    print("Detecting opening candidates...")

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

    # --------------------------------------------------
    # 7. Pixel evidence
    # --------------------------------------------------

    print()
    print("Analyzing pixel evidence...")

    pixel_analyzer = (
        OpeningPixelEvidenceAnalyzer()
    )

    all_evidence = []

    for index, opening in enumerate(
        openings,
        start=1,
    ):

        pixel_result = pixel_analyzer.analyze(
            image,
            opening,
        )

        all_evidence.append(
            pixel_result.as_dict()
        )

        crop_path = (
            OUTPUT_DIR
            / f"opening_{index:02d}_crop.png"
        )

        saved = save_opening_crop(
            image,
            opening,
            pixel_result.crop_bbox,
            crop_path,
        )

        print()
        print(
            f"Opening {index}"
        )

        print(
            f"  ID: {opening.id}"
        )

        print(
            f"  Wall: {opening.wall_id}"
        )

        print(
            f"  Width: "
            f"{opening.width:.2f}"
        )

        print(
            f"  Thickness: "
            f"{opening.wall_thickness:.2f}"
        )

        print(
            f"  Crop: "
            f"{pixel_result.crop_bbox}"
        )

        print(
            f"  Dark ratio: "
            f"{pixel_result.dark_pixel_ratio:.4f}"
        )

        print(
            f"  Edge density: "
            f"{pixel_result.edge_density:.4f}"
        )

        print(
            f"  Crop saved: "
            f"{saved}"
        )

    # --------------------------------------------------
    # 8. Save JSON
    # --------------------------------------------------

    json_path = (
        OUTPUT_DIR
        / "opening_evidence.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_evidence,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Evidence JSON saved:",
        json_path,
    )

    print()
    print("==============================")
    print("Integration test completed.")
    print("==============================")


if __name__ == "__main__":
    main()