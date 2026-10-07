import cv2

from engine.walls.mask import WallMaskBuilder
from engine.geometry_v2.opening_candidates_v2 import (
    OpeningCandidateGeneratorV2,
)


IMAGE_PATH = "test_floorplan.png"


def main():

    print("Opening Candidate V2 Test")
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

    # Build existing wall mask.
    builder = WallMaskBuilder(image)
    wall_mask = builder.build()

    generator = OpeningCandidateGeneratorV2()

    candidates = generator.detect(
        image=image,
        wall_mask=wall_mask,
    )

    print()
    print(
        f"Candidates detected: "
        f"{len(candidates)}"
    )

    for candidate in candidates:

        print(
            f"{candidate.id} | "
            f"{candidate.orientation} | "
            f"width={candidate.width:.1f} | "
            f"source={candidate.source} | "
            f"confidence={candidate.confidence:.2f}"
        )

    print()
    print("=========================")
    print("Test completed.")
    print("=========================")


if __name__ == "__main__":
    main()