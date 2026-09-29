import cv2
import sys

sys.path.insert(
    0,
    "engine"
)

from structural.wall_centerline import (
    WallCenterlineExtractor
)


mask = cv2.imread(
    "output/wall_regions.png",
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise FileNotFoundError(
        "wall_regions.png not found"
    )


extractor = WallCenterlineExtractor(
    mask,
    min_distance=3
)

centerline, distance = (
    extractor.extract()
)


output_path = (
    "output/wall_centerline.png"
)

cv2.imwrite(
    output_path,
    centerline
)


print()
print("Wall Centerline Extraction")
print("---------------------------")

print(
    "Input wall pixels:",
    int(
        (mask > 0).sum()
    )
)

print(
    "Centerline pixels:",
    int(
        (centerline > 0).sum()
    )
)

print(
    "Maximum wall half-thickness:",
    round(
        float(distance.max()),
        2
    )
)

print(
    "Output:",
    output_path
)
