import cv2
import sys

sys.path.insert(
    0,
    "engine"
)

from structural.wall_geometry import (
    WallGeometryExtractor
)


wall_mask = cv2.imread(
    "output/wall_regions.png",
    cv2.IMREAD_GRAYSCALE
)

if wall_mask is None:
    raise FileNotFoundError(
        "wall_regions.png not found"
    )


extractor = WallGeometryExtractor(
    wall_mask
)

walls = extractor.extract()


print()
print("Wall Geometry")
print("-------------")
print(
    f"Walls detected: {len(walls)}"
)

for wall in walls:
    print(
        wall["id"],
        "|",
        wall["orientation"],
        "|",
        "length:",
        wall["length_pixels"],
        "|",
        "thickness:",
        wall["thickness_pixels"],
        "|",
        "solidity:",
        wall["solidity"]
    )