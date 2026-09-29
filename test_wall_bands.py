import cv2
import sys

sys.path.insert(
    0,
    "engine"
)

from structural.wall_band_detector import (
    WallBandDetector
)


mask = cv2.imread(
    "output/wall_regions.png",
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise FileNotFoundError(
        "wall_regions.png not found"
    )


detector = WallBandDetector(
    mask
)

walls = detector.detect()


horizontal = [
    wall
    for wall in walls
    if wall["orientation"] == "horizontal"
]

vertical = [
    wall
    for wall in walls
    if wall["orientation"] == "vertical"
]


print()
print("Wall Band Detection")
print("-------------------")
print(
    "Total walls:",
    len(walls)
)
print(
    "Horizontal:",
    len(horizontal)
)
print(
    "Vertical:",
    len(vertical)
)

print()
print("Detected Walls")
print("--------------")

for wall in walls:

    print(
        wall["orientation"],
        "|",
        wall["start"],
        "->",
        wall["end"],
        "| length:",
        wall["length"],
        "| thickness:",
        wall["thickness"]
    )