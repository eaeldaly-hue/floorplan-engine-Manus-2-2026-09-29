import cv2
import sys

sys.path.insert(
    0,
    "engine"
)

from structural.wall_candidate_detector import (
    WallCandidateDetector
)


mask = cv2.imread(
    "output/wall_regions.png",
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise FileNotFoundError(
        "wall_regions.png not found"
    )


detector = WallCandidateDetector(
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
print("Wall Candidate Detection")
print("------------------------")

print(
    "Total:",
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
print("Walls")
print("-----")

for i, wall in enumerate(
    walls,
    start=1
):

    start = wall["start"]
    end = wall["end"]

    if wall["orientation"] == "horizontal":
        length = abs(
            end[0] - start[0]
        )
    else:
        length = abs(
            end[1] - start[1]
        )

    if length < 100:
        continue

    print(
        f"{i:03d}",
        "|",
        wall["orientation"],
        "|",
        start,
        "->",
        end,
        "| length:",
        length
    )