import cv2
import sys

sys.path.insert(0, "engine")

from structural.wall_candidate_detector import WallCandidateDetector


mask = cv2.imread(
    "output/wall_regions.png",
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise FileNotFoundError(
        "output/wall_regions.png not found"
    )


detector = WallCandidateDetector(mask)

walls = detector.detect()


# Convert mask to color
visual = cv2.cvtColor(
    mask,
    cv2.COLOR_GRAY2BGR
)


for wall in walls:

    x1, y1 = wall["start"]
    x2, y2 = wall["end"]

    cv2.line(
        visual,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        3
    )


output_path = "output/wall_candidates_visual.png"

cv2.imwrite(
    output_path,
    visual
)

print()
print("Wall Candidate Visualization")
print("-----------------------------")
print("Candidates:", len(walls))
print("Output:", output_path)