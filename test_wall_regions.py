import cv2
import sys

sys.path.insert(0, "engine/structural")

from wall_region_detector import WallRegionDetector


image = cv2.imread("test_floorplan.png")

if image is None:
    raise FileNotFoundError(
        "Could not load test_floorplan.png"
    )

detector = WallRegionDetector(image)

wall_mask = detector.detect()

cv2.imwrite(
    "output/wall_regions.png",
    wall_mask
)

print()
print("Wall Region Detection")
print("---------------------")
print("Image:", image.shape)
print("Wall mask generated successfully")
print("Output: output/wall_regions.png")