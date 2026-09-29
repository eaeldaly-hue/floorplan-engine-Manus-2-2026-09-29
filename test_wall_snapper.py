import cv2
import sys

sys.path.insert(
    0,
    "engine"
)

from structural.wall_centerline import (
    WallCenterlineExtractor
)

from structural.wall_segment_extractor import (
    WallSegmentExtractor
)

from structural.wall_network import (
    WallNetworkBuilder
)

from structural.wall_snapper import (
    WallSnapper
)


# -------------------------
# Load mask
# -------------------------

mask = cv2.imread(
    "output/wall_regions.png",
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise FileNotFoundError(
        "wall_regions.png not found"
    )


# -------------------------
# Centerline
# -------------------------

centerline_extractor = (
    WallCenterlineExtractor(
        mask,
        min_distance=3
    )
)

centerline, distance = (
    centerline_extractor.extract()
)


# -------------------------
# Segments
# -------------------------

segment_extractor = (
    WallSegmentExtractor(
        centerline,
        distance,
        min_segment_length=40
    )
)

segments = (
    segment_extractor.extract()
)


# -------------------------
# Network
# -------------------------

network_builder = (
    WallNetworkBuilder(
        segments,
        line_tolerance=12,
        thickness_tolerance=10,
        max_gap=60
    )
)

network = (
    network_builder.build()
)


# -------------------------
# Snapper
# -------------------------

snapper = WallSnapper(
    network["walls"],
    snap_tolerance=35
)

intersections = (
    snapper.snap()
)


# -------------------------
# Output
# -------------------------

print()
print("Wall Snapping")
print("-------------")

print(
    "Walls:",
    len(network["walls"])
)

print(
    "Intersections:",
    len(intersections)
)

print()
print("Intersections")
print("-------------")


for intersection in intersections:

    print(
        intersection["horizontal_wall"],
        "+",
        intersection["vertical_wall"],
        "| point:",
        intersection["point"],
        "| confidence:",
        intersection["confidence"]
    )
    # -------------------------
# Visualize intersections
# -------------------------

image = cv2.imread(
    "test_floorplan.png"
)

if image is None:
    raise FileNotFoundError(
        "test_floorplan.png not found"
    )


for i, intersection in enumerate(
    intersections,
    start=1
):

    x, y = intersection["point"]

    # Draw intersection
    cv2.circle(
        image,
        (int(x), int(y)),
        10,
        (255, 0, 0),
        -1
    )

    # Draw small label
    cv2.putText(
        image,
        str(i),
        (
            int(x) + 12,
            int(y) - 12
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
        cv2.LINE_AA
    )


cv2.imwrite(
    "output/wall_intersections.png",
    image
)

print()
print(
    "Visualization:",
    "output/wall_intersections.png"
)
