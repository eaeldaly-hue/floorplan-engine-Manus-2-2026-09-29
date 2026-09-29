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


# -------------------------
# Load wall mask
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
# Wall segments
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
# Wall Network
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
# Visualize Wall Network
# -------------------------

image = cv2.imread(
    "test_floorplan.png"
)

if image is None:
    raise FileNotFoundError(
        "test_floorplan.png not found"
    )

for wall in network["walls"]:

    color = (
        0,
        0,
        255
    )

    thickness = max(
        2,
        int(
            wall["thickness"] / 4
        )
    )

    for start, end in wall["intervals"]:

        if wall["orientation"] == "horizontal":

            y = (
                wall["geometry"]["start"][1]
            )

            x1 = start
            x2 = end

            cv2.line(
                image,
                (x1, y),
                (x2, y),
                color,
                thickness
            )

        else:

            x = (
                wall["geometry"]["start"][0]
            )

            y1 = start
            y2 = end

            cv2.line(
                image,
                (x, y1),
                (x, y2),
                color,
                thickness
            )

# Draw intersections if any

for intersection in network["intersections"]:

    x, y = intersection["point"]

    cv2.circle(
        image,
        (int(x), int(y)),
        8,
        (255, 0, 0),
        -1
    )

cv2.imwrite(
    "output/wall_network.png",
    image
)

print(
    "Visualization:",
    "output/wall_network.png"
)


# -------------------------
# Output
# -------------------------

print()
print("Wall Network")
print("------------")

print(
    "Input segments:",
    len(segments)
)

print(
    "Walls:",
    network["wall_count"]
)

print(
    "Intersections:",
    network["intersection_count"]
)

print()
print("Walls")
print("-----")


for wall in network["walls"]:

    print()
    print(wall["id"])
    print("  orientation:", wall["orientation"])
    print("  geometry:", wall["geometry"])
    print("  intervals:", wall["intervals"])
    print("  gaps:", wall["gaps"])
    print("  length:", wall["length"])
    print("  thickness:", wall["thickness"])
    print("  segments:", wall["segment_count"])
    print("  confidence:", wall["confidence"])