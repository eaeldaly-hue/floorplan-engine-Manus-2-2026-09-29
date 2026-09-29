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


mask = cv2.imread(
    "output/wall_regions.png",
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise FileNotFoundError(
        "wall_regions.png not found"
    )


# -------------------------
# Extract wall centerline
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
# Extract wall segments
# -------------------------

segment_extractor = (
    WallSegmentExtractor(
        centerline,
        distance,
        min_segment_length=40
    )
)

segments = segment_extractor.extract()


horizontal = [
    s
    for s in segments
    if s["orientation"] == "horizontal"
]

vertical = [
    s
    for s in segments
    if s["orientation"] == "vertical"
]


print()
print("Wall Segment Extraction")
print("-----------------------")

print(
    "Total segments:",
    len(segments)
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
print("Segments")
print("--------")


for i, segment in enumerate(
    segments,
    start=1
):

    print(
        f"{i:03d}",
        "|",
        segment["orientation"],
        "|",
        segment["start"],
        "->",
        segment["end"],
        "| length:",
        segment["length"],
        "| wall thickness:",
        segment["wall_thickness"],
        "| median half:",
        segment["median_half_thickness"]
    )