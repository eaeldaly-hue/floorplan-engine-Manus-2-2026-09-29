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

from structural.wall_graph import (
    WallGraphBuilder
)


# --------------------------------------------------
# Load
# --------------------------------------------------

mask = cv2.imread(
    "output/wall_regions.png",
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise FileNotFoundError(
        "wall_regions.png not found"
    )


# --------------------------------------------------
# Centerline
# --------------------------------------------------

centerline_extractor = (
    WallCenterlineExtractor(
        mask,
        min_distance=3
    )
)

centerline, distance = (
    centerline_extractor.extract()
)


# --------------------------------------------------
# Segments
# --------------------------------------------------

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


# --------------------------------------------------
# Network
# --------------------------------------------------

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


# --------------------------------------------------
# Intersections
# --------------------------------------------------

snapper = WallSnapper(
    network["walls"],
    snap_tolerance=35,
    min_confidence=0.75
)

intersections = (
    snapper.snap()
)


# --------------------------------------------------
# Graph
# --------------------------------------------------

graph_builder = WallGraphBuilder(
    network["walls"],
    intersections,
    endpoint_tolerance=30
)

graph = (
    graph_builder.build()
)


# --------------------------------------------------
# Output
# --------------------------------------------------

print()
print("Wall Graph")
print("----------")

print(
    "Walls:",
    len(network["walls"])
)

print(
    "Intersections:",
    len(intersections)
)

print(
    "Nodes:",
    graph["node_count"]
)

print(
    "Edges:",
    graph["edge_count"]
)

print()
print("Node Types")
print("----------")

intersection_count = len([
    n
    for n in graph["nodes"]
    if n["type"] == "intersection"
])

endpoint_count = len([
    n
    for n in graph["nodes"]
    if n["type"] == "endpoint"
])

print(
    "Intersection nodes:",
    intersection_count
)

print(
    "Endpoint nodes:",
    endpoint_count
)

print()
print("Edges")
print("-----")

for edge in graph["edges"]:

    print(
        edge["id"],
        "|",
        edge["wall"],
        "|",
        edge["from"],
        "->",
        edge["to"],
        "| length:",
        edge["length"],
        "|",
        edge["orientation"]
    )