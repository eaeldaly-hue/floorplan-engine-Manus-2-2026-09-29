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
# Build graph
# --------------------------------------------------

mask = cv2.imread(
    "output/wall_regions.png",
    cv2.IMREAD_GRAYSCALE
)

centerline_extractor = WallCenterlineExtractor(
    mask,
    min_distance=3
)

centerline, distance = (
    centerline_extractor.extract()
)

segment_extractor = WallSegmentExtractor(
    centerline,
    distance,
    min_segment_length=40
)

segments = segment_extractor.extract()

network_builder = WallNetworkBuilder(
    segments,
    line_tolerance=12,
    thickness_tolerance=10,
    max_gap=60
)

network = network_builder.build()

snapper = WallSnapper(
    network["walls"],
    snap_tolerance=35,
    min_confidence=0.75
)

intersections = snapper.snap()

graph_builder = WallGraphBuilder(
    network["walls"],
    intersections,
    endpoint_tolerance=30
)

graph = graph_builder.build()


# --------------------------------------------------
# Image
# --------------------------------------------------

image = cv2.imread(
    "test_floorplan.png"
)

if image is None:
    raise FileNotFoundError(
        "test_floorplan.png not found"
    )


# --------------------------------------------------
# Draw edges
# --------------------------------------------------

for edge in graph["edges"]:

    wall_id = edge["wall"]

    wall = next(
        w
        for w in network["walls"]
        if w["id"] == wall_id
    )

    start_node = next(
        n
        for n in graph["nodes"]
        if n["id"] == edge["from"]
    )

    end_node = next(
        n
        for n in graph["nodes"]
        if n["id"] == edge["to"]
    )

    p1 = start_node["point"]
    p2 = end_node["point"]

    cv2.line(
        image,
        (
            int(p1[0]),
            int(p1[1])
        ),
        (
            int(p2[0]),
            int(p2[1])
        ),
        (0, 0, 255),
        3
    )


# --------------------------------------------------
# Draw nodes
# --------------------------------------------------

for node in graph["nodes"]:

    x, y = node["point"]

    if node["type"] == "intersection":

        radius = 9
        thickness = -1

    else:

        radius = 6
        thickness = -1

    cv2.circle(
        image,
        (
            int(x),
            int(y)
        ),
        radius,
        (255, 0, 0),
        thickness
    )

    cv2.putText(
        image,
        node["id"],
        (
            int(x) + 8,
            int(y) - 8
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 0, 0),
        1,
        cv2.LINE_AA
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

output = (
    "output/wall_graph.png"
)

cv2.imwrite(
    output,
    image
)

print()
print("Wall Graph Visualization")
print("------------------------")
print(
    "Nodes:",
    graph["node_count"]
)
print(
    "Edges:",
    graph["edge_count"]
)
print(
    "Output:",
    output
)