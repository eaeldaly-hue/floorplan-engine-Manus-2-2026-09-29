import math


class WallGraphBuilder:

    def __init__(
        self,
        walls,
        intersections,
        endpoint_tolerance=30
    ):
        self.walls = walls
        self.intersections = intersections
        self.endpoint_tolerance = endpoint_tolerance

    # --------------------------------------------------
    # Distance
    # --------------------------------------------------

    @staticmethod
    def distance(p1, p2):

        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]

        return math.sqrt(
            dx * dx + dy * dy
        )

    # --------------------------------------------------
    # Create intersection nodes
    # --------------------------------------------------

    def create_intersection_nodes(self):

        nodes = []

        for index, intersection in enumerate(
            self.intersections,
            start=1
        ):

            x, y = intersection["point"]

            nodes.append({

                "id":
                    f"N{index:03d}",

                "type":
                    "intersection",

                "point":
                    (x, y),

                "walls": [
                    intersection[
                        "horizontal_wall"
                    ],
                    intersection[
                        "vertical_wall"
                    ]
                ],

                "confidence":
                    intersection[
                        "confidence"
                    ]
            })

        return nodes

    # --------------------------------------------------
    # Get wall endpoints
    # --------------------------------------------------

    def get_wall_endpoints(self, wall):

        return [
            wall["geometry"]["start"],
            wall["geometry"]["end"]
        ]

    # --------------------------------------------------
    # Find endpoints not represented
    # by an intersection
    # --------------------------------------------------

    def create_endpoint_nodes(
        self,
        existing_nodes
    ):

        nodes = []

        next_id = (
            len(existing_nodes)
            + 1
        )

        for wall in self.walls:

            endpoints = (
                self.get_wall_endpoints(
                    wall
                )
            )

            for point in endpoints:

                nearest = None
                nearest_distance = None

                for node in existing_nodes:

                    distance = (
                        self.distance(
                            point,
                            node["point"]
                        )
                    )

                    if (
                        nearest_distance
                        is None
                        or
                        distance
                        <
                        nearest_distance
                    ):

                        nearest = node
                        nearest_distance = (
                            distance
                        )

                if (
                    nearest_distance
                    is not None
                    and
                    nearest_distance
                    <=
                    self.endpoint_tolerance
                ):
                    continue

                # ----------------------------------
                # Check duplicate endpoint
                # ----------------------------------

                duplicate = False

                for node in nodes:

                    if (
                        self.distance(
                            point,
                            node["point"]
                        )
                        <=
                        self.endpoint_tolerance
                    ):

                        duplicate = True
                        break

                if duplicate:
                    continue

                nodes.append({

                    "id":
                        f"N{next_id:03d}",

                    "type":
                        "endpoint",

                    "point":
                        point,

                    "walls": [
                        wall["id"]
                    ],

                    "confidence":
                        1.0
                })

                next_id += 1

        return nodes

    # --------------------------------------------------
    # Build edges
    # --------------------------------------------------

    def create_edges(
        self,
        nodes
    ):

        edges = []

        for wall in self.walls:

            wall_nodes = []

            x1, y1 = (
                wall["geometry"]["start"]
            )

            x2, y2 = (
                wall["geometry"]["end"]
            )

            # --------------------------------------
            # Find nodes belonging to this wall
            # --------------------------------------

            for node in nodes:

                x, y = node["point"]

                if wall["orientation"] == "horizontal":

                    if (
                        abs(y - y1)
                        <= self.endpoint_tolerance
                        and
                        min(x1, x2)
                        - self.endpoint_tolerance
                        <= x
                        <=
                        max(x1, x2)
                        + self.endpoint_tolerance
                    ):
                        wall_nodes.append(node)

                else:

                    if (
                        abs(x - x1)
                        <= self.endpoint_tolerance
                        and
                        min(y1, y2)
                        - self.endpoint_tolerance
                        <= y
                        <=
                        max(y1, y2)
                        + self.endpoint_tolerance
                    ):
                        wall_nodes.append(node)

            # --------------------------------------
            # Sort nodes along wall
            # --------------------------------------

            if wall["orientation"] == "horizontal":

                wall_nodes.sort(
                    key=lambda n:
                    n["point"][0]
                )

            else:

                wall_nodes.sort(
                    key=lambda n:
                    n["point"][1]
                )

            # --------------------------------------
            # Connect consecutive nodes
            # --------------------------------------

            for i in range(
                len(wall_nodes) - 1
            ):

                node_a = wall_nodes[i]
                node_b = wall_nodes[i + 1]

                if (
                    node_a["id"]
                    ==
                    node_b["id"]
                ):
                    continue

                length = (
                    self.distance(
                        node_a["point"],
                        node_b["point"]
                    )
                )

                if length < 5:
                    continue

                edges.append({

                    "id":
                        f"E{len(edges)+1:03d}",

                    "wall":
                        wall["id"],

                    "from":
                        node_a["id"],

                    "to":
                        node_b["id"],

                    "length":
                        round(
                            length,
                            2
                        ),

                    "orientation":
                        wall["orientation"],

                    "thickness":
                        wall["thickness"]
                })

        return edges

    # --------------------------------------------------
    # Build graph
    # --------------------------------------------------

    def build(self):

        intersection_nodes = (
            self.create_intersection_nodes()
        )

        endpoint_nodes = (
            self.create_endpoint_nodes(
                intersection_nodes
            )
        )

        nodes = (
            intersection_nodes
            +
            endpoint_nodes
        )

        edges = (
            self.create_edges(
                nodes
            )
        )

        return {

            "nodes":
                nodes,

            "edges":
                edges,

            "node_count":
                len(nodes),

            "edge_count":
                len(edges)
        }