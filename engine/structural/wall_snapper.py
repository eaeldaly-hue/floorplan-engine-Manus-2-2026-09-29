class WallSnapper:

    def __init__(
        self,
        walls,
        snap_tolerance=35,
        min_confidence=0.75
    ):
        self.walls = walls
        self.snap_tolerance = snap_tolerance
        self.min_confidence = min_confidence

    # --------------------------------------------------
    # Geometry helpers
    # --------------------------------------------------

    def get_horizontal_geometry(self, wall):

        x1, y1 = wall["geometry"]["start"]
        x2, y2 = wall["geometry"]["end"]

        return (
            min(x1, x2),
            max(x1, x2),
            y1
        )

    def get_vertical_geometry(self, wall):

        x1, y1 = wall["geometry"]["start"]
        x2, y2 = wall["geometry"]["end"]

        return (
            min(y1, y2),
            max(y1, y2),
            x1
        )

    # --------------------------------------------------
    # Confidence
    # --------------------------------------------------

    def calculate_confidence(
        self,
        dx,
        dy,
        thickness_tolerance
    ):

        distance = (
            dx ** 2
            +
            dy ** 2
        ) ** 0.5

        if distance <= thickness_tolerance:
            return 1.0

        if distance <= self.snap_tolerance:

            remaining = (
                self.snap_tolerance
                - distance
            )

            return round(
                0.5
                +
                0.5
                *
                (
                    remaining
                    /
                    self.snap_tolerance
                ),
                2
            )

        return 0.0

    # --------------------------------------------------
    # Detect candidate intersections
    # --------------------------------------------------

    def find_snap_candidates(self):

        horizontal = [
            wall
            for wall in self.walls
            if wall["orientation"] == "horizontal"
        ]

        vertical = [
            wall
            for wall in self.walls
            if wall["orientation"] == "vertical"
        ]

        candidates = []

        for h in horizontal:

            hx1, hx2, hy = (
                self.get_horizontal_geometry(h)
            )

            h_half = h["thickness"] / 2

            for v in vertical:

                vy1, vy2, vx = (
                    self.get_vertical_geometry(v)
                )

                v_half = v["thickness"] / 2

                # ------------------------------
                # Distance from vertical line
                # to horizontal segment
                # ------------------------------

                if vx < hx1:
                    dx = hx1 - vx
                elif vx > hx2:
                    dx = vx - hx2
                else:
                    dx = 0

                # ------------------------------
                # Distance from horizontal line
                # to vertical segment
                # ------------------------------

                if hy < vy1:
                    dy = vy1 - hy
                elif hy > vy2:
                    dy = hy - vy2
                else:
                    dy = 0

                centerline_distance = (
                    dx ** 2
                    +
                    dy ** 2
                ) ** 0.5

                thickness_tolerance = (
                    h_half
                    +
                    v_half
                )

                allowed_distance = max(
                    self.snap_tolerance,
                    thickness_tolerance
                )

                if centerline_distance > allowed_distance:
                    continue

                confidence = (
                    self.calculate_confidence(
                        dx,
                        dy,
                        thickness_tolerance
                    )
                )

                if confidence < self.min_confidence:
                    continue

                point = (
                    int(round(vx)),
                    int(round(hy))
                )

                candidates.append({

                    "horizontal_wall":
                        h["id"],

                    "vertical_wall":
                        v["id"],

                    "point":
                        point,

                    "dx":
                        round(dx, 2),

                    "dy":
                        round(dy, 2),

                    "centerline_distance":
                        round(
                            centerline_distance,
                            2
                        ),

                    "thickness_tolerance":
                        round(
                            thickness_tolerance,
                            2
                        ),

                    "confidence":
                        confidence
                })

        return candidates

    # --------------------------------------------------
    # Apply snapping
    # --------------------------------------------------

    def snap(self):

        candidates = (
            self.find_snap_candidates()
        )

        wall_map = {
            wall["id"]: wall
            for wall in self.walls
        }

        intersections = []

        for candidate in candidates:

            h = wall_map[
                candidate["horizontal_wall"]
            ]

            v = wall_map[
                candidate["vertical_wall"]
            ]

            x, y = candidate["point"]

            # --------------------------------------
            # Add intersection metadata
            # --------------------------------------

            intersection = {

                "point": (
                    x,
                    y
                ),

                "horizontal_wall":
                    h["id"],

                "vertical_wall":
                    v["id"],

                "confidence":
                    candidate["confidence"]
            }

            intersections.append(
                intersection
            )

        return intersections