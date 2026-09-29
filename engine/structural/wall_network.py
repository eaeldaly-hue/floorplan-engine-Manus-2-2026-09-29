class WallNetworkBuilder:

    def __init__(
        self,
        segments,
        line_tolerance=12,
        thickness_tolerance=10,
        max_gap=60,
        endpoint_tolerance=30
    ):
        self.segments = segments
        self.line_tolerance = line_tolerance
        self.thickness_tolerance = thickness_tolerance
        self.max_gap = max_gap
        self.endpoint_tolerance = endpoint_tolerance

    # --------------------------------------------------
    # Basic Geometry
    # --------------------------------------------------

    def segment_axis_position(self, segment):

        x1, y1 = segment["start"]
        x2, y2 = segment["end"]

        if segment["orientation"] == "horizontal":
            return (y1 + y2) / 2

        return (x1 + x2) / 2

    def segment_range(self, segment):

        x1, y1 = segment["start"]
        x2, y2 = segment["end"]

        if segment["orientation"] == "horizontal":
            return min(x1, x2), max(x1, x2)

        return min(y1, y2), max(y1, y2)

    def segment_thickness(self, segment):

        return segment.get(
            "wall_thickness",
            0
        )

    # --------------------------------------------------
    # Compatibility
    # --------------------------------------------------

    def compatible(self, a, b):

        if a["orientation"] != b["orientation"]:
            return False

        axis_a = self.segment_axis_position(a)
        axis_b = self.segment_axis_position(b)

        if abs(axis_a - axis_b) > self.line_tolerance:
            return False

        thickness_a = self.segment_thickness(a)
        thickness_b = self.segment_thickness(b)

        if abs(thickness_a - thickness_b) > self.thickness_tolerance:
            return False

        a_start, a_end = self.segment_range(a)
        b_start, b_end = self.segment_range(b)

        if a_end < b_start:
            gap = b_start - a_end
        elif b_end < a_start:
            gap = a_start - b_end
        else:
            gap = 0

        if gap > self.max_gap:
            return False

        return True

    # --------------------------------------------------
    # Group compatible segments
    # --------------------------------------------------

    def build_groups(self):

        groups = []

        for segment in self.segments:

            assigned = False

            for group in groups:

                if any(
                    self.compatible(
                        segment,
                        existing
                    )
                    for existing in group["segments"]
                ):

                    group["segments"].append(segment)
                    assigned = True
                    break

            if not assigned:

                groups.append({
                    "segments": [segment]
                })

        return groups

    # --------------------------------------------------
    # Build Wall
    # --------------------------------------------------

    def build_wall(self, group, wall_id):

        segments = group["segments"]

        orientation = segments[0]["orientation"]

        axis_values = [
            self.segment_axis_position(s)
            for s in segments
        ]

        axis = round(
            sum(axis_values) / len(axis_values)
        )

        ranges = [
            self.segment_range(s)
            for s in segments
        ]

        ranges.sort()

        intervals = []

        current_start, current_end = ranges[0]

        for start, end in ranges[1:]:

            if start <= current_end:

                current_end = max(
                    current_end,
                    end
                )

            elif start - current_end <= self.max_gap:

                intervals.append(
                    (current_start, current_end)
                )

                current_start = start
                current_end = end

            else:

                intervals.append(
                    (current_start, current_end)
                )

                current_start = start
                current_end = end

        intervals.append(
            (current_start, current_end)
        )

        # ----------------------------------------------
        # Gaps
        # ----------------------------------------------

        gaps = []

        for i in range(
            len(intervals) - 1
        ):

            gap_start = intervals[i][1]
            gap_end = intervals[i + 1][0]

            gaps.append({
                "start": gap_start,
                "end": gap_end,
                "width": gap_end - gap_start
            })

        # ----------------------------------------------
        # Overall geometry
        # ----------------------------------------------

        start = min(
            interval[0]
            for interval in intervals
        )

        end = max(
            interval[1]
            for interval in intervals
        )

        if orientation == "horizontal":

            geometry = {
                "start": (
                    start,
                    axis
                ),
                "end": (
                    end,
                    axis
                )
            }

        else:

            geometry = {
                "start": (
                    axis,
                    start
                ),
                "end": (
                    axis,
                    end
                )
            }

        # ----------------------------------------------
        # Thickness
        # ----------------------------------------------

        thicknesses = [
            self.segment_thickness(s)
            for s in segments
            if self.segment_thickness(s) > 0
        ]

        if thicknesses:

            average_thickness = (
                sum(thicknesses)
                /
                len(thicknesses)
            )

        else:

            average_thickness = 0

        # ----------------------------------------------
        # Confidence
        # ----------------------------------------------

        confidence = self.calculate_confidence(
            segments,
            gaps
        )

        return {
            "id": wall_id,
            "orientation": orientation,
            "geometry": geometry,
            "segments": segments,
            "intervals": intervals,
            "gaps": gaps,
            "length": round(
                end - start,
                2
            ),
            "thickness": round(
                average_thickness,
                2
            ),
            "segment_count": len(
                segments
            ),
            "confidence": confidence
        }

    # --------------------------------------------------
    # Confidence
    # --------------------------------------------------

    def calculate_confidence(
        self,
        segments,
        gaps
    ):

        score = 0.0

        # More supporting segments
        score += min(
            len(segments) * 0.15,
            0.45
        )

        # Similar thickness
        thicknesses = [
            self.segment_thickness(s)
            for s in segments
            if self.segment_thickness(s) > 0
        ]

        if thicknesses:

            average = (
                sum(thicknesses)
                /
                len(thicknesses)
            )

            variation = max(
                abs(
                    t - average
                )
                for t in thicknesses
            )

            if variation <= 3:
                score += 0.30

            elif variation <= 8:
                score += 0.20

            else:
                score += 0.05

        # Long wall
        total_length = sum(
            s["length"]
            for s in segments
        )

        if total_length >= 500:
            score += 0.25

        elif total_length >= 250:
            score += 0.15

        else:
            score += 0.05

        # Gap penalty
        if len(gaps) == 0:
            score += 0.10

        elif len(gaps) <= 2:
            score += 0.05

        return round(
            min(score, 1.0),
            3
        )

    # --------------------------------------------------
    # Check if coordinate lies inside wall interval
    # --------------------------------------------------

    def coordinate_in_intervals(
        self,
        coordinate,
        intervals
    ):

        for start, end in intervals:

            if (
                start <= coordinate <= end
            ):
                return True

        return False

    # --------------------------------------------------
    # Intersection Detection
    # --------------------------------------------------

    def find_intersections(
        self,
        walls
    ):

        intersections = []

        horizontal = [
            wall
            for wall in walls
            if wall["orientation"]
            == "horizontal"
        ]

        vertical = [
            wall
            for wall in walls
            if wall["orientation"]
            == "vertical"
        ]

        for h in horizontal:

            hx1, hy = h["geometry"]["start"]
            hx2, _ = h["geometry"]["end"]

            for v in vertical:

                vx, vy1 = v["geometry"]["start"]
                _, vy2 = v["geometry"]["end"]

                # --------------------------------------
                # Check horizontal wall intervals
                # --------------------------------------

                horizontal_hit = (
                    self.coordinate_in_intervals(
                        vx,
                        h["intervals"]
                    )
                )

                if not horizontal_hit:
                    continue

                # --------------------------------------
                # Check vertical wall intervals
                # --------------------------------------

                vertical_hit = (
                    self.coordinate_in_intervals(
                        hy,
                        v["intervals"]
                    )
                )

                if not vertical_hit:
                    continue

                # --------------------------------------
                # Actual intersection
                # --------------------------------------

                point = (
                    vx,
                    hy
                )

                # --------------------------------------
                # Determine endpoint proximity
                # --------------------------------------

                horizontal_endpoint = False

                for start, end in h["intervals"]:

                    if (
                        abs(vx - start)
                        <= self.endpoint_tolerance
                        or
                        abs(vx - end)
                        <= self.endpoint_tolerance
                    ):

                        horizontal_endpoint = True
                        break

                vertical_endpoint = False

                for start, end in v["intervals"]:

                    if (
                        abs(hy - start)
                        <= self.endpoint_tolerance
                        or
                        abs(hy - end)
                        <= self.endpoint_tolerance
                    ):

                        vertical_endpoint = True
                        break

                # --------------------------------------
                # Intersection type
                # --------------------------------------

                if (
                    horizontal_endpoint
                    and vertical_endpoint
                ):

                    intersection_type = (
                        "endpoint"
                    )

                elif (
                    horizontal_endpoint
                    or vertical_endpoint
                ):

                    intersection_type = "T"

                else:

                    intersection_type = "cross"

                intersections.append({
                    "horizontal_wall": h["id"],
                    "vertical_wall": v["id"],
                    "point": point,
                    "type": intersection_type
                })
        print()
        print("Intersection Debug")
        print("-------------------")

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

        print(
            "Horizontal walls:",
            len(horizontal)
        )

        print(
            "Vertical walls:",
            len(vertical)
        )

        print(
            "Possible pairs:",
            len(horizontal) * len(vertical)
        )

        for h in horizontal:

            hx1, hy = h["geometry"]["start"]
            hx2, _ = h["geometry"]["end"]

            for v in vertical:

                vx, vy1 = v["geometry"]["start"]
                _, vy2 = v["geometry"]["end"]

                horizontal_hit = (
                    self.coordinate_in_intervals(
                        vx,
                        h["intervals"]
                    )
                )

                vertical_hit = (
                    self.coordinate_in_intervals(
                        hy,
                        v["intervals"]
                    )
                )

                if horizontal_hit and vertical_hit:

                    print(
                        "CANDIDATE:",
                        h["id"],
                        "+",
                        v["id"],
                        "=>",
                        (vx, hy)
                    )

        return intersections

    # --------------------------------------------------
    # Build Network
    # --------------------------------------------------

    def build(self):

        groups = self.build_groups()

        walls = []

        for index, group in enumerate(
            groups,
            start=1
        ):

            wall = self.build_wall(
                group,
                wall_id=f"W{index:03d}"
            )

            walls.append(wall)

        intersections = (
            self.find_intersections(
                walls
            )
        )

        return {
            "walls": walls,
            "intersections": intersections,
            "wall_count": len(walls),
            "intersection_count": len(
                intersections
            )
        }