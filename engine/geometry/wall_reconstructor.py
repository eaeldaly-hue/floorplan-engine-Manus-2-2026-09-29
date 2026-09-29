from geometry.geometry import LineGeometry


class WallReconstructor:

    def __init__(self, lines, wall_pairs):
        self.lines = lines
        self.wall_pairs = wall_pairs

    def get_orientation(self, line):

        angle = LineGeometry.angle(line)

        if angle <= 10 or angle >= 170:
            return "horizontal"

        if 80 <= angle <= 100:
            return "vertical"

        return None

    def reconstruct_wall(self, pair):

        line1 = self.lines[pair["line1_index"]]
        line2 = self.lines[pair["line2_index"]]

        orientation = self.get_orientation(line1)

        if orientation is None:
            return None

        # Make sure both lines have the same orientation.
        if self.get_orientation(line2) != orientation:
            return None

        # --------------------------------------------------
        # Horizontal Wall
        # --------------------------------------------------

        if orientation == "horizontal":

            x_values = [
                line1[0],
                line1[2],
                line2[0],
                line2[2]
            ]

            y1 = (
                line1[1]
                + line1[3]
            ) / 2

            y2 = (
                line2[1]
                + line2[3]
            ) / 2

            center_y = (
                y1 + y2
            ) / 2

            start = (
                round(min(x_values)),
                round(center_y)
            )

            end = (
                round(max(x_values)),
                round(center_y)
            )

        # --------------------------------------------------
        # Vertical Wall
        # --------------------------------------------------

        else:

            y_values = [
                line1[1],
                line1[3],
                line2[1],
                line2[3]
            ]

            x1 = (
                line1[0]
                + line1[2]
            ) / 2

            x2 = (
                line2[0]
                + line2[2]
            ) / 2

            center_x = (
                x1 + x2
            ) / 2

            start = (
                round(center_x),
                round(min(y_values))
            )

            end = (
                round(center_x),
                round(max(y_values))
            )

        length = LineGeometry.length(
            (
                start[0],
                start[1],
                end[0],
                end[1]
            )
        )

        # Ignore extremely short walls.
        if length < 80:
            return None

        thickness = pair["distance"]

        return {
            "geometry": {
                "start": start,
                "end": end
            },
            "thickness": round(
                thickness,
                2
            ),
            "orientation": orientation,
            "confidence": pair["score"],
            "source_lines": [
                pair["line1_index"],
                pair["line2_index"]
            ]
        }

    def reconstruct_all(self):

        walls = []

        for pair in self.wall_pairs:

            wall = self.reconstruct_wall(
                pair
            )

            if wall is not None:
                walls.append(wall)

        return walls