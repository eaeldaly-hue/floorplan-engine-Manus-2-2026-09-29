import math


class LineGeometry:

    @staticmethod
    def length(line):
        x1, y1, x2, y2 = line

        dx = x2 - x1
        dy = y2 - y1

        return math.sqrt(dx ** 2 + dy ** 2)

    @staticmethod
    def angle(line):
        x1, y1, x2, y2 = line

        dx = x2 - x1
        dy = y2 - y1

        angle = math.degrees(math.atan2(dy, dx))

        return angle % 180

    @staticmethod
    def midpoint(line):
        x1, y1, x2, y2 = line

        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2
        )

    @staticmethod
    def angle_difference(angle1, angle2):
        difference = abs(angle1 - angle2)

        return min(difference, 180 - difference)

    @staticmethod
    def is_parallel(line1, line2, angle_tolerance=5):
        angle1 = LineGeometry.angle(line1)
        angle2 = LineGeometry.angle(line2)

        difference = LineGeometry.angle_difference(
            angle1,
            angle2
        )

        return difference <= angle_tolerance

    @staticmethod
    def point_to_line_distance(point, line):
        px, py = point
        x1, y1, x2, y2 = line

        dx = x2 - x1
        dy = y2 - y1

        line_length_squared = dx ** 2 + dy ** 2

        if line_length_squared == 0:
            return math.sqrt(
                (px - x1) ** 2 +
                (py - y1) ** 2
            )

        distance = abs(
            dy * px
            - dx * py
            + x2 * y1
            - y2 * x1
        ) / math.sqrt(line_length_squared)

        return distance

    @staticmethod
    def are_collinear(
        line1,
        line2,
        angle_tolerance=5,
        distance_tolerance=10
    ):
        if not LineGeometry.is_parallel(
            line1,
            line2,
            angle_tolerance
        ):
            return False

        midpoint2 = LineGeometry.midpoint(line2)

        distance = LineGeometry.point_to_line_distance(
            midpoint2,
            line1
        )

        return distance <= distance_tolerance


if __name__ == "__main__":

    line1 = (100, 200, 500, 200)

    line2 = (450, 203, 800, 203)

    line3 = (100, 300, 500, 300)

    print("Line 1 length:")
    print(LineGeometry.length(line1))

    print("\nLine 1 angle:")
    print(LineGeometry.angle(line1))

    print("\nLine 1 midpoint:")
    print(LineGeometry.midpoint(line1))

    print("\nLine 1 and Line 2 parallel:")
    print(LineGeometry.is_parallel(line1, line2))

    print("\nLine 1 and Line 2 collinear:")
    print(LineGeometry.are_collinear(line1, line2))

    print("\nLine 1 and Line 3 collinear:")
    print(LineGeometry.are_collinear(line1, line3))