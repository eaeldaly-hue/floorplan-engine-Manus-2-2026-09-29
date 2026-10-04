from enum import Enum

from engine.geometry_v2.geometry import GeometryEngine, Line


class Orientation(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    DIAGONAL = "diagonal"


class OrientationAnalyzer:
    """
    Classifies line orientation using geometric angle analysis.

    The analyzer does not know anything about walls, doors, windows,
    or rooms. It only determines the orientation of a line.
    """

    def __init__(
        self,
        horizontal_tolerance: float = 10.0,
        vertical_tolerance: float = 10.0,
    ):
        self.horizontal_tolerance = horizontal_tolerance
        self.vertical_tolerance = vertical_tolerance

    def classify(self, line: Line) -> Orientation:
        """
        Classify a line as horizontal, vertical, or diagonal.
        """

        angle = GeometryEngine.angle(line)

        distance_from_horizontal = min(
            angle,
            180.0 - angle,
        )

        distance_from_vertical = abs(
            angle - 90.0
        )

        if distance_from_horizontal <= self.horizontal_tolerance:
            return Orientation.HORIZONTAL

        if distance_from_vertical <= self.vertical_tolerance:
            return Orientation.VERTICAL

        return Orientation.DIAGONAL

    def analyze(self, line: Line) -> dict:
        """
        Return orientation information for a line.
        """

        angle = GeometryEngine.angle(line)
        orientation = self.classify(line)

        return {
            "angle": angle,
            "orientation": orientation.value,
        }


if __name__ == "__main__":

    analyzer = OrientationAnalyzer()

    test_lines = {
        "horizontal": (100, 200, 500, 200),
        "almost_horizontal": (100, 200, 500, 240),
        "vertical": (300, 100, 300, 500),
        "almost_vertical": (300, 100, 340, 500),
        "diagonal": (100, 100, 300, 200),
    }

    print("Orientation V2 Test")
    print("--------------------")

    for name, line in test_lines.items():
        result = analyzer.analyze(line)

        print(
            f"{name:20} "
            f"angle={result['angle']:.2f}° "
            f"orientation={result['orientation']}"
        )