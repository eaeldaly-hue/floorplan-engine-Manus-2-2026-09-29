from geometry.geometry import LineGeometry


class LineAnalyzer:

    def __init__(self, lines):
        self.lines = lines

    def analyze(self):

        horizontal = []
        vertical = []
        diagonal = []

        very_short = []
        medium = []
        long = []

        for line in self.lines:

            length = LineGeometry.length(line)
            angle = LineGeometry.angle(line)

            # Direction
            if angle <= 10 or angle >= 170:
                horizontal.append(line)

            elif 80 <= angle <= 100:
                vertical.append(line)

            else:
                diagonal.append(line)

            # Length
            if length < 50:
                very_short.append(line)

            elif length < 200:
                medium.append(line)

            else:
                long.append(line)

        return {
            "total": len(self.lines),

            "horizontal": len(horizontal),
            "vertical": len(vertical),
            "diagonal": len(diagonal),

            "very_short": len(very_short),
            "medium": len(medium),
            "long": len(long)
        }


if __name__ == "__main__":

    test_lines = [
        (100, 200, 400, 200),
        (100, 300, 500, 300),
        (200, 100, 200, 600),
        (50, 50, 80, 50),
        (100, 100, 250, 150)
    ]

    analyzer = LineAnalyzer(test_lines)

    result = analyzer.analyze()

    print("Line Analysis")
    print("----------------")

    for key, value in result.items():
        print(f"{key}: {value}")