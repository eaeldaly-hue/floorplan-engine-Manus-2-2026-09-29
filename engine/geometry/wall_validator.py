from geometry.geometry import LineGeometry


class WallValidator:

    def __init__(
        self,
        walls,
        min_length=80,
        min_confidence=0.60
    ):
        self.walls = walls
        self.min_length = min_length
        self.min_confidence = min_confidence

    def is_valid(self, wall):
        start = wall["geometry"]["start"]
        end = wall["geometry"]["end"]

        line = (
            start[0],
            start[1],
            end[0],
            end[1]
        )

        length = LineGeometry.length(line)

        if length < self.min_length:
            return False

        if wall["confidence"] < self.min_confidence:
            return False

        return True

    def validate(self):
        valid_walls = []

        for wall in self.walls:
            if self.is_valid(wall):
                valid_walls.append(wall)

        return valid_walls

    def is_horizontal(self, wall):
        start = wall["geometry"]["start"]
        end = wall["geometry"]["end"]

        return abs(end[0] - start[0]) >= abs(end[1] - start[1])

    def can_merge(self, wall1, wall2):
        horizontal1 = self.is_horizontal(wall1)
        horizontal2 = self.is_horizontal(wall2)

        if horizontal1 != horizontal2:
            return False

        if horizontal1:

            y1 = (
                wall1["geometry"]["start"][1]
                + wall1["geometry"]["end"][1]
            ) / 2

            y2 = (
                wall2["geometry"]["start"][1]
                + wall2["geometry"]["end"][1]
            ) / 2

            if abs(y1 - y2) > 25:
                return False

            a1 = min(
                wall1["geometry"]["start"][0],
                wall1["geometry"]["end"][0]
            )

            b1 = max(
                wall1["geometry"]["start"][0],
                wall1["geometry"]["end"][0]
            )

            a2 = min(
                wall2["geometry"]["start"][0],
                wall2["geometry"]["end"][0]
            )

            b2 = max(
                wall2["geometry"]["start"][0],
                wall2["geometry"]["end"][0]
            )

        else:

            x1 = (
                wall1["geometry"]["start"][0]
                + wall1["geometry"]["end"][0]
            ) / 2

            x2 = (
                wall2["geometry"]["start"][0]
                + wall2["geometry"]["end"][0]
            ) / 2

            if abs(x1 - x2) > 25:
                return False

            a1 = min(
                wall1["geometry"]["start"][1],
                wall1["geometry"]["end"][1]
            )

            b1 = max(
                wall1["geometry"]["start"][1],
                wall1["geometry"]["end"][1]
            )

            a2 = min(
                wall2["geometry"]["start"][1],
                wall2["geometry"]["end"][1]
            )

            b2 = max(
                wall2["geometry"]["start"][1],
                wall2["geometry"]["end"][1]
            )

        if b1 < a2:
            gap = a2 - b1
        elif b2 < a1:
            gap = a1 - b2
        else:
            gap = 0

        return gap <= 60

    def merge_pair(self, wall1, wall2):
        if not self.can_merge(wall1, wall2):
            return None

        horizontal = self.is_horizontal(wall1)

        if horizontal:

            x_values = [
                wall1["geometry"]["start"][0],
                wall1["geometry"]["end"][0],
                wall2["geometry"]["start"][0],
                wall2["geometry"]["end"][0]
            ]

            y = round((
                wall1["geometry"]["start"][1]
                + wall1["geometry"]["end"][1]
                + wall2["geometry"]["start"][1]
                + wall2["geometry"]["end"][1]
            ) / 4)

            return {
                "geometry": {
                    "start": (min(x_values), y),
                    "end": (max(x_values), y)
                },
                "thickness": max(
                    wall1["thickness"],
                    wall2["thickness"]
                ),
                "orientation": "horizontal",
                "confidence": max(
                    wall1["confidence"],
                    wall2["confidence"]
                )
            }

        y_values = [
            wall1["geometry"]["start"][1],
            wall1["geometry"]["end"][1],
            wall2["geometry"]["start"][1],
            wall2["geometry"]["end"][1]
        ]

        x = round((
            wall1["geometry"]["start"][0]
            + wall1["geometry"]["end"][0]
            + wall2["geometry"]["start"][0]
            + wall2["geometry"]["end"][0]
        ) / 4)

        return {
            "geometry": {
                "start": (x, min(y_values)),
                "end": (x, max(y_values))
            },
            "thickness": max(
                wall1["thickness"],
                wall2["thickness"]
            ),
            "orientation": "vertical",
            "confidence": max(
                wall1["confidence"],
                wall2["confidence"]
            )
        }

    def validate_and_merge(self):
        walls = self.validate()

        changed = True

        while changed:

            changed = False
            merged_walls = []
            used = set()

            for i in range(len(walls)):

                if i in used:
                    continue

                current = walls[i]

                for j in range(i + 1, len(walls)):

                    if j in used:
                        continue

                    merged = self.merge_pair(
                        current,
                        walls[j]
                    )

                    if merged is not None:
                        current = merged
                        used.add(j)
                        changed = True

                merged_walls.append(current)
                used.add(i)

            walls = merged_walls

        return walls