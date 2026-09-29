from geometry.geometry import LineGeometry


class LineMerger:

    def __init__(self, lines):
        self.lines = lines

    def can_merge(self, line1, line2, max_gap=20):
        """
        Check whether two lines can be merged.
        """

        # Must be approximately parallel
        if not LineGeometry.is_parallel(
            line1,
            line2,
            angle_tolerance=5
        ):
            return False

        # Must be approximately on the same line
        if not LineGeometry.are_collinear(
            line1,
            line2,
            angle_tolerance=5,
            distance_tolerance=10
        ):
            return False

        # Horizontal
        if abs(line1[1] - line1[3]) <= abs(line1[0] - line1[2]):

            line1_start = min(line1[0], line1[2])
            line1_end = max(line1[0], line1[2])

            line2_start = min(line2[0], line2[2])
            line2_end = max(line2[0], line2[2])

        # Vertical
        else:

            line1_start = min(line1[1], line1[3])
            line1_end = max(line1[1], line1[3])

            line2_start = min(line2[1], line2[3])
            line2_end = max(line2[1], line2[3])

        # Calculate gap between the two intervals
        if line1_end < line2_start:
            gap = line2_start - line1_end

        elif line2_end < line1_start:
            gap = line1_start - line2_end

        else:
            gap = 0

        return gap <= max_gap

    def merge_pair(self, line1, line2):
        """
        Merge two compatible lines.
        """

        if not self.can_merge(line1, line2):
            return None

        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Horizontal
        if abs(y1 - y2) <= abs(x1 - x2):

            min_x = min(x1, x2, x3, x4)
            max_x = max(x1, x2, x3, x4)

            average_y = round(
                (y1 + y2 + y3 + y4) / 4
            )

            return (
                min_x,
                average_y,
                max_x,
                average_y
            )

        # Vertical
        else:

            min_y = min(y1, y2, y3, y4)
            max_y = max(y1, y2, y3, y4)

            average_x = round(
                (x1 + x2 + x3 + x4) / 4
            )

            return (
                average_x,
                min_y,
                average_x,
                max_y
            )

    def merge_all(self):
        """
        Merge all compatible lines.
        """

        lines = self.lines.copy()

        changed = True

        while changed:

            changed = False
            merged_lines = []
            used = set()

            for i in range(len(lines)):

                if i in used:
                    continue

                current = lines[i]

                for j in range(i + 1, len(lines)):

                    if j in used:
                        continue

                    merged = self.merge_pair(
                        current,
                        lines[j]
                    )

                    if merged is not None:

                        current = merged

                        used.add(j)

                        changed = True

                merged_lines.append(current)
                used.add(i)

            lines = merged_lines

        return lines


if __name__ == "__main__":

    lines = [
        (100, 200, 400, 200),
        (390, 202, 700, 202),
        (720, 201, 900, 201),

        (100, 300, 300, 300),
        (305, 301, 500, 301),

        (200, 100, 200, 400),
        (202, 390, 202, 600)
    ]

    merger = LineMerger(lines)

    print("Original lines:")
    print(len(lines))

    merged_lines = merger.merge_all()

    print("Merged lines:")
    print(len(merged_lines))

    for line in merged_lines:
        print(line)