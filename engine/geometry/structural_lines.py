from geometry.geometry import LineGeometry


class StructuralLineAnalyzer:

    def __init__(self, lines):
        self.lines = lines

    def line_interval(self, line):
        x1, y1, x2, y2 = line

        angle = LineGeometry.angle(line)

        if angle <= 10 or angle >= 170:
            return (
                min(x1, x2),
                max(x1, x2)
            )

        if 80 <= angle <= 100:
            return (
                min(y1, y2),
                max(y1, y2)
            )

        return (
            0,
            LineGeometry.length(line)
        )

    def overlap_ratio(self, line1, line2):
        start1, end1 = self.line_interval(line1)
        start2, end2 = self.line_interval(line2)

        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)

        overlap = max(
            0,
            overlap_end - overlap_start
        )

        length1 = end1 - start1
        length2 = end2 - start2

        if length1 <= 0 or length2 <= 0:
            return 0

        shorter_length = min(
            length1,
            length2
        )

        return overlap / shorter_length

    def length_similarity(self, line1, line2):
        length1 = LineGeometry.length(line1)
        length2 = LineGeometry.length(line2)

        if length1 <= 0 or length2 <= 0:
            return 0

        return (
            min(length1, length2)
            / max(length1, length2)
        )

    def calculate_score(
        self,
        overlap,
        length_similarity,
        distance
    ):

        if distance <= 10:
            distance_score = 1.0

        elif distance >= 40:
            distance_score = 0.0

        else:
            distance_score = 1 - (
                (distance - 10) / 30
            )

        score = (
            overlap * 0.45
            + length_similarity * 0.35
            + distance_score * 0.20
        )

        return round(score, 3)

    def find_parallel_pairs(
        self,
        angle_tolerance=5,
        distance_min=8,
        distance_max=40,
        min_length=80,
        min_overlap=0.50
    ):

        candidates = []

        for i in range(len(self.lines)):

            line1 = self.lines[i]

            length1 = LineGeometry.length(line1)

            if length1 < min_length:
                continue

            for j in range(
                i + 1,
                len(self.lines)
            ):

                line2 = self.lines[j]

                length2 = LineGeometry.length(line2)

                if length2 < min_length:
                    continue

                if not LineGeometry.is_parallel(
                    line1,
                    line2,
                    angle_tolerance
                ):
                    continue

                midpoint1 = LineGeometry.midpoint(line1)

                distance = LineGeometry.point_to_line_distance(
                    midpoint1,
                    line2
                )

                if distance < distance_min:
                    continue

                if distance > distance_max:
                    continue

                overlap = self.overlap_ratio(
                    line1,
                    line2
                )

                if overlap < min_overlap:
                    continue

                similarity = self.length_similarity(
                    line1,
                    line2
                )

                score = self.calculate_score(
                    overlap,
                    similarity,
                    distance
                )

                candidates.append({
                    "line1_index": i,
                    "line2_index": j,
                    "distance": round(
                        distance,
                        2
                    ),
                    "overlap": round(
                        overlap,
                        3
                    ),
                    "length_similarity": round(
                        similarity,
                        3
                    ),
                    "score": score
                })

        candidates.sort(
            key=lambda candidate:
                candidate["score"],
            reverse=True
        )

        return candidates