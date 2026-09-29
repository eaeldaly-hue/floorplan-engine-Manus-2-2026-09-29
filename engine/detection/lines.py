import cv2
import numpy as np
import math


class LineDetector:

    def __init__(self, edges):
        self.edges = edges

    def detect(self):
        lines = cv2.HoughLinesP(
            self.edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=50,
            maxLineGap=10
        )

        if lines is None:
            return []

        detected_lines = []

        for line in lines:
            x1, y1, x2, y2 = line

            detected_lines.append(
                (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                )
            )

        return detected_lines

    def classify_angle(self, line):
        x1, y1, x2, y2 = line

        dx = x2 - x1
        dy = y2 - y1

        angle = math.degrees(math.atan2(dy, dx))

        # Normalize angle to 0-180
        angle = angle % 180

        if angle <= 10 or angle >= 170:
            direction = "horizontal"

        elif 80 <= angle <= 100:
            direction = "vertical"

        else:
            direction = "diagonal"

        return {
            "line": line,
            "angle": angle,
            "direction": direction
        }