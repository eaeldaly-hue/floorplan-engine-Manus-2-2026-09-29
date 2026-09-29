import cv2
import numpy as np


class TextMasker:

    def __init__(self, image):
        self.image = image

    def create_mask(self):

        gray = cv2.cvtColor(
            self.image,
            cv2.COLOR_BGR2GRAY
        )

        binary = cv2.threshold(
            gray,
            200,
            255,
            cv2.THRESH_BINARY_INV
        )[1]

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )

        text_mask = np.zeros_like(gray)

        for i in range(1, num_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if area < 10 or area > 2000:
                continue

            aspect_ratio = w / max(h, 1)

            if (
                h <= 80
                and w <= 300
                and aspect_ratio <= 15
            ):
                cv2.rectangle(
                    text_mask,
                    (x - 2, y - 2),
                    (x + w + 2, y + h + 2),
                    255,
                    -1
                )

        return text_mask

    def remove_text(self):

        mask = self.create_mask()

        result = self.image.copy()

        result[mask > 0] = 255

        return result, mask