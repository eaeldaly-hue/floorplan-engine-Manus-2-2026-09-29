import cv2


class ImagePreprocessor:

    def __init__(self, image):
        self.original = image

    def grayscale(self):
        return cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)

    def denoise(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def threshold(self, image):
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10
        )

    def edges(self, image):
        return cv2.Canny(image, 50, 150)