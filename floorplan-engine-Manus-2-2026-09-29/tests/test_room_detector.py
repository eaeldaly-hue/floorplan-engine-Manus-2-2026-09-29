import cv2
import numpy as np

from engine.rooms.detect import RoomDetector


def test_exterior_is_flood_filled_from_padded_border_not_corner_wall():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (60, 60), (140, 140), 255, thickness=5)
    mask[0, 0] = 255  # A drawing stroke touches the source image corner.

    rooms = RoomDetector(mask, min_area_ratio=0.001, max_area_ratio=0.95).detect()

    assert len(rooms) == 1
    assert 5_000 < rooms[0]["area_pixels"] < 7_000
    assert all(0 <= x < 200 and 0 <= y < 200 for x, y in rooms[0]["polygon"])
