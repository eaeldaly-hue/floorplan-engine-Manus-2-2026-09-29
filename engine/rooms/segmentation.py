from geometry.geometry import LineGeometry


class RoomSegmenter:

    def __init__(self, image, walls):
        self.image = image
        self.walls = walls

    def is_horizontal(self, wall):
        start = wall["geometry"]["start"]
        end = wall["geometry"]["end"]

        return abs(end[0] - start[0]) >= abs(end[1] - start[1])

    def get_horizontal_walls(self):
        return [
            wall
            for wall in self.walls
            if self.is_horizontal(wall)
        ]

    def get_vertical_walls(self):
        return [
            wall
            for wall in self.walls
            if not self.is_horizontal(wall)
        ]

    def intersection(self, horizontal, vertical):

        h_start = horizontal["geometry"]["start"]
        h_end = horizontal["geometry"]["end"]

        v_start = vertical["geometry"]["start"]
        v_end = vertical["geometry"]["end"]

        hx1 = min(h_start[0], h_end[0])
        hx2 = max(h_start[0], h_end[0])
        hy = (h_start[1] + h_end[1]) / 2

        vx = (v_start[0] + v_end[0]) / 2
        vy1 = min(v_start[1], v_end[1])
        vy2 = max(v_start[1], v_end[1])

        if hx1 <= vx <= hx2 and vy1 <= hy <= vy2:
            return (round(vx), round(hy))

        return None

    def build_intersections(self):

        horizontal = self.get_horizontal_walls()
        vertical = self.get_vertical_walls()

        intersections = []

        for h in horizontal:
            for v in vertical:

                point = self.intersection(h, v)

                if point is not None:
                    intersections.append(point)

        return intersections

    def detect_rooms(self):

        horizontal = self.get_horizontal_walls()
        vertical = self.get_vertical_walls()

        rooms = []

        room_id = 1

        for top in horizontal:

            for bottom in horizontal:

                if top is bottom:
                    continue

                top_y = (
                    top["geometry"]["start"][1]
                    + top["geometry"]["end"][1]
                ) / 2

                bottom_y = (
                    bottom["geometry"]["start"][1]
                    + bottom["geometry"]["end"][1]
                ) / 2

                if bottom_y <= top_y:
                    continue

                for left in vertical:

                    for right in vertical:

                        if left is right:
                            continue

                        left_x = (
                            left["geometry"]["start"][0]
                            + left["geometry"]["end"][0]
                        ) / 2

                        right_x = (
                            right["geometry"]["start"][0]
                            + right["geometry"]["end"][0]
                        ) / 2

                        if right_x <= left_x:
                            continue

                        top_left = self.intersection(top, left)
                        top_right = self.intersection(top, right)
                        bottom_left = self.intersection(bottom, left)
                        bottom_right = self.intersection(bottom, right)

                        if not all([
                            top_left,
                            top_right,
                            bottom_left,
                            bottom_right
                        ]):
                            continue

                        width = right_x - left_x
                        height = bottom_y - top_y

                        if width < 100 or height < 100:
                            continue

                        area = width * height

                        room = {
                            "id": f"room_{room_id}",
                            "polygon": [
                                top_left,
                                top_right,
                                bottom_right,
                                bottom_left
                            ],
                            "center": (
                                round((left_x + right_x) / 2),
                                round((top_y + bottom_y) / 2)
                            ),
                            "width_pixels": round(width),
                            "height_pixels": round(height),
                            "area_pixels": round(area)
                        }

                        rooms.append(room)

                        room_id += 1

        return rooms