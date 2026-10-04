import cv2
import numpy as np


class RoomMaskBuilder:
    """
    Wall segments correctly leave a gap at every doorway (that gap *is*
    the door). That is exactly right for reporting wall geometry, but it
    means a naive flood fill leaks straight through every doorway and
    merges the whole floor plan into one "room".

    This builder draws the walls, then additionally bridges gaps between
    two collinear, same-thickness-class segments when the gap is in the
    size range of a door or window rather than a genuine open-concept
    opening. The bridge threshold is derived from the wall thickness
    actually measured in this drawing, not a fixed pixel count, so it
    scales with image resolution automatically.
    """

    def __init__(self, segments, shape, gap_factor=16.0, source_image=None):
        self.segments = segments
        self.height, self.width = shape
        self.gap_factor = gap_factor
        self.source_image = source_image
        self.last_unsealed_gap = None

    def _median_thickness(self):
        thicknesses = [s["thickness"] for s in self.segments]
        if not thicknesses:
            return 10
        thicknesses.sort()
        return thicknesses[len(thicknesses) // 2]

    def _draw_segment(self, mask, segment):
        x1, y1 = segment["start"]
        x2, y2 = segment["end"]
        thickness = max(2, int(segment["thickness"]))
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

    def _bridge_gaps(self, mask, segments, max_gap):
        by_orientation = {"horizontal": [], "vertical": []}
        for s in segments:
            by_orientation[s["orientation"]].append(s)

        for orientation, axis_index, range_index in (
            ("horizontal", 1, 0),
            ("vertical", 0, 1),
        ):
            items = by_orientation[orientation]

            def axis_pos(s):
                return (s["start"][axis_index] + s["end"][axis_index]) / 2

            # Cluster segments that sit on (roughly) the same centerline,
            # regardless of small measurement jitter in the axis position.
            items.sort(key=axis_pos)

            groups = []
            for s in items:
                s_thickness = s["thickness"]
                placed = False
                for group in groups:
                    if abs(axis_pos(s) - group["axis"]) <= max(
                        s_thickness, group["thickness"]
                    ):
                        group["items"].append(s)
                        group["axis"] = (
                            group["axis"] * group["thickness"]
                            + axis_pos(s) * s_thickness
                        ) / (group["thickness"] + s_thickness)
                        group["thickness"] = max(group["thickness"], s_thickness)
                        placed = True
                        break
                if not placed:
                    groups.append({
                        "axis": axis_pos(s),
                        "thickness": s["thickness"],
                        "items": [s],
                    })

            for group in groups:
                group_items = sorted(
                    group["items"], key=lambda s: s["start"][range_index]
                )

                for i in range(len(group_items) - 1):
                    a, b = group_items[i], group_items[i + 1]

                    a_end = a["end"][range_index]
                    b_start = b["start"][range_index]
                    gap = b_start - a_end

                    if 0 < gap <= max_gap:
                        thickness = max(2, int(min(a["thickness"], b["thickness"])))
                        axis_value = round(group["axis"])

                        if orientation == "horizontal":
                            p1 = (a_end, axis_value)
                            p2 = (b_start, axis_value)
                        else:
                            p1 = (axis_value, a_end)
                            p2 = (axis_value, b_start)

                        cv2.line(mask, p1, p2, 255, thickness)

    def _max_collinear_gap(self):
        """
        Largest real gap between two segments that sit on the same wall
        line, across the whole drawing. Used only to size the exterior
        envelope's closing kernel: the envelope needs to bridge whatever
        the biggest genuine discontinuity in the drawing is (in this
        kind of floor plan, that is usually a wall drawn with a much
        fainter or thinner stroke somewhere, not a real hole), while the
        finer interior gap-bridging in `_bridge_gaps` uses its own,
        smaller, door/window-scaled threshold.
        """

        by_orientation = {"horizontal": [], "vertical": []}
        for s in self.segments:
            by_orientation[s["orientation"]].append(s)

        max_gap = 0

        for orientation, axis_index, range_index in (
            ("horizontal", 1, 0),
            ("vertical", 0, 1),
        ):
            items = by_orientation[orientation]

            def axis_pos(s):
                return (s["start"][axis_index] + s["end"][axis_index]) / 2

            items.sort(key=axis_pos)

            groups = []
            for s in items:
                s_thickness = s["thickness"]
                placed = False
                for group in groups:
                    if abs(axis_pos(s) - group["axis"]) <= max(
                        s_thickness, group["thickness"]
                    ):
                        group["items"].append(s)
                        placed = True
                        break
                if not placed:
                    groups.append({"axis": axis_pos(s), "thickness": s_thickness, "items": [s]})

            for group in groups:
                group_items = sorted(group["items"], key=lambda s: s["start"][range_index])
                for i in range(len(group_items) - 1):
                    gap = group_items[i + 1]["start"][range_index] - group_items[i]["end"][range_index]
                    max_gap = max(max_gap, gap)

        return max_gap

    def _exterior_envelope(self):
        """
        Guarantees the outer building boundary is sealed for flood-fill
        purposes, even where it happens to be drawn with a thinner or
        lighter stroke than the rest of the walls (a real but awkward
        case: a big picture window, a faint anti-aliased line, a scan
        artifact). We don't need wall-accurate thickness here - we only
        need "inside vs outside" - so we fall back to the full drawing
        (every dark pixel, not just thick ones) and keep only its
        single largest outline, closed with a modest kernel.

        The kernel is deliberately capped low (a few % of the image
        diagonal). A kernel large enough to bridge an exceptionally wide
        gap would also fuse every nearby scrap of text and every close
        pair of interior walls into the same blob, which silently wrecks
        the whole envelope. A gap wider than this cap is reported, not
        silently patched over - see `unsealed_gap` on the result.
        """

        if self.source_image is None:
            return None, None

        gray = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        base_k = max(15, int(np.hypot(self.width, self.height) * 0.015))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (base_k, base_k))
        closed = cv2.dilate(binary, kernel, iterations=1)
        closed = cv2.erode(closed, kernel, iterations=1)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, None

        largest = max(contours, key=cv2.contourArea)

        filled = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.drawContours(filled, [largest], -1, 255, thickness=-1)

        border_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (base_k, base_k)
        )
        eroded = cv2.erode(filled, border_kernel, iterations=1)
        envelope = cv2.bitwise_and(filled, cv2.bitwise_not(eroded))

        max_gap = self._max_collinear_gap()
        unsealed_gap = max_gap if max_gap > base_k * 2 else None

        return envelope, unsealed_gap

    def build(self):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        for segment in self.segments:
            self._draw_segment(mask, segment)

        max_gap = self._median_thickness() * self.gap_factor
        self._bridge_gaps(mask, self.segments, max_gap)

        envelope, unsealed_gap = self._exterior_envelope()
        if envelope is not None:
            mask = cv2.bitwise_or(mask, envelope)

        # Close 1-2px anti-aliasing seams left at corners/junctions.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        self.last_unsealed_gap = unsealed_gap

        return mask
