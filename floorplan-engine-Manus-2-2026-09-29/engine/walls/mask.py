import cv2
import numpy as np


class WallMaskBuilder:
    """
    Builds a clean binary mask of "wall pixels" from a floor plan image.

    Design notes
    ------------
    Floor plans are usually drawn as solid, thick, dark strokes for walls,
    and thin strokes for everything else structural (door swings, window
    ticks, dimension lines). We lean on that thickness difference instead
    of hardcoded pixel counts, so the same code works whether the image is
    800px or 4000px wide: every size-based threshold below is derived from
    ``reference_size`` (the image diagonal), not a fixed pixel count.
    """

    def __init__(self, image, min_wall_thickness_ratio=0.0035):
        self.image = image
        self.height, self.width = image.shape[:2]
        self.reference_size = float(np.hypot(self.width, self.height))

        # Minimum stroke thickness (in pixels) that counts as "wall" rather
        # than a thin annotation line. Scales with image resolution.
        self.min_wall_thickness = max(
            3,
            int(round(self.reference_size * min_wall_thickness_ratio))
        )

    def _binary_from_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Otsu picks the split point automatically instead of a fixed 200,
        # so it still works on scans with a slightly grey background.
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        return binary

    def _connect_disconnected_pieces(self, mask, max_bridges=8):
        """
        Real walls interconnect at every corner, so the true wall network
        should be a single connected shape. If it isn't, that means one
        specific stroke was too faint/thin to survive thresholding (a
        real but awkward case in scanned or lightly-rendered drawings).
        Rather than guessing a global closing kernel size (which risks
        bridging unrelated walls together across the whole image), we
        bridge exactly the disconnected pieces, at their closest
        approach, and nowhere else.
        """

        min_component_area = max(50, self.min_wall_thickness * 20)

        for _ in range(max_bridges):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            components = [
                i for i in range(1, num_labels)
                if stats[i, cv2.CC_STAT_AREA] >= min_component_area
            ]
            if len(components) <= 1:
                break

            components.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
            main_label = components[0]

            main_points = np.column_stack(
                np.where(labels == main_label)
            )[:, ::-1]  # (x, y)

            # Compare against boundary points only, for speed.
            main_contours, _ = cv2.findContours(
                (labels == main_label).astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
            )
            if main_contours:
                main_points = np.concatenate(
                    [c.reshape(-1, 2) for c in main_contours]
                )

            other_label = components[1]
            other_contours, _ = cv2.findContours(
                (labels == other_label).astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
            )
            other_points = np.concatenate(
                [c.reshape(-1, 2) for c in other_contours]
            )

            # Nearest pair between the two point sets (vectorised, chunked
            # so it stays cheap even for a few thousand boundary points).
            best_dist = None
            best_pair = None
            chunk = 500
            for start in range(0, len(other_points), chunk):
                chunk_pts = other_points[start:start + chunk]
                diffs = main_points[:, None, :] - chunk_pts[None, :, :]
                dists = np.sum(diffs ** 2, axis=2)
                idx = np.unravel_index(np.argmin(dists), dists.shape)
                d = dists[idx]
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_pair = (
                        tuple(main_points[idx[0]]),
                        tuple(chunk_pts[idx[1]]),
                    )

            cv2.line(
                mask, best_pair[0], best_pair[1], 255,
                thickness=self.min_wall_thickness,
            )

        return mask

    def build(self):
        """
        Returns a uint8 mask (0/255) where 255 marks pixels that belong to
        a wall stroke thick enough to be structural.
        """

        binary = self._binary_from_image()

        # Erode first: thin strokes (door swings, window ticks, leftover
        # text fragments) disappear completely; thick wall strokes survive
        # but shrink. Kernel size = the thinnest stroke we still want to
        # call a "wall".
        k = self.min_wall_thickness
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        eroded = cv2.erode(binary, erode_kernel, iterations=1)

        # Grow back only the wall-sized regions that survived, restoring
        # their original thickness without bringing the thin lines back.
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (k + 2, k + 2)
        )
        restored = cv2.dilate(eroded, dilate_kernel, iterations=1)

        # Keep the restored shape only where the original binary was also
        # foreground, so we don't balloon past the true wall edges.
        wall_mask = cv2.bitwise_and(restored, binary)

        # Close tiny gaps left by anti-aliasing at corners/junctions.
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        wall_mask = cv2.morphologyEx(
            wall_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1
        )

        wall_mask = self._connect_disconnected_pieces(wall_mask)

        return wall_mask
