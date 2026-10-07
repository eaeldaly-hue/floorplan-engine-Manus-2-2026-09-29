"""Wall mask (thick, solid strokes) — thin wrapper over engine.structure.

Kept for backward compatibility with older callers and diagnostics. The wall mask is
produced by the same thickness filter as the production structure pass: the erosion
kernel is chosen from the image (see engine.structure.estimate_wall_kernel), small
medium-thickness jamb blocks attached to walls are kept, and specks are removed.

The previous version also "connected disconnected pieces" by drawing straight wall
lines between the two largest components (up to 8 times). That fabricated walls that
are not in the drawing — walls interrupted by doors and windows are genuinely
disconnected — so it was removed. The old behaviour is preserved, for comparison
only, in benchmark/legacy_openings.py.
"""

import cv2

from engine.structure import _binarize, build_wall_mask, estimate_wall_kernel


class WallMaskBuilder:
    def __init__(self, image, min_wall_thickness_ratio=None):
        self.image = image
        self.height, self.width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        self._ink, _ = _binarize(gray)
        self.min_wall_thickness, self.line_width, self.thinnest_wall = estimate_wall_kernel(self._ink)

    def build(self, connect_disconnected=False):
        """uint8 mask (0/255) of wall strokes. `connect_disconnected` is accepted for
        compatibility and ignored (see module docstring)."""
        return build_wall_mask(self._ink, self.min_wall_thickness)
