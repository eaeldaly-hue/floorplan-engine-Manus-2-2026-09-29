import numpy as np

from engine.geometry_v2.local_opening_analyzer import LocalOpeningAnalyzer
from engine.geometry_v2.segments import WallSegmentBuilder


def _segments():
    builder = WallSegmentBuilder()
    return builder.build_many(
        [
            {"start": (20, 50), "end": (80, 50), "thickness": 12, "confidence": 0.9},
            {"start": (120, 50), "end": (200, 50), "thickness": 12, "confidence": 0.9},
        ]
    )


def _wall_mask(fill_gap):
    mask = np.zeros((100, 220), dtype=np.uint8)
    mask[44:57, 20:81] = 255
    mask[44:57, 120:201] = 255
    if fill_gap:
        mask[44:57, 81:120] = 255
    return mask


def test_mask_validation_keeps_supported_empty_opening():
    candidates = LocalOpeningAnalyzer().detect(_segments(), _wall_mask(fill_gap=False))
    assert len(candidates) == 1
    assert candidates[0].source == "local_segment_gap_mask_validated"
    assert 0 < candidates[0].confidence <= 1


def test_mask_validation_rejects_gap_filled_with_wall_pixels():
    candidates = LocalOpeningAnalyzer().detect(_segments(), _wall_mask(fill_gap=True))
    assert candidates == []
