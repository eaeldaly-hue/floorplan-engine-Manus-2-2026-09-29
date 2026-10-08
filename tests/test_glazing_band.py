"""A glazing band (several parallel lines) between two walls is an opening, not text or hatching."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine.structure import _parallel_strokes, analyze_structure


REAL_PLANS = Path(__file__).resolve().parents[1].parent / "Test Cases"


@pytest.mark.skipif(not (REAL_PLANS / "12.png").exists(), reason="real test plans not available")
def test_real_glazed_facade_separates_bedroom_and_balcony():
    """12.png: the bedroom's balcony wall is a sliding-glass band from wall to wall."""
    spec = json.loads((Path(__file__).resolve().parents[1] / "benchmark/fixtures/real_plans.json").read_text())["plans"]["12.png"]
    pts = {r["name"]: r["point"] for r in spec["rooms"]}
    s = analyze_structure(cv2.imread(str(REAL_PLANS / "12.png")))
    bed, balcony = (int(s.space_labels[y, x]) for x, y in (pts["Bedroom"], pts["Balcony"]))
    assert bed > 0 and balcony > 0 and bed != balcony


def test_rows_of_a_glazing_band_vs_a_pattern():
    mask = np.zeros((200, 300), np.uint8)
    for y in (96, 100, 104):
        mask[y, 50:250] = 255
    s = np.linspace(10, 190, 60)
    d, n = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert _parallel_strokes(mask, np.array([50.0, 96.0]), d, n, s, 0.0, 8.0)
    for y in range(108, 140, 4):                                        # the lines go on: a pattern
        mask[y, 50:250] = 255
    assert not _parallel_strokes(mask, np.array([50.0, 96.0]), d, n, s, 0.0, 8.0)
    text = np.zeros((200, 300), np.uint8)
    text[96, 50:250] = text[104, 50:250] = 255
    for x in range(55, 245, 9):                                         # letters: short pieces per row
        text[99:102, x:x + 4] = 255
    assert not _parallel_strokes(text, np.array([50.0, 96.0]), d, n, s, 0.0, 8.0)
