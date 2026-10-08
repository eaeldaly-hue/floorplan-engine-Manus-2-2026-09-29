"""A drawn door keeps a long gap that only one wall end found (openings -> walls)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

import engine.structure as st


def _plan():
    img = np.full((460, 700, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 20), (680, 440), (0, 0, 0), 10)
    cv2.line(img, (360, 20), (360, 150), (0, 0, 0), 10)                 # stub with a free end at y=150
    cv2.line(img, (360, 310), (680, 310), (0, 0, 0), 10)                # perpendicular wall: the far jamb is its face
    for hy, sgn in ((155, 1), (305, -1)):                               # double door, leaves open to the left
        pts = [(int(355 - 75 * math.sin(a)), int(hy + sgn * 75 * math.cos(a))) for a in np.linspace(0, math.pi / 2, 40)]
        cv2.polylines(img, [np.array(pts).reshape(-1, 1, 2)], False, (60, 60, 60), 1)
        cv2.line(img, (355, hy), (280, hy), (60, 60, 60), 1)
    return img


@pytest.mark.parametrize("rescue,separated", [(True, True), (False, False)])
def test_double_door_keeps_the_long_gap(monkeypatch, rescue, separated):
    monkeypatch.setattr(st, "DOOR_RESCUE", rescue)
    s = st.analyze_structure(_plan())
    left, right = int(s.space_labels[230, 150]), int(s.space_labels[230, 520])
    assert left > 0 and right > 0
    assert (left != right) is separated
