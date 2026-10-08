"""Context-aware reading: text lines inside unnamed spaces are read one at a time."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from engine.analysis import context_ocr as C
from engine.analysis.ocr import OCRLine, pytesseract
from engine.analysis.room_lexicon import room_name
from engine.structure import analyze_structure

pytestmark = pytest.mark.skipif(pytesseract is None, reason="Tesseract not installed")


def _plan():
    img = np.full((500, 800, 3), 255, np.uint8)
    cv2.rectangle(img, (30, 30), (770, 470), (0, 0, 0), 10)
    cv2.line(img, (400, 30), (400, 470), (0, 0, 0), 10)
    cv2.putText(img, "KITCHEN", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "BED 4", (540, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img


def _space_at(s, line):
    return int(s.space_labels[line.center[1], line.center[0]])


def test_unnamed_space_gets_the_name_written_in_it():
    img = _plan()
    s = analyze_structure(img)
    out, notes = C.reread(img, s, [])
    kitchen = [ln for ln in out if room_name(ln.text) == "Kitchen"]
    assert kitchen and kitchen[0].source == "context-ocr"
    assert _space_at(s, kitchen[0]) == int(s.space_labels[250, 200])


def test_label_that_lost_its_number_is_completed():
    img = _plan()
    s = analyze_structure(img)
    bed = OCRLine(text="BED", x=540, y=233, width=50, height=18, confidence=60.0, source="robust-ocr")
    out, notes = C.reread(img, s, [bed])
    assert any(ln.text == "BED 4" for ln in out), notes


def test_nothing_is_read_where_no_text_is(monkeypatch):
    img = np.full((500, 800, 3), 255, np.uint8)
    cv2.rectangle(img, (30, 30), (770, 470), (0, 0, 0), 10)
    s = analyze_structure(img)
    calls = []
    monkeypatch.setattr(C, "_read", lambda *a: calls.append(1) or ("", 0.0))
    out, _ = C.reread(img, s, [])
    assert out == [] and not calls


def test_switch_off(monkeypatch):
    monkeypatch.setenv("FLOORPLAN_CONTEXT_OCR", "0")
    img = _plan()
    s = analyze_structure(img)
    out, notes = C.reread(img, s, [])
    assert out == [] and notes == []
