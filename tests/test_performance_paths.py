"""Performance mechanisms keep results identical: OCR memo, structural job overlapping OCR, large
uploads reduced to the working resolution."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from engine.analysis import ocr as O


def _text_image():
    img = np.full((400, 900, 3), 255, np.uint8)
    cv2.putText(img, "KITCHEN 12'-6", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    return img


@pytest.mark.skipif(O.pytesseract is None, reason="Tesseract not installed")
def test_page_ocr_memo_returns_the_same_result_without_rereading(monkeypatch):
    O.clear_ocr_memo()
    calls = []
    real = O._extract_ocr_uncached
    monkeypatch.setattr(O, "_extract_ocr_uncached", lambda *a, **k: calls.append(1) or real(*a, **k))
    img = _text_image()
    first = O.extract_ocr(img)
    second = O.extract_ocr(img.copy())                 # same pixels, different array
    assert len(calls) == 1 and second is first
    other = img.copy()
    other[0, 0] = 0
    O.extract_ocr(other)                               # one pixel differs: read again
    assert len(calls) == 2
    monkeypatch.setenv("FLOORPLAN_OCR_MEMO", "0")
    O.extract_ocr(img)                                 # memo off
    assert len(calls) == 3


@pytest.mark.skipif(O.pytesseract is None, reason="Tesseract not installed")
def test_line_ocr_memo_returns_a_fresh_list():
    O.clear_ocr_memo()
    img = _text_image()
    a = O.ocr_lines(img, 6)
    a.append("mutated")
    b = O.ocr_lines(img, 6)
    assert "mutated" not in b


def test_large_image_upload_is_reduced_not_rejected():
    from app import ANALYSIS_PIXELS, _decode_upload

    big = np.full((5200, 5200, 3), 255, np.uint8)        # 27 MP
    cv2.rectangle(big, (500, 500), (4700, 4700), (0, 0, 0), 30)
    ok, png = cv2.imencode(".png", big)
    image, warnings = _decode_upload(png.tobytes(), "large.png")
    assert image.shape[0] * image.shape[1] <= ANALYSIS_PIXELS
    assert any("Large image" in w for w in warnings)


def test_structural_job_runs_plan_model_off_the_ocr_path(monkeypatch):
    """The Plan Model fallback is prepared in the structural job; the result equals the serial run."""
    from benchmark.generator import generate
    from benchmark.wall_styles import layout_specs, styled
    from engine import analyzer as A
    from engine.plan import adapter

    p = generate(styled(layout_specs(4)[1], "hollow"))          # the legacy structure finds nothing
    threads = []
    real = adapter.prepare
    monkeypatch.setattr(adapter, "prepare", lambda *a, **k: threads.append(__import__("threading").current_thread().name) or real(*a, **k))
    monkeypatch.setenv("FLOORPLAN_OCR_WORKERS", "4")
    parallel = A.FloorPlanAnalyzer().analyze(p.image, "h.png")
    monkeypatch.setenv("FLOORPLAN_OCR_WORKERS", "1")
    serial = A.FloorPlanAnalyzer().analyze(p.image, "h.png")
    assert threads and threads[0].startswith("floorplan-structure")
    for key in ("room_count", "unlabeled_space_count", "opening_count"):
        assert parallel[key] == serial[key]
    assert parallel["plan_model"]["summary"] == serial["plan_model"]["summary"]


@pytest.mark.skipif(O.pytesseract is None, reason="Tesseract not installed")
def test_page_passes_are_upright_and_vertical_text_is_still_read(monkeypatch):
    """Rotated text is read by the text-line reader (each vertical line turned upright), so the
    full-page passes read upright only: 4 variants x 2 modes = 8 instead of 32."""
    img = np.full((900, 700, 3), 255, np.uint8)
    cv2.rectangle(img, (40, 40), (660, 860), (0, 0, 0), 8)
    cv2.putText(img, "KITCHEN", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    side = np.full((70, 420, 3), 255, np.uint8)
    cv2.putText(side, "BEDROOM", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    img[300:720, 500:570] = cv2.rotate(side, cv2.ROTATE_90_COUNTERCLOCKWISE)     # vertical label
    calls = []
    real = O._read_words
    monkeypatch.setattr(O, "_read_words", lambda im, **k: calls.append((k["variant_name"], k["rotation"])) or real(im, **k))
    O.clear_ocr_memo()
    texts = {b.text.upper() for b in O.extract_ocr(img).boxes}
    page = [c for c in calls if c[0] != "text-lines"]
    assert len(page) == 8 and {r for _, r in page} == {0}
    assert "KITCHEN" in texts and "BEDROOM" in texts
    calls.clear()
    monkeypatch.setenv("FLOORPLAN_OCR_ROTATIONS", "all")
    O.clear_ocr_memo()
    O.extract_ocr(img)
    assert len([c for c in calls if c[0] != "text-lines"]) == 32
