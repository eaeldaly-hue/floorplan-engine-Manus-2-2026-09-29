"""Measurement layer: wall-style synthetic variants, wall metrics, benchmark OCR cache."""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

from benchmark import ocr_cache
from benchmark.generator import Style, generate
from benchmark.wall_metrics import WallStats, drawn_wall
from benchmark.wall_styles import STYLES, gt_digest, layout_specs, styled


@pytest.fixture(scope="module")
def plans():
    spec = layout_specs(4)[1]
    return {style: generate(styled(spec, style)) for style in STYLES}


def test_ground_truth_is_identical_in_every_style(plans):
    digests = {style: gt_digest(p) for style, p in plans.items()}
    assert len(set(digests.values())) == 1
    images = {style: p.image.tobytes() for style, p in plans.items()}
    assert len(set(images.values())) == len(STYLES)              # ... while the drawings differ


def test_default_style_is_the_original_filled_rendering(plans):
    assert Style().wall_render == "filled"
    p = plans["filled"]
    gray = cv2.cvtColor(p.image, cv2.COLOR_BGR2GRAY)
    band = p.wall_mask > 0
    assert (gray[band] < 60).mean() > 0.97                        # the band is solid ink
    hollow = cv2.cvtColor(plans["hollow"].image, cv2.COLOR_BGR2GRAY)
    core = cv2.erode(p.wall_mask, np.ones((7, 7), np.uint8)) > 0
    assert (hollow[core] > 200).mean() > 0.9                       # hollow: the band interior is paper


def test_wall_metrics_perfect_empty_and_fabricated(plans):
    p = plans["filled"]
    perfect = WallStats().add(p.wall_mask, p, "filled", typical_thickness=None)
    assert perfect["pixel_precision"] == 1.0 and perfect["pixel_recall"] == 1.0
    assert perfect["centerline_recall"] == 1.0 and perfect["segment_recall"] == 1.0 and perfect["fabricated_share"] == 0.0
    w = WallStats()
    empty = w.add(np.zeros_like(p.wall_mask), p, "filled")
    assert empty["pixel_recall"] == 0.0 and empty["pixel_precision"] is None and w.metrics()["plans_without_walls"] == 1
    junk = p.wall_mask.copy()
    labels = p.room_labels
    ys, xs = np.nonzero(labels == labels.max())
    cy, cx = int(ys.mean()), int(xs.mean())
    junk[cy - 15:cy + 15, cx - 15:cx + 15] = 255                  # a furniture block taken as wall
    fab = WallStats().add(junk, p, "filled")
    assert fab["fabricated_share"] > 0 and fab["phantom_components"] == 1 and fab["pixel_recall"] == 1.0


def test_thin_walls_are_measured_against_the_drawn_line(plans):
    p = plans["thin"]
    drawn = drawn_wall(p, "thin")
    assert 0 < drawn.sum() < 0.6 * (p.wall_mask > 0).sum()
    row = WallStats().add(drawn.astype(np.uint8) * 255, p, "thin")
    assert row["pixel_recall"] == 1.0 and row["centerline_recall"] > 0.95


def test_hollow_outline_alone_does_not_count_as_understood_walls(plans):
    """Finding only the two faces of a hollow wall leaves the wall axis uncovered."""
    p = plans["hollow"]
    outline = cv2.morphologyEx(p.wall_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    row = WallStats().add(outline, p, "hollow")
    assert row["centerline_recall"] < 0.5 and row["pixel_precision"] > 0.95


# ---------- OCR cache ----------

def text_image():
    img = np.full((420, 760, 3), 255, np.uint8)
    cv2.rectangle(img, (40, 40), (720, 380), (0, 0, 0), 10)
    cv2.putText(img, "KITCHEN", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    cv2.putText(img, "12'0\" x 10'6\"", (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img


@pytest.mark.skipif(__import__("engine.analysis.ocr", fromlist=["x"]).pytesseract is None, reason="Tesseract not installed")
def test_cached_analysis_is_identical_and_served_from_disk(tmp_path, monkeypatch):
    from engine.analyzer import FloorPlanAnalyzer

    monkeypatch.setenv("FLOORPLAN_OCR_CACHE", str(tmp_path))
    img = text_image()
    reference = FloorPlanAnalyzer().analyze(img, "p")
    for k in ocr_cache.STATS:
        ocr_cache.STATS[k] = 0
    with ocr_cache.enabled():
        cold = FloorPlanAnalyzer().analyze(img, "p")
        misses = ocr_cache.STATS["misses"]
        warm = FloorPlanAnalyzer().analyze(img, "p")
    assert reference == cold == warm
    assert misses >= 1 and ocr_cache.STATS["hits"] == misses     # second run: every read from disk
    assert any((tmp_path / ocr_cache.code_version()).rglob("*.pkl"))


def test_cache_key_contains_the_ocr_code_version(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOORPLAN_OCR_CACHE", str(tmp_path))
    assert ocr_cache._path("page", "ab" * 32).parts[-4] == ocr_cache.code_version()
    monkeypatch.setenv("FLOORPLAN_OCR_CACHE", "off")
    assert ocr_cache.cache_dir() is None


# ---------- hold-out runner: PDF pages through the production path, labelled regions ----------

def test_holdout_pdf_plan_is_scored_in_the_engine_frame(tmp_path):
    from benchmark import holdout_inputs
    from benchmark.holdout_metrics import normalize_spec
    from benchmark.real_plans import evaluate
    from tests.test_pdf_input import plan_ops
    from tests.test_pdf_text_layer import text_pdf

    W, H = 842, 595
    (tmp_path / "pdf").mkdir()
    (tmp_path / "pdf" / "p.pdf").write_bytes(text_pdf([{"w": W, "h": H, "ops": plan_ops(W, H)}]))
    image, _, info = holdout_inputs.load(tmp_path / "pdf" / "p.pdf", 1)
    assert info["kind"] == "pdf" and info["dpi"] == 150
    assert abs(image.shape[0] - H * 150 / 72) <= 1 and abs(image.shape[1] - W * 150 / 72) <= 1   # poppler rounds up
    k = 150 / 72
    left, right = [int(0.31 * W * k), int(0.5 * H * k)], [int(0.69 * W * k), int(0.5 * H * k)]
    spec = {"file": "pdf/p.pdf", "page": 1, "holdout_input": True,
            "spaces": [{"id": "S1", "kind": "room", "points": [left]}, {"id": "S2", "kind": "room", "points": [right]}],
            "rooms": [{"name": "Bedroom", "printed": None, "point": left, "space": "S1"}]}
    full = evaluate("pdf/p.pdf", normalize_spec(spec), tmp_path)["holdout"]
    assert full["physical_spaces"]["gt"] == 2 and full["physical_spaces"]["recovered"] == 2
    # Label only the left room: the right room's engine space is outside the labelled region,
    # so it is not a false space.
    half = dict(spec, spaces=spec["spaces"][:1], roi=[[0, 0, int(0.5 * W * k), int(H * k)]])
    part = evaluate("pdf/p.pdf", normalize_spec(half), tmp_path)["holdout"]
    no_roi = evaluate("pdf/p.pdf", normalize_spec(dict(half, roi=None)), tmp_path)["holdout"]
    assert part["false_spaces"]["engine_spaces_without_gt_point"] < no_roi["false_spaces"]["engine_spaces_without_gt_point"]


def test_large_images_are_reduced_to_the_analysis_limit(tmp_path):
    from benchmark import holdout_inputs

    big = np.full((5200, 5200, 3), 255, np.uint8)
    cv2.rectangle(big, (400, 400), (4800, 4800), (0, 0, 0), 40)
    cv2.imwrite(str(tmp_path / "big.png"), big)
    image, evidence, info = holdout_inputs.load(tmp_path / "big.png")
    assert evidence == () and info["analysis_scale"] < 1
    assert image.shape[0] * image.shape[1] <= holdout_inputs.MAX_PIXELS
    again, _, _ = holdout_inputs.load(tmp_path / "big.png", analysis_scale=info["analysis_scale"])
    assert holdout_inputs.frame_digest(again) == holdout_inputs.frame_digest(image)     # recorded scale reproduces the frame
