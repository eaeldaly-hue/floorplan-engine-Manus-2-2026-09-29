"""The OCR speed-ups must not change any result: concurrent passes are consumed in pass order,
the aggregation index only skips comparisons that cannot succeed, the dimension re-reads and the
structural pass run concurrently but are applied exactly where they were before."""

from __future__ import annotations

import math
import random
import threading

import cv2
import numpy as np
import pytest

import engine.analysis.ocr as ocr
import engine.analysis.ocr_aggregation as agg
from engine.analyzer import FloorPlanAnalyzer


# ---------- reference implementations: the linear scans the index replaces ----------

def linear_group_regions(observations):
    seeds, members = [], []
    for o in sorted(observations, key=lambda o: -o.confidence):
        for k, seed in enumerate(seeds):
            if agg._same_place(o, seed):
                members[k].append(o)
                break
        else:
            seeds.append(o)
            members.append([o])
    return [agg.Region(observations=obs) for obs in members]


def linear_merge_regions(regions, is_vocabulary):
    order = sorted(regions, key=lambda r: -(r.support * r.confidence))
    kept = []
    for r in order:
        for k in kept:
            ratio = max(r.box[3], k.box[3]) / max(1, min(r.box[3], k.box[3]))
            overlap = agg._box_overlap(r.box, k.box)
            same_word = r.norm == k.norm or agg._one_substitution(r.norm, k.norm)
            if (overlap >= 0.5 and (same_word or ratio <= 1.65)) or (overlap >= 0.85 and ratio <= 2.2):
                k.observations.extend(r.observations)
                agg.resolve(k, is_vocabulary)
                break
        else:
            kept.append(r)
    return kept


def random_observations(seed: int, n: int = 1500):
    """Dense, adversarial readings: repeated reads of the same words with jitter, tiny boxes
    (the 4 px floor of the centre rule), long boxes spanning many grid cells, ties in confidence."""
    rng = random.Random(seed)
    words = ["BEDROOM", "BATH", "BETH", "KITCHEN", "12'-0\"", "10'6\"", "CL", "LIVING", "DINING", "x", "WC"]
    anchors = [(rng.randint(0, 1500), rng.randint(0, 1000), rng.choice([2, 3, 8, 14, 22]), rng.choice([1, 2, 30, 90, 400]))
               for _ in range(n // 6)]
    out = []
    for _ in range(n):
        x, y, h, w = rng.choice(anchors)
        jx, jy = rng.randint(-3, 3), rng.randint(-3, 3)
        sw, sh = rng.choice([1.0, 0.8, 1.3, 2.1, 0.45]), rng.choice([1.0, 0.7, 1.5, 1.7])
        text = rng.choice(words)
        out.append(agg.Observation(text=text, norm=agg.normalize_reading(text), x=max(0, x + jx), y=max(0, y + jy),
                                   width=max(1, int(w * sw)), height=max(1, int(h * sh)),
                                   confidence=float(rng.choice([25, 40, 60, 61.5, 80, 95])),
                                   variant=rng.choice(["a", "b"]), rotation=rng.choice([0, 90]), psm=rng.choice([6, 11]),
                                   kind="OTHER"))
    return out


def snapshot(regions):
    return [(r.text, r.norm, r.box, r.accepted, r.reason, r.confidence, r.support, r.rivals,
             [(o.text, o.x, o.y, o.width, o.height, o.confidence, o.variant, o.rotation, o.psm) for o in r.observations])
            for r in regions]


vocab = lambda reading: reading in {"BEDROOM", "BATH", "KITCHEN", "LIVING", "DINING"}


@pytest.mark.parametrize("seed", range(6))
def test_indexed_grouping_equals_the_linear_scan(seed):
    obs = random_observations(seed)
    fast = agg.group_regions(obs)
    slow = linear_group_regions(obs)
    assert [[id(o) for o in r.observations] for r in fast] == [[id(o) for o in r.observations] for r in slow]


@pytest.mark.parametrize("seed", range(6))
def test_indexed_merging_equals_the_linear_scan(seed):
    obs = random_observations(100 + seed)
    a = [agg.resolve(r, vocab) for r in linear_group_regions(obs)]
    b = [agg.resolve(agg.Region(observations=list(r.observations)), vocab) for r in a]
    assert snapshot(linear_merge_regions(a, vocab)) == snapshot(agg._merge_regions(b, vocab))


def test_reach_box_covers_every_centre_match():
    """Two boxes accepted by the centre-distance rule always have intersecting reach boxes."""
    rng = random.Random(7)
    for _ in range(20000):
        a = agg.Observation("ab", "AB", rng.randint(0, 40), rng.randint(0, 40), rng.randint(1, 30), rng.randint(1, 30), 50, "v", 0, 6, "OTHER")
        b = agg.Observation("ab", "AB", rng.randint(0, 40), rng.randint(0, 40), rng.randint(1, 30), rng.randint(1, 30), 50, "v", 0, 6, "OTHER")
        if agg._same_place(a, b):
            ra, rb = agg._reach_box(a), agg._reach_box(b)
            assert ra[0] <= rb[2] and rb[0] <= ra[2] and ra[1] <= rb[3] and rb[1] <= ra[3]


# ---------- the OCR pool ----------

def test_pool_returns_results_in_submission_order(monkeypatch):
    monkeypatch.setenv("FLOORPLAN_OCR_WORKERS", "4")
    import time

    def slow(i):
        time.sleep(0.002 * (10 - i % 10))
        return i * i, threading.current_thread().name
    out = ocr.run_ocr_tasks(slow, range(40))
    assert [v for v, _ in out] == [i * i for i in range(40)]
    assert any(name.startswith("floorplan-ocr") for _, name in out)


def test_single_worker_and_nested_calls_run_inline(monkeypatch):
    monkeypatch.setenv("FLOORPLAN_OCR_WORKERS", "1")
    assert {name for name in ocr.run_ocr_tasks(lambda i: threading.current_thread().name, range(5))} == {threading.current_thread().name}
    monkeypatch.setenv("FLOORPLAN_OCR_WORKERS", "3")
    inner = ocr.run_ocr_tasks(lambda i: ocr.run_ocr_tasks(lambda j: j, range(3)), range(6))   # no deadlock
    assert inner == [[0, 1, 2]] * 6


# ---------- end to end: concurrent == sequential ----------

def text_plan():
    img = np.full((700, 1000, 3), 255, np.uint8)
    cv2.rectangle(img, (60, 60), (940, 640), (0, 0, 0), 12)
    cv2.line(img, (500, 60), (500, 640), (0, 0, 0), 10)
    img[300:380, 494:506] = 255
    cv2.putText(img, "BEDROOM", (150, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    cv2.putText(img, "12'0\" x 10'6\"", (150, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "KITCHEN", (620, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    return img


@pytest.mark.skipif(ocr.pytesseract is None, reason="Tesseract not installed")
def test_concurrent_analysis_equals_sequential_analysis(monkeypatch):
    img = text_plan()
    strip = lambda r: {k: v for k, v in r.items()}
    monkeypatch.setenv("FLOORPLAN_OCR_WORKERS", "1")
    seq_ocr = ocr.extract_ocr(img)
    sequential = strip(FloorPlanAnalyzer().analyze(img, "plan"))
    monkeypatch.setenv("FLOORPLAN_OCR_WORKERS", "6")
    par_ocr = ocr.extract_ocr(img)
    concurrent = strip(FloorPlanAnalyzer().analyze(img, "plan"))
    assert par_ocr.boxes == seq_ocr.boxes and par_ocr.passes == seq_ocr.passes
    assert [(r.text, r.box, r.accepted) for r in par_ocr.regions] == [(r.text, r.box, r.accepted) for r in seq_ocr.regions]
    assert concurrent == sequential
    assert sequential["room_count"] >= 1
