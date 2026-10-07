"""OCR evidence aggregation, room-label assembly and open-plan room naming (no Tesseract needed)."""

import cv2
import numpy as np

import engine.analyzer as analyzer
from engine.analysis.ocr import OCRBox, OCRResult, classify_text
from engine.analysis.ocr_aggregation import Observation, aggregate, normalize_reading
from engine.analysis.room_labels import build_room_labels
from engine.analysis.room_lexicon import is_room_word


def obs(text, x, y, w, h, conf, rotation=0, variant="v", psm=11):
    return Observation(text=text, norm=normalize_reading(text), x=x, y=y, width=w, height=h, confidence=conf,
                       variant=variant, rotation=rotation, psm=psm, kind=classify_text(text))


def vocab(reading):
    return is_room_word(reading) or classify_text(reading) == "DIMENSION"


def test_upright_reads_are_kept_when_a_rotated_pass_produces_more_noise():
    observations = [obs("Kitchen", 330, 144, 50, 14, 96, 0, v) for v in ("a", "b", "c")]
    # a rotated pass reads many confident fragments elsewhere and thin slivers over the word
    observations += [obs(t, 300 + 10 * i, 20, 6, 30, 70, 270, "a") for i, t in enumerate(("woe", "ge", "On", "GE", "rain"))]
    observations += [obs("ee", 340, 140, 3, 28, 48, 270, "a")]
    regions = aggregate(observations, vocab)
    kitchen = [r for r in regions if r.norm == "KITCHEN"]
    assert len(kitchen) == 1 and kitchen[0].accepted and kitchen[0].support == 3


def test_two_words_are_not_chained_into_one_region_through_a_spanning_box():
    observations = [obs("CLOSET", 1781, 1084, 134, 25, 96, 0, v) for v in ("a", "b")]
    observations += [obs("5'3\"", 1767, 1123, 53, 25, 90, 0, v) for v in ("a", "b")]
    observations += [obs("COE", 1781, 1084, 147, 64, 28, 0, "c", 6)]          # psm-6 box over both lines
    regions = aggregate(observations, vocab)
    readings = {r.norm for r in regions if r.accepted}
    assert "CLOSET" in readings and "5'3\"" in readings


def test_one_letter_misread_yields_to_a_strong_vocabulary_reading():
    observations = [obs("Beth", 512, 445, 30, 14, 93, 0, v) for v in ("a", "b", "c", "d")]
    observations += [obs("Bath", 512, 445, 30, 14, 78, 0, "e")]
    [region] = aggregate(observations, vocab)
    assert region.norm == "BATH" and region.accepted and "misread" in region.reason


def test_weak_unsupported_text_is_rejected_with_a_reason():
    [region] = aggregate([obs("xqzt", 10, 10, 30, 12, 40)], vocab)
    assert not region.accepted and "weak" in region.reason


def box(text, x, y, w, h, conf=95, rotation=0):
    return OCRBox(text=text, x=x, y=y, width=w, height=h, confidence=conf, source="t", variant="v",
                  rotation=rotation, psm=11, kind=classify_text(text))


def test_multi_word_and_stacked_room_names_are_assembled():
    boxes = [box("Living", 316, 403, 38, 14), box("Room", 360, 402, 41, 14),          # one line
             box("Bonus", 140, 120, 40, 12), box("Room", 142, 135, 36, 12),           # stacked
             box("Kitchen", 600, 120, 55, 14), box("23.51", 605, 138, 30, 10), box("m²", 637, 138, 10, 10)]
    names = sorted(g.text for g in build_room_labels(boxes))
    assert names == ["Bonus Room", "Kitchen", "Living Room"]


def test_area_numbers_and_other_orientations_do_not_join_a_name():
    boxes = [box("Bedroom", 100, 100, 60, 14), box("84", 110, 118, 14, 10),
             box("Hall", 300, 100, 30, 14), box("woe", 335, 100, 25, 14, rotation=270)]
    names = sorted(g.text for g in build_room_labels(boxes))
    assert names == ["Bedroom", "Hall"]


def test_open_plan_labels_share_one_structural_space_without_invented_walls(monkeypatch):
    """Kitchen, hall and living area in one walled space with no partition between them:
    each label is named and refers to the shared space; no extra space is created."""
    img = np.full((700, 900, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 100), (800, 600), (0, 0, 0), 20)
    cv2.line(img, (450, 100), (450, 600), (0, 0, 0), 14)                 # a bedroom on the right
    cv2.rectangle(img, (445, 300), (455, 380), (255, 255, 255), -1)      # its door
    result = OCRResult(boxes=(box("Kitchen", 150, 180, 70, 16), box("Hall", 300, 330, 40, 16),
                              box("Living", 150, 480, 60, 16), box("Room", 215, 480, 48, 16),
                              box("Bedroom", 580, 330, 80, 16)))
    monkeypatch.setattr(analyzer, "extract_ocr", lambda image: result)
    response = analyzer.FloorPlanAnalyzer().analyze(img, "open-plan")
    rooms = {r["name"]: r for r in response["rooms"]}
    assert set(rooms) == {"Kitchen", "Hall", "Living room", "Bedroom"}
    shared = [rooms[n] for n in ("Kitchen", "Hall", "Living room")]
    # one physical space (shared), three functional zones inside it (estimated extents, no walls)
    assert all(r["boundary"]["method"] == "open-plan-zone" for r in shared)
    assert len({r["space_id"] for r in shared}) == 1
    assert all(r["physical_space"]["kind"] == "open-plan" and r["physical_space"]["id"] == shared[0]["space_id"] for r in shared)
    assert sorted(shared[0]["shares_space_with"]) == sorted(r["id"] for r in shared[1:])
    assert rooms["Bedroom"]["boundary"]["method"] == "wall-region"
    assert response["topology"]["spaces"].__len__() == 2


def test_a_word_with_a_too_short_ocr_box_still_joins_its_line():
    """11.png: OCR boxed 'LIVING' 5 px tall next to 'ROOM' 13 px tall. The words sit side by side
    on one line, so they are one label; an overlapping duplicate reading is not a second word."""
    boxes = [box("LIVING", 321, 216, 21, 5), box("ROOM", 346, 213, 19, 13)]
    assert [g.text for g in build_room_labels(boxes)] == ["LIVING ROOM"]
    dup = [box("Kitchen", 600, 120, 55, 14), box("itchen", 610, 124, 40, 6)]
    assert [g.text for g in build_room_labels(dup)] == ["Kitchen"]


def test_a_junk_fragment_with_a_stray_box_does_not_join_a_room_name():
    boxes = [box("CORRIDOR.", 300, 200, 70, 14), box("wot", 284, 205, 14, 4)]
    assert [g.text for g in build_room_labels(boxes)] == ["CORRIDOR."]
