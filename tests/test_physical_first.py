"""Physical space first: room names and printed sizes never redefine physical geometry."""

import cv2
import numpy as np

import engine.analyzer as analyzer
from engine.analysis.ocr import OCRBox, OCRResult, classify_text
from engine.structure import analyze_structure


def box(text, x, y, w, h, conf=95):
    return OCRBox(text=text, x=x, y=y, width=w, height=h, confidence=conf, source="t", variant="v",
                  rotation=0, psm=11, kind=classify_text(text))


def test_compact_inches_prefer_a_room_sized_value():
    d = analyzer._parse_dimensions('108" x 120"')                 # 10'8" x 12'0", not 1'8" x 12'0"
    assert (round(d["width"], 2), round(d["height"], 2)) == (round(10 + 8 / 12, 2), 12.0)


def test_implausible_printed_dimensions_are_not_used_for_geometry():
    assert not analyzer._plausible_dimensions({"width": 0.92, "height": 7.6, "unit": "ft"})
    assert analyzer._plausible_dimensions({"width": 3.0, "height": 4.0, "unit": "ft"})
    assert not analyzer._plausible_dimensions({"width": 0.5, "height": 3.0, "unit": "m"})


def _two_room_plan():
    img = np.full((700, 900, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 100), (800, 600), (0, 0, 0), 20)
    cv2.line(img, (450, 100), (450, 600), (0, 0, 0), 14)
    cv2.rectangle(img, (445, 300), (455, 380), (255, 255, 255), -1)
    return img


def test_open_plan_zones_keep_the_shared_physical_space_even_with_printed_sizes(monkeypatch):
    """Kitchen and living area share one walled space; their printed sizes give zone extents
    as metadata, never a rectangle that replaces the physical space."""
    img = _two_room_plan()
    result = OCRResult(boxes=(box("Kitchen", 150, 180, 70, 16), box("12'0\" x 10'0\"", 150, 200, 90, 14),
                              box("Living", 150, 480, 60, 16), box("Room", 215, 480, 48, 16), box("14'0\" x 12'0\"", 150, 500, 90, 14),
                              box("Bedroom", 580, 330, 80, 16), box("13'0\" x 15'0\"", 580, 350, 90, 14)))
    monkeypatch.setattr(analyzer, "extract_ocr", lambda image: result)
    response = analyzer.FloorPlanAnalyzer().analyze(img, "open-plan-sizes")
    rooms = {r["name"]: r for r in response["rooms"]}
    for name in ("Kitchen", "Living room"):
        assert rooms[name]["boundary"]["method"] == "open-plan-zone", rooms[name]["boundary"]["method"]
        assert rooms[name]["physical_space"]["kind"] == "open-plan"
    assert rooms["Kitchen"]["space_id"] == rooms["Living room"]["space_id"]
    _zones_partition_the_space(img.shape[:2], [rooms["Kitchen"], rooms["Living room"]])
    assert rooms["Bedroom"]["boundary"]["method"] == "wall-region"


def test_enclosed_rooms_sharing_a_space_are_reported_as_a_merge_not_an_open_plan(monkeypatch):
    """A bedroom and a bath in one structural space (their separating wall was not found) are
    not an open plan; the merge is reported as such and no wall is invented."""
    img = np.full((700, 900, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 100), (800, 600), (0, 0, 0), 20)
    result = OCRResult(boxes=(box("Bedroom", 200, 300, 80, 16), box("Bath", 600, 300, 50, 16)))
    monkeypatch.setattr(analyzer, "extract_ocr", lambda image: result)
    response = analyzer.FloorPlanAnalyzer().analyze(img, "merged")
    methods = {r["name"]: r["boundary"]["method"] for r in response["rooms"]}
    assert methods == {"Bedroom": "merged-space-zone", "Bath": "merged-space-zone"}
    assert all(r["shares_space_with"] and r["physical_space"]["kind"] == "merged" for r in response["rooms"])
    _zones_partition_the_space(img.shape[:2], response["rooms"])


def _zones_partition_the_space(shape, rooms):
    """Each room's zone holds its own label, zones do not overlap, and all stay inside the one
    physical space (no zone extends beyond it)."""
    def fill(poly):
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [np.array([[p["x"], p["y"]] for p in poly], np.int32)], 1)
        return m.astype(bool)
    space = fill(rooms[0]["physical_space"]["polygon"])
    zones = [fill(r["boundary"]["polygon"]) for r in rooms]
    for r, z in zip(rooms, zones):
        assert z[r["label_center"]["y"], r["label_center"]["x"]]
        assert (z & ~space).sum() <= 0.02 * z.sum()
    for i in range(len(zones)):
        for j in range(i + 1, len(zones)):
            assert (zones[i] & zones[j]).sum() <= 0.02 * min(zones[i].sum(), zones[j].sum())


def test_door_jambs_with_frame_ticks_on_thin_walls_are_found():
    """3 px walls whose door ends carry 1-2 px jamb ticks (as in real low-resolution plans): the
    end face looks 5 px long, wider than the wall behind it. The band test samples the wall's
    core, so the jambs are accepted, the door is found and the two rooms stay separate."""
    img = np.full((420, 640, 3), 255, np.uint8)
    img[38:41, 40:601] = 0; img[379:382, 40:601] = 0; img[40:381, 38:41] = 0; img[40:381, 598:601] = 0
    img[40:181, 319:322] = 0; img[214:381, 319:322] = 0                 # partition with a 33 px door gap
    img[180:182, 318:323] = 0; img[213:215, 318:323] = 0                # jamb ticks
    s = analyze_structure(img)
    gaps = [c for c in s.candidates if abs(c.center[0] - 320) < 6 and 175 < c.center[1] < 219]
    assert gaps, [(c.start, c.end) for c in s.candidates]
    left, right = s.space_labels[200, 180], s.space_labels[200, 460]
    assert left > 0 and right > 0 and left != right


def _open_connection_plan():
    """Four rooms; the two on the left are joined by a wide door-less opening (2.4 doors wide)
    next to a short wall; every other connection is a swing door (50 px)."""
    img = np.full((760, 1000, 3), 255, np.uint8)
    k = (0, 0, 0)
    cv2.rectangle(img, (60, 60), (940, 700), k, 12)
    cv2.line(img, (500, 60), (500, 700), k, 10)
    cv2.line(img, (60, 380), (500, 380), k, 10)
    img[375:386, 375:496] = 255                               # wide opening (2.4 doors), no door symbol
    cv2.line(img, (500, 380), (940, 380), k, 10)

    def door(x0, y, w):
        img[y - 6:y + 6, x0:x0 + w] = 255
        cv2.line(img, (x0, y), (x0, y - w), k, 2)
        cv2.ellipse(img, (x0, y), (w, w), 0, -90, 0, k, 1, cv2.LINE_AA)

    def vdoor(x, y0, w):
        img[y0:y0 + w, x - 6:x + 6] = 255
        cv2.line(img, (x, y0), (x + w, y0), k, 2)
        cv2.ellipse(img, (x, y0), (w, w), 0, 0, 90, k, 1, cv2.LINE_AA)
    door(620, 380, 50); door(800, 380, 50); vdoor(500, 150, 50); vdoor(500, 520, 50)
    return img


def _labels(top_left, bottom_left):
    return OCRResult(boxes=(box(top_left, 200, 200, 80, 16), box(bottom_left, 200, 540, 80, 16),
                            box("Bedroom", 650, 200, 80, 16), box("Bath", 680, 540, 50, 16)))


def test_wide_doorless_opening_between_open_plan_functions_is_one_physical_space(monkeypatch):
    img = _open_connection_plan()
    monkeypatch.setattr(analyzer, "extract_ocr", lambda image: _labels("Kitchen", "Living"))
    response = analyzer.FloorPlanAnalyzer().analyze(img, "open")
    rooms = {r["name"]: r for r in response["rooms"]}
    assert rooms["Kitchen"]["space_id"] == rooms["Living"]["space_id"]
    assert rooms["Kitchen"]["physical_space"]["kind"] == "open-plan"
    assert rooms["Bedroom"]["boundary"]["method"] == "wall-region"            # doors still separate rooms
    assert rooms["Bedroom"]["space_id"] != rooms["Bath"]["space_id"]


def test_same_opening_does_not_join_an_enclosed_room(monkeypatch):
    """The identical opening with a bedroom on one side stays a boundary between two rooms:
    semantics only interprets an ambiguous opening, it never merges enclosed rooms."""
    img = _open_connection_plan()
    monkeypatch.setattr(analyzer, "extract_ocr", lambda image: _labels("Bedroom", "Living"))
    rooms = analyzer.FloorPlanAnalyzer().analyze(img, "closed")["rooms"]
    ids = {r["label_center"]["y"] < 380: r["space_id"] for r in rooms if r["label_center"]["x"] < 500}
    assert ids[True] != ids[False]
    assert all("physical_space" not in r for r in rooms)



def test_one_open_plan_cell_is_not_split_into_rooms_by_its_labels(monkeypatch):
    """Kitchen and living area in one cell (their 160 px opening is too wide to be sealed): the
    labels name functional zones of one physical space; they do not cut it at the narrowing."""
    img = _open_connection_plan()
    img[375:386, 335:496] = 255
    monkeypatch.setattr(analyzer, "extract_ocr", lambda image: _labels("Kitchen", "Living"))
    rooms = {r["name"]: r for r in analyzer.FloorPlanAnalyzer().analyze(img, "open-cell")["rooms"]}
    assert rooms["Kitchen"]["space_id"] == rooms["Living"]["space_id"]
    assert rooms["Kitchen"]["boundary"]["method"] == "open-plan-zone"


def _corner_door_plan(arc: bool):
    """A wall ends at an L-corner; the doorway below it runs from that corner to the next wall.
    The corner has no end face of its own, so only the corner-door step can find this door."""
    img = np.full((760, 1000, 3), 255, np.uint8)
    k = (0, 0, 0)
    cv2.rectangle(img, (60, 60), (940, 700), k, 12)
    cv2.line(img, (500, 60), (500, 420), k, 10)
    cv2.line(img, (500, 420), (940, 420), k, 10)
    cv2.line(img, (60, 490), (505, 490), k, 10)

    def hdoor(x0, y, w):
        img[y - 6:y + 6, x0:x0 + w] = 255
        cv2.line(img, (x0, y), (x0, y - w), k, 2)
        cv2.ellipse(img, (x0, y), (w, w), 0, -90, 0, k, 1, cv2.LINE_AA)

    def vdoor(x, y0, w):
        img[y0:y0 + w, x - 6:x + 6] = 255
        cv2.line(img, (x, y0), (x + w, y0), k, 2)
        cv2.ellipse(img, (x, y0), (w, w), 0, 0, 90, k, 1, cv2.LINE_AA)
    hdoor(200, 490, 60); hdoor(700, 420, 60); vdoor(500, 200, 60)
    if arc:
        cv2.line(img, (500, 489), (560, 489), k, 2)
        cv2.ellipse(img, (500, 489), (63, 63), 0, -90, 0, k, 1, cv2.LINE_AA)
    return img


def test_door_at_a_wall_corner_separates_the_rooms_when_the_door_is_drawn():
    s = analyze_structure(_corner_door_plan(arc=True))
    assert s.space_labels[300, 300] != s.space_labels[600, 700]
    assert any("corner door" in j for c in s.candidates for j in c.jambs)


def test_bare_corner_gap_stays_open():
    """The same gap with no door symbol is not sealed: corner gaps also bound open areas, and
    without door evidence the engine does not invent a boundary."""
    s = analyze_structure(_corner_door_plan(arc=False))
    assert s.space_labels[300, 300] == s.space_labels[600, 700] > 0
