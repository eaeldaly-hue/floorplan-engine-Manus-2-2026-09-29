"""Furniture / fixture recognition rules (engine.arch.objects) on small synthetic drawings."""

from __future__ import annotations

import math
from types import SimpleNamespace

import cv2
import numpy as np

from engine.arch.objects import (Obj, _beds, _crossed_boxes, _dedupe, _lone_seats, _oval_basin, _segments, _tags,
                                 arcs, basins, circles, within_walls)

PPC = 2.0


class _E:
    def __init__(self, pts):
        self.geometry = pts


def _arc(cx, cy, r, a0, a1, n=12):
    return _E([(cx + r * math.cos(t), cy + r * math.sin(t)) for t in np.radians(np.linspace(a0, a1, n))])


def _obj(i, kind, box, conf=0.5):
    return Obj(f"OB{i:03d}", kind, tuple(float(v) for v in box), {}, conf, [kind])


def _rounded_rect(x0, y0, x1, y1, r):
    return [_arc(x0, y0, r, 180, 270), _arc(x1, y0, r, 270, 360), _arc(x1, y1, r, 0, 90), _arc(x0, y1, r, 90, 180)]


def test_a_rounded_rectangle_needs_its_drain_to_be_a_basin():
    """Basins and pillows are both rounded rectangles; only the basin has a drain."""
    basin = _rounded_rect(100, 100, 160, 150, 8)
    drain = [_arc(130, 125, 3, q * 90, q * 90 + 90) for q in range(4)]
    al = arcs(basin + drain)
    found = basins(al, PPC, {i for c in circles(al, PPC) for i in c[3]})
    drains = circles(al, PPC, r_cm=(0.8, 6.0))
    assert len(found) == 1 and any(found[0][0] < d[0] < found[0][2] and found[0][1] < d[1] < found[0][3] for d in drains)
    al2 = arcs(basin)                                        # the same outline without a drain
    assert not circles(al2, PPC, r_cm=(0.8, 6.0))


def test_an_oval_bowl_with_a_drain_is_a_basin_a_plain_box_is_not():
    comp = np.zeros((90, 70), np.uint8)
    cv2.ellipse(comp, (35, 45), (32, 42), 0, 0, 360, 1, 3)
    drains = [(35, 45, 4, ())]
    assert _oval_basin(comp, (0, 0), drains, PPC)[0] == "sink"
    assert _oval_basin(comp, (0, 0), [], PPC) is None
    box = np.zeros((90, 70), np.uint8)
    cv2.rectangle(box, (1, 1), (68, 88), 1, 3)
    assert _oval_basin(box, (0, 0), drains, PPC) is None


def test_a_crossed_rectangle_is_a_shower():
    s = 2.0
    elems = [_E([(0, 0), (180 * s, 90 * s)]), _E([(0, 90 * s), (180 * s, 0)]), _E([(0, 0), (60 * s, 0)])]
    boxes = _crossed_boxes(_segments(elems), s)
    assert len(boxes) == 1 and boxes[0][2] == 180 * s


def test_pieces_at_a_bed_corner_are_nightstands_pieces_inside_are_part_of_the_bed():
    objs = [_obj(1, "bed", (0, 0, 400, 420)), _obj(2, "seat", (0, 0, 70, 70)), _obj(3, "side table", (100, 20, 180, 120)),
            _obj(4, "counter", (150, 150, 300, 300)), _obj(5, "seat", (900, 900, 960, 960))]
    _beds(objs, PPC)
    kinds = {o.id: o.kind for o in objs}
    assert kinds == {"OB001": "bed", "OB002": "nightstand", "OB005": "seat"}


def test_a_lone_seat_is_dropped_seats_beside_a_counter_stay():
    objs = [_obj(1, "seat", (0, 0, 70, 70)), _obj(2, "counter", (500, 0, 800, 120)), _obj(3, "seat", (480, 130, 540, 190))]
    _lone_seats(objs, PPC)
    assert [o.id for o in objs] == ["OB002", "OB003"]


def test_a_box_around_a_coded_tag_is_not_furniture():
    objs = [_obj(1, "armchair", (0, 0, 150, 130)), _obj(2, "table", (400, 0, 600, 100))]
    words = [SimpleNamespace(text="KIT-1", x=40, y=40, width=60, height=25, confidence=99),
             SimpleNamespace(text="LIVING", x=450, y=30, width=90, height=25, confidence=99)]
    _tags(objs, words)
    assert [o.id for o in objs] == ["OB002"]                 # a room name over a table keeps the table


def test_duplicates_keep_the_most_confident_reading():
    objs = [_obj(1, "side table", (0, 0, 100, 100), 0.3), _obj(2, "nightstand", (1, 1, 100, 100), 0.5)]
    _dedupe(objs)
    assert [o.kind for o in objs] == ["nightstand"]


def test_objects_outside_the_walls_are_not_plan_furniture():
    b = {"walls": [{"p0": [0, 0], "p1": [1000, 0]}, {"p0": [1000, 0], "p1": [1000, 800]},
                   {"p0": [1000, 800], "p1": [0, 800]}, {"p0": [0, 800], "p1": [0, 0]}],
         "buildings": [{"polygon": [[0, 0], [1000, 0], [1000, 800], [0, 800]]}]}
    objs = [_obj(7, "bed", (100, 100, 300, 300)), _obj(8, "range", (1200, 100, 1300, 200))]
    kept = within_walls(objs, b)
    assert [o.kind for o in kept] == ["bed"] and kept[0].id == "OB001"


def test_tables_beside_a_sofa_are_part_of_a_living_arrangement():
    from engine.arch.objects import _seating
    objs = [_obj(1, "loveseat", (100, 100, 360, 230)), _obj(2, "side table", (20, 120, 95, 200)),
            _obj(3, "side table", (900, 900, 980, 980))]
    _seating(objs, PPC)
    assert [o.kind for o in objs] == ["loveseat", "end table", "side table"]
    assert objs[1].functions == {"living": 0.35}


def test_a_bowl_among_kitchen_fixtures_is_a_kitchen_sink():
    from engine.arch.objects import _context
    objs = [_obj(1, "range", (0, 0, 150, 130)), Obj("OB002", "washbasin", (300, 0, 400, 100), {"bath": 0.6}, 0.6, ["bowl"]),
            _obj(3, "toilet", (3000, 0, 3100, 140)), Obj("OB004", "washbasin", (3200, 0, 3300, 100), {"bath": 0.6}, 0.6, ["bowl"])]
    _context(objs, PPC)
    assert objs[1].kind == "sink" and objs[1].functions == {"kitchen": 0.9}
    assert objs[3].kind == "washbasin"
