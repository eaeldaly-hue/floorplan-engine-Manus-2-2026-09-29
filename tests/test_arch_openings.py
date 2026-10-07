"""Openings from typed elements: role decided by topology against the final spaces."""

from __future__ import annotations

import types

import numpy as np

from engine.arch.openings import typed_openings


def _structure():
    # left room 1 | right room 2 | bottom strip 3 ; outside -1 ; a wall column at x 95..105
    labels = np.full((300, 400), -1, np.int32)
    labels[20:180, 20:95] = 1
    labels[20:180, 105:380] = 2
    labels[185:280, 20:380] = 3
    ids = {1: "space_1", 2: "space_2", 3: "space_3"}
    return types.SimpleNamespace(space_labels=labels, wall_thickness=10.0, space_id=lambda k: ids.get(k))


def _cleaned(openings):
    return types.SimpleNamespace(openings=openings, wall_thickness=10.0, source="vector-wall-pen")


def test_role_follows_topology():
    door = lambda y0, y1: {"p0": (100.0, y0), "p1": (100.0, y1), "kind": "door", "evidence": "swing arc"}  # noqa: E731
    ops = [door(40, 80), door(90, 130), door(135, 175),                                   # 3 swing doors (door width 40)
           {"p0": (100.0, 30.0), "p1": (100.0, 36.0), "kind": "glazing", "evidence": "2 lines"},     # joint (< half a door)
           {"p0": (60.0, 182.0), "p1": (110.0, 182.0), "kind": "glazing", "evidence": "2 lines"},    # between rooms 1 and 3
           {"p0": (140.0, 20.0), "p1": (190.0, 20.0), "kind": "glazing", "evidence": "2 lines"},     # room 2 to outside
           {"p0": (200.0, 60.0), "p1": (200.0, 110.0), "kind": "gap-window", "evidence": "1 line"}]  # inside room 2
    out = typed_openings(_cleaned(ops), _structure())
    by_start = {tuple(o["start"]): o for o in out}
    assert len(out) == 6                                                     # the joint is not reported
    assert by_start[(140, 20)]["type"] == "window" and by_start[(140, 20)]["evidence"]["room_relation"] == "room_to_exterior"
    assert by_start[(60, 182)]["type"] == "door" and "sliding" in by_start[(60, 182)]["evidence"]["role"]
    assert by_start[(200, 60)]["type"] == "opening" and by_start[(200, 60)]["evidence"]["room_relation"] == "same_space"
    d = by_start[(100, 40)]
    assert d["type"] == "door" and set(d["evidence"]["adjacent_space_ids"]) == {"space_1", "space_2"}
    assert [o["id"] for o in out if o["type"] == "door"] == ["D01", "D02", "D03", "D04"]


def test_door_wins_over_glazing_in_the_same_gap():
    ops = [{"p0": (100.0, 40.0), "p1": (100.0, 80.0), "kind": "door", "evidence": "swing arc"},
           {"p0": (100.0, 41.0), "p1": (100.0, 79.0), "kind": "glazing", "evidence": "2 lines"}]
    out = typed_openings(_cleaned(ops), _structure())
    assert len(out) == 1 and out[0]["type"] == "door"
