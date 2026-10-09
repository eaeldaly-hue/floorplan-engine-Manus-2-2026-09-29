"""engine.arch.projection: the structured plan's zones and objects reach the flat API records the
Rooms / Cleaned / Elements views read, without a second inference."""

from __future__ import annotations

from engine.arch.projection import project
from engine.rescale import to_original


def _square(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _result(room_method="open-plan-zone"):
    zones = [
        {"id": "space_1.Z1", "function": "kitchen", "confidence": 0.9, "polygon": _square(0, 0, 100, 100),
         "center": [50, 50], "area_m2": 10.0, "evidence": [{"source": "object", "kind": "range", "id": "OB001"}],
         "boundaries": []},
        {"id": "space_1.Z2", "function": "living", "confidence": 0.8, "polygon": _square(100, 0, 200, 100),
         "center": [150, 50], "area_m2": 10.0, "evidence": [{"source": "label", "kind": "label", "id": "Living"}],
         "boundaries": []},
    ]
    building = {
        "spaces": [
            {"id": "space_1", "names": [], "polygon": _square(0, 0, 200, 100), "zones": zones, "circulation_m2": 1.5},
            {"id": "space_2", "names": [], "polygon": _square(0, 100, 100, 200),
             "function": {"function": "bedroom", "confidence": 0.6, "evidence": [{"kind": "bed", "id": "OB003"}]}},
        ],
        "objects": [
            {"id": "OB001", "kind": "range", "bbox": [20, 20, 40, 40], "functions": {"kitchen": 1.0}, "confidence": 0.9, "evidence": []},
            {"id": "OB003", "kind": "bed", "bbox": [20, 120, 80, 180], "functions": {"bedroom": 1.0}, "confidence": 0.8, "evidence": []},
        ],
    }
    room = {"id": "room-01", "name": "Living", "space_id": "space_1", "label_center": {"x": 150, "y": 50},
            "area": {"value": 5.0, "unit": "m²", "source": "zone-estimate"},
            "boundary": {"method": room_method, "polygon": [{"x": 120, "y": 10}, {"x": 190, "y": 10}, {"x": 190, "y": 90}],
                         "bbox": {"x": 120, "y": 10, "width": 70, "height": 80}, "confidence": 0.5}}
    return {"building": building, "rooms": [room], "pixel_scale": {"pixels_per_unit": 10.0, "unit": "m"},
            "unlabeled_spaces": [{"id": "space_1", "boundary": {}}, {"id": "space_2", "boundary": {}}]}


def test_zones_and_objects_reach_the_flat_records():
    r = _result()
    project(r)
    assert [z["id"] for z in r["zones"]] == ["space_1.Z1", "space_1.Z2"] and r["zone_count"] == 2
    assert all(z["space_id"] == "space_1" for z in r["zones"])
    assert r["object_count"] == 2
    obj = {o["id"]: o for o in r["objects"]}
    assert obj["OB001"]["space_id"] == "space_1" and obj["OB001"]["zone_id"] == "space_1.Z1"
    assert obj["OB003"]["space_id"] == "space_2" and obj["OB003"]["zone_id"] is None
    spaces = {u["id"]: u for u in r["unlabeled_spaces"]}
    assert spaces["space_1"]["zone_ids"] == ["space_1.Z1", "space_1.Z2"]
    assert spaces["space_2"]["function"]["function"] == "bedroom"


def test_a_labelled_open_plan_room_takes_its_zone_extent():
    """One labelled zone has one geometry: the room record's boundary is the zone's polygon."""
    r = _result()
    project(r)
    room = r["rooms"][0]
    assert room["zone_id"] == "space_1.Z2" and room["boundary"]["zone_id"] == "space_1.Z2"
    assert [[p["x"], p["y"]] for p in room["boundary"]["polygon"]] == _square(100, 0, 200, 100)
    assert room["boundary"]["method"] == "open-plan-zone"                 # still a zone, not a walled room
    assert room["area"] == {"value": 100.0, "unit": "m²", "source": "zone-estimate"}
    assert r["zones"][1]["room_ids"] == ["room-01"] and r["zones"][0]["room_ids"] == []


def test_a_walled_room_keeps_its_own_boundary():
    r = _result(room_method="wall-region")
    project(r)
    room = r["rooms"][0]
    assert room["zone_id"] == "space_1.Z2" and "zone_id" not in room["boundary"]
    assert room["boundary"]["polygon"][0] == {"x": 120, "y": 10}


def test_object_boxes_return_to_the_original_frame():
    r = _result()
    project(r)
    out = to_original({k: v for k, v in r.items()} | {"image": {}}, 2.0, 100, 100)
    assert out["objects"][0]["bbox"] == [10, 10, 20, 20]
    assert out["zones"][0]["polygon"][1] == [50, 0]
