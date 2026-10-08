"""The structured plan (engine.arch.building): buildings, walls, openings and the space graph."""

from __future__ import annotations

import json

import cv2
import numpy as np

from engine.analyzer import FloorPlanAnalyzer
from engine.arch.building import build_building
from engine.opening_detection import classify_openings
from engine.structure import analyze_structure


def _two_rooms():
    """Two rooms side by side, a door gap between them and an entrance gap in the outer wall."""
    img = np.full((420, 760, 3), 255, np.uint8)
    cv2.rectangle(img, (40, 40), (720, 380), (0, 0, 0), 12)
    cv2.line(img, (380, 40), (380, 170), (0, 0, 0), 12)
    cv2.line(img, (380, 250), (380, 380), (0, 0, 0), 12)          # interior door 170-250
    img[374:387, 520:600] = 255                                     # entrance in the bottom wall
    return img


def test_two_rooms_door_and_entrance():
    img = _two_rooms()
    s = analyze_structure(img)
    openings = classify_openings(img, s)
    b = build_building(s, s.spaces, [], [], openings)
    assert b["summary"]["buildings"] == 1
    assert b["summary"]["spaces"] == 2
    left, right = (int(s.space_labels[200, x]) for x in (200, 560))
    lid, rid = s.space_id(left), s.space_id(right)
    links = {frozenset(o["connects"]) for o in b["openings"]}
    assert frozenset({lid, rid}) in links                           # the interior door
    assert frozenset({rid, "exterior"}) in links                    # the entrance
    assert rid in next(sp for sp in b["spaces"] if sp["id"] == lid)["neighbors"]["through_openings"]
    assert all(sp["building"] == "B1" for sp in b["spaces"])
    assert b["summary"]["exterior_walls"] >= 3 and any(not w["exterior"] for w in b["walls"])
    json.dumps(b)


def test_analyzer_response_carries_the_building_model():
    r = FloorPlanAnalyzer().analyze(_two_rooms(), "two.png")
    assert r["building"]["schema"] == "floorplan.building/1"
    assert r["building"]["summary"]["spaces"] == len(r["building"]["spaces"])
    json.dumps(r["building"])


def test_app_serves_the_building_document(tmp_path):
    import io

    from app import create_app

    app = create_app({"TESTING": True, "UPLOAD_FOLDER": str(tmp_path / "results"), "PDF_FOLDER": str(tmp_path / "pdfs")})
    client = app.test_client()
    ok, png = cv2.imencode(".png", _two_rooms())
    data = client.post("/api/analyze", data={"file": (io.BytesIO(png.tobytes()), "two.png")},
                       content_type="multipart/form-data").get_json()
    doc = client.get(data["building_url"])
    assert doc.status_code == 200 and doc.get_json() == data["building"]
    assert client.get("/api/results/" + "0" * 32 + "/building.json").status_code == 404
