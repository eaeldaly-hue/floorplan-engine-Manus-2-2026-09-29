"""Architectural Cleaner stage: modes, representation, artifacts, app integration."""

from __future__ import annotations

import io
import json

import cv2
import numpy as np
import pytest

from engine.cleaning import cleaner as C


def test_cleaning_is_off_by_default(monkeypatch):
    for var in ("FLOORPLAN_CLEANING", "FLOORPLAN_STRUCTURAL_LAYER"):
        monkeypatch.delenv(var, raising=False)
    assert C.enabled() == "off" and C.enabled({}) == "off"
    assert C.enabled({"cleaning": "on"}) == "on" and C.enabled({"cleaning": True}) == "on"
    assert C.enabled({"cleaning": "vector"}) == "vector" and C.enabled({"cleaning": "nonsense"}) == "off"
    assert C.enabled({"cleaning": "raster"}) == "raster"
    assert C.enabled({"structural_layer": True}) == "vector"          # the earlier experiment flag
    monkeypatch.setenv("FLOORPLAN_CLEANING", "on")
    assert C.enabled() == "on" and C.enabled({"cleaning": "off"}) == "off"


@pytest.fixture(scope="module")
def vector_page():
    from tests.test_semantic_layer import H, W, _page
    return _page(), (H, W, 3)


def test_vector_cleaning_keeps_architecture_and_suppresses_the_rest(vector_page, monkeypatch):
    from engine.semantic import layer as SL

    paths, shape = vector_page
    layer = SL.build_layer(paths, shape)
    res = C._from_layer(layer, "vector-wall-pen", 0.0)
    rep = res.representation()
    assert res.applicable and rep["source"] == "vector-wall-pen"
    assert rep["symbols_kept"]["DOOR"] == 2 and rep["symbols_kept"]["WINDOW"] == 2
    assert rep["suppressed"]["OTHER"] >= 4                             # sofa, dimension line, frame
    assert {o["kind"] for o in rep["openings"]} >= {"door", "glazing"}
    json.dumps(rep)
    sk = res.skeleton_image
    assert sk.shape == shape and (sk[600:681, 250:421] == 255).all()   # furniture gone from the skeleton
    assert (res.recognition_image[..., 0] < 128).sum() > 0


def test_not_applicable_leaves_the_original(monkeypatch):
    blank = np.full((300, 400, 3), 255, np.uint8)
    res = C.clean(blank, mode="vector")
    assert not res.applicable and res.source == "none" and res.recognition_image is None


def test_artifacts_are_written(vector_page, tmp_path):
    from engine.semantic import layer as SL

    paths, shape = vector_page
    res = C._from_layer(SL.build_layer(paths, shape), "vector-wall-pen", 0.0)
    original = np.full(shape, 255, np.uint8)
    files = C.save_artifacts(tmp_path, original, res, original)
    for name in ("cleaned.png", "recognition-input.png", "elements.png", "comparison.png", "cleaning.json"):
        assert name in files and (tmp_path / name).stat().st_size > 0
    sheet = cv2.imread(str(tmp_path / "comparison.png"))
    assert sheet.shape[1] > sheet.shape[0]                              # 2 x 2 panels


def test_app_returns_cleaning_artifacts(tmp_path, monkeypatch):
    """Image upload with raster cleaning (Plan Model source); URLs served by the app."""
    from app import create_app
    from benchmark.generator import generate
    from benchmark.wall_styles import layout_specs, styled

    app = create_app({"TESTING": True, "UPLOAD_FOLDER": str(tmp_path / "results")})
    client = app.test_client()
    p = generate(styled(layout_specs(4)[1], "hollow"))
    ok, png = cv2.imencode(".png", p.image)
    data = {"file": (io.BytesIO(png.tobytes()), "hollow.png"), "cleaning": "raster"}
    res = client.post("/api/analyze", data=data, content_type="multipart/form-data").get_json()
    assert res["cleaning"]["source"] == "raster" and res["cleaning"]["mode_used"] == "cleaned"
    for url in res["cleaning"]["urls"].values():
        assert client.get(url).status_code == 200
    plain = client.post("/api/analyze", data={"file": (io.BytesIO(png.tobytes()), "hollow.png")},
                        content_type="multipart/form-data").get_json()
    assert "cleaning" not in plain                                      # off: response unchanged


def test_on_mode_does_not_use_the_raster_source():
    blank = np.full((300, 400, 3), 255, np.uint8)
    blank[50:250, 50:60] = 0
    assert C.clean(blank, mode="on").source == "none"                 # raster only when asked for
