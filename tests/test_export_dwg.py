"""DWG export (engine.export.dwg) and the CAD download routes: a genuine DWG or a clear error,
never a DXF under a .dwg name."""

from __future__ import annotations

import io
import stat

import ezdxf
import pytest

from engine.export import dwg
from engine.export.dwg import DwgConversionError, DwgUnavailable, dwg_bytes
from tests.test_export_dxf import _model

HAVE_LIBREDWG = dwg._libredwg() is not None


def test_switched_off_dwg_is_an_error_not_a_dxf(monkeypatch):
    monkeypatch.setenv("FLOORPLAN_DWG_BACKEND", "off")
    assert dwg.backend()["available"] is False
    with pytest.raises(DwgUnavailable):
        dwg_bytes(_model())


def test_unsupported_version_is_rejected(monkeypatch):
    for version in ("R2004", "R2018"):           # R2004: LibreDWG cannot read its own R2004 back
        with pytest.raises(ValueError):
            dwg_bytes(_model(), version=version)


def test_a_converter_that_does_not_write_a_dwg_is_an_error(tmp_path, monkeypatch):
    # a stand-in dxf2dwg that writes something else: the header check must refuse it
    for name in ("dxf2dwg", "dwg2dxf"):
        script = tmp_path / name
        script.write_text('#!/bin/sh\nwhile [ "$1" != "-o" ]; do shift; done; printf "0\\nSECTION" > "$2"\n')
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FLOORPLAN_DWG_BACKEND", "libredwg")
    monkeypatch.setenv("FLOORPLAN_DXF2DWG", str(tmp_path / "dxf2dwg"))
    with pytest.raises(DwgConversionError):
        dwg_bytes(_model())


@pytest.mark.skipif(not HAVE_LIBREDWG, reason="GNU LibreDWG (dxf2dwg) not installed")
def test_libredwg_writes_a_dwg_holding_the_drawing(monkeypatch):
    monkeypatch.setenv("FLOORPLAN_DWG_BACKEND", "libredwg")
    data, report = dwg_bytes(_model(pixel_scale={"unit": "m", "pixels_per_unit": 50.0}))
    assert data[:6] == b"AC1015" and not data.lstrip().startswith(b"0")
    assert report["units"] == "mm" and report["dwg"]["backend"] == "libredwg"
    assert report["dwg"]["entities"] > 10


# ---------------------------------------------------------------- routes

@pytest.fixture
def analysed(tmp_path):
    from app import create_app
    app = create_app({"TESTING": True, "UPLOAD_FOLDER": str(tmp_path / "results")})
    client = app.test_client()
    data = client.post("/api/sample").json
    return client, data


def test_the_dxf_route_serves_a_real_dxf_of_the_result(analysed):
    client, data = analysed
    assert data["exports"]["dxf_url"] == f"/api/results/{data['result_id']}/export.dxf"
    r = client.get(data["exports"]["dxf_url"])
    assert r.status_code == 200 and r.mimetype == "application/dxf"
    assert r.headers["Content-Disposition"].endswith('.dxf"')
    doc = ezdxf.read(io.StringIO(r.data.decode("utf8")))
    assert not doc.audit().errors
    assert len(doc.modelspace().query('LINE[layer=="WALL_CENTERLINES"]')) == len(data["building"]["walls"])
    assert client.get("/api/results/" + "0" * 32 + "/export.dxf").status_code == 404


def test_the_dwg_route_never_disguises_a_dxf(analysed, monkeypatch):
    client, data = analysed
    monkeypatch.setenv("FLOORPLAN_DWG_BACKEND", "off")
    assert client.get("/api/export/formats").json["dwg"]["available"] is False
    r = client.get(f"/api/results/{data['result_id']}/export.dwg")
    assert r.status_code == 503 and r.json["code"] == "dwg_unavailable"


@pytest.mark.skipif(not HAVE_LIBREDWG, reason="GNU LibreDWG (dxf2dwg) not installed")
def test_the_dwg_route_serves_a_dwg_when_a_converter_is_installed(analysed, monkeypatch):
    client, data = analysed
    monkeypatch.setenv("FLOORPLAN_DWG_BACKEND", "libredwg")
    assert data["exports"].get("dwg_url")
    r = client.get(data["exports"]["dwg_url"])
    assert r.status_code == 200 and r.mimetype == "application/acad"
    assert r.data[:6] == b"AC1015" and r.headers["Content-Disposition"].endswith('.dwg"')
