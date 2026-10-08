from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter

from app import _decode_upload, create_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(tmp_path):
    app = create_app({
        "TESTING": True,
        "UPLOAD_FOLDER": str(tmp_path / "results"),
        "MAX_CONTENT_LENGTH": 20 * 1024 * 1024,
    })
    return app.test_client()


def test_home_page_and_health(client):
    page = client.get("/")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "Analysis Workbench" in html
    assert "sample-button" in html
    assert "download-json" in html
    assert "js/main.js" in html
    assert client.get("/static/js/main.js").status_code == 200
    assert client.get("/static/css/app.css").status_code == 200
    assert client.get("/health").json["status"] == "ok"


def test_sample_image_is_served_for_the_ui(client):
    response = client.get("/api/sample.png")
    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert response.data == (PROJECT_ROOT / "test_floorplan.png").read_bytes()


def test_missing_and_unsupported_uploads_are_rejected(client):
    assert client.post("/api/analyze").status_code == 400
    response = client.post(
        "/api/analyze",
        data={"file": (BytesIO(b"not an image"), "not-a-plan.exe")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "PNG" in response.json["error"]


def test_corrupt_image_is_rejected(client):
    response = client.post(
        "/api/analyze",
        data={"file": (BytesIO(b"broken image data"), "broken.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "صورة" in response.json["error"]


def test_valid_image_upload_is_decoded_and_only_overlay_is_retained(tmp_path):
    class StubAnalyzer:
        def analyze(self, image, source_name):
            assert image.ndim == 3
            return {
                "source_name": source_name,
                "image": {"width": image.shape[1], "height": image.shape[0]},
                "room_count": 0,
                "unlabeled_space_count": 0,
                "pixel_scale": None,
                "rooms": [],
                "unlabeled_spaces": [],
                "warnings": [],
                "overlay_png": b"\x89PNG\r\n\x1a\n",
                "openings_overlay_png": b"\x89PNG\r\n\x1a\n",
            }

    app = create_app({
        "TESTING": True,
        "UPLOAD_FOLDER": str(tmp_path / "results"),
        "ANALYZER": StubAnalyzer(),
    })
    client = app.test_client()
    payload = (PROJECT_ROOT / "test_floorplan.png").read_bytes()
    response = client.post(
        "/api/analyze",
        data={"file": (BytesIO(payload), "plan.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    data = response.json
    assert data["source_name"] == "plan.png"
    assert data["image"]["width"] >= 100
    overlay = client.get(data["overlay_url"])
    assert overlay.status_code == 200
    assert overlay.data.startswith(b"\x89PNG\r\n\x1a\n")
    result_dir = Path(app.config["UPLOAD_FOLDER"]) / data["result_id"]
    assert sorted(path.name for path in result_dir.iterdir()) == ["openings-overlay.png", "rooms-overlay.png"]


def test_pdf_decoder_uses_first_page_and_reports_multipage_file():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    payload = BytesIO()
    writer.write(payload)

    image, warnings = _decode_upload(payload.getvalue(), "plan.pdf")
    assert image.shape[0] >= 100
    assert image.shape[1] >= 100
    assert any("الصفحة الأولى" in warning for warning in warnings)


def test_pdf_decoder_rejects_oversized_page_before_rasterizing():
    writer = PdfWriter()
    writer.add_blank_page(width=10_000, height=10_000)
    payload = BytesIO()
    writer.write(payload)

    with pytest.raises(ValueError, match="25 ميغابكسل"):
        _decode_upload(payload.getvalue(), "oversized.pdf")


def test_sample_endpoint_returns_rooms_and_a_fetchable_overlay(client):
    response = client.post("/api/sample")
    assert response.status_code == 200, response.get_data(as_text=True)
    data = response.json
    assert data["room_count"] >= 14
    assert len(data["rooms"]) == data["room_count"]
    assert data["overlay_url"].startswith("/api/results/")
    assert all("name" in room and "area" in room and "boundary" in room for room in data["rooms"])

    overlay = client.get(data["overlay_url"])
    assert overlay.status_code == 200
    assert overlay.mimetype == "image/png"
    assert overlay.data.startswith(b"\x89PNG\r\n\x1a\n")

    result_dir = Path(client.application.config["UPLOAD_FOLDER"]) / data["result_id"]
    assert sorted(path.name for path in result_dir.iterdir()) == ["building.json", "openings-overlay.png", "rooms-overlay.png"]


def test_result_id_path_is_validated(client):
    assert client.get("/api/results/../../etc/passwd/overlay.png").status_code in (404, 308)
    assert client.get("/api/results/not-a-valid-id/overlay.png").status_code == 404
