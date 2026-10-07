"""PDF input layer (ingest/pdf.py and the /api/pdf endpoints): inspection, page selection,
rendering and analysis of selected pages. The recognition engine itself is not involved
beyond receiving the rendered page."""

from __future__ import annotations

import io
import zlib

import cv2
import numpy as np
import pytest
from PIL import Image
from pypdf import PdfWriter

from app import create_app
from ingest import pdf as pdfin


# ---------- tiny PDF builders (no PDF library needed for writing) ----------

def vector_pdf(pages: list[dict]) -> bytes:
    """pages: [{w, h, rotate?, ops?}] with ops a PDF content stream (points, origin bottom-left)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None]
    kids = []
    for p in pages:
        ops = p.get("ops", "").encode()
        stream = zlib.compress(ops)
        content_id = len(objs) + 1
        objs.append(f"<< /Length {len(stream)} /Filter /FlateDecode >>\nstream\n".encode() + stream + b"\nendstream")
        page_id = len(objs) + 1
        rot = f" /Rotate {p['rotate']}" if p.get("rotate") else ""
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {p['w']} {p['h']}]{rot} /Contents {content_id} 0 R /Resources << >> >>")
        kids.append(page_id)
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode())
        out.write(body if isinstance(body, bytes) else body.encode())
        out.write(b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def plan_ops(w: float, h: float) -> str:
    """Two rooms with thick walls and a door gap, as filled rectangles (vector)."""
    t = 9  # wall thickness in points
    x0, y0, x1, y1 = 0.12 * w, 0.15 * h, 0.88 * w, 0.85 * h
    xm = (x0 + x1) / 2
    door = 0.09 * (y1 - y0)
    rects = [(x0, y0, x1 - x0, t), (x0, y1 - t, x1 - x0, t), (x0, y0, t, y1 - y0), (x1 - t, y0, t, y1 - y0),
             (xm - t / 2, y0, t, (y1 - y0) / 2 - door), (xm - t / 2, y0 + (y1 - y0) / 2 + door, t, (y1 - y0) / 2 - door)]
    return "0 g\n" + "".join(f"{a:.1f} {b:.1f} {c:.1f} {d:.1f} re f\n" for a, b, c, d in rects)


def raster_pdf(images: list[np.ndarray], dpi: int = 100) -> bytes:
    pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in images]
    buf = io.BytesIO()
    pil[0].save(buf, format="PDF", resolution=dpi, save_all=True, append_images=pil[1:])
    return buf.getvalue()


def plan_image(w=900, h=640):
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.rectangle(img, (80, 80), (w - 80, h - 80), (0, 0, 0), 14)
    cv2.line(img, (w // 2, 80), (w // 2, h - 80), (0, 0, 0), 10)
    img[h // 2 - 40:h // 2 + 40, w // 2 - 6:w // 2 + 6] = 255
    return img


@pytest.fixture
def client(tmp_path):
    app = create_app({"TESTING": True, "UPLOAD_FOLDER": str(tmp_path / "results"), "PDF_FOLDER": str(tmp_path / "pdfs")})
    return app.test_client()


def upload(client, payload: bytes, name="plans.pdf"):
    return client.post("/api/pdf", data={"file": (io.BytesIO(payload), name)}, content_type="multipart/form-data")


# ---------- inspection ----------

def test_multi_page_pdf_lists_pages_with_sizes_orientation_and_thumbnails(client):
    payload = vector_pdf([{"w": 595, "h": 842}, {"w": 1684, "h": 1191, "ops": plan_ops(1684, 1191)},
                          {"w": 612, "h": 792, "rotate": 90}])
    r = upload(client, payload)
    assert r.status_code == 200, r.get_json()
    data = r.get_json()
    assert data["page_count"] == 3
    p1, p2, p3 = data["pages"]
    assert (p1["sheet"], p1["orientation"]) == ("A4", "portrait")
    assert (p2["sheet"], p2["orientation"], p2["kind"]) == ("A2", "landscape", "vector")
    assert (p3["orientation"], p3["rotation"], p3["width_in"], p3["height_in"]) == ("landscape", 90, 11.0, 8.5)   # /Rotate applied
    thumb = client.get(p3["thumbnail_url"])
    assert thumb.status_code == 200 and thumb.mimetype == "image/png"
    im = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
    assert im.shape[1] > im.shape[0] and max(im.shape[:2]) <= pdfin.THUMBNAIL_SIDE + 1    # small, landscape


def test_large_sheet_resolution_is_lowered_to_the_pixel_limit_and_scans_are_not_upsampled(tmp_path):
    big = tmp_path / "a0.pdf"
    big.write_bytes(vector_pdf([{"w": 2384, "h": 3370}]))                    # A0
    page = pdfin.inspect_pdf(big).pages[0]
    assert page.sheet == "A0" and page.analysis_dpi < pdfin.DEFAULT_DPI
    assert page.analysis_size[0] * page.analysis_size[1] <= pdfin.MAX_PIXELS
    scan = tmp_path / "scan.pdf"
    scan.write_bytes(raster_pdf([plan_image()], dpi=100))
    sp = pdfin.inspect_pdf(scan).pages[0]
    assert sp.kind == "raster" and round(sp.raster_ppi) == 100 and sp.analysis_dpi == 100


@pytest.mark.parametrize("payload,code", [
    (b"\x89PNG\r\n\x1a\n" + b"0" * 200, "not_pdf"),
    (b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog", "corrupt"),
])
def test_invalid_and_corrupted_files_are_rejected_with_a_readable_reason(client, payload, code):
    r = upload(client, payload)
    assert r.status_code == 400 and r.get_json()["code"] == code and r.get_json()["error"]


def test_encrypted_and_empty_pdfs_are_explained(client):
    w = PdfWriter()
    w.add_blank_page(width=300, height=300)
    w.encrypt("secret")
    buf = io.BytesIO(); w.write(buf)
    r = upload(client, buf.getvalue())
    assert r.status_code == 400 and r.get_json()["code"] == "encrypted"
    empty = io.BytesIO(); PdfWriter().write(empty)
    r = upload(client, empty.getvalue())
    assert r.status_code == 400 and r.get_json()["code"] == "no_pages"


# ---------- analysis of selected pages ----------

def test_selected_page_is_analysed_and_keeps_its_page_number(client):
    """Page 2 of 3 holds the plan; analysing it returns rooms and the source page, and the
    rendered page image the Workbench draws on."""
    img = plan_image()
    payload = raster_pdf([np.full_like(img, 255), img, np.full_like(img, 255)], dpi=100)
    doc = upload(client, payload).get_json()
    r = client.post(f"/api/pdf/{doc['pdf_id']}/pages/2/analyze", json={})
    assert r.status_code == 200, r.get_json()
    data = r.get_json()
    assert data["source"]["page"] == 2 and data["source"]["page_count"] == 3 and data["source"]["type"] == "pdf"
    assert data["image"] == {"width": 900, "height": 640}                    # native scan resolution, not resampled
    assert data["room_count"] + data["unlabeled_space_count"] >= 2
    page_png = client.get(data["page_image_url"])
    assert page_png.status_code == 200
    assert cv2.imdecode(np.frombuffer(page_png.data, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (640, 900)


def test_pdf_page_gives_the_same_result_as_the_same_image(client):
    """The PDF path only rasterizes: a scan embedded at its own resolution analyses exactly like
    the image uploaded directly."""
    img = plan_image()
    doc = upload(client, raster_pdf([img], dpi=100)).get_json()
    via_pdf = client.post(f"/api/pdf/{doc['pdf_id']}/pages/1/analyze", json={}).get_json()
    ok, png = cv2.imencode(".png", img)
    direct = client.post("/api/analyze", data={"file": (io.BytesIO(png.tobytes()), "plan.png")},
                         content_type="multipart/form-data").get_json()
    keys = lambda d: (d["room_count"], d["unlabeled_space_count"], d["door_count"], d["window_count"],
                      sorted(tuple(o["center"]) for o in d["openings"]))
    assert keys(via_pdf) == keys(direct)


def test_multiple_pages_are_independent_analyses(client):
    payload = vector_pdf([{"w": 842, "h": 595, "ops": plan_ops(842, 595)}, {"w": 595, "h": 842},
                          {"w": 595, "h": 842, "ops": plan_ops(595, 842)}])
    doc = upload(client, payload).get_json()
    results = {n: client.post(f"/api/pdf/{doc['pdf_id']}/pages/{n}/analyze", json={}).get_json() for n in (1, 3)}
    assert results[1]["source"]["page"] == 1 and results[3]["source"]["page"] == 3
    assert results[1]["result_id"] != results[3]["result_id"]
    assert results[1]["image"]["width"] > results[1]["image"]["height"]      # landscape page
    assert results[3]["image"]["width"] < results[3]["image"]["height"]      # portrait page
    for res in results.values():
        assert res["unlabeled_space_count"] + res["room_count"] >= 2           # two vector rooms found on each


def test_page_out_of_range_and_unknown_document(client):
    doc = upload(client, vector_pdf([{"w": 595, "h": 842}])).get_json()
    assert client.post(f"/api/pdf/{doc['pdf_id']}/pages/2/analyze", json={}).status_code == 400
    assert client.post(f"/api/pdf/{'0' * 32}/pages/1/analyze", json={}).status_code == 404
    assert client.get(f"/api/pdf/{doc['pdf_id']}/pages/9/thumbnail.png").status_code == 404


def test_rendering_failure_is_reported_not_crashed(client, monkeypatch):
    doc = upload(client, vector_pdf([{"w": 595, "h": 842}])).get_json()

    def boom(*a, **k):
        raise RuntimeError("poppler crashed")
    monkeypatch.setattr("pdf2image.convert_from_path", boom)
    r = client.post(f"/api/pdf/{doc['pdf_id']}/pages/1/analyze", json={})
    assert r.status_code == 400 and r.get_json()["code"] == "render_failed"


def test_page_selection_parser():
    assert pdfin.parse_pages("3, 4 7-9", 12) == [3, 4, 7, 8, 9]
    with pytest.raises(pdfin.PdfInputError):
        pdfin.parse_pages("13", 12)
    with pytest.raises(pdfin.PdfInputError):
        pdfin.parse_pages("", 12)
