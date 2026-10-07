"""The PDF text layer as text evidence (ingest/pdf_text.py, merge_document_text, the
text_evidence input of the analyzer). Image inputs and scans must be unaffected."""

from __future__ import annotations

import io
import zlib

import cv2
import numpy as np
import pytest

from app import create_app
from engine.analysis.ocr import OCRBox, OCRResult, merge_document_text
from ingest import pdf as pdfin
from ingest import pdf_text
from tests.test_pdf_input import plan_ops, raster_pdf, plan_image

FONT = "<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>"


def text_pdf(pages: list[dict]) -> bytes:
    """pages: [{w, h, rotate?, ops, resources?, image?}] (points, origin bottom-left)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None]
    kids = []
    for p in pages:
        stream = zlib.compress(p["ops"].encode("latin-1"))
        content_id = len(objs) + 1
        objs.append(f"<< /Length {len(stream)} /Filter /FlateDecode >>\nstream\n".encode() + stream + b"\nendstream")
        rot = f" /Rotate {p['rotate']}" if p.get("rotate") else ""
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {p['w']} {p['h']}]{rot} /Contents {content_id} 0 R "
                    f"/Resources {p.get('resources', FONT)} >>")
        kids.append(len(objs))
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


def text(x: float, y: float, s: str, size: int = 12, mode: int = 0) -> str:
    return f"BT /F1 {size} Tf {mode} Tr {x} {y} Td ({s}) Tj ET\n"


W, H = 842, 595


def labelled_plan() -> str:
    """Two vector rooms; the left one is labelled in the text layer, with a dimension written as
    three words the way CAD programs export it."""
    return (plan_ops(W, H) + "0 g\n" + text(180, 330, "BEDROOM") + text(175, 300, "12'-0\"") + text(210, 300, "x")
            + text(219, 300, "10'-6\""))


def write(tmp_path, payload: bytes, name="t.pdf"):
    path = tmp_path / name
    path.write_bytes(payload)
    return path


# ---------- reading and mapping the text layer ----------

def test_words_are_mapped_to_the_pixels_of_the_rendered_page(tmp_path):
    path = write(tmp_path, text_pdf([{"w": W, "h": H, "ops": labelled_plan()}]))
    page = pdfin.inspect_pdf(path).pages[0]
    image, _ = pdfin.render_page(path, page, dpi=100)
    pt = pdf_text.read_page_text(path, 1, page.kind)
    assert pt.usable and pt.quality == 1.0
    boxes = pdf_text.text_boxes(pt, page.width_pt, page.height_pt, image.shape)
    bed = next(b for b in boxes if b.text == "BEDROOM")
    k = 100 / 72
    assert abs(bed.x - 180 * k) <= 2 and abs(bed.bottom - (H - 330) * k) <= 4     # baseline at y = 330 pt
    # the ink of the rendered word lies inside the box
    ink = image[bed.y:bed.bottom, bed.x:bed.right].min()
    assert ink < 100 and bed.kind == "ROOM_LABEL" and bed.source == "pdf-text"


def test_a_dimension_split_into_words_is_read_as_one_dimension(tmp_path):
    path = write(tmp_path, text_pdf([{"w": W, "h": H, "ops": labelled_plan()}]))
    pt = pdf_text.read_page_text(path, 1)
    dims = [w[0] for w in pt.words if " x " in w[0]]
    assert dims == ["12'-0\" x 10'-6\""]
    assert not any(w[0] == "x" for w in pt.words)


def test_rotated_page_text_follows_the_displayed_page(tmp_path):
    path = write(tmp_path, text_pdf([{"w": W, "h": H, "rotate": 90, "ops": labelled_plan()}]))
    page = pdfin.inspect_pdf(path).pages[0]
    image, _ = pdfin.render_page(path, page, dpi=72)
    boxes = pdf_text.text_boxes(pdf_text.read_page_text(path, 1), page.width_pt, page.height_pt, image.shape)
    bed = next(b for b in boxes if b.text == "BEDROOM")
    assert image.shape[0] > image.shape[1]                                   # displayed portrait
    assert image[bed.y:bed.bottom, bed.x:bed.right].min() < 100              # the box covers the drawn word


def test_unusable_text_layers_are_not_used(tmp_path):
    empty = write(tmp_path, text_pdf([{"w": W, "h": H, "ops": plan_ops(W, H)}]), "empty.pdf")
    assert pdf_text.read_page_text(empty, 1).reason == "no text layer"
    garbled = write(tmp_path, text_pdf([{"w": W, "h": H, "ops": "".join(
        text(100, 100 + 20 * i, "\\247\\266\\244") for i in range(10))}]), "garbled.pdf")
    pt = pdf_text.read_page_text(garbled, 1)
    assert not pt.usable and "unreadable" in pt.reason
    assert pdf_text.text_boxes(pt, W, H, (H, W, 3)) == ()


def test_invisible_ocr_layer_of_a_scan_is_not_used(tmp_path):
    ops = text(100, 300, "BEDROOM", mode=3)
    path = write(tmp_path, text_pdf([{"w": W, "h": H, "ops": ops}]), "scan.pdf")
    assert pdf_text.read_page_text(path, 1, kind="raster").reason == "invisible OCR layer of a scan"
    assert pdf_text.read_page_text(path, 1, kind="vector").usable          # vector drawings: visible CAD text


# ---------- merging with OCR ----------

def box(text, x, y, w=60, h=14, source="ocr", kind="ROOM_LABEL"):
    return OCRBox(text=text, x=x, y=y, width=w, height=h, confidence=80.0, source=source, variant="v",
                  rotation=0, psm=11, kind=kind)


def test_document_text_wins_where_both_read_and_ocr_keeps_the_rest():
    doc = (box("BEDROOM", 100, 100, source="pdf-text"),)
    ocr = OCRResult(boxes=(box("BEDR0OM", 104, 102), box("den", 400, 300)))
    merged = merge_document_text(doc, ocr)
    assert [b.text for b in merged.boxes] == ["BEDROOM", "den"]
    assert merge_document_text(doc, None).boxes == doc                       # no OCR available


def test_without_text_evidence_the_analysis_is_unchanged(monkeypatch):
    import engine.analyzer as analyzer_module

    calls = []
    real = analyzer_module.extract_ocr
    monkeypatch.setattr(analyzer_module, "extract_ocr", lambda image, **k: calls.append(1) or real(image, **k))
    monkeypatch.setattr(analyzer_module, "merge_document_text",
                        lambda *a, **k: pytest.fail("merge must not run without text evidence"))
    img = plan_image()
    a = analyzer_module.FloorPlanAnalyzer().analyze(img, "plan")
    b = analyzer_module.FloorPlanAnalyzer().analyze(img, "plan", text_evidence=())
    strip = lambda r: {k: v for k, v in r.items() if k not in ("overlay_png", "openings_overlay_png")}
    assert strip(a) == strip(b) and len(calls) == 2


# ---------- the PDF page endpoint ----------

@pytest.fixture
def client(tmp_path):
    app = create_app({"TESTING": True, "UPLOAD_FOLDER": str(tmp_path / "results"), "PDF_FOLDER": str(tmp_path / "pdfs")})
    return app.test_client()


def upload(client, payload: bytes):
    return client.post("/api/pdf", data={"file": (io.BytesIO(payload), "plans.pdf")}, content_type="multipart/form-data").get_json()


def test_vector_page_room_is_named_from_the_text_layer(client):
    doc = upload(client, text_pdf([{"w": W, "h": H, "ops": labelled_plan()}]))
    data = client.post(f"/api/pdf/{doc['pdf_id']}/pages/1/analyze", json={}).get_json()
    layer = data["source"]["text_layer"]
    assert layer["used"] and layer["boxes"] >= 2
    bedroom = [r for r in data["rooms"] if r["name"] == "Bedroom"]
    assert bedroom and bedroom[0]["label_source"] == "pdf-text"
    assert any("PDF text layer" in w for w in data["warnings"])


def test_scan_pages_are_analysed_exactly_as_before(client):
    """A scan has no text layer: nothing is merged and the result equals the direct image upload."""
    img = plan_image()
    doc = upload(client, raster_pdf([img], dpi=100))
    via_pdf = client.post(f"/api/pdf/{doc['pdf_id']}/pages/1/analyze", json={}).get_json()
    assert via_pdf["source"]["text_layer"]["used"] is False
    ok, png = cv2.imencode(".png", img)
    direct = client.post("/api/analyze", data={"file": (io.BytesIO(png.tobytes()), "plan.png")},
                         content_type="multipart/form-data").get_json()
    keys = lambda d: (d["room_count"], d["unlabeled_space_count"], [r["name"] for r in d["rooms"]],
                      sorted(tuple(o["center"]) for o in d["openings"]))
    assert keys(via_pdf) == keys(direct)
