from __future__ import annotations

import io
import json
import logging
import os
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from engine.analyzer import FloorPlanAnalyzer
from engine.ocr_runtime import tesseract_status

logger = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "pdf"}
MAX_PIXELS = 25_000_000           # PDF pages (first-page upload path)
ANALYSIS_PIXELS = 24_000_000      # images above this are reduced to it
MAX_DECODED_PIXELS = 200_000_000  # refuse beyond (memory)


def _decode_upload(payload: bytes, filename: str) -> tuple[np.ndarray, list[str]]:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("ارفع صورة PNG أو JPG أو TIFF أو ملف PDF.")
    if not payload:
        raise ValueError("الملف فارغ.")

    warnings = []
    if suffix == "pdf":
        try:
            from pdf2image import convert_from_bytes
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(payload))
            if not reader.pages:
                raise ValueError("ملف PDF لا يحتوي صفحات.")
            page_count = len(reader.pages)
            dpi = 150
            first_page = reader.pages[0]
            width_points = float(first_page.mediabox.width)
            height_points = float(first_page.mediabox.height)
            estimated_pixels = width_points * height_points * (dpi / 72) ** 2
            if estimated_pixels > MAX_PIXELS:
                raise ValueError("صفحة PDF كبيرة جدًا؛ الحد الأقصى 25 ميغابكسل بعد التحويل.")
            page = convert_from_bytes(payload, dpi=dpi, first_page=1, last_page=1, fmt="png")[0]
            rgb = np.asarray(page.convert("RGB"))
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if page_count > 1:
                warnings.append(f"تم تحليل الصفحة الأولى من ملف PDF المكوّن من {page_count} صفحات.")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"تعذّر قراءة PDF: {type(exc).__name__}") from exc
    else:
        encoded = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("تعذّر فك الصورة. تأكد أن الملف صورة سليمة.")

    height, width = image.shape[:2]
    if height < 100 or width < 100:
        raise ValueError("أبعاد الصورة صغيرة جدًا؛ الحد الأدنى 100 × 100 بكسل.")
    if height * width > MAX_DECODED_PIXELS:
        raise ValueError("الصورة كبيرة جدًا؛ الحد الأقصى 200 ميغابكسل.")
    if height * width > ANALYSIS_PIXELS:
        # large scans and exports are analysed at the engine's working resolution (as PDF pages are)
        scale = (ANALYSIS_PIXELS / (height * width)) ** 0.5
        image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        warnings.append(f"Large image ({width} × {height} px) analysed at {image.shape[1]} × {image.shape[0]} px "
                        f"(the engine's {ANALYSIS_PIXELS // 1_000_000} MP working resolution).")
    return image, warnings


def _trim_old_results(result_root: Path, keep: int = 30) -> None:
    result_dirs = [path for path in result_root.iterdir() if path.is_dir() and len(path.name) == 32]
    result_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for old in result_dirs[keep:]:
        shutil.rmtree(old, ignore_errors=True)


def _valid_id(value: str) -> bool:
    return len(value) == 32 and all(ch in "0123456789abcdef" for ch in value)


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        # PDF drawing sets are often larger than single images; raster uploads are still bounded
        # by their pixel count (MAX_PIXELS) and the Workbench limits images to 20 MB.
        MAX_CONTENT_LENGTH=100 * 1024 * 1024,
        UPLOAD_FOLDER=os.environ.get("FLOORPLAN_UPLOAD_DIR", str(Path(app.root_path) / "instance" / "results")),
    )
    if test_config:
        app.config.update(test_config)
    result_root = Path(app.config["UPLOAD_FOLDER"])
    result_root.mkdir(parents=True, exist_ok=True)
    analyzer = app.config.get("ANALYZER") or FloorPlanAnalyzer()

    pdf_root = Path(app.config.get("PDF_FOLDER") or result_root.parent / "pdfs")
    pdf_root.mkdir(parents=True, exist_ok=True)

    def analyze_and_store(image: np.ndarray, filename: str, load_warnings: list[str], extra: dict | None = None,
                          page_image: np.ndarray | None = None, text_evidence: tuple = (),
                          cleaned=None):
        cleaner = cleaned                     # deferred cleaning step (run_cleaner) or None
        try:
            if cleaner is not None:
                result = analyzer.analyze(image, filename, text_evidence=text_evidence or None, cleaner=cleaner)
            elif text_evidence:
                result = analyzer.analyze(image, filename, text_evidence=text_evidence)
            else:
                result = analyzer.analyze(image, filename)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            logger.exception("Floor-plan analysis failed")
            return jsonify({"error": "حدث خطأ أثناء تحليل المخطط. تأكد من وضوح الملف وحاول مرة أخرى."}), 500

        result_id = uuid.uuid4().hex
        result_dir = result_root / result_id
        result_dir.mkdir(parents=True, exist_ok=False)
        cleaned = result.pop("_clean_result", None) if cleaner is not None else None
        overlay_png = result.pop("overlay_png")
        (result_dir / "rooms-overlay.png").write_bytes(overlay_png)
        if cleaned is not None:
            # the cleaned plan as inspectable artifacts next to the result
            from engine.cleaning.cleaner import save_artifacts
            recognized = cv2.imdecode(np.frombuffer(overlay_png, np.uint8), cv2.IMREAD_COLOR)
            files = save_artifacts(result_dir / "cleaning", image, cleaned, recognized)
            result["cleaning"] = cleaned.representation() | {
                "mode_used": "cleaned" if cleaned.applicable else "original",
                "urls": {name.split(".")[0]: f"/api/results/{result_id}/cleaning/{name}" for name in files}}
        (result_dir / "openings-overlay.png").write_bytes(result.pop("openings_overlay_png"))
        result["warnings"] = load_warnings + result["warnings"]
        result["result_id"] = result_id
        result["overlay_url"] = f"/api/results/{result_id}/overlay.png"
        if result.get("building"):
            # the structured plan (engine.arch.building) as a downloadable document
            (result_dir / "building.json").write_text(json.dumps(result["building"]))
            result["building_url"] = f"/api/results/{result_id}/building.json"
        result["openings_overlay_url"] = f"/api/results/{result_id}/openings.png"
        if page_image is not None:
            # the exact image that was analysed, so the Workbench draws over the same pixels
            ok, encoded = cv2.imencode(".png", page_image)
            if ok:
                (result_dir / "page.png").write_bytes(encoded.tobytes())
                result["page_image_url"] = f"/api/results/{result_id}/page.png"
        if extra:
            result.update(extra)
        _trim_old_results(result_root, keep=30)
        return jsonify(result)

    def run_cleaner(image, warnings, pdf_path, page, evidence, body):
        """Architectural Cleaning (engine.cleaning) as a deferred step: None when off (the analysis
        is unchanged), else a callable the analyzer runs in its structural job, concurrently with
        OCR."""
        from engine.cleaning import enabled

        mode = enabled(dict(body) if body else None)
        if mode == "off":
            return None
        return lambda: _clean_now(image, warnings, pdf_path, page, evidence, mode)

    def _clean_now(image, warnings, pdf_path, page, evidence, mode):
        from engine.cleaning import clean

        try:
            cleaned = clean(image, pdf_path, page, evidence, mode=mode)
        except Exception:
            logger.exception("Architectural cleaning failed")
            warnings.append("Architectural cleaning failed; the original plan was analysed.")
            return None
        if cleaned.applicable:
            rep = cleaned.representation()
            warnings.append(f"Architectural cleaning ({cleaned.source}): recognition ran on the cleaned plan - "
                            f"walls, doors and windows kept, {len(cleaned.openings)} openings sealed, "
                            f"text/dimensions/furniture suppressed {rep['suppressed'] or ''}. {cleaned.reason}.")
        else:
            warnings.append(f"Architectural cleaning not applicable ({cleaned.reason}); the original plan was analysed.")
        return cleaned

    @app.get("/api/results/<result_id>/cleaning/<name>")
    def get_cleaning_artifact(result_id: str, name: str):
        if len(result_id) != 32 or any(ch not in "0123456789abcdef" for ch in result_id) or name not in (
                "cleaned.png", "recognition-input.png", "elements.png", "comparison.png", "cleaning.json"):
            return jsonify({"error": "النتيجة غير موجودة."}), 404
        directory = result_root / result_id / "cleaning"
        if not (directory / name).is_file():
            return jsonify({"error": "انتهت صلاحية النتيجة أو لم تعد موجودة."}), 404
        return send_from_directory(directory, name, max_age=0)

    # ---------- PDF documents: inspect, preview pages, analyse selected pages ----------

    def _pdf_dir(doc_id: str) -> Path | None:
        if not _valid_id(doc_id):
            return None
        directory = pdf_root / doc_id
        return directory if (directory / "source.pdf").is_file() else None

    def _pdf_error(exc: Exception, status: int = 400):
        from ingest.pdf import PdfInputError
        if isinstance(exc, PdfInputError):
            return jsonify({"error": str(exc), "code": exc.code}), status
        logger.exception("PDF handling failed")
        return jsonify({"error": "The PDF could not be processed.", "code": "internal"}), 500

    @app.post("/api/pdf")
    def pdf_inspect():
        """Store an uploaded PDF and describe its pages (no analysis yet; pages are chosen by the user)."""
        from ingest.pdf import PdfInputError, inspect_pdf
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "Choose a PDF file first."}), 400
        payload = uploaded.read()
        if not payload:
            return jsonify({"error": "The file is empty.", "code": "empty"}), 400
        doc_id = uuid.uuid4().hex
        directory = pdf_root / doc_id
        directory.mkdir(parents=True)
        (directory / "source.pdf").write_bytes(payload)
        try:
            info = inspect_pdf(directory / "source.pdf")
        except (PdfInputError, Exception) as exc:
            shutil.rmtree(directory, ignore_errors=True)
            return _pdf_error(exc)
        name = secure_filename(uploaded.filename) or "document.pdf"
        data = {"pdf_id": doc_id, "filename": name, "size_bytes": len(payload), **info.to_dict()}
        for page in data["pages"]:
            page["thumbnail_url"] = f"/api/pdf/{doc_id}/pages/{page['number']}/thumbnail.png"
        (directory / "info.json").write_text(json.dumps(data))
        _trim_old_results(pdf_root, keep=20)
        return jsonify(data)

    @app.get("/api/pdf/<doc_id>/pages/<int:number>/thumbnail.png")
    def pdf_thumbnail(doc_id: str, number: int):
        from ingest.pdf import render_thumbnail
        directory = _pdf_dir(doc_id)
        if directory is None:
            return jsonify({"error": "This PDF is no longer available; upload it again."}), 404
        thumb = directory / f"thumb-{number}.png"
        if not thumb.is_file():
            info = json.loads((directory / "info.json").read_text())
            if not 1 <= number <= info["page_count"]:
                return jsonify({"error": f"Page {number} does not exist."}), 404
            try:
                thumb.write_bytes(render_thumbnail(directory / "source.pdf", number))
            except Exception as exc:
                return _pdf_error(exc)
        return send_from_directory(directory, thumb.name, mimetype="image/png", max_age=3600)

    @app.post("/api/pdf/<doc_id>/pages/<int:number>/analyze")
    def pdf_analyze_page(doc_id: str, number: int):
        """Render one selected page at the analysis resolution and run the engine on it."""
        from ingest.pdf import PageInfo, render_page
        directory = _pdf_dir(doc_id)
        if directory is None:
            return jsonify({"error": "This PDF is no longer available; upload it again."}), 404
        info = json.loads((directory / "info.json").read_text())
        if not 1 <= number <= info["page_count"]:
            return jsonify({"error": f"Page {number} does not exist (the PDF has {info['page_count']} pages).",
                            "code": "bad_selection"}), 400
        stored = info["pages"][number - 1]
        page = PageInfo(**{k: stored[k] for k in PageInfo.__dataclass_fields__ if k in stored})
        page.analysis_size = tuple(page.analysis_size)
        body = request.get_json(silent=True) or {}
        try:
            image, render = render_page(directory / "source.pdf", page, body.get("dpi"))
        except Exception as exc:
            return _pdf_error(exc)
        source = {"type": "pdf", "filename": info["filename"], "pdf_id": doc_id, "page": number,
                  "page_count": info["page_count"], "page_kind": page.kind, "sheet": page.sheet,
                  "page_size_in": [page.width_in, page.height_in], "render": render, "dpi_reason": page.dpi_reason}
        stem = Path(info["filename"]).stem
        warnings = [f"PDF page {number} of {info['page_count']} rendered at {render['dpi']} DPI "
                    f"({render['width']} × {render['height']} px; {page.dpi_reason})."]
        # The page's own text layer (exact names, dimensions) is merged with OCR of the page.
        from ingest.pdf_text import read_page_text, text_boxes
        page_text = read_page_text(directory / "source.pdf", number, page.kind)
        evidence = text_boxes(page_text, page.width_pt, page.height_pt, image.shape)
        source["text_layer"] = page_text.summary() | {"boxes": len(evidence)}
        if evidence:
            warnings.append(f"Text read from the PDF text layer ({len(evidence)} words) together with OCR of the page.")
        elif page_text.reason and page_text.reason != "no text layer":
            warnings.append(f"PDF text layer not used: {page_text.reason}; text was read by OCR.")
        cleaned = run_cleaner(image, warnings, directory / "source.pdf", number, evidence, body)
        return analyze_and_store(image, f"{stem}-page-{number}", warnings, {"source": source}, page_image=image,
                                 text_evidence=evidence, cleaned=cleaned)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        ocr_ready, ocr_error = tesseract_status()
        return jsonify({"status": "ok", "ocr_available": ocr_ready, "ocr_error": ocr_error})

    @app.post("/api/analyze")
    def analyze_upload():
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "اختر ملفًا قبل بدء التحليل."}), 400
        original_name = secure_filename(uploaded.filename) or "floor-plan"
        try:
            image, load_warnings = _decode_upload(uploaded.read(), original_name)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        cleaned = run_cleaner(image, load_warnings, None, None, (), request.form)
        reduced = any(w.startswith("Large image") for w in load_warnings)
        return analyze_and_store(image, original_name, load_warnings, cleaned=cleaned,
                                 page_image=image if reduced else None)

    @app.post("/api/sample")
    def analyze_sample():
        sample_path = Path(app.root_path) / "test_floorplan.png"
        image = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "لم يتم تضمين ملف المخطط التجريبي."}), 404
        return analyze_and_store(image, "test_floorplan.png", [])

    @app.get("/api/sample.png")
    def sample_image():
        # Lets the UI load the bundled plan as a normal upload (and show it).
        return send_from_directory(app.root_path, "test_floorplan.png", mimetype="image/png", max_age=0)

    @app.get("/api/results/<result_id>/overlay.png")
    def get_overlay(result_id: str):
        if len(result_id) != 32 or any(ch not in "0123456789abcdef" for ch in result_id):
            return jsonify({"error": "النتيجة غير موجودة."}), 404
        directory = result_root / result_id
        if not (directory / "rooms-overlay.png").is_file():
            return jsonify({"error": "انتهت صلاحية النتيجة أو لم تعد موجودة."}), 404
        return send_from_directory(directory, "rooms-overlay.png", mimetype="image/png", max_age=0)

    @app.get("/api/results/<result_id>/openings.png")
    def get_openings_overlay(result_id: str):
        if len(result_id) != 32 or any(ch not in "0123456789abcdef" for ch in result_id):
            return jsonify({"error": "النتيجة غير موجودة."}), 404
        directory = result_root / result_id
        if not (directory / "openings-overlay.png").is_file():
            return jsonify({"error": "انتهت صلاحية النتيجة أو لم تعد موجودة."}), 404
        return send_from_directory(directory, "openings-overlay.png", mimetype="image/png", max_age=0)

    @app.get("/api/results/<result_id>/building.json")
    def get_building(result_id: str):
        if not _valid_id(result_id) or not (result_root / result_id / "building.json").is_file():
            return jsonify({"error": "انتهت صلاحية النتيجة أو لم تعد موجودة."}), 404
        return send_from_directory(result_root / result_id, "building.json", mimetype="application/json", max_age=0)

    @app.get("/api/results/<result_id>/page.png")
    def get_page_image(result_id: str):
        if not _valid_id(result_id) or not (result_root / result_id / "page.png").is_file():
            return jsonify({"error": "انتهت صلاحية النتيجة أو لم تعد موجودة."}), 404
        return send_from_directory(result_root / result_id, "page.png", mimetype="image/png", max_age=0)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify({"error": f"حجم الملف يتجاوز الحد الأقصى ({limit} ميغابايت)."}), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        debug=False,
        threaded=True,
    )
