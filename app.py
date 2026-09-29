from __future__ import annotations

import io
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

logger = logging.getLogger(__name__)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "pdf"}
MAX_PIXELS = 25_000_000


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
    if height * width > MAX_PIXELS:
        raise ValueError("الصورة كبيرة جدًا؛ الحد الأقصى 25 ميغابكسل.")
    return image, warnings


def _trim_old_results(result_root: Path, keep: int = 30) -> None:
    result_dirs = [path for path in result_root.iterdir() if path.is_dir() and len(path.name) == 32]
    result_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for old in result_dirs[keep:]:
        shutil.rmtree(old, ignore_errors=True)


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=20 * 1024 * 1024,
        UPLOAD_FOLDER=os.environ.get("FLOORPLAN_UPLOAD_DIR", str(Path(app.root_path) / "instance" / "results")),
    )
    if test_config:
        app.config.update(test_config)
    result_root = Path(app.config["UPLOAD_FOLDER"])
    result_root.mkdir(parents=True, exist_ok=True)
    analyzer = app.config.get("ANALYZER") or FloorPlanAnalyzer()

    def analyze_and_store(image: np.ndarray, filename: str, load_warnings: list[str]):
        try:
            result = analyzer.analyze(image, filename)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            logger.exception("Floor-plan analysis failed")
            return jsonify({"error": "حدث خطأ أثناء تحليل المخطط. تأكد من وضوح الملف وحاول مرة أخرى."}), 500

        result_id = uuid.uuid4().hex
        result_dir = result_root / result_id
        result_dir.mkdir(parents=True, exist_ok=False)
        (result_dir / "rooms-overlay.png").write_bytes(result.pop("overlay_png"))
        result["warnings"] = load_warnings + result["warnings"]
        result["result_id"] = result_id
        result["overlay_url"] = f"/api/results/{result_id}/overlay.png"
        _trim_old_results(result_root, keep=30)
        return jsonify(result)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            ocr_ready = True
        except Exception:
            ocr_ready = False
        return jsonify({"status": "ok", "ocr_available": ocr_ready})

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
        return analyze_and_store(image, original_name, load_warnings)

    @app.post("/api/sample")
    def analyze_sample():
        sample_path = Path(app.root_path) / "test_floorplan.png"
        image = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "لم يتم تضمين ملف المخطط التجريبي."}), 404
        return analyze_and_store(image, "test_floorplan.png", [])

    @app.get("/api/results/<result_id>/overlay.png")
    def get_overlay(result_id: str):
        if len(result_id) != 32 or any(ch not in "0123456789abcdef" for ch in result_id):
            return jsonify({"error": "النتيجة غير موجودة."}), 404
        directory = result_root / result_id
        if not (directory / "rooms-overlay.png").is_file():
            return jsonify({"error": "انتهت صلاحية النتيجة أو لم تعد موجودة."}), 404
        return send_from_directory(directory, "rooms-overlay.png", mimetype="image/png", max_age=0)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify({"error": "حجم الملف يتجاوز الحد الأقصى (20 ميغابايت)."}), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        debug=False,
        threaded=True,
    )
