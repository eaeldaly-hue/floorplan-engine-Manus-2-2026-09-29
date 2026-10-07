"""The PDF's own text layer as text evidence for the engine.

CAD exports carry the room names, dimensions and scale notes as real text: exact strings at
exact positions, read in ~0.1 s. This module turns the words of one page into OCR-style boxes
in the pixel frame of the rendered page (`render_page`), so the engine can use them alongside
its own OCR (see `engine.analysis.ocr.merge_document_text`).

The text layer is evidence, not a replacement for OCR: CAD programs often export part of the
text as drawn strokes (e.g. AutoCAD SHX fonts), which only OCR can read. The engine therefore
still runs OCR on the unchanged page and the text layer wins only where both read the same
place (docs/PDF_TEXT_LAYER.md).

A page's text layer is not used when
* it has no words, or too few ordinary characters (broken font encodings);
* it is an invisible OCR layer added to a scan by another program (text render mode 3 or the
  GlyphLessFont of Tesseract/OCRmyPDF): that is somebody else's OCR, not drawing text.
"""

from __future__ import annotations

import html
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

MIN_QUALITY = 0.8              # share of words made of ordinary drawing characters
TEXT_TIMEOUT_S = 60
_WORD = re.compile(r'<word xMin="([\d.-]+)" yMin="([\d.-]+)" xMax="([\d.-]+)" yMax="([\d.-]+)">(.*?)</word>')
_ORDINARY = re.compile(r"""[A-Za-z0-9.,:;'"()/\-+=×x%&²³°#@\s]+""")
_QUOTES = str.maketrans({"”": '"', "“": '"', "″": '"', "’": "'", "‘": "'", "′": "'"})
_DIM_TOKEN = re.compile(r"""^\(?\d{1,3}'(-?\d{1,2}(\s*\d/\d)?")?\)?$|^\(?\d{1,2}"\)?$|^\(?\d+([.,]\d+)?\s*(m|mm)?\)?$""")


@dataclass
class PageText:
    """Words of one page in PDF points (top-left origin, CropBox, page rotation applied)."""

    words: list[tuple[str, float, float, float, float]] = field(default_factory=list)
    quality: float = 0.0
    usable: bool = False
    reason: str = ""
    seconds: float = 0.0

    def summary(self) -> dict:
        return {"words": len(self.words), "quality": round(self.quality, 3), "used": self.usable,
                "reason": self.reason, "seconds": round(self.seconds, 3)}


def _read_words(path: Path, number: int) -> list[tuple[str, float, float, float, float]]:
    out = subprocess.run(["pdftotext", "-bbox", "-cropbox", "-f", str(number), "-l", str(number), str(path), "-"],
                         capture_output=True, text=True, timeout=TEXT_TIMEOUT_S).stdout
    words = []
    for m in _WORD.finditer(out):
        text = html.unescape(m.group(5)).translate(_QUOTES).strip()
        if text:
            words.append((text, *map(float, m.group(1, 2, 3, 4))))
    return words


def _invisible_ocr_layer(path: Path, number: int) -> bool:
    """True when the page's text is an invisible OCR layer (text render mode 3 or GlyphLessFont)."""
    try:
        from pypdf import PdfReader

        page = PdfReader(str(path)).pages[number - 1]
        fonts = page.get("/Resources", {}).get("/Font", {}) or {}
        for ref in fonts.values():
            if "GlyphLess" in str(ref.get_object().get("/BaseFont", "")):
                return True
        data = page.get_contents().get_data() if page.get_contents() is not None else b""
        return re.search(rb"(?<![\d.])3\s+Tr\b", data) is not None
    except Exception:
        return False


def merge_dimension_pairs(words):
    """`(12'-0"`, `x`, `7'-0")` on one baseline -> one word, the way OCR reads the line."""
    dims = [i for i, w in enumerate(words) if _DIM_TOKEN.match(w[0])]
    used, merged = set(), []

    def neighbour(x, left):
        h = max(x[4] - x[2], 1.0)
        best = None
        for i in dims:
            w = words[i]
            if i in used or abs((w[2] + w[4]) / 2 - (x[2] + x[4]) / 2) > 0.4 * h:
                continue
            gap = x[1] - w[3] if left else w[1] - x[3]
            if -0.1 * h <= gap < 1.2 * h and (best is None or gap < best[0]):
                best = (gap, i)
        return best[1] if best else None

    for k, x in enumerate(words):
        if x[0] not in ("x", "X", "×"):
            continue
        a = neighbour(x, left=True)
        b = neighbour(x, left=False)
        if a is None or b is None or a == b:
            continue
        wa, wb = words[a], words[b]
        used.update((a, b, k))
        merged.append((f"{wa[0]} x {wb[0]}", wa[1], min(wa[2], x[2], wb[2]), wb[3], max(wa[4], x[4], wb[4])))
    return [w for i, w in enumerate(words) if i not in used] + merged


def read_page_text(path: Path, number: int, kind: str = "vector") -> PageText:
    """The page's text layer and whether it can be used as evidence."""
    t = time.perf_counter()
    result = PageText()
    try:
        result.words = _read_words(Path(path), number)
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.reason = f"text layer could not be read ({type(exc).__name__})"
        result.seconds = time.perf_counter() - t
        return result
    if not result.words:
        result.reason = "no text layer"
    else:
        result.quality = sum(bool(_ORDINARY.fullmatch(w[0])) for w in result.words) / len(result.words)
        if result.quality < MIN_QUALITY:
            result.reason = "text layer unreadable (font encoding)"
        elif kind != "vector" and _invisible_ocr_layer(Path(path), number):
            result.reason = "invisible OCR layer of a scan"
        else:
            result.usable = True
            result.words = merge_dimension_pairs(result.words)
    result.seconds = time.perf_counter() - t
    return result


def text_boxes(page_text: PageText, page_width_pt: float, page_height_pt: float, image_shape) -> tuple:
    """OCR-style boxes (engine.analysis.ocr.OCRBox) in the pixel frame of the rendered page."""
    from engine.analysis.ocr import OCRBox, classify_text

    if not page_text.usable:
        return ()
    height, width = image_shape[:2]
    sx, sy = width / page_width_pt, height / page_height_pt
    boxes = []
    for text, x0, y0, x1, y1 in page_text.words:
        x, y = int(round(x0 * sx)), int(round(y0 * sy))
        w, h = max(1, int(round((x1 - x0) * sx))), max(1, int(round((y1 - y0) * sy)))
        if x >= width or y >= height or x + w <= 0 or y + h <= 0:
            continue
        rotation = 90 if h > 1.5 * w and len(text) > 2 else 0      # vertical word
        boxes.append(OCRBox(text=text, x=x, y=y, width=w, height=h, confidence=99.0, source="pdf-text",
                            variant="pdf-text", rotation=rotation, psm=0, kind=classify_text(text)))
    boxes.sort(key=lambda b: (b.y, b.x))
    return tuple(boxes)
