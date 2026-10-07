"""PDF input layer: inspect a PDF, preview its pages, render chosen pages for the engine.

The recognition engine takes one raster image. This module turns one PDF page into that
image and nothing else: it never decides which page is a floor plan (the user selects pages)
and never changes the image content beyond rasterizing it.

Libraries: pypdf (structure: page count, page boxes, rotation, encryption) and poppler
(pdftoppm via pdf2image for rendering, pdfimages for embedded-image metadata). Poppler applies
the page /Rotate and renders the CropBox, i.e. what a PDF viewer shows.

Rendering resolution (analysis_dpi): 150 DPI by default. At 150 DPI, 2.5 mm drawing text is
~15 px tall (readable by OCR) and a 100 mm wall at 1:50 is ~12 px thick, the range the engine
is built for. The DPI is lowered for large sheets so the page stays within the engine's pixel
limit (MAX_PIXELS); a page that is a scan is not rendered above the resolution of its image
(upsampling adds no detail). See docs/PDF_INPUT.md.
"""

from __future__ import annotations

import io
import math
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DPI = 150
MIN_DPI = 72
MAX_DPI = 300
MAX_PIXELS = 24_000_000          # under the engine's 25 MP limit
RENDER_TIMEOUT_S = 180
THUMBNAIL_SIDE = 360

# Named sheet sizes, inches (short, long)
SHEETS = {
    "A4": (8.27, 11.69), "A3": (11.69, 16.54), "A2": (16.54, 23.39), "A1": (23.39, 33.11), "A0": (33.11, 46.81),
    "Letter": (8.5, 11.0), "Legal": (8.5, 14.0), "Tabloid / ANSI B": (11.0, 17.0), "ANSI C": (17.0, 22.0),
    "ANSI D": (22.0, 34.0), "ANSI E": (34.0, 44.0), "ARCH C": (18.0, 24.0), "ARCH D": (24.0, 36.0),
    "ARCH E1": (30.0, 42.0), "ARCH E": (36.0, 48.0),
}


class PdfInputError(ValueError):
    """A PDF problem the user can understand and act on."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


@dataclass
class PageInfo:
    number: int                 # 1-based
    width_pt: float             # as displayed (after /Rotate)
    height_pt: float
    rotation: int
    width_in: float
    height_in: float
    orientation: str            # portrait | landscape
    sheet: str | None           # named sheet size when it matches
    kind: str                   # vector | raster | mixed
    images: int
    raster_ppi: float | None    # effective resolution of the images covering the page (raster / mixed)
    analysis_dpi: int = 0
    analysis_size: tuple = (0, 0)
    dpi_reason: str = ""


@dataclass
class DocumentInfo:
    page_count: int
    pages: list[PageInfo] = field(default_factory=list)
    encrypted: bool = False
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {"page_count": self.page_count, "encrypted": self.encrypted, "seconds": round(self.seconds, 3),
                "pages": [asdict(p) for p in self.pages]}


def _sheet_name(w_in: float, h_in: float) -> str | None:
    short, long = sorted((w_in, h_in))
    for name, (s, l) in SHEETS.items():
        if abs(short - s) <= 0.03 * s and abs(long - l) <= 0.03 * l:
            return name
    return None


def _image_table(path: Path) -> dict[int, list[dict]]:
    """Embedded images per page from `pdfimages -list` (size, effective ppi)."""
    try:
        out = subprocess.run(["pdfimages", "-list", str(path)], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    pages: dict[int, list[dict]] = {}
    for line in out.splitlines()[2:]:
        f = line.split()
        if len(f) < 14 or f[2] not in ("image", "stencil"):
            continue
        try:
            pages.setdefault(int(f[0]), []).append({"w": int(f[3]), "h": int(f[4]), "xppi": float(f[12]), "yppi": float(f[13])})
        except ValueError:
            continue
    return pages


def _kind(page_in2: float, images: list[dict]) -> tuple[str, float | None]:
    """vector: no image content; raster: images cover (almost) the whole page (a scan);
    mixed: both. Coverage = area of the images at their effective resolution."""
    if not images:
        return "vector", None
    covered = 0.0
    weighted = []
    for im in images:
        if im["xppi"] <= 0 or im["yppi"] <= 0:
            continue
        a = (im["w"] / im["xppi"]) * (im["h"] / im["yppi"])
        covered += a
        weighted.append((a, min(im["xppi"], im["yppi"])))
    share = covered / max(page_in2, 1e-6)
    if not weighted or share < 0.02:
        return "vector", None
    ppi = max(weighted)[1]                       # resolution of the largest image
    return ("raster" if share >= 0.6 else "mixed"), ppi


def analysis_dpi(page: PageInfo, target: int = DEFAULT_DPI, max_pixels: int = MAX_PIXELS) -> tuple[int, str]:
    """Resolution for analysing a page; see the module docstring."""
    dpi = float(min(max(target, MIN_DPI), MAX_DPI))
    reason = f"default {int(dpi)} DPI"
    if page.kind == "raster" and page.raster_ppi and page.raster_ppi < dpi:
        dpi = max(float(MIN_DPI), page.raster_ppi)
        reason = f"scanned page: its own resolution ({page.raster_ppi:.0f} ppi), no upsampling"
    limit = math.sqrt(max_pixels / max(page.width_in * page.height_in, 1e-6))
    if dpi > limit:
        dpi = limit
        reason = f"large sheet ({page.width_in:.1f} x {page.height_in:.1f} in): lowered to stay within {max_pixels / 1e6:.0f} MP"
    if dpi < MIN_DPI * 0.5:
        raise PdfInputError(
            f"Page {page.number} is too large to analyse ({page.width_in:.0f} x {page.height_in:.0f} in).", "page_too_large")
    return int(math.floor(dpi)), reason


def _check_header(payload_or_path) -> None:
    head = payload_or_path[:1024] if isinstance(payload_or_path, (bytes, bytearray)) else Path(payload_or_path).read_bytes()[:1024]
    if b"%PDF" not in head:
        raise PdfInputError("This file is not a PDF document.", "not_pdf")


def inspect_pdf(path: Path, target_dpi: int = DEFAULT_DPI) -> DocumentInfo:
    """Page count, page sizes / orientation / kind and the planned analysis resolution."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    t = time.perf_counter()
    path = Path(path)
    _check_header(path)
    try:
        reader = PdfReader(str(path), strict=False)
        encrypted = bool(reader.is_encrypted)
        if encrypted:
            try:
                ok = reader.decrypt("")                 # owner-password-only PDFs open without a password
            except Exception:
                ok = 0
            if not ok:
                raise PdfInputError("This PDF is password-protected. Remove the password and upload it again.", "encrypted")
        count = len(reader.pages)
    except PdfInputError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
        raise PdfInputError(f"This PDF is damaged or cannot be read ({type(exc).__name__}).", "corrupt") from exc
    if count == 0:
        raise PdfInputError("This PDF has no pages.", "no_pages")
    images = _image_table(path)
    pages = []
    for i in range(count):
        try:
            p = reader.pages[i]
            box = p.cropbox
            w, h = float(box.width), float(box.height)
            rot = int(p.rotation or 0) % 360
        except Exception as exc:
            raise PdfInputError(f"Page {i + 1} of this PDF cannot be read ({type(exc).__name__}).", "bad_page") from exc
        if rot in (90, 270):
            w, h = h, w
        w_in, h_in = w / 72.0, h / 72.0
        kind, ppi = _kind(w_in * h_in, images.get(i + 1, []))
        info = PageInfo(number=i + 1, width_pt=round(w, 1), height_pt=round(h, 1), rotation=rot,
                        width_in=round(w_in, 2), height_in=round(h_in, 2),
                        orientation="landscape" if w > h else "portrait", sheet=_sheet_name(w_in, h_in),
                        kind=kind, images=len(images.get(i + 1, [])), raster_ppi=round(ppi, 1) if ppi else None)
        try:
            dpi, why = analysis_dpi(info, target_dpi)
            info.analysis_dpi, info.dpi_reason = dpi, why
            info.analysis_size = (int(round(w_in * dpi)), int(round(h_in * dpi)))
        except PdfInputError as exc:
            info.dpi_reason = str(exc)
        pages.append(info)
    return DocumentInfo(page_count=count, pages=pages, encrypted=encrypted, seconds=time.perf_counter() - t)


def _render(path: Path, number: int, **kwargs):
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError

    try:
        pages = convert_from_path(str(path), first_page=number, last_page=number, use_cropbox=True,
                                  timeout=RENDER_TIMEOUT_S, **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise PdfInputError(f"Rendering page {number} took too long and was stopped.", "render_timeout") from exc
    except (PDFPageCountError, PDFSyntaxError) as exc:
        raise PdfInputError(f"Page {number} could not be rendered: the PDF is damaged.", "render_failed") from exc
    except MemoryError as exc:
        raise PdfInputError(f"Not enough memory to render page {number}.", "out_of_memory") from exc
    except Exception as exc:
        raise PdfInputError(f"Page {number} could not be rendered ({type(exc).__name__}).", "render_failed") from exc
    if not pages:
        raise PdfInputError(f"Page {number} could not be rendered.", "render_failed")
    return pages[0]


def render_thumbnail(path: Path, number: int, side: int = THUMBNAIL_SIDE) -> bytes:
    """Small PNG preview for page selection (never the analysis resolution)."""
    image = _render(Path(path), number, size=side)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_page(path: Path, page: PageInfo, dpi: int | None = None) -> tuple[np.ndarray, dict]:
    """The page as the engine's input image (BGR) at the analysis resolution."""
    if dpi is None:
        if not page.analysis_dpi:
            raise PdfInputError(page.dpi_reason or f"Page {page.number} cannot be analysed.", "page_too_large")
        dpi = page.analysis_dpi
    dpi = int(min(max(dpi, MIN_DPI // 2), MAX_DPI))
    if page.width_in * page.height_in * dpi * dpi > MAX_PIXELS * 1.05:
        raise PdfInputError(f"Page {page.number} at {dpi} DPI would exceed {MAX_PIXELS / 1e6:.0f} MP.", "page_too_large")
    t = time.perf_counter()
    image = _render(Path(path), page.number, dpi=dpi)
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, {"dpi": dpi, "width": int(bgr.shape[1]), "height": int(bgr.shape[0]), "seconds": round(time.perf_counter() - t, 3)}


def parse_pages(text: str, page_count: int) -> list[int]:
    """'3, 4, 7-9' -> [3, 4, 7, 8, 9] (validated against the document)."""
    pages: list[int] = []
    for part in re.split(r"[,\s]+", text.strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise PdfInputError(f"'{part}' is not a page number.", "bad_selection")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        for n in range(min(a, b), max(a, b) + 1):
            if not 1 <= n <= page_count:
                raise PdfInputError(f"Page {n} does not exist (the PDF has {page_count} pages).", "bad_selection")
            if n not in pages:
                pages.append(n)
    if not pages:
        raise PdfInputError("Select at least one page.", "bad_selection")
    return pages


__all__ = ["PdfInputError", "PageInfo", "DocumentInfo", "inspect_pdf", "render_thumbnail", "render_page",
           "analysis_dpi", "parse_pages", "DEFAULT_DPI", "MAX_PIXELS"]
