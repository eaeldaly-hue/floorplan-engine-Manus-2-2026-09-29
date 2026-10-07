"""Robust OCR helpers for floor-plan text extraction.

OCR is treated as evidence for the structural engine.
It must not modify or replace the original floor-plan image.
"""

from __future__ import annotations

import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from ..ocr_runtime import configure_tesseract
from .room_lexicon import is_room_word
from .ocr_preprocessing import prepare_ocr_variants

try:
    import pytesseract
    from pytesseract import Output
except ImportError:
    pytesseract = None
    Output = None


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class OCRLine:
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    source: str

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class OCRBox:
    """A single OCR observation mapped to original image coordinates."""

    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    source: str
    variant: str
    rotation: int
    psm: int
    kind: str

    @property
    def center(self) -> tuple[int, int]:
        return (
            self.x + self.width // 2,
            self.y + self.height // 2,
        )

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class OCRResult:
    """Structured OCR evidence for a floor-plan image."""

    boxes: tuple[OCRBox, ...]
    selected_variant: str | None = None
    selected_rotation: int = 0
    selected_psm: int | None = None
    # Evidence from all passes (engine.analysis.ocr_aggregation): every text region with its
    # readings, support and accept/reject decision, and a summary of each pass.
    regions: tuple = ()
    passes: tuple = ()

    @property
    def room_labels(self) -> tuple[OCRBox, ...]:
        return tuple(box for box in self.boxes if box.kind == "ROOM_LABEL")

    @property
    def dimensions(self) -> tuple[OCRBox, ...]:
        return tuple(box for box in self.boxes if box.kind == "DIMENSION")

    @property
    def other_text(self) -> tuple[OCRBox, ...]:
        return tuple(box for box in self.boxes if box.kind == "OTHER")


# ---------------------------------------------------------------------------
# Tesseract configuration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OCR worker pool
# ---------------------------------------------------------------------------
# Every Tesseract read runs as its own single-threaded process (the build has no OpenMP), so
# the independent reads of a page (the passes of extract_ocr, the dimension retries of its
# rooms) run concurrently on one pool shared by the whole process; the pool bounds how many
# Tesseract processes are alive at once. Results come back in submission order, so everything
# downstream sees exactly the sequential order. FLOORPLAN_OCR_WORKERS=1 reads one at a time.

_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()
_POOL_PREFIX = "floorplan-ocr"
DEFAULT_OCR_WORKERS = 8


def ocr_workers() -> int:
    value = os.environ.get("FLOORPLAN_OCR_WORKERS", "").strip()
    if value.isdigit() and int(value) >= 1:
        return int(value)
    return max(1, min(DEFAULT_OCR_WORKERS, os.cpu_count() or 1))


def run_ocr_tasks(fn, items) -> list:
    """[fn(item) for item in items], with the items read concurrently on the OCR pool."""
    items = list(items)
    workers = ocr_workers()
    nested = threading.current_thread().name.startswith(_POOL_PREFIX)   # never wait on the pool from inside it
    if workers <= 1 or len(items) <= 1 or nested:
        return [fn(item) for item in items]
    configured_ocr_languages()          # resolved once, before the reads start
    global _POOL
    with _POOL_LOCK:
        if _POOL is None or _POOL._max_workers != workers:
            _POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=_POOL_PREFIX)
        pool = _POOL
    return list(pool.map(fn, items))


@lru_cache(maxsize=1)
def configured_ocr_languages() -> str:
    requested = os.environ.get("FLOORPLAN_OCR_LANG", "eng").strip() or "eng"

    if pytesseract is None:
        return ""

    configure_tesseract()

    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception:
        return "eng"

    requested_parts = requested.split("+")

    if all(language in available for language in requested_parts):
        return requested

    if "eng" in available:
        return "eng"

    return next(iter(sorted(available)), "eng")


# ---------------------------------------------------------------------------
# Text classification
# ---------------------------------------------------------------------------


_DIMENSION_PATTERNS = (
    # 12' 6" x 14' 2"
    r"\d+(?:\.\d+)?\s*(?:'|ft|feet)\s*"
    r"\d*(?:\.\d+)?\s*(?:\"|in|inch|inches)?",

    # 12'-6"
    r"\d+(?:\.\d+)?\s*['′]\s*[-–]?\s*"
    r"\d+(?:\.\d+)?\s*[\"″]",

    # 12 x 14 / 12' x 14'
    r"\d+(?:\.\d+)?\s*['′\"]?\s*[x×]\s*"
    r"\d+(?:\.\d+)?\s*['′\"]?",

    # Plain architectural dimensions such as 12'-0"
    r"\d+\s*[-–]\s*\d+\s*[\"″]?",

    # Decimal dimensions: 12.5 x 14.2
    r"\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?",
)

_DIMENSION_REGEX = re.compile(
    "|".join(f"(?:{pattern})" for pattern in _DIMENSION_PATTERNS),
    re.IGNORECASE,
)


_COMMON_ROOM_TERMS = {
    "BEDROOM",
    "MASTER",
    "KITCHEN",
    "LIVING",
    "ROOM",
    "DINING",
    "BATH",
    "BATHROOM",
    "PANTRY",
    "HALL",
    "HALLWAY",
    "CLOSET",
    "LAUNDRY",
    "OFFICE",
    "GARAGE",
    "PORCH",
    "FOYER",
    "ENTRY",
    "ENTRANCE",
    "STORAGE",
    "UTILITY",
    "WIC",
    "WC",
    "TOILET",
    "CORRIDOR",
    "DEN",
    "STUDY",
    "FAMILY",
    "LOUNGE",
    "KIDS",
    "GUEST",
    "PRIMARY",
}


def classify_text(text: str) -> str:
    """Classify OCR text using deterministic pattern recognition."""
    normalized = " ".join(text.upper().split())

    if not normalized:
        return "OTHER"

    if _DIMENSION_REGEX.search(normalized):
        return "DIMENSION"

    words = re.findall(r"[A-Z]+", normalized)

    if any(word in _COMMON_ROOM_TERMS or is_room_word(word) for word in words):
        return "ROOM_LABEL"

    return "OTHER"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def rotate_image(
    image: np.ndarray,
    quarter_turns_clockwise: int,
) -> np.ndarray:
    """Rotate image by 0/90/180/270 degrees clockwise."""
    k = quarter_turns_clockwise % 4

    if k == 0:
        return image

    if k == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if k == 2:
        return cv2.rotate(image, cv2.ROTATE_180)

    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _inverse_rotate_point(
    px: float,
    py: float,
    rotation: int,
    rotated_width: int,
    rotated_height: int,
) -> tuple[float, float]:
    """Map a point from rotated coordinates back to pre-rotation coordinates."""

    rotation %= 4

    if rotation == 0:
        return px, py

    if rotation == 1:
        # Inverse of clockwise 90°.
        return py, rotated_height - 1 - px

    if rotation == 2:
        return (
            rotated_width - 1 - px,
            rotated_height - 1 - py,
        )

    # Inverse of counter-clockwise 90°.
    return rotated_width - 1 - py, px


def map_box_to_original(
    x: int,
    y: int,
    width: int,
    height: int,
    rotation: int,
    rotated_width: int,
    rotated_height: int,
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, int, int]:
    """Map an OCR box back to original source-image pixels."""

    corners = (
        (x, y),
        (x + width, y),
        (x, y + height),
        (x + width, y + height),
    )

    mapped = [
        _inverse_rotate_point(
            float(px),
            float(py),
            rotation,
            rotated_width,
            rotated_height,
        )
        for px, py in corners
    ]

    xs = [point[0] for point in mapped]
    ys = [point[1] for point in mapped]

    left = max(0, round(min(xs) / scale_x))
    top = max(0, round(min(ys) / scale_y))
    right = round(max(xs) / scale_x)
    bottom = round(max(ys) / scale_y)

    return (
        left,
        top,
        max(0, right - left),
        max(0, bottom - top),
    )


# ---------------------------------------------------------------------------
# Legacy line OCR
# ---------------------------------------------------------------------------


def ocr_lines(image: np.ndarray, psm: int) -> list[OCRLine]:
    """Run Tesseract and group word observations into OCR lines (memoized: the same crop is read
    once per process, see `_memo`)."""
    return list(_memo("lines", _ocr_lines_uncached, image, (psm,), 512, psm))


def _ocr_lines_uncached(
    image: np.ndarray,
    psm: int,
) -> list[OCRLine]:
    """Run Tesseract and group word observations into OCR lines.

    This function is retained for compatibility with the existing analyzer.
    """
    if pytesseract is None:
        return []

    data = pytesseract.image_to_data(
        image,
        config=f"--oem 3 --psm {psm}",
        output_type=Output.DICT,
        lang=configured_ocr_languages(),
    )

    grouped: dict[
        tuple[int, int, int, int],
        list[tuple[int, int, int, int, str, float]],
    ] = {}

    keys = (
        "page_num",
        "block_num",
        "par_num",
        "line_num",
    )

    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()

        if not text:
            continue

        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue

        if confidence < 18:
            continue

        key = tuple(
            int(data[field][index])
            for field in keys
        )

        grouped.setdefault(key, []).append(
            (
                int(data["left"][index]),
                int(data["top"][index]),
                int(data["width"][index]),
                int(data["height"][index]),
                text,
                confidence,
            )
        )

    lines: list[OCRLine] = []

    for tokens in grouped.values():
        tokens.sort(key=lambda token: token[0])

        left = min(token[0] for token in tokens)
        top = min(token[1] for token in tokens)
        right = max(
            token[0] + token[2]
            for token in tokens
        )
        bottom = max(
            token[1] + token[3]
            for token in tokens
        )

        lines.append(
            OCRLine(
                text=" ".join(token[4] for token in tokens),
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
                confidence=sum(
                    token[5] for token in tokens
                ) / len(tokens),
                source=f"tesseract-psm-{psm}",
            )
        )

    return lines


# ---------------------------------------------------------------------------
# Robust multi-pass OCR
# ---------------------------------------------------------------------------


def _read_words(
    image: np.ndarray,
    *,
    psm: int,
    variant_name: str,
    rotation: int,
    min_confidence: float,
) -> list[OCRBox]:
    """Read word-level OCR observations from one image configuration."""
    if pytesseract is None:
        return []

    config = (
        f"--oem 3 --psm {psm} "
        "-c preserve_interword_spaces=1"
    )

    try:
        data = pytesseract.image_to_data(
            image,
            lang=configured_ocr_languages(),
            config=config,
            output_type=Output.DICT,
        )
    except Exception:
        return []

    boxes: list[OCRBox] = []

    for index, raw_text in enumerate(data["text"]):
        text = " ".join(str(raw_text).split())

        if not text:
            continue

        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue

        if confidence < min_confidence:
            continue

        width = int(data["width"][index])
        height = int(data["height"][index])

        if width <= 0 or height <= 0:
            continue

        boxes.append(
            OCRBox(
                text=text,
                x=int(data["left"][index]),
                y=int(data["top"][index]),
                width=width,
                height=height,
                confidence=confidence,
                source=f"tesseract-psm-{psm}",
                variant=variant_name,
                rotation=rotation,
                psm=psm,
                kind=classify_text(text),
            )
        )

    return boxes


def _normalize_text(text: str) -> str:
    """Normalize OCR text for deterministic duplicate comparison."""
    text = text.upper()
    text = text.replace("×", "X")
    text = text.replace("′", "'")
    text = text.replace("″", '"')
    text = re.sub(r"[^A-Z0-9'\"X.\- ]+", " ", text)
    return " ".join(text.split())


def _box_iou(a: OCRBox, b: OCRBox) -> float:
    """Calculate intersection-over-union between two boxes."""
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.right, b.right)
    bottom = min(a.bottom, b.bottom)

    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)

    area_a = max(1, a.width * a.height)
    area_b = max(1, b.width * b.height)

    union = area_a + area_b - intersection

    return intersection / max(1, union)


def _center_distance(a: OCRBox, b: OCRBox) -> float:
    ax, ay = a.center
    bx, by = b.center

    return math.hypot(
        float(ax - bx),
        float(ay - by),
    )


def _same_observation(a: OCRBox, b: OCRBox) -> bool:
    """Determine whether two OCR boxes represent the same text."""
    if _normalize_text(a.text) != _normalize_text(b.text):
        return False

    distance = _center_distance(a, b)

    max_dimension = max(
        a.width,
        a.height,
        b.width,
        b.height,
        1,
    )

    if distance <= max(20.0, max_dimension * 0.75):
        return True

    if _box_iou(a, b) >= 0.25:
        return True

    return False


def deduplicate_ocr_boxes(
    boxes: list[OCRBox],
) -> list[OCRBox]:
    """Merge repeated OCR observations while retaining the strongest one."""
    ordered = sorted(
        boxes,
        key=lambda box: box.confidence,
        reverse=True,
    )

    kept: list[OCRBox] = []

    for candidate in ordered:
        duplicate = False

        for existing in kept:
            if _same_observation(candidate, existing):
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    kept.sort(
        key=lambda box: (
            box.y,
            box.x,
        )
    )

    return kept


def _configuration_score(
    boxes: list[OCRBox],
    *,
    min_confidence: float,
) -> float:
    """Score an OCR configuration without rewarding raw noisy counts."""
    reliable = [
        box
        for box in boxes
        if box.confidence >= min_confidence
    ]

    if not reliable:
        return 0.0

    confidence_score = sum(
        max(
            0.0,
            box.confidence - min_confidence,
        )
        for box in reliable
    )

    readable_count_score = 12.0 * math.sqrt(
        len(reliable)
    )

    return confidence_score + readable_count_score


def extract_ocr(
    image: np.ndarray,
    *,
    min_confidence: float = 25.0,
    psms: tuple[int, ...] = (11, 6),
    max_side: int = 5000,
) -> OCRResult:
    """Full-page OCR (see `_extract_ocr_uncached`). Memoized per process: analysing the same page
    again (another page tab, "Analyze again" with or without cleaning) does not re-read it."""
    return _memo("page", _extract_ocr_uncached, image, (min_confidence, tuple(psms), max_side), _page_memo_size(),
                 min_confidence=min_confidence, psms=psms, max_side=max_side)


def _page_memo_size() -> int:
    try:
        return max(0, int(os.environ.get("FLOORPLAN_OCR_MEMO", "4")))
    except ValueError:
        return 4


_MEMO: dict = {}
_MEMO_LOCK = threading.Lock()


def _memo(kind: str, fn, image, extra, size: int, *args, **kwargs):
    """Process-local LRU of OCR results keyed by the exact pixels and parameters. The OCR code
    cannot change while the process runs, and results are immutable (tuples / new lists per call
    site), so a hit returns exactly what a fresh read would. size 0 disables it."""
    if size <= 0 or image is None or getattr(image, "size", 0) == 0:
        return fn(image, *args, **kwargs)
    import hashlib

    a = np.ascontiguousarray(image)
    h = hashlib.blake2b(f"{a.shape}|{a.dtype}|{extra!r}|{configured_ocr_languages()}".encode(), digest_size=20)
    h.update(memoryview(a).cast("B"))
    key = h.hexdigest()
    with _MEMO_LOCK:
        cache = _MEMO.setdefault(kind, {})
        if key in cache:
            value = cache.pop(key)
            cache[key] = value                      # most recently used
            return value
    value = fn(image, *args, **kwargs)
    with _MEMO_LOCK:
        cache = _MEMO.setdefault(kind, {})
        cache[key] = value
        while len(cache) > size:
            cache.pop(next(iter(cache)))
    return value


def clear_ocr_memo() -> None:
    with _MEMO_LOCK:
        _MEMO.clear()


def _extract_ocr_uncached(
    image: np.ndarray,
    *,
    min_confidence: float = 25.0,
    psms: tuple[int, ...] = (11, 6),
    max_side: int = 5000,
) -> OCRResult:
    """
    Run robust multi-variant, multi-orientation OCR and aggregate the evidence.

    Every pass (preprocessing variant x rotation x PSM) is read once. Instead of keeping the
    pass with the best global score, all word observations are mapped to original image
    coordinates, grouped into text regions, and each region is decided on its own evidence
    (see engine.analysis.ocr_aggregation). The original image is never modified.
    """
    if pytesseract is None:
        return OCRResult(boxes=tuple())

    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    from .ocr_aggregation import extract_aggregated

    accepted, regions, passes = extract_aggregated(
        image, min_confidence=min_confidence, psms=psms, max_side=max_side
    )
    # For reference only: the configuration that contributed most accepted regions.
    votes: dict[tuple, int] = {}
    for region in regions:
        if region.accepted and region.best is not None:
            key = (region.best.variant, region.best.rotation, region.best.psm)
            votes[key] = votes.get(key, 0) + 1
    top = max(votes, key=votes.get) if votes else (None, 0, None)
    accepted.sort(key=lambda box: (box.y, box.x))
    return OCRResult(
        boxes=tuple(accepted),
        selected_variant=top[0],
        selected_rotation=top[1],
        selected_psm=top[2],
        regions=tuple(regions),
        passes=tuple(passes),
    )


def merge_document_text(document_boxes, ocr_result: OCRResult | None) -> OCRResult:
    """Combine a document's own text (e.g. a PDF text layer: exact strings and positions) with
    OCR of the same page. The document text wins wherever both read the same place; OCR keeps
    everything the document text does not cover (text drawn as strokes, raster drawings)."""
    document_boxes = tuple(document_boxes)
    ocr_boxes = ocr_result.boxes if ocr_result is not None else ()

    def covered(box: OCRBox) -> bool:
        return any(
            box.x < d.right and d.x < box.right and box.y < d.bottom and d.y < box.bottom
            for d in document_boxes
        )

    boxes = list(document_boxes) + [box for box in ocr_boxes if not covered(box)]
    boxes.sort(key=lambda box: (box.y, box.x))
    if ocr_result is None:
        return OCRResult(boxes=tuple(boxes), selected_variant="document-text")
    return OCRResult(
        boxes=tuple(boxes),
        selected_variant=ocr_result.selected_variant,
        selected_rotation=ocr_result.selected_rotation,
        selected_psm=ocr_result.selected_psm,
        regions=ocr_result.regions,
        passes=ocr_result.passes,
    )


__all__ = [
    "merge_document_text",
    "run_ocr_tasks",
    "ocr_workers",
    "OCRLine",
    "OCRBox",
    "OCRResult",
    "configured_ocr_languages",
    "ocr_lines",
    "rotate_image",
    "map_box_to_original",
    "classify_text",
    "deduplicate_ocr_boxes",
    "extract_ocr",
]
