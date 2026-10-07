"""Benchmark-only OCR cache: structural experiments re-run in minutes, not hours.

OCR is ~85-95 % of the analysis time and does not depend on the structural code. While a
benchmark runs, the analyzer's two OCR entry points are served from disk when the same pixels
were read before by the same OCR code:

  extract_ocr(image)        the 32-pass page OCR        key: image bytes + arguments
  ocr_lines(crop, psm)      the dimension re-reads      key: crop bytes + psm

Every key also contains the OCR code version: a hash of the OCR source files, the Tesseract
version and the OCR language. Changing any of them starts a new cache, so a stale reading can
never be served. Production (the app, the desktop build) never uses this module.

    from benchmark import ocr_cache
    with ocr_cache.enabled():           # or ocr_cache.enable() for a whole process
        FloorPlanAnalyzer().analyze(image)
    print(ocr_cache.STATS)

Location: benchmark/.ocr_cache (override with FLOORPLAN_OCR_CACHE). Disable with
FLOORPLAN_OCR_CACHE=off or the benchmarks' --no-ocr-cache flag.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
import subprocess
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OCR_SOURCES = ("engine/analysis/ocr.py", "engine/analysis/ocr_aggregation.py", "engine/analysis/ocr_preprocessing.py",
               "engine/analysis/room_lexicon.py", "engine/ocr_runtime.py", "engine/analysis/text_lines.py")

STATS = {"hits": 0, "misses": 0, "seconds_saved": 0.0, "seconds_reading": 0.0}
_LOCK = threading.Lock()
_ORIGINAL: dict = {}


def cache_dir() -> Path | None:
    value = os.environ.get("FLOORPLAN_OCR_CACHE", "").strip()
    if value.lower() in ("off", "0", "no", "false"):
        return None
    return Path(value) if value else ROOT / "benchmark" / ".ocr_cache"


@lru_cache(maxsize=1)
def code_version() -> str:
    h = hashlib.sha256()
    for rel in OCR_SOURCES:
        h.update(rel.encode())
        h.update((ROOT / rel).read_bytes())
    try:
        from engine.analysis.ocr import configured_ocr_languages, pytesseract
        h.update(str(pytesseract.get_tesseract_version()).encode() if pytesseract else b"no-tesseract")
        h.update(configured_ocr_languages().encode())
        h.update(os.environ.get("FLOORPLAN_TEXT_LINES", "on").encode())
    except Exception as exc:                       # version unknown -> never share a cache with a known one
        h.update(f"unknown:{type(exc).__name__}".encode())
    return h.hexdigest()[:16]


def _key(kind: str, image: np.ndarray, extra) -> str:
    a = np.ascontiguousarray(image)
    h = hashlib.sha256()
    h.update(f"{kind}|{a.shape}|{a.dtype}|{extra!r}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _path(kind: str, key: str) -> Path:
    return cache_dir() / code_version() / kind / key[:2] / f"{key}.pkl"


def _cached(kind: str, fn, image, extra, *args, **kwargs):
    if cache_dir() is None:
        return fn(image, *args, **kwargs)
    path = _path(kind, _key(kind, image, extra))
    if path.exists():
        t = time.perf_counter()
        with open(path, "rb") as f:
            value, cost = pickle.load(f)
        with _LOCK:
            STATS["hits"] += 1
            STATS["seconds_reading"] += time.perf_counter() - t
            STATS["seconds_saved"] += cost
        return value
    t = time.perf_counter()
    value = fn(image, *args, **kwargs)
    cost = time.perf_counter() - t
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as f:   # atomic: threads, re-runs
        pickle.dump((value, cost), f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(f.name, path)
    with _LOCK:
        STATS["misses"] += 1
    return value


def enable() -> None:
    """Serve the analyzer's OCR from the cache for the rest of the process."""
    import engine.analyzer as analyzer

    if _ORIGINAL:
        return
    _ORIGINAL["extract_ocr"] = analyzer.extract_ocr
    _ORIGINAL["ocr_lines"] = analyzer.ocr_lines

    def extract_ocr(image, **kwargs):
        return _cached("page", _ORIGINAL["extract_ocr"], image, sorted(kwargs.items()), **kwargs)

    def ocr_lines(image, psm):
        return _cached("lines", _ORIGINAL["ocr_lines"], image, psm, psm)

    analyzer.extract_ocr = extract_ocr
    analyzer.ocr_lines = ocr_lines


def disable() -> None:
    import engine.analyzer as analyzer

    if _ORIGINAL:
        analyzer.extract_ocr = _ORIGINAL.pop("extract_ocr")
        analyzer.ocr_lines = _ORIGINAL.pop("ocr_lines")


@contextlib.contextmanager
def enabled():
    already = bool(_ORIGINAL)
    enable()
    try:
        yield STATS
    finally:
        if not already:
            disable()


def summary() -> str:
    s = STATS
    return (f"OCR cache {cache_dir() or 'off'} (version {code_version()}): {s['hits']} hits, {s['misses']} misses, "
            f"{s['seconds_saved']:.0f} s of OCR served in {s['seconds_reading']:.1f} s")
