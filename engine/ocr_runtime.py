"""Resolve and report the local Tesseract executable used by OCR."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

try:
    import pytesseract
except ImportError:  # Keep geometric analysis importable without the Python wrapper.
    pytesseract = None


_KNOWN_TESSERACT_PATHS = (
    "/opt/homebrew/bin/tesseract",  # Apple Silicon Homebrew
    "/usr/local/bin/tesseract",  # Intel Homebrew
    "/usr/bin/tesseract",
)


def configure_tesseract() -> str | None:
    """Configure pytesseract and return an actionable error when unavailable."""
    if pytesseract is None:
        return "Python package pytesseract is not installed"

    configured = os.environ.get("TESSERACT_CMD", "").strip()
    if configured:
        resolved = shutil.which(configured) if not Path(configured).is_absolute() else configured
        candidates = [resolved] if resolved else []
    else:
        candidates = [shutil.which("tesseract"), *_KNOWN_TESSERACT_PATHS]

    for candidate in candidates:
        if not candidate:
            continue
        executable = Path(candidate).expanduser()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            continue
        pytesseract.pytesseract.tesseract_cmd = str(executable)
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            continue
        return None

    if configured:
        return f"Tesseract was not found or could not run at TESSERACT_CMD={configured}"
    return "Tesseract was not found; install it or set TESSERACT_CMD to its executable path"


def tesseract_status() -> tuple[bool, str | None]:
    """Return whether Tesseract and its English language data are usable."""
    error = configure_tesseract()
    if error:
        return False, error

    try:
        languages = set(pytesseract.get_languages(config=""))
    except Exception as exc:
        return False, f"Tesseract language data could not be read ({type(exc).__name__})"
    if "eng" not in languages:
        return False, "Tesseract English language data (eng.traineddata) is missing"
    return True, None
