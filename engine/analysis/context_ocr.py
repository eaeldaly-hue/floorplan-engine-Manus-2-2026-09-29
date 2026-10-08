"""Context-aware reading: read the text where the structure says a room name should be.

The page passes (engine.analysis.ocr) read the whole sheet at once. Small labels then get lost
or half-read: in a packed text-line mosaic or a full-page pass, 'BED 1' at a 5 px glyph height
may come back empty or as 'BED'. Once the structure is known, the question is narrower: a space
with no name, holding a few text lines, probably has its name among them. Each of those lines is
read on its own (scale-normalised crop, single-line mode, grey and Otsu), and accepted only when
the lexicon finds a room term in it with good confidence. Labels without their room number
('BED' for 'BED 4') are re-read the same way, and the number is kept when the re-read finds the
same room term followed by one.

Evidence, not invention: only drawn text lines inside spaces are read, nothing is added where
no line is, and every added label still goes through the normal association. FLOORPLAN_CONTEXT_OCR=0
turns it off.
"""

from __future__ import annotations

import os
import re

import cv2
import numpy as np

from .ocr import OCRLine, _read_words, pytesseract
from .room_lexicon import normalize, room_name
from .text_lines import TextLine, _normalised_crop, find_text_lines

MIN_CONFIDENCE = 60.0
MAX_READS = 60                 # Tesseract calls per plan (each a single short line)
LINES_PER_SPACE = 3
GENERIC = "Room"               # 'ROOM' alone (or a fragment + ROOM) names no particular room
_NUMBER = re.compile(r"^(.*?)[\s#-]*(\d{1,2})$")


def enabled() -> bool:
    return pytesseract is not None and os.environ.get("FLOORPLAN_CONTEXT_OCR", "1").strip().lower() not in ("0", "off", "false", "no")


def _read(gray: np.ndarray, line: TextLine) -> tuple[str, float]:
    """Best single-line reading of one text line: (text, mean word confidence)."""
    crop, _ = _normalised_crop(gray, line)
    pad = cv2.copyMakeBorder(crop, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
    _, bw = cv2.threshold(pad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best = ("", 0.0)
    for name, img in (("context-gray", pad), ("context-otsu", bw)):
        words = _read_words(img, psm=7, variant_name=name, rotation=0, min_confidence=0)
        words = [w for w in words if w.text.strip()]
        if not words:
            continue
        text = " ".join(w.text for w in sorted(words, key=lambda w: w.x))
        conf = float(np.mean([w.confidence for w in words]))
        if room_name(text) not in (None, GENERIC) and conf > best[1]:
            best = (text, conf)
    return best


def _inside(labels: np.ndarray, x: int, y: int) -> int:
    h, w = labels.shape
    return int(labels[min(h - 1, max(0, y)), min(w - 1, max(0, x))])


def reread(image: np.ndarray, structure, room_lines: list[OCRLine]) -> tuple[list[OCRLine], list[str]]:
    """(room lines with context readings added / completed, notes)."""
    if not enabled() or structure is None or not getattr(structure, "spaces", None):
        return room_lines, []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    labels = structure.space_labels
    if labels.shape != gray.shape:
        return room_lines, []
    lines = find_text_lines(gray)
    if not lines:
        return room_lines, []
    heights = [ln.height for ln in room_lines if ln.height > 0]
    ref = float(np.median(heights)) if heights else None
    out = list(room_lines)
    notes = []
    budget = _Budget()

    # 1. labels that lost their number: re-read the text line under them
    jobs = []
    for i, ln in enumerate(out):
        if re.search(r"\d", ln.text) or not room_name(ln.text):
            continue
        cx, cy = ln.center
        under = [t for t in lines if not t.vertical and t.x <= cx <= t.x + t.w and t.y - 2 <= cy <= t.y + t.h + 2
                 and t.w >= 1.1 * ln.width]
        if under:
            jobs.append((i, min(under, key=lambda t: t.w)))
    for (i, t), (text, conf) in budget.run(gray, jobs):
        ln = out[i]
        m = _NUMBER.match(normalize(text))
        if m and conf >= MIN_CONFIDENCE and room_name(m.group(1)) == room_name(ln.text):
            out[i] = OCRLine(text=f"{ln.text} {m.group(2)}", x=ln.x, y=ln.y, width=ln.width, height=ln.height,
                             confidence=ln.confidence, source=ln.source)
            notes.append(f"{ln.text} -> {out[i].text}")
            budget.hit()

    # 2. spaces without a name: read the text lines inside them, one at a time (largest glyphs
    #    first; a space's next line only while it is still unnamed)
    named = {_inside(labels, *ln.center) for ln in out}
    by_space: dict[int, list[TextLine]] = {}
    for t in lines:
        k = _inside(labels, t.x + t.w // 2, t.y + t.h // 2)
        if k <= 0 or k in named:
            continue
        g = t.glyph_h
        if t.w < 2.0 * g or t.w > 40 * g:          # a mark or a paragraph, not a name
            continue
        if ref and not 0.5 * ref <= g <= 2.0 * ref:
            continue
        by_space.setdefault(k, []).append(t)
    ranked = {k: sorted(c, key=lambda t: (-t.glyph_h, t.y))[:LINES_PER_SPACE] for k, c in by_space.items()}
    for r in range(LINES_PER_SPACE):
        jobs = [(k, c[r]) for k, c in sorted(ranked.items(), key=lambda kv: -kv[1][0].glyph_h)
                if r < len(c) and k not in named]
        for (k, t), (text, conf) in budget.run(gray, jobs):
            if k not in named and text and conf >= MIN_CONFIDENCE:
                out.append(OCRLine(text=text, x=t.x, y=t.y, width=t.w, height=t.h, confidence=round(conf, 1), source="context-ocr"))
                notes.append(f"+{text}")
                named.add(k)
                budget.hit()
        if budget.exhausted:
            break
    return out, notes


class _Budget:
    """Reads run in parallel batches; reading stops when it does not pay: after MAX_READS, or
    when the first STOP_AFTER reads found nothing, or the yield falls below MIN_YIELD."""

    BATCH = 8
    STOP_AFTER = 16
    MIN_YIELD = 1 / 12

    def __init__(self):
        self.reads = 0
        self.hits = 0
        self.exhausted = False

    def hit(self):
        self.hits += 1

    def _stop(self) -> bool:
        if self.reads >= MAX_READS:
            return True
        return self.reads >= self.STOP_AFTER and self.hits < max(1.0, self.MIN_YIELD * self.reads)

    def run(self, gray, jobs):
        """Yield (job, reading) batch by batch until the budget says stop."""
        from concurrent.futures import ThreadPoolExecutor

        from .ocr import ocr_workers
        workers = max(1, min(self.BATCH, ocr_workers()))
        for start in range(0, len(jobs), self.BATCH):
            if self._stop():
                self.exhausted = True
                return
            batch = jobs[start:start + self.BATCH][:MAX_READS - self.reads]
            self.reads += len(batch)
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="context-ocr") as pool:
                    results = list(pool.map(lambda job: _read(gray, job[1]), batch))
            else:
                results = [_read(gray, job[1]) for job in batch]
            yield from zip(batch, results)
