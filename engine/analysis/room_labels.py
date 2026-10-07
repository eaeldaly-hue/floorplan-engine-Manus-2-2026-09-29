"""Assemble room labels from aggregated OCR words.

Room names are often several words ("LIVING ROOM", "MASTER BEDROOM", "BR. 2") and are
sometimes stacked over two or more lines ("BONUS / ROOM", "SECOND FLOOR / MASTER SUITE").
This module builds text lines from all accepted OCR words — not only words that are room
terms on their own — so that qualifiers belong to the label, and then decides which lines
are room labels:

* words join a line when they share a baseline (vertical centres within half a word
  height), have similar heights and are separated by at most about one word height;
* short alphabetic lines stack into one label when they are centred on each other, have
  similar heights and are separated by less than a line height; dimension and area text
  never stacks (it is not alphabetic);
* a line is a room label when the lexicon finds a room term in it, or when a two-letter room
  abbreviation (BR, WC, CL) stands with a room number;
* tokens that are measurements (dimensions, areas, units) are not part of the name.

Nearby but unrelated labels stay apart: they are on different baselines, or too far apart,
or not stacked on a common centre.
"""

from __future__ import annotations

import re

from .ocr import OCRBox
from .ocr_fusion import OCRTextGroup, normalize_ocr_text
from .room_lexicon import abbreviations_active, is_room_word, is_short_abbreviation, normalize, room_name

_MEASURE = re.compile(r"\d+\s*['′\"″]|\d+[.,]\d+|\bM2\b|M²|\bSQ\b|\bFT\b|\d{3,}", re.IGNORECASE)


def _is_name_token(text: str) -> bool:
    """Words that can be part of a room name: alphabetic words of 3+ letters, room-term
    abbreviations (BR, WC, CL), and 1-2 digit room numbers. Measurements, areas, units and
    mixed letter/digit fragments (typical OCR debris) are not."""
    t = text.strip().strip(".,:;")
    if not t or _MEASURE.search(t.upper()):
        return False
    if abbreviations_active():
        # drafting room tags: a room term with a number ("BDRM.#1", "BA-3", "KIT-2", "STAIR#1", "W/D")
        m = re.fullmatch(r"([A-Za-z][A-Za-z./]*?)[.\s]*(?:[#-]\s*\d{1,2})?", t)
        if m and room_name(m.group(1)):
            return True
    if re.fullmatch(r"\d{1,2}", t):
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z'./-]*", t):
        letters = sum(c.isalpha() for c in t)
        return letters >= 3 or is_short_abbreviation(t)
    return False


def _contained_vertically(a: OCRBox, b: OCRBox) -> bool:
    """One word's box lies within the other's vertical extent (centres aligned): OCR sometimes
    returns a word box far too short for its letters (a 13 px word boxed 5 px tall), and the
    height test alone would then split one printed label ("LIVING ROOM") in two."""
    lo, hi = sorted((a, b), key=lambda w: w.height)
    inside = hi.y - 1 <= lo.y and lo.y + lo.height <= hi.y + hi.height + 1
    return inside and abs((a.y + a.height / 2) - (b.y + b.height / 2)) <= 0.25 * hi.height


def _same_line(a: OCRBox, b: OCRBox) -> bool:
    if a.rotation != b.rotation:          # words of one line are read in one orientation
        return False
    ha, hb = a.height, b.height
    contained = _contained_vertically(a, b)
    if not (0.6 <= ha / max(1, hb) <= 1.65 or contained):
        return False
    if abs((a.y + ha / 2) - (b.y + hb / 2)) > 0.5 * max(ha, hb):
        return False
    gap = max(a.x, b.x) - min(a.x + a.width, b.x + b.width)
    if not 0.6 <= ha / max(1, hb) <= 1.65:
        # joined only on vertical containment: two different words side by side (not two
        # overlapping readings of one word), both room vocabulary ('LIVING' + 'ROOM'), so a junk
        # fragment with a stray box does not join a name
        return -1 <= gap <= 0.6 * max(ha, hb) and is_room_word(a.text) and is_room_word(b.text)
    return gap <= 1.2 * max(ha, hb)


def _lines(words: list[OCRBox]) -> list[list[OCRBox]]:
    words = sorted(words, key=lambda w: (w.y, w.x))
    parent = list(range(len(words)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if _same_line(words[i], words[j]):
                parent[find(i)] = find(j)
    groups: dict[int, list[OCRBox]] = {}
    for i, w in enumerate(words):
        groups.setdefault(find(i), []).append(w)
    return [sorted(g, key=lambda w: w.x) for g in groups.values()]


def _box(line: list[OCRBox]) -> tuple[int, int, int, int]:
    x0 = min(w.x for w in line); y0 = min(w.y for w in line)
    x1 = max(w.x + w.width for w in line); y1 = max(w.y + w.height for w in line)
    return x0, y0, x1, y1


def _stack(lines: list[list[OCRBox]]) -> list[list[list[OCRBox]]]:
    """Group lines that are stacked parts of one label."""
    order = sorted(range(len(lines)), key=lambda i: _box(lines[i])[1])
    used = set()
    stacks = []
    for i in order:
        if i in used:
            continue
        stack = [i]
        used.add(i)
        while True:
            x0, y0, x1, y1 = _box(lines[stack[-1]])
            h = y1 - y0
            cx = (x0 + x1) / 2
            nxt = None
            for j in order:
                if j in used:
                    continue
                a0, b0, a1, b1 = _box(lines[j])
                hj = b1 - b0
                if not (0 <= b0 - y1 <= 0.8 * max(h, hj)):
                    continue
                if not 0.6 <= hj / max(1, h) <= 1.65:
                    continue
                if lines[j][0].rotation != lines[stack[-1]][0].rotation:
                    continue
                if abs((a0 + a1) / 2 - cx) > 0.35 * max(x1 - x0, a1 - a0):
                    continue
                nxt = j
                break
            if nxt is None:
                break
            stack.append(nxt)
            used.add(nxt)
        stacks.append([lines[k] for k in stack])
    return stacks


def build_room_labels(boxes) -> list[OCRTextGroup]:
    """Room labels (possibly multi-word / multi-line) from accepted OCR words."""
    words = [b for b in boxes if b.kind != "DIMENSION" and _is_name_token(b.text)]
    lines = _lines(words)
    # Only lines made of name tokens can stack; a stack must contain a room term somewhere.
    # Number-only lines (areas, codes) never stack onto a name: room numbers are on the name's line.
    lines = [line for line in lines if any(any(c.isalpha() for c in w.text) for w in line)]
    labels = []
    for stack in _stack(lines):
        tokens = [w for line in stack for w in line]
        text = normalize_ocr_text(" ".join(" ".join(w.text for w in line) for line in stack))
        name = room_name(text)
        if name is None:
            parts = normalize(text).split()
            has_abbrev_number = any(is_short_abbreviation(p) for p in parts) and any(p.isdigit() for p in parts)
            if not has_abbrev_number:
                # a stack that is not a room label may still contain one on a single line
                for line in stack:
                    line_text = normalize_ocr_text(" ".join(w.text for w in line))
                    if room_name(line_text):
                        labels.append(OCRTextGroup(text=line_text, boxes=tuple(line),
                                                   confidence=round(sum(w.confidence for w in line) / len(line), 1)))
                continue
        labels.append(OCRTextGroup(text=text, boxes=tuple(tokens),
                                   confidence=round(sum(w.confidence for w in tokens) / len(tokens), 1)))
    labels.sort(key=lambda g: (g.y, g.x))
    return labels


__all__ = ["build_room_labels"]
