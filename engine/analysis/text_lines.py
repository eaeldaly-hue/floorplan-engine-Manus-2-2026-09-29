"""Scale-normalised text reading: find text lines first, read each at its own optimal scale.

Full-page OCR reads the whole drawing at one scale (engine.analysis.ocr_preprocessing upscales a
page by at most 2x and downscales large sheets to 5000 px). Text on floor plans varies from 4 px
(small web images) to 40 px (large sheets), so whole-page passes miss the smallest names entirely.

1. Text lines are found geometrically on the ink: glyph-sized connected components (not long
   strokes, not large blobs) grouped into horizontal or vertical runs of similar height.
2. Each line is cropped, turned upright and rescaled so its glyphs are TARGET_HEIGHT px tall.
3. The normalised crops are packed into mosaics, one Tesseract call per mosaic, and every word is
   mapped back to page coordinates.

The readings are additional evidence for engine.analysis.ocr_aggregation (variant "text-lines"),
never a replacement for the full-page passes.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

TARGET_HEIGHT = 30          # px: glyph height Tesseract reads best
MOSAIC_WIDTH = 2400         # px
MOSAIC_MAX_HEIGHT = 6000    # px per Tesseract call
GAP = 24                    # px of paper between packed lines


@dataclass
class TextLine:
    x: int
    y: int
    w: int
    h: int
    vertical: bool           # reads bottom-to-top (rotated 90 degrees counter-clockwise)
    glyph_h: float


def _components(gray: np.ndarray, local: bool = False):
    """Global (Otsu) ink, or global OR local contrast: the global threshold is set by the darkest
    drawing (walls) and drops light-grey text; the local one keeps it but can fuse small glyphs."""
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if local:
        ink = cv2.bitwise_or(ink, cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 12))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    return ink, stats[1:]


def find_text_lines(gray: np.ndarray, min_h: int = 3) -> list[TextLine]:
    """Text lines under both binarisations (distinct lines only)."""
    out: list[TextLine] = []
    for local in (False, True):
        for ln in _find_lines(gray, min_h, local):
            if not any(_iou(ln, o) > 0.6 for o in out):
                out.append(ln)
    return out


def _iou(a: TextLine, b: TextLine) -> float:
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    return inter / float(a.w * a.h + b.w * b.h - inter) if inter else 0.0


def _find_lines(gray: np.ndarray, min_h: int, local: bool) -> list[TextLine]:
    """Runs of glyph-like components of similar height on one baseline (horizontal, and vertical
    runs read sideways). Walls, long lines, hatching and symbols rarely form such runs. Pairs are
    found through a grid index and joined with union-find (near-linear on large sheets)."""
    H, W = gray.shape[:2]
    _, st = _components(gray, local)
    max_h = max(12, int(0.05 * min(H, W)))
    out: list[TextLine] = []
    for vertical in (False, True):
        boxes = []
        for x, y, w, h, area in st:
            gw, gh = (h, w) if vertical else (w, h)          # glyph width / height in reading frame
            if not (min_h <= gh <= max_h) or area < 4:
                continue
            fill = area / float(w * h)
            if fill > 0.9 and gw > 0.6 * gh:                   # solid block (fill, column), not a glyph
                continue
            if gw > 2.5 * gh and not (gw <= 14 * gh and 0.15 <= fill <= 0.75):
                continue                                       # long stroke; a fused word is allowed
            boxes.append((int(x), int(y), int(w), int(h), int(gh)))
        if not boxes:
            continue
        cell = max(8, int(np.median([b[4] for b in boxes]) * 2))
        grid: dict = {}
        for i, (x, y, w, h, gh) in enumerate(boxes):
            for gx in range(x // cell, (x + w) // cell + 1):
                for gy in range(y // cell, (y + h) // cell + 1):
                    grid.setdefault((gx, gy), []).append(i)
        used = [False] * len(boxes)
        # seeds in reading order across the page; each line grows as a box (new glyphs are tested
        # against the line so far, which keeps stacked lines apart)
        order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]) if vertical else (boxes[i][0], boxes[i][1]))
        for i in order:
            if used[i]:
                continue
            x, y, w, h, gh = boxes[i]
            used[i] = True
            members = [i]
            x0, y0, x1, y1 = x, y, x + w, y + h
            grew = True
            while grew:
                grew = False
                reach = int(1.4 * gh * 1.7) + 1
                cand = set()
                for gx in range((x0 - reach) // cell, (x1 + reach) // cell + 1):
                    for gy in range((y0 - reach) // cell, (y1 + reach) // cell + 1):
                        cand.update(grid.get((gx, gy), ()))
                for j in sorted(cand, key=lambda k: (boxes[k][1], boxes[k][0]) if vertical else (boxes[k][0], boxes[k][1])):
                    if used[j]:
                        continue
                    bx, by, bw, bh, bgh = boxes[j]
                    if not 0.6 <= bgh / gh <= 1.7:
                        continue
                    if vertical:
                        overlap = min(x1, bx + bw) - max(x0, bx)
                        gap = max(by - y1, y0 - (by + bh))
                        ok = overlap >= 0.5 * min(x1 - x0, bw) and gap <= 1.4 * gh
                    else:
                        overlap = min(y1, by + bh) - max(y0, by)
                        gap = max(bx - x1, x0 - (bx + bw))
                        ok = overlap >= 0.5 * min(y1 - y0, bh) and gap <= 1.4 * gh
                    if ok:
                        used[j] = True
                        members.append(j)
                        x0, y0, x1, y1 = min(x0, bx), min(y0, by), max(x1, bx + bw), max(y1, by + bh)
                        grew = True
            hs = sorted(boxes[k][4] for k in members)
            gmed = float(hs[len(hs) // 2])
            length = (y1 - y0) if vertical else (x1 - x0)
            if (len(members) >= 2 or length >= 2.5 * gmed) and length >= 1.5 * gmed:
                out.append(TextLine(x0, y0, x1 - x0, y1 - y0, vertical, gmed))
    return out


def _normalised_crop(gray: np.ndarray, line: TextLine) -> tuple[np.ndarray, float]:
    pad = max(2, int(round(0.5 * line.glyph_h)))
    H, W = gray.shape[:2]
    x0, y0 = max(0, line.x - pad), max(0, line.y - pad)
    x1, y1 = min(W, line.x + line.w + pad), min(H, line.y + line.h + pad)
    crop = gray[y0:y1, x0:x1]
    if line.vertical:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    s = float(np.clip(TARGET_HEIGHT / max(line.glyph_h, 1.0), 0.4, 8.0))
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    return crop, s


def build_mosaics(gray: np.ndarray, lines: list[TextLine]) -> list[tuple[np.ndarray, list]]:
    """Normalised line crops packed into mosaics: [(mosaic, placements)], placement =
    (x, y, crop_w, crop_h, scale, line) in mosaic pixels."""
    tiles = []
    for ln in lines:
        crop, s = _normalised_crop(gray, ln)
        if crop.shape[1] > MOSAIC_WIDTH - 2 * GAP:
            f = (MOSAIC_WIDTH - 2 * GAP) / crop.shape[1]
            crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            s *= f
        tiles.append((crop, s, ln))
    mosaics = []
    i = 0
    while i < len(tiles):
        placements, cx, cy, row_h = [], GAP, GAP, 0
        while i < len(tiles):
            crop, s, ln = tiles[i]
            th, tw = crop.shape[:2]
            if cx + tw + GAP > MOSAIC_WIDTH:
                cx, cy, row_h = GAP, cy + row_h + GAP, 0
            if cy + th + GAP > MOSAIC_MAX_HEIGHT and placements:
                break
            placements.append((cx, cy, crop, s, ln))
            cx += tw + 2 * GAP
            row_h = max(row_h, th)
            i += 1
        height = min(MOSAIC_MAX_HEIGHT, cy + row_h + GAP)
        mosaic = np.full((height, MOSAIC_WIDTH), 255, np.uint8)
        for px, py, crop, s, ln in placements:
            mosaic[py:py + crop.shape[0], px:px + crop.shape[1]] = crop
        mosaics.append((mosaic, [(px, py, crop.shape[1], crop.shape[0], s, ln) for px, py, crop, s, ln in placements]))
    return mosaics


def to_page(wx, wy, ww, wh, placements, page_shape):
    """A word box of a mosaic -> (x, y, w, h, line) on the page, or None (between tiles)."""
    H, W = page_shape[:2]
    cxm, cym = wx + ww / 2, wy + wh / 2
    for px, py, cw, ch_, s, ln in placements:
        if px <= cxm < px + cw and py <= cym < py + ch_:
            pad = max(2, int(round(0.5 * ln.glyph_h)))
            ox, oy = max(0, ln.x - pad), max(0, ln.y - pad)
            lx, ly, lw, lh = (wx - px) / s, (wy - py) / s, ww / s, wh / s
            if ln.vertical:                # crop rotated clockwise: (u, v) -> (x = ox + v, y = oy + CH - u)
                crop_h = min(H, ln.y + ln.h + pad) - oy
                gx, gy, gw, gh = ox + ly, oy + crop_h - (lx + lw), lh, lw
            else:
                gx, gy, gw, gh = ox + lx, oy + ly, lw, lh
            return int(round(gx)), int(round(gy)), max(1, int(round(gw))), max(1, int(round(gh))), ln
    return None


def read_text_lines(gray: np.ndarray, lines: list[TextLine], read_words) -> list[tuple]:
    """Read all lines via mosaics. read_words(image) -> [(text, x, y, w, h, conf)] in that image.
    Returns [(text, x, y, w, h, conf, line)] in page coordinates."""
    out = []
    for mosaic, placements in build_mosaics(gray, lines):
        for text, wx, wy, ww, wh, conf in read_words(mosaic):
            hit = to_page(wx, wy, ww, wh, placements, gray.shape)
            if hit:
                out.append((text, *hit[:4], conf, hit[4]))
    return out
