"""Vector drawing of a PDF page: every stroked or filled path with its pen, in the pixel frame of
the analysis render (experimental; used by engine.semantic).

The page is converted to SVG by poppler (`pdftocairo -svg`, media box frame) and mapped onto the
crop box that ingest.pdf renders. Glyphs of the PDF text layer
are definitions (<use>) and are skipped - the text layer is read separately (ingest.pdf_text).
Text that CAD programs export as strokes (SHX fonts) stays here as ordinary paths.
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_TOKENS = re.compile(r"[MLCZmlcz]|-?\d*\.?\d+(?:e-?\d+)?")
_NUM = re.compile(r"-?\d*\.?\d+(?:e-?\d+)?")


@dataclass
class VPath:
    kind: str                  # stroke | fill
    width: float               # pen width in px (0 for fills)
    color: str
    segs: list = field(default_factory=list)   # ("L", p0, p1) | ("C", p0, p3, [c1, c2, p3]) in px

    @property
    def pen(self) -> tuple:
        return (self.kind, round(self.width, 2), self.color)

    def length(self) -> float:
        return sum(math.dist(s[1], s[2]) for s in self.segs)


def _parse_d(d: str) -> list:
    toks = _TOKENS.findall(d)
    segs, cur, start, i, cmd = [], None, None, 0, None
    while i < len(toks):
        t = toks[i]
        if t in "MLCZmlcz":
            cmd = t
            i += 1
            if cmd in "Zz" and cur is not None and start is not None:
                if cur != start:
                    segs.append(("L", cur, start))
                cur = start
            continue
        if cmd == "M":
            cur = start = (float(toks[i]), float(toks[i + 1]))
            i += 2
            cmd = "L"
        elif cmd == "L":
            p = (float(toks[i]), float(toks[i + 1]))
            segs.append(("L", cur, p))
            cur = p
            i += 2
        elif cmd == "C":
            pts = [(float(toks[i + k]), float(toks[i + k + 1])) for k in (0, 2, 4)]
            segs.append(("C", cur, pts[2], pts))
            cur = pts[2]
            i += 6
        else:
            i += 1
    return segs


def _matrix(tr: str) -> np.ndarray:
    m = np.eye(3)
    for kind, args in re.findall(r"(matrix|translate|scale)\(([^)]*)\)", tr or ""):
        a = [float(x) for x in _NUM.findall(args)]
        if kind == "matrix":
            t = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1]])
        elif kind == "translate":
            t = np.array([[1, 0, a[0]], [0, 1, a[1] if len(a) > 1 else 0], [0, 0, 1]])
        else:
            t = np.diag([a[0], a[1] if len(a) > 1 else a[0], 1])
        m = m @ t
    return m


def parse_svg(svg_text: str, image_shape, crop=None) -> list[VPath]:
    """Paths of a cairo SVG (media box frame), mapped onto an image of `image_shape` that covers
    `crop` = (left, top, width, height) of the page in points, measured from the media box's
    top-left corner (None: the whole SVG page)."""
    size = re.search(r'<svg[^>]*width="([\d.]+)(?:pt)?"[^>]*height="([\d.]+)', svg_text)
    if not size:
        return []
    pw, ph = float(size.group(1)), float(size.group(2))
    left, top, cw, ch = crop if crop else (0.0, 0.0, pw, ph)
    h, w = image_shape[:2]
    page = np.array([[w / cw, 0, -left * w / cw], [0, h / ch, -top * h / ch], [0, 0, 1.0]])
    body = re.sub(r"<defs>.*?</defs>", "", svg_text, flags=re.S)
    stack = [page]
    out: list[VPath] = []
    for tag in re.finditer(r"<(/?)(g|path)\b([^>]*?)(/?)>", body):
        close, name, attrs, selfclose = tag.groups()
        if name == "g":
            if close:
                if len(stack) > 1:
                    stack.pop()
            else:
                tr = re.search(r'transform="([^"]*)"', attrs)
                stack.append(stack[-1] @ _matrix(tr.group(1) if tr else ""))
                if selfclose and len(stack) > 1:
                    stack.pop()
            continue
        d = re.search(r'\sd="([^"]*)"', attrs)
        if not d:
            continue
        tr = re.search(r'transform="([^"]*)"', attrs)
        m = stack[-1] @ _matrix(tr.group(1) if tr else "")
        fill = re.search(r'fill="([^"]*)"', attrs)
        stroke = re.search(r'stroke="([^"]*)"', attrs)
        sw = re.search(r'stroke-width="([^"]*)"', attrs)
        kind = "stroke" if stroke and stroke.group(1) != "none" else "fill"
        if kind == "fill" and fill and fill.group(1) == "none":
            continue
        color = stroke.group(1) if kind == "stroke" else (fill.group(1) if fill else "#000")
        scale = math.sqrt(abs(np.linalg.det(m[:2, :2]))) or 1.0
        width = (float(sw.group(1)) if sw else 1.0) * scale if kind == "stroke" else 0.0

        def tf(p, m=m):
            v = m @ np.array([p[0], p[1], 1.0])
            return (float(v[0]), float(v[1]))
        segs = []
        for s in _parse_d(d.group(1)):
            if s[0] == "L":
                segs.append(("L", tf(s[1]), tf(s[2])))
            else:
                segs.append(("C", tf(s[1]), tf(s[2]), [tf(q) for q in s[3]]))
        if segs:
            out.append(VPath(kind, width, color, segs))
    return out


def page_paths(pdf_path: Path, page_number: int, image_shape, timeout: float = 120.0) -> list[VPath]:
    """All vector paths of one page in the frame of its analysis render (crop box, as
    ingest.pdf renders it). [] when poppler is missing, or for a rotated page with a crop box
    different from its media box (not mapped yet)."""
    from pypdf import PdfReader

    try:
        page = PdfReader(str(pdf_path)).pages[page_number - 1]
    except Exception:
        return []
    media, crop = page.mediabox, page.cropbox
    same = all(abs(float(a) - float(b)) < 0.5 for a, b in zip(media, crop))
    if int(page.rotation or 0) % 360:
        if not same:
            return []                     # rotated page with a crop box: not mapped yet
        box = None                        # poppler writes the rotated page: its SVG page is the render
    else:
        box = (float(crop.left) - float(media.left), float(media.top) - float(crop.top), float(crop.width), float(crop.height))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "page.svg"
        try:
            subprocess.run(["pdftocairo", "-svg", "-f", str(page_number), "-l", str(page_number),
                            str(pdf_path), str(out)], check=True, capture_output=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError):
            return []
        return parse_svg(out.read_text(errors="ignore"), image_shape, box)
