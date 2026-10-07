"""How a hold-out plan reaches the engine — one definition, used by the benchmark runner and by
the labelling tool, so labels are made in exactly the pixel frame the engine analyses.

* PDF pages go through the production PDF path (ingest.pdf): rendered at the page's analysis
  DPI (150, lowered for large sheets to stay within 24 MP), and the page's text layer is passed
  to the analyzer as text evidence, as the Workbench does.
* Images above 24 MP (the PDF policy's limit; the upload path rejects > 25 MP) are reduced with
  INTER_AREA by the factor recorded in the fixture (`analysis_scale`); smaller images are used
  as they are. The original files are never modified.

A fixture entry names the file relative to the hold-out directory and may give `page` (PDF) and
`analysis_scale` (image). `analysis_frame` records the resulting size for the freeze.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

MAX_PIXELS = 24_000_000


def image_scale(width: int, height: int) -> float:
    """Reduction applied to an image above MAX_PIXELS (rounded so it can be written down)."""
    if width * height <= MAX_PIXELS:
        return 1.0
    scale = math.floor(math.sqrt(MAX_PIXELS / (width * height)) * 10_000) / 10_000
    while round(width * scale) * round(height * scale) > MAX_PIXELS:   # rounding can overshoot by a row
        scale = round(scale - 0.0001, 4)
    return scale


def load(path: Path, page: int | None = None, analysis_scale: float | None = None):
    """(image BGR, text evidence boxes, info) for one hold-out plan."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from ingest.pdf import inspect_pdf, render_page
        from ingest.pdf_text import read_page_text, text_boxes

        number = page or 1
        info = inspect_pdf(path).pages[number - 1]
        image, meta = render_page(path, info)
        page_text = read_page_text(path, number, info.kind)
        evidence = text_boxes(page_text, info.width_pt, info.height_pt, image.shape)
        return image, evidence, {"kind": "pdf", "page": number, "dpi": meta["dpi"], "size": [meta["width"], meta["height"]],
                                 "page_kind": info.kind, "text_layer": page_text.summary()}
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{path.name}: not a readable image")
    h, w = image.shape[:2]
    scale = analysis_scale if analysis_scale is not None else image_scale(w, h)
    if scale != 1.0:
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return image, (), {"kind": "image", "original_size": [w, h], "analysis_scale": scale,
                       "size": [int(image.shape[1]), int(image.shape[0])]}


def load_spec(name: str, spec: dict, plans_dir: Path):
    return load(Path(plans_dir) / spec.get("file", name), spec.get("page"), spec.get("analysis_scale"))


def frame_digest(image: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
