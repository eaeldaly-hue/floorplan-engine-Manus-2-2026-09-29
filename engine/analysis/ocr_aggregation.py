"""Evidence aggregation across OCR passes.

The OCR stage reads the plan in several configurations (preprocessing variant x rotation x
page-segmentation mode). Keeping only the configuration with the best global score throws
away everything the other passes read: a rotated pass full of confident noise can win over
upright passes that read every room label. This module keeps all passes:

1. every word observation, mapped to original image coordinates, is kept with its text,
   box, confidence, variant, rotation and PSM;
2. obvious noise is dropped (empty, one character, repeated characters);
3. observations of the same place on the page are grouped into text regions (box overlap
   or near-coincident centres with similar size);
4. inside a region, observations are grouped by normalized reading; each reading is scored
   from its support — how many passes produced it and how confidently — with a small prior
   for readings that are recognisable vocabulary (room terms, dimension patterns);
5. the best reading represents the region, and the region is accepted on its own evidence
   (strong single read, or repeated reads), not because its pass won globally.

Each region keeps its evidence (supporting passes, rotations, rival readings, decision and
reason), so the outcome can be inspected.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# Acceptance thresholds for a region's chosen reading.
STRONG_CONFIDENCE = 60.0      # one pass reading the word this confidently is enough
SUPPORTED_CONFIDENCE = 40.0   # ... or at least two passes agreeing with this mean confidence
VOCAB_CONFIDENCE = 25.0       # a vocabulary term (room word / dimension) read once at the OCR floor
VOCAB_PRIOR = 1.25            # score multiplier for vocabulary readings when choosing among rivals


@dataclass
class Observation:
    text: str
    norm: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    variant: str
    rotation: int        # degrees
    psm: int
    kind: str            # ROOM_LABEL / DIMENSION / OTHER (from the word alone)

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def pass_key(self) -> tuple:
        return (self.variant, self.rotation, self.psm)


@dataclass
class Region:
    observations: list[Observation]
    text: str = ""
    norm: str = ""
    kind: str = "OTHER"
    confidence: float = 0.0
    support: int = 0                 # distinct passes producing the chosen reading
    box: tuple[int, int, int, int] = (0, 0, 0, 0)
    rotations: tuple[int, ...] = ()
    rivals: list[tuple[str, float, int]] = field(default_factory=list)   # (reading, score, support)
    accepted: bool = False
    reason: str = ""
    best: Observation | None = None


def normalize_reading(text: str) -> str:
    t = text.upper().replace("×", "X").replace("′", "'").replace("″", '"')
    t = re.sub(r"[^A-Z0-9'\"X.\- /]+", " ", t)
    t = t.strip(" .,:;-")
    return " ".join(t.split())


def is_noise(text: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", text)
    if len(compact) <= 1:
        return True
    if re.fullmatch(r"(.)\1{2,}", compact.lower()):
        return True
    return False


def _overlap(a: Observation, b: Observation) -> float:
    """Intersection over the smaller box."""
    w = min(a.right, b.right) - max(a.x, b.x)
    h = min(a.bottom, b.bottom) - max(a.y, b.y)
    if w <= 0 or h <= 0:
        return 0.0
    return (w * h) / max(1, min(a.width * a.height, b.width * b.height))


def _same_place(a: Observation, b: Observation) -> bool:
    """Two observations of the same piece of text: boxes of similar height and width that
    strongly overlap (or have near-identical centres). Neighbouring words on a line, or a
    word and a box spanning two lines, do not qualify."""
    if not 0.6 <= a.height / max(1, b.height) <= 1.65:
        return False
    if not 0.5 <= a.width / max(1, b.width) <= 2.0:
        return False
    if _overlap(a, b) >= 0.6:
        return True
    (ax, ay), (bx, by) = a.center, b.center
    return math.hypot(ax - bx, ay - by) <= 0.25 * max(min(a.height, a.width), min(b.height, b.width), 4)


class _BoxGrid:
    """Uniform grid over boxes: which stored items can touch a query box. Used only to skip
    comparisons that cannot succeed; every candidate is still tested with the exact rule, and
    the lowest index wins, so the outcome equals a scan of all items in order."""

    CELL = 64

    def __init__(self):
        self.cells: dict[tuple[int, int], list[int]] = {}

    def _keys(self, x0: float, y0: float, x1: float, y1: float):
        c = self.CELL
        for i in range(math.floor(x0 / c), math.floor(x1 / c) + 1):
            for j in range(math.floor(y0 / c), math.floor(y1 / c) + 1):
                yield i, j

    def add(self, index: int, box: tuple[float, float, float, float]) -> None:
        for key in self._keys(*box):
            self.cells.setdefault(key, []).append(index)

    def candidates(self, box: tuple[float, float, float, float]) -> list[int]:
        found: set[int] = set()
        for key in self._keys(*box):
            found.update(self.cells.get(key, ()))
        return sorted(found)


def _reach_box(o: Observation) -> tuple[float, float, float, float]:
    """The box grown by the centre-distance allowance of _same_place. Two observations that
    match (overlapping boxes, or centres within max of both allowances) always have
    intersecting reach boxes."""
    r = 0.25 * max(min(o.height, o.width), 4)
    return (o.x - r, o.y - r, o.right + r, o.bottom + r)


def group_regions(observations: list[Observation]) -> list[Region]:
    """Seeded grouping: observations are taken strongest first; each joins the first region
    whose seed (its strongest observation) it matches, else starts a new region. Matching
    only against seeds prevents chaining two different words through an intermediate box."""
    seeds: list[Observation] = []
    members: list[list[Observation]] = []
    grid = _BoxGrid()
    for o in sorted(observations, key=lambda o: -o.confidence):
        reach = _reach_box(o)
        for k in grid.candidates(reach):                       # in seed order: first match wins
            if _same_place(o, seeds[k]):
                members[k].append(o)
                break
        else:
            grid.add(len(seeds), reach)
            seeds.append(o)
            members.append([o])
    return [Region(observations=obs) for obs in members]


def _one_substitution(a: str, b: str) -> bool:
    return len(a) == len(b) and len(a) >= 3 and sum(x != y for x, y in zip(a, b)) == 1


def resolve(region: Region, is_vocabulary) -> Region:
    """Choose the region's reading from cross-pass support and decide acceptance."""
    by_reading: dict[str, list[Observation]] = {}
    for o in region.observations:
        by_reading.setdefault(o.norm, []).append(o)
    scored = []
    for reading, obs in by_reading.items():
        best_per_pass: dict[tuple, float] = {}
        for o in obs:
            best_per_pass[o.pass_key] = max(best_per_pass.get(o.pass_key, 0.0), o.confidence)
        support = len(best_per_pass)
        score = sum(best_per_pass.values()) / 100.0
        vocab = is_vocabulary(reading)
        if vocab:
            score *= VOCAB_PRIOR
        scored.append((score, support, reading, obs, vocab))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    # Lexicon-constrained correction: when the best-supported reading is not a known term but
    # some pass directly read a known term at the same place, strongly, and the two differ by a
    # single character substitution (BETH / BATH), the known term is the word on the drawing.
    if not scored[0][4]:
        for cand in scored[1:]:
            c_obs, c_vocab = cand[3], cand[4]
            if (c_vocab and max(o.confidence for o in c_obs) >= STRONG_CONFIDENCE
                    and _one_substitution(cand[2], scored[0][2])):
                region.reason_prefix = f"vocabulary reading '{cand[2]}' preferred over '{scored[0][2]}' (one-letter misread); "
                scored.remove(cand)
                scored.insert(0, cand)
                break
    score, support, reading, obs, vocab = scored[0]
    best = max(obs, key=lambda o: o.confidence)
    xs = sorted(o.x for o in obs); ys = sorted(o.y for o in obs)
    rs = sorted(o.right for o in obs); bs = sorted(o.bottom for o in obs)
    mid = len(obs) // 2
    region.text = best.text
    region.norm = reading
    region.kind = best.kind
    region.confidence = best.confidence
    region.support = support
    region.box = (xs[mid], ys[mid], max(1, rs[mid] - xs[mid]), max(1, bs[mid] - ys[mid]))
    region.rotations = tuple(sorted({o.rotation for o in obs}))
    region.rivals = [(r, round(s, 2), sup) for s, sup, r, _, _ in scored[1:6]]
    region.best = best
    mean = sum(o.confidence for o in obs) / len(obs)
    prefix = getattr(region, "reason_prefix", "")
    if best.confidence >= STRONG_CONFIDENCE:
        region.accepted, region.reason = True, prefix + f"strong read ({best.confidence:.0f}%)"
    elif support >= 2 and mean >= SUPPORTED_CONFIDENCE:
        region.accepted, region.reason = True, prefix + f"read by {support} passes (mean {mean:.0f}%)"
    elif vocab and best.confidence >= VOCAB_CONFIDENCE:
        region.accepted, region.reason = True, prefix + f"vocabulary term read at {best.confidence:.0f}%"
    else:
        region.accepted, region.reason = False, (
            f"weak: {support} pass(es), best {best.confidence:.0f}%, not vocabulary")
    return region


def _box_overlap(a: tuple, b: tuple) -> float:
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return (w * h) / max(1, min(a[2] * a[3], b[2] * b[3]))


def _merge_regions(regions: list[Region], is_vocabulary) -> list[Region]:
    """Second stage: regions whose representative boxes (medians over their observations)
    cover the same place are one text region read slightly differently by different passes."""
    order = sorted(regions, key=lambda r: -(r.support * r.confidence))
    kept: list[Region] = []
    grid = _BoxGrid()                     # both merge rules need overlapping boxes

    def extent(box):
        return (box[0], box[1], box[0] + box[2], box[1] + box[3])

    for r in order:
        for index in grid.candidates(extent(r.box)):          # in kept order: first match wins
            k = kept[index]
            ratio = max(r.box[3], k.box[3]) / max(1, min(r.box[3], k.box[3]))
            overlap = _box_overlap(r.box, k.box)
            same_word = r.norm == k.norm or _one_substitution(r.norm, k.norm)
            if ((overlap >= 0.5 and (same_word or ratio <= 1.65))
                    or (overlap >= 0.85 and ratio <= 2.2)):          # a word inside a two-line box
                k.observations.extend(r.observations)
                resolve(k, is_vocabulary)
                grid.add(index, extent(k.box))                       # its median box may have moved
                break
        else:
            grid.add(len(kept), extent(r.box))
            kept.append(r)
    return kept


def aggregate(observations: list[Observation], is_vocabulary) -> list[Region]:
    clean = [o for o in observations if not is_noise(o.text) and o.norm]
    regions = [resolve(r, is_vocabulary) for r in group_regions(clean)]
    regions = _merge_regions(regions, is_vocabulary)
    regions.sort(key=lambda r: (r.box[1], r.box[0]))
    return regions


__all__ = ["Observation", "Region", "aggregate", "group_regions", "normalize_reading", "is_noise"]


def text_lines_enabled() -> bool:
    import os
    return os.environ.get("FLOORPLAN_TEXT_LINES", "on").strip().lower() not in ("off", "0", "no", "false")


def text_line_mosaics(image) -> list:
    import cv2
    from .text_lines import build_mosaics, find_text_lines

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return build_mosaics(gray, find_text_lines(gray))


def to_page(*args):
    from .text_lines import to_page as _to_page
    return _to_page(*args)


def extract_aggregated(image, *, min_confidence: float = 25.0, psms=(11, 6), max_side: int = 5000):
    """Run every OCR pass once, aggregate the evidence, and return
    (accepted OCRBox list, all regions, per-pass summary). No pass is re-run."""
    from . import ocr as _ocr
    from .ocr_preprocessing import prepare_ocr_variants
    from .room_lexicon import is_room_word

    variants, scale_x, scale_y = prepare_ocr_variants(image, max_side=max_side)
    # The passes are independent reads: run them concurrently, then consume the results in the
    # fixed pass order (variant, rotation, psm), so the evidence is identical to reading them
    # one after another.
    jobs = [(variant, rotation, psm) for variant in variants for rotation in range(4) for psm in psms]
    # Scale-normalised text lines (engine.analysis.text_lines): every detected line read at its
    # own optimal glyph height, packed into mosaics; read in the same pool as the page passes.
    mosaics = text_line_mosaics(image) if text_lines_enabled() else []
    line_jobs = [(m, psm) for m in range(len(mosaics)) for psm in psms]

    def read(job):
        if job[0] == "lines":
            m, psm = job[1]
            return _ocr._read_words(mosaics[m][0], psm=psm, variant_name="text-lines", rotation=0,
                                    min_confidence=min_confidence)
        variant, rotation, psm = job
        oriented = _ocr.rotate_image(variant.image, rotation)
        return _ocr._read_words(oriented, psm=psm, variant_name=variant.name, rotation=rotation,
                                min_confidence=min_confidence)

    results = _ocr.run_ocr_tasks(read, jobs + [("lines", j) for j in line_jobs])
    line_results = results[len(jobs):]
    results = results[:len(jobs)]
    observations: list[Observation] = []
    passes = []
    for (m, psm), boxes in zip(line_jobs, line_results):
        mosaic, placements = mosaics[m]
        kept = 0
        for b in boxes:
            hit = to_page(b.x, b.y, b.width, b.height, placements, image.shape)
            if hit is None:
                continue
            x, y, w, h, ln = hit
            kept += 1
            observations.append(Observation(text=b.text, norm=normalize_reading(b.text), x=x, y=y, width=w,
                                            height=h, confidence=b.confidence, variant="text-lines",
                                            rotation=90 if ln.vertical else 0, psm=psm, kind=b.kind))
        passes.append({"variant": "text-lines", "rotation": 0, "psm": psm, "words": kept, "mosaic": m,
                       "score": round(_ocr._configuration_score(boxes, min_confidence=min_confidence), 1)})
    for (variant, rotation, psm), boxes in zip(jobs, results):
        vh, vw = variant.image.shape[:2]
        rh, rw = (vw, vh) if rotation % 2 else (vh, vw)      # shape of the rotated image
        passes.append({"variant": variant.name, "rotation": rotation * 90, "psm": psm, "words": len(boxes),
                       "score": round(_ocr._configuration_score(boxes, min_confidence=min_confidence), 1)})
        for b in boxes:
            x, y, w, h = _ocr.map_box_to_original(b.x, b.y, b.width, b.height, rotation, rw, rh, scale_x, scale_y)
            if w <= 0 or h <= 0:
                continue
            observations.append(Observation(text=b.text, norm=normalize_reading(b.text), x=x, y=y, width=w,
                                            height=h, confidence=b.confidence, variant=variant.name,
                                            rotation=rotation * 90, psm=psm, kind=b.kind))

    def vocabulary(reading: str) -> bool:
        return is_room_word(reading) or _ocr.classify_text(reading) == "DIMENSION"

    regions = aggregate(observations, vocabulary)
    for r in regions:
        # text seen only by the text-line reader is accepted only when it is domain text (a room
        # term, a dimension, a room number): the new evidence source adds meaning, not noise
        if r.accepted and all(o.variant == "text-lines" for o in r.observations) \
                and not (vocabulary(r.norm) or r.norm.replace(" ", "").isdigit()):
            r.accepted, r.reason = False, "text-line reading only, not vocabulary"
    accepted = []
    for r in regions:
        if not r.accepted:
            continue
        b = r.best
        accepted.append(_ocr.OCRBox(text=r.text, x=r.box[0], y=r.box[1], width=r.box[2], height=r.box[3],
                                    confidence=r.confidence, source=f"tesseract-psm-{b.psm}", variant=b.variant,
                                    rotation=b.rotation, psm=b.psm, kind=_ocr.classify_text(r.text)))
    return accepted, regions, passes
