"""Which OCR passes earn their cost? Record every pass once, re-aggregate subsets offline.

    python -m benchmark.ocr_passes record      # one OCR run per scored input, all observations kept
    python -m benchmark.ocr_passes evaluate    # printed room names recovered and CPU per pass subset

`record` runs the production OCR (engine.analysis.ocr_aggregation.extract_aggregated) on every input
with ground-truth room names (fixtures/real_plans.json dev images, fixtures/pdf_plans.json pages),
keeping each pass's observations and Tesseract time. `evaluate` rebuilds the accepted text from a
subset of the passes exactly as production aggregates it, attaches room labels, and counts the
printed names recovered (same canonical name, nearest unmatched label within 8 % of the page
diagonal of the room's point). The proxy is for comparing subsets; a chosen configuration is then
confirmed with the full benchmarks (real_plans names, pdf_plans).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT.parent / "Test Cases"
OUT = ROOT / "output/perf/ocr_passes"


def _inputs():
    real = json.loads((ROOT / "benchmark/fixtures/real_plans.json").read_text())["plans"]
    pdf = json.loads((ROOT / "benchmark/fixtures/pdf_plans.json").read_text())["plans"]
    for name, spec in list(real.items()) + list(pdf.items()):
        rooms = [(r["printed"], r["point"]) for r in spec.get("rooms", []) if r.get("printed")]
        if rooms:
            yield name, rooms


def _load(case: str):
    name, _, page = case.partition(":")
    if page:
        from ingest.pdf import inspect_pdf, render_page
        image, _ = render_page(REAL / name, inspect_pdf(REAL / name).pages[int(page) - 1])
    else:
        from app import _decode_upload
        image, _ = _decode_upload((REAL / name).read_bytes(), name)
    return image


def record() -> None:
    import engine.analysis.ocr as O
    import engine.analysis.ocr_aggregation as AG
    OUT.mkdir(parents=True, exist_ok=True)
    for case, _rooms in _inputs():
        path = OUT / (case.replace(":", "_p") + ".pkl")
        if path.exists():
            continue
        if not (REAL / case.partition(":")[0]).exists():
            continue
        image = _load(case)
        times: dict = {}
        captured: dict = {}
        orig_read, orig_agg = O._read_words, AG.aggregate

        def timed(img, **k):
            t = time.perf_counter()
            r = orig_read(img, **k)
            key = (k["variant_name"], k["psm"])
            times[key] = times.get(key, 0.0) + time.perf_counter() - t
            return r

        def grab(observations, vocabulary):
            captured["obs"] = list(observations)
            return orig_agg(observations, vocabulary)
        O._read_words, AG.aggregate = timed, grab
        try:
            O.clear_ocr_memo()
            t = time.perf_counter()
            AG.extract_aggregated(image)
            wall = time.perf_counter() - t
        finally:
            O._read_words, AG.aggregate = orig_read, orig_agg
        pickle.dump({"obs": captured["obs"], "times": times, "wall": wall, "shape": image.shape}, path.open("wb"))
        print(f"{case:<20} {len(captured['obs']):>6} observations  {wall:6.1f} s", flush=True)


def accepted_text(observations, keep):
    """Production aggregation (engine.analysis.ocr_aggregation.extract_aggregated) over a subset."""
    from engine.analysis import ocr as O
    from engine.analysis.ocr_aggregation import aggregate
    from engine.analysis.room_lexicon import is_room_word

    obs = [o for o in observations if (o.variant, o.psm) in keep]

    def vocabulary(reading: str) -> bool:
        return is_room_word(reading) or O.classify_text(reading) == "DIMENSION"
    regions = aggregate(obs, vocabulary)
    out = []
    for r in regions:
        if r.accepted and all(o.variant == "text-lines" for o in r.observations) \
                and not (vocabulary(r.norm) or r.norm.replace(" ", "").isdigit()):
            continue
        if r.accepted:
            b = r.best
            out.append(O.OCRBox(text=r.text, x=r.box[0], y=r.box[1], width=r.box[2], height=r.box[3],
                                confidence=r.confidence, source=f"tesseract-psm-{b.psm}", variant=b.variant,
                                rotation=b.rotation, psm=b.psm, kind=O.classify_text(r.text)))
    return out


def names_found(boxes, rooms, shape) -> int:
    from engine.analysis.room_labels import build_room_labels
    from engine.analyzer import _room_name
    labels = [(_room_name(g.text), (g.x + g.width / 2, g.y + g.height / 2)) for g in build_room_labels(boxes)]
    labels = [(n, c) for n, c in labels if n]
    reach = 0.08 * math.hypot(shape[0], shape[1])
    used, found = set(), 0
    for printed, point in rooms:
        want = _room_name(printed)
        best = None
        for i, (n, c) in enumerate(labels):
            if i in used or n != want:
                continue
            d = math.dist(c, point)
            if d <= reach and (best is None or d < best[0]):
                best = (d, i)
        if best:
            used.add(best[1])
            found += 1
    return found


def evaluate() -> None:
    data = {}
    for case, rooms in _inputs():
        path = OUT / (case.replace(":", "_p") + ".pkl")
        if path.exists():
            data[case] = (pickle.load(path.open("rb")), rooms)
    passes = sorted({k for d, _ in data.values() for k in d["times"]})
    page = [p for p in passes if p[0] != "text-lines"]
    lines = [p for p in passes if p[0] == "text-lines"]
    subsets = {"all (production)": set(passes)}
    for p in passes:
        subsets[f"without {p[0]} psm{p[1]}"] = set(passes) - {p}
    for psm in (11, 6):
        subsets[f"page psm{psm} only + lines"] = {p for p in page if p[1] == psm} | set(lines)
    for pair in itertools.combinations(sorted({p[0] for p in page}), 2):
        subsets[f"page {'+'.join(pair)} + lines"] = {p for p in page if p[0] in pair} | set(lines)
    subsets["lines only"] = set(lines)
    printed = sum(len(r) for _, r in data.values())
    print(f"{len(data)} inputs, {printed} printed room names; CPU = summed Tesseract time of the passes")
    rows = []
    for label, keep in subsets.items():
        found = sum(names_found(accepted_text(d["obs"], keep), rooms, d["shape"]) for d, rooms in data.values())
        cpu = sum(t for d, _ in data.values() for k, t in d["times"].items() if k in keep)
        rows.append((label, found, cpu))
        print(f"{label:<44} names {found:>4} / {printed}   CPU {cpu:7.1f} s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["record", "evaluate"])
    args = ap.parse_args()
    record() if args.what == "record" else evaluate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
