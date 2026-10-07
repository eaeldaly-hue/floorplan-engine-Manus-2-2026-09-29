"""End-to-end Analyze timing on the canonical test cases (../Test Cases), stage by stage.

Runs exactly what the app runs for each input (image upload: decode + analyze; PDF page: render +
text layer + [cleaning] + analyze) with real OCR (no benchmark cache unless --ocr-cache) and reports
per-stage time. Stages run concurrently (OCR and structure overlap), so the stage sum can exceed the
wall time.

    python -m benchmark.perf_suite [--cases a.png,22.pdf:6] [--cleaning] [--save F] [--ocr-cache]
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = Path(os.environ.get("FLOORPLAN_REAL_PLANS", ROOT.parent / "Test Cases"))
PDF_PAGES = {"22.pdf": [5, 6, 7, 8], "3.pdf": [4], "23.pdf": [1]}

_STAGES: dict = defaultdict(float)
_LOCK = threading.Lock()


def _timed(module, name, label=None):
    fn = getattr(module, name)
    if getattr(fn, "_perf_wrapped", False):
        return

    @functools.wraps(fn)
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            with _LOCK:
                _STAGES[label or name] += time.perf_counter() - t
    wrapper._perf_wrapped = True
    setattr(module, name, wrapper)


def instrument():
    import engine.analyzer as A
    import engine.plan.adapter as PA
    import engine.semantic.layer as SL
    import engine.cleaning.cleaner as CL
    import ingest.pdf as IP
    import ingest.pdf_text as IT
    import ingest.pdf_vectors as IV

    for name, label in [("extract_ocr", "ocr"), ("analyze_structure", "structure"), ("classify_openings", "openings"),
                        ("_split_shared_spaces", "split_spaces"), ("_join_open_connections", "join_open"),
                        ("_build_room_records", "room_records"), ("_infer_anonymous_dimensions", "anon_dims"),
                        ("_draw_overlay", "overlay"), ("draw_openings_overlay", "openings_overlay"),
                        ("_topology", "topology"), ("build_room_labels", "room_labels"), ("merge_document_text", "merge_text")]:
        if hasattr(A, name):
            _timed(A, name, label)
    _timed(A.plan_adapter, "run", "plan_model")
    _timed(PA, "run", "plan_model")
    _timed(IP, "render_page", "pdf_render")
    _timed(IT, "read_page_text", "pdf_text")
    _timed(IV, "page_paths", "pdf_vectors")
    _timed(SL, "_gap_closures", "clean_gap_closures")
    _timed(SL, "_wall_test", "clean_pen_tests")
    _timed(CL, "clean", "cleaning_total")


def run_case(case: str, cleaning: bool) -> dict:
    from app import _decode_upload
    from engine.analyzer import FloorPlanAnalyzer

    _STAGES.clear()
    name, _, page = case.partition(":")
    path = CASES_DIR / name
    t0 = time.perf_counter()
    if page:
        from ingest.pdf import inspect_pdf, render_page
        from ingest.pdf_text import read_page_text, text_boxes
        import ingest.pdf as IP
        info = inspect_pdf(path).pages[int(page) - 1]
        image, _ = IP.render_page(path, info)
        import ingest.pdf_text as IT
        pt = IT.read_page_text(path, int(page), info.kind)
        evidence = text_boxes(pt, info.width_pt, info.height_pt, image.shape)
        structure_image = None
        if cleaning:
            import engine.cleaning.cleaner as CL
            res = CL.clean(image, path, int(page), evidence, mode="on")
            structure_image = res.recognition_image if res.applicable else None
        result = FloorPlanAnalyzer().analyze(image, name, text_evidence=evidence or None, structure_image=structure_image)
    else:
        image, _ = _decode_upload(path.read_bytes(), name)
        result = FloorPlanAnalyzer().analyze(image, name)
    wall = time.perf_counter() - t0
    return {"case": case, "pixels": int(image.shape[0] * image.shape[1]), "seconds": round(wall, 2),
            "rooms": result["room_count"], "unlabeled": result["unlabeled_space_count"], "openings": result["opening_count"],
            "stages": {k: round(v, 2) for k, v in sorted(_STAGES.items(), key=lambda kv: -kv[1])}}


def default_cases() -> list[str]:
    cases = []
    for p in sorted(CASES_DIR.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            cases.append(p.name)
    for pdf, pages in PDF_PAGES.items():
        if (CASES_DIR / pdf).exists():
            cases += [f"{pdf}:{n}" for n in pages]
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None)
    ap.add_argument("--cleaning", action="store_true")
    ap.add_argument("--save", default=None)
    ap.add_argument("--ocr-cache", action="store_true")
    args = ap.parse_args()
    if args.ocr_cache:
        from benchmark import ocr_cache
        ocr_cache.enable()
    instrument()
    cases = args.cases.split(",") if args.cases else default_cases()
    rows = []
    for case in cases:
        try:
            row = run_case(case, args.cleaning)
        except Exception as exc:                      # report and continue
            row = {"case": case, "error": repr(exc)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    ok = [r for r in rows if "seconds" in r]
    total = sum(r["seconds"] for r in ok)
    stage_tot = defaultdict(float)
    for r in ok:
        for k, v in r["stages"].items():
            stage_tot[k] += v
    summary = {"cases": len(ok), "total_seconds": round(total, 1),
               "stage_totals": {k: round(v, 1) for k, v in sorted(stage_tot.items(), key=lambda kv: -kv[1])}}
    print("SUMMARY", json.dumps(summary))
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
