"""Stage-by-stage timing of exactly what the Workbench runs for one input (cold by default).

PDF page: render + text layer + analyze with the vector-cleaned candidate (the app's default path);
image: decode + analyze. Each case runs in its own process (peak memory is per case).

    python -m benchmark.profile_app [--cases 3.pdf:5,22.pdf:6,7.png] [--ocr-cache] [--save F]

Stages overlap (OCR runs concurrently with the structural job), so their sum exceeds wall time;
'wall' is the end-to-end time and 'after_ocr' the time from OCR's end to the result.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import resource
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = Path(os.environ.get("FLOORPLAN_REAL_PLANS", ROOT.parent / "Test Cases"))
DEFAULT = ["3.pdf:5", "3.pdf:4", "22.pdf:6", "27.pdf:1", "29.pdf:1", "24.pdf:1", "1.png", "9.jpg", "26.jpg", "20.jpg", "7.png"]

_T: dict = defaultdict(float)
_N: dict = defaultdict(int)
_END: dict = {}
_START: dict = {}
_T0 = [0.0]
_LOCK = threading.Lock()


def _wrap(module, name, label):
    fn = getattr(module, name, None)
    if fn is None or getattr(fn, "_prof", False):
        return

    @functools.wraps(fn)
    def w(*a, **k):
        t = time.perf_counter()
        with _LOCK:
            _START.setdefault(label, t)
        try:
            return fn(*a, **k)
        finally:
            with _LOCK:
                _T[label] += time.perf_counter() - t
                _N[label] += 1
                _END[label] = time.perf_counter()
    w._prof = True
    setattr(module, name, w)


def instrument():
    import engine.analyzer as A
    import engine.arch.building as B
    import engine.arch.building_view as BV
    import engine.arch.objects as O
    import engine.arch.projection as PR
    import engine.arch.zones as Z
    import engine.cleaning.cleaner as CL
    import engine.reconstruction as R
    import engine.structure as S
    import engine.analysis.ocr as OCR
    import ingest.pdf as IP
    import ingest.pdf_text as IT
    import ingest.pdf_vectors as IV
    import engine.semantic.layer as SL

    for mod, name, label in [
        (A, "extract_ocr", "ocr"), (A, "merge_document_text", "merge_text"), (A, "build_room_labels", "room_labels"),
        (A, "analyze_structure", "structure"), (A, "classify_openings", "openings_classify"),
        (A, "_split_shared_spaces", "split_spaces"), (A, "_join_open_connections", "join_open"),
        (A, "_build_room_records", "room_records"), (A, "_infer_anonymous_dimensions", "anon_dims"),
        (A, "_draw_overlay", "overlay_rooms"), (A, "draw_openings_overlay", "overlay_openings"),
        (A, "_topology", "topology"), (A, "_functional_zones", "objects+zones"),
        (B, "build_building", "building_model"), (BV, "render_png", "overlay_building"),
        (O, "vector_objects", "objects"), (Z, "infer_zones", "zones"), (PR, "project", "projection"),
        (R, "generate", "hyp_generate"), (R, "choose", "hyp_choose"), (R, "text_blocks", "hyp_text_blocks"),
        (CL, "clean", "cleaning"), (SL, "build_layer", "clean_build_layer"),
        (IP, "render_page", "pdf_render"), (IT, "read_page_text", "pdf_text"), (IV, "page_paths", "pdf_vectors"),
        (OCR, "run_ocr_tasks", "ocr_passes"),
    ]:
        _wrap(mod, name, label)
    import engine.plan.adapter as PA
    for name in ("prepare", "finish", "run"):
        _wrap(PA, name, f"plan_model_{name}")
    import engine.arch.openings as AO
    _wrap(AO, "typed_openings", "openings_typed")
    _wrap(S, "analyze_structure", "structure_hyp")      # alternative readings (local imports)
    import engine.rescale as RS
    _wrap(RS, "to_original", "rescale")


def _rss_sampler(samples: list, stop: threading.Event, t0: list) -> None:
    """Resident memory of this process every 0.25 s (via ps: no extra dependency)."""
    pid = str(os.getpid())
    while not stop.is_set():
        try:
            kb = int(subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True).stdout.strip() or 0)
            samples.append((time.perf_counter() - t0[0], kb / 1024))
        except Exception:
            pass
        stop.wait(0.25)


def run_one(case: str, ocr_cache: bool, rss: bool = False) -> dict:
    if ocr_cache:
        from benchmark import ocr_cache as oc
        oc.enable()
    instrument()
    from engine.analyzer import FloorPlanAnalyzer
    name, _, page = case.partition(":")
    path = CASES_DIR / name
    t0 = time.perf_counter()
    _T0[0] = t0
    samples: list = []
    stop = threading.Event()
    if rss:
        threading.Thread(target=_rss_sampler, args=(samples, stop, _T0), daemon=True).start()
    if page:
        from ingest.pdf import inspect_pdf, render_page
        from ingest.pdf_text import read_page_text, text_boxes
        from engine.cleaning.cleaner import clean
        info = inspect_pdf(path).pages[int(page) - 1]
        image, _ = render_page(path, info)
        pt = read_page_text(path, int(page), info.kind)
        ev = text_boxes(pt, info.width_pt, info.height_pt, image.shape)
        r = FloorPlanAnalyzer().analyze(image, name, text_evidence=ev or None,
                                        cleaner_candidate=lambda: clean(image, path, int(page), ev, mode="vector"))
    else:
        from app import _decode_upload
        image, _ = _decode_upload(path.read_bytes(), name)
        r = FloorPlanAnalyzer().analyze(image, name)
    wall = time.perf_counter() - t0
    stop.set()
    b = r.get("building") or {}
    rss_info = None
    if samples:
        tp, mp = max(samples, key=lambda x: x[1])
        active = sorted(k for k in _START if _START[k] - t0 <= tp <= _END.get(k, 0) - t0)
        rss_info = {"peak_mb": round(mp), "at_s": round(tp, 1), "active": active,
                    "trace": [(round(a, 1), round(m)) for a, m in samples[::4]]}
    ocr_end = _END.get("ocr", t0)
    return {"case": case, "mp": round(image.shape[0] * image.shape[1] / 1e6, 1), "wall": round(wall, 2),
            "after_ocr": round(time.perf_counter() - ocr_end, 2) if "ocr" in _END else None,
            "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 0),
            "reading": (r.get("reconstruction") or {}).get("chosen", "default"),
            "rooms": r["room_count"], "spaces": len(b.get("spaces", [])), "openings": r["opening_count"],
            "objects": r.get("object_count", 0), "zones": r.get("zone_count", 0),
            "stages": {k: round(v, 2) for k, v in sorted(_T.items(), key=lambda kv: -kv[1])},
            "calls": {k: _N[k] for k in _T if _N[k] > 1},
            "timeline": {k: [round(_START[k] - t0, 1), round(_END[k] - t0, 1)] for k in sorted(_START, key=_START.get)},
            "rss": rss_info}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases")
    ap.add_argument("--ocr-cache", action="store_true")
    ap.add_argument("--save")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--rss", action="store_true", help="sample resident memory over time (peak and active stages)")
    args = ap.parse_args()
    if args.one:
        print("RESULT " + json.dumps(run_one(args.one, args.ocr_cache, args.rss)), flush=True)
        return 0
    rows = []
    for case in (args.cases.split(",") if args.cases else DEFAULT):
        cmd = [sys.executable, "-m", "benchmark.profile_app", "--one", case] + (["--ocr-cache"] if args.ocr_cache else []) \
            + (["--rss"] if args.rss else [])
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
        row = json.loads(line[7:]) if line else {"case": case, "error": out.stderr[-600:]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
