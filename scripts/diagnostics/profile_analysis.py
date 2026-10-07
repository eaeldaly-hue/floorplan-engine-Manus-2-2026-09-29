"""Profile the production /api/analyze path stage by stage (measurement only).

Nothing in the engine, app, frontend or desktop wrapper is modified: timers
are attached at runtime by wrapping the real functions on the real modules
(``time.perf_counter``), and requests go through the real Flask app.
Delete this file to remove the instrumentation completely.

    .venv/bin/python -m scripts.diagnostics.profile_analysis                 # 3 in-process runs + breakdown
    .venv/bin/python -m scripts.diagnostics.profile_analysis --http          # + real HTTP round trips
    .venv/bin/python -m scripts.diagnostics.profile_analysis --webview       # + desktop WKWebView round trips
    .venv/bin/python -m scripts.diagnostics.profile_analysis --cprofile      # + cProfile hot functions
    .venv/bin/python -m scripts.diagnostics.profile_analysis --image synthetic/generated/floorplan_0001.png

Results are also written as JSON to output/profiling/ (git-ignored).
"""

from __future__ import annotations

import argparse
import cProfile
import functools
import io
import json
import platform
import pstats
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
import pytesseract

import app as app_module
import engine.analysis.ocr as ocr
import engine.analyzer as analyzer
import engine.opening_detection as opening_detection
import engine.structure as structure

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Runtime timing wrappers (thread-local call stacks so a threaded server works)
# ---------------------------------------------------------------------------

_local = threading.local()
RECORDS: list[dict] = []
_records_lock = threading.Lock()


def _stack() -> list[str]:
    if not hasattr(_local, "stack"):
        _local.stack = []
    return _local.stack


def instrument(owner, name: str, label: str, meta=None) -> None:
    is_mapping = isinstance(owner, dict)  # e.g. Flask's view_functions
    original = owner[name] if is_mapping else getattr(owner, name)

    @functools.wraps(original)
    def timed(*args, **kwargs):
        stack = _stack()
        stack.append(label)
        path = tuple(stack)
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            stack.pop()
            record = {"path": path, "seconds": elapsed}
            if meta is not None:
                record["meta"] = meta(args, kwargs)
            with _records_lock:
                RECORDS.append(record)

    if is_mapping:
        owner[name] = timed
    else:
        setattr(owner, name, timed)


def install_instrumentation(flask_app) -> None:
    # Request handler and app-level steps.
    instrument(flask_app.view_functions, "analyze_upload", "POST /api/analyze handler")
    instrument(app_module, "_decode_upload", "decode upload (imdecode)")
    instrument(app_module, "_trim_old_results", "trim stored results")
    instrument(app_module, "jsonify", "JSON response (jsonify)")
    instrument(analyzer.FloorPlanAnalyzer, "analyze", "FloorPlanAnalyzer.analyze")

    # OCR.
    instrument(analyzer, "extract_ocr", "OCR: extract_ocr")
    instrument(ocr, "prepare_ocr_variants", "OCR preprocessing (resize, CLAHE, denoise, thresholds)")
    instrument(cv2, "fastNlMeansDenoising", "cv2.fastNlMeansDenoising")
    instrument(ocr, "rotate_image", "rotate image")
    instrument(ocr, "_read_words", "Tesseract pass (_read_words)",
               meta=lambda a, k: {"variant": k.get("variant_name"), "rotation": k.get("rotation"), "psm": k.get("psm")})
    instrument(ocr, "deduplicate_ocr_boxes", "OCR dedup")
    instrument(pytesseract, "image_to_data", "pytesseract.image_to_data")
    instrument(pytesseract.pytesseract, "run_tesseract", "tesseract subprocess")
    instrument(analyzer, "group_room_labels", "group room labels")
    instrument(analyzer, "group_dimensions", "group dimensions")

    # Structure (walls → gaps → spaces → topology).
    instrument(analyzer, "analyze_structure", "structure: analyze_structure")
    instrument(structure, "_analyze_frame", "structure pass (one frame)")
    instrument(structure, "estimate_skew", "skew estimate")
    instrument(structure, "estimate_wall_kernel", "wall kernel (plateau search)")
    instrument(structure, "build_wall_mask", "wall mask")
    instrument(structure, "_axis_bands", "axis wall bands")
    instrument(structure, "_diagonal_bands", "diagonal wall bands")
    instrument(structure, "find_gaps", "opening candidates (find_gaps)")
    instrument(structure, "_line_bridged_gaps", "line-bridged openings")
    instrument(structure, "segment_spaces", "sealed spaces")
    instrument(structure, "assign_relations", "opening topology")
    instrument(structure, "wall_adjacency", "room wall adjacency")

    # Room records / scale.
    instrument(analyzer, "_build_room_records", "room records (_build_room_records)")
    instrument(analyzer, "_dimension_candidates", "dimension candidates")
    instrument(analyzer, "_associate_dimensions", "associate dimensions")
    instrument(analyzer, "_retry_dimensions_near_label", "OCR retry near label (crop, psm 6+7)")
    instrument(analyzer, "_component_map", "label→space mapping")
    instrument(analyzer, "_scale_from_rooms", "scale (_scale_from_rooms)")
    instrument(analyzer, "_snap_rectangle", "snap rectangle to walls")
    instrument(analyzer, "_room_box_from_wall_rays", "wall-ray room box")
    instrument(analyzer, "_infer_anonymous_dimensions", "fallback: infer anonymous dimensions")

    # Openings.
    instrument(analyzer, "classify_openings", "classify openings")
    instrument(opening_detection, "band_lines", "evidence: band lines")
    instrument(opening_detection, "arc_scores", "evidence: swing arcs")
    instrument(opening_detection, "leaf_score", "evidence: door leaf")
    instrument(opening_detection, "sliding_shift", "evidence: sliding panels")

    # Overlays.
    instrument(analyzer, "_draw_overlay", "rooms overlay PNG")
    instrument(analyzer, "draw_openings_overlay", "openings overlay PNG")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(runs: list[list[dict]]) -> dict[tuple, dict]:
    """Per call path: mean seconds per request and calls per request."""
    totals: dict[tuple, list[float]] = defaultdict(lambda: [0.0] * len(runs))
    counts: dict[tuple, list[int]] = defaultdict(lambda: [0] * len(runs))
    for index, records in enumerate(runs):
        for record in records:
            totals[record["path"]][index] += record["seconds"]
            counts[record["path"]][index] += 1
    return {
        path: {"mean": statistics.mean(totals[path]), "calls": statistics.mean(counts[path])}
        for path in totals
    }


def print_tree(stats: dict[tuple, dict], root: tuple, min_seconds: float = 0.005) -> None:
    def children(path):
        return sorted((p for p in stats if len(p) == len(path) + 1 and p[:len(path)] == path),
                      key=lambda p: -stats[p]["mean"])

    def walk(path, depth):
        entry = stats[path]
        calls = f"×{entry['calls']:.0f}" if entry["calls"] != 1 else ""
        print(f"{'  ' * depth}{path[-1]:<{66 - 2 * depth}} {entry['mean']:8.3f} s {calls}")
        kids = children(path)
        if kids:
            covered = sum(stats[k]["mean"] for k in kids)
            remainder = entry["mean"] - covered
            for kid in kids:
                if stats[kid]["mean"] >= min_seconds or depth < 1:
                    walk(kid, depth + 1)
            small = sum(stats[k]["mean"] for k in kids if stats[k]["mean"] < min_seconds and depth >= 1)
            if small >= 0.001:
                print(f"{'  ' * (depth + 1)}{'(smaller steps)':<{64 - 2 * depth}} {small:8.3f} s")
            if remainder >= 0.01:
                print(f"{'  ' * (depth + 1)}{'(other work in ' + path[-1][:28] + ')':<{64 - 2 * depth}} {remainder:8.3f} s")

    walk(root, 0)


def ocr_breakdown(runs: list[list[dict]]) -> dict:
    """Tesseract passes by variant/rotation/psm, the duplicate re-run, and retries."""
    per_run = []
    for records in runs:
        passes = [r for r in records if r["path"][-1] == "Tesseract pass (_read_words)"
                  and "OCR: extract_ocr" in r["path"]]
        # Records are appended at exit, in call order; the last search pass is the re-run.
        search, rerun = passes[:-1], passes[-1:]
        variant, rotation, psm = defaultdict(float), defaultdict(float), defaultdict(float)
        for r in search:
            variant[r["meta"]["variant"]] += r["seconds"]
            rotation[r["meta"]["rotation"] * 90] += r["seconds"]
            psm[r["meta"]["psm"]] += r["seconds"]
        to_data = [r for r in records if r["path"][-1] == "pytesseract.image_to_data"]
        subproc = [r for r in records if r["path"][-1] == "tesseract subprocess"]
        retry = [r for r in records if r["path"][-1] == "OCR retry near label (crop, psm 6+7)"]
        per_run.append({
            "search_passes": len(search), "search_seconds": sum(r["seconds"] for r in search),
            "rerun_passes": len(rerun), "rerun_seconds": sum(r["seconds"] for r in rerun),
            "by_variant": dict(variant), "by_rotation": dict(rotation), "by_psm": dict(psm),
            "tesseract_calls": len(to_data), "image_to_data_seconds": sum(r["seconds"] for r in to_data),
            "subprocess_seconds": sum(r["seconds"] for r in subproc),
            "retry_calls": len(retry), "retry_seconds": sum(r["seconds"] for r in retry),
        })

    def mean_dict(key):
        keys = per_run[0][key].keys()
        return {k: statistics.mean(run[key][k] for run in per_run) for k in keys}

    scalar_keys = [k for k, v in per_run[0].items() if not isinstance(v, dict)]
    result = {k: statistics.mean(run[k] for run in per_run) for k in scalar_keys}
    for key in ("by_variant", "by_rotation", "by_psm"):
        result[key] = mean_dict(key)
    return result


def repeated_work(runs: list[list[dict]]) -> list[dict]:
    """Expensive operations executed more than once per request."""
    watched = ["wall mask", "structure pass (one frame)", "OCR preprocessing (resize, CLAHE, denoise, thresholds)",
               "sealed spaces", "classify openings"]
    out = []
    for label in watched:
        sites = defaultdict(lambda: [0, 0.0])
        for records in runs:
            for r in records:
                if r["path"][-1] == label:
                    caller = r["path"][-2] if len(r["path"]) > 1 else "(top)"
                    sites[caller][0] += 1
                    sites[caller][1] += r["seconds"]
        n = len(runs)
        out.append({"operation": label, "sites": {k: {"calls": v[0] / n, "seconds": v[1] / n} for k, v in sites.items()}})
    return out


# ---------------------------------------------------------------------------
# Measurement modes
# ---------------------------------------------------------------------------

def run_in_process(flask_app, payload: bytes, name: str, runs: int):
    client = flask_app.test_client()
    totals, record_sets, responses = [], [], []
    for index in range(runs):
        RECORDS.clear()
        started = time.perf_counter()
        response = client.post("/api/analyze", data={"file": (io.BytesIO(payload), name)},
                               content_type="multipart/form-data")
        totals.append(time.perf_counter() - started)
        assert response.status_code == 200, response.get_data(as_text=True)
        record_sets.append(list(RECORDS))
        responses.append(response.get_json())
        print(f"  in-process run {index + 1}: {totals[-1]:.3f} s", flush=True)
    return totals, record_sets, responses


def multipart_body(payload: bytes, name: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


class ThreadedServer:
    """The instrumented app on a real localhost socket (werkzeug, threaded)."""

    def __init__(self, flask_app):
        from werkzeug.serving import make_server

        self.server = make_server("127.0.0.1", 0, flask_app, threaded=True)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()


def handler_seconds() -> float:
    return sum(r["seconds"] for r in RECORDS if r["path"] == ("POST /api/analyze handler",))


def run_http(flask_app, payload: bytes, name: str, runs: int):
    rows = []
    with ThreadedServer(flask_app) as server:
        for index in range(runs):
            RECORDS.clear()
            body, content_type = multipart_body(payload, name)
            request = urllib.request.Request(f"{server.url}/api/analyze", data=body, method="POST",
                                             headers={"Content-Type": content_type})
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=600) as response:
                response.read()
            rtt = time.perf_counter() - started
            server_side = handler_seconds()
            rows.append({"round_trip": rtt, "server_handler": server_side, "overhead": rtt - server_side})
            print(f"  HTTP run {index + 1}: round trip {rtt:.3f} s, server handler {server_side:.3f} s", flush=True)
    return rows


def run_webview(flask_app, runs: int):
    """Drive the real Workbench UI in a WKWebView window against the instrumented server."""
    import webview

    rows, startup = [], {}

    def drive(window, url):
        def js(code):
            return window.evaluate_js(code)

        def wait(expr, timeout=600):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                value = js(expr)
                if value:
                    return value
                time.sleep(0.2)
            return None

        try:
            wait(f"location.href.startsWith('{url}') && document.readyState === 'complete' && !!document.getElementById('sample-button')", 60)
            startup["page_load_ms"] = js("performance.getEntriesByType('navigation')[0].duration")
            for index in range(runs):
                js("document.getElementById('sample-button').click(); true")
                wait("document.getElementById('file-name').textContent === 'test_floorplan.png' && !document.getElementById('analyze-button').disabled", 30)
                RECORDS.clear()
                js("performance.clearResourceTimings(); document.getElementById('analyze-button').click(); true")
                wait("document.getElementById('status-card').dataset.state === 'running'", 10)
                state = wait("['done','error'].includes(document.getElementById('status-card').dataset.state) && document.getElementById('status-card').dataset.state")
                rendered = js("performance.now()")
                entry = json.loads(js("JSON.stringify(performance.getEntriesByType('resource').filter(e => e.name.endsWith('/api/analyze')).map(e => ({start: e.startTime, end: e.responseEnd, duration: e.duration})).pop() || null)"))
                server_side = handler_seconds()
                xhr = entry["duration"] / 1000 if entry else float("nan")
                rows.append({"state": state, "webview_xhr": xhr, "server_handler": server_side, "overhead": xhr - server_side,
                             "render_after_response": (rendered - entry["end"]) / 1000 if entry else float("nan")})
                print(f"  WebView run {index + 1}: XHR {xhr:.3f} s, server handler {server_side:.3f} s, state {state}", flush=True)
        finally:
            window.destroy()

    with ThreadedServer(flask_app) as server:
        window = webview.create_window("Floor Plan Engine — profiling", server.url, width=1400, height=900)
        webview.start(drive, (window, server.url))
    return rows, startup


def disable_app_nap() -> object:
    """Experiment only: tell macOS this process does user-initiated work (no App Nap)."""
    from Foundation import NSProcessInfo

    options = 0x00FFFFFF | 0xFF00000000  # NSActivityUserInitiated | NSActivityLatencyCritical
    return NSProcessInfo.processInfo().beginActivityWithOptions_reason_(options, "profiling")


def run_real_desktop(runs: int):
    """The shipped desktop flow (desktop.run_desktop → Flask child process), driven via the UI.
    The child server is not instrumented, so only the WebView round trip is measured."""
    import desktop

    rows: list[dict] = []

    def drive(window, server):
        def js(code):
            return window.evaluate_js(code)

        def wait(expr, timeout=600):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                value = js(expr)
                if value:
                    return value
                time.sleep(0.2)
            return None

        try:
            wait(f"location.href.startsWith('{server.url}') && document.readyState === 'complete' && !!document.getElementById('sample-button')", 60)
            for index in range(runs):
                js("document.getElementById('sample-button').click(); true")
                wait("document.getElementById('file-name').textContent === 'test_floorplan.png' && !document.getElementById('analyze-button').disabled", 30)
                js("performance.clearResourceTimings(); document.getElementById('analyze-button').click(); true")
                wait("document.getElementById('status-card').dataset.state === 'running'", 10)
                wait("['done','error'].includes(document.getElementById('status-card').dataset.state)")
                entry = json.loads(js("JSON.stringify(performance.getEntriesByType('resource').filter(e => e.name.endsWith('/api/analyze')).pop() || null)"))
                xhr = entry["duration"] / 1000 if entry else float("nan")
                rows.append({"webview_xhr": xhr, "child_pid": server.process.pid})
                print(f"  real desktop run {index + 1}: XHR {xhr:.3f} s", flush=True)
        finally:
            window.destroy()

    desktop.run_desktop(after_load=drive)
    return rows


def desktop_startup(samples: int = 3) -> list[float]:
    """Time desktop.start_server(): spawn child, import app/engine, answer /health."""
    import desktop

    times = []
    for _ in range(samples):
        started = time.perf_counter()
        server = desktop.start_server(desktop.free_port())
        times.append(time.perf_counter() - started)
        server.stop()
    return times


def run_cprofile(flask_app, payload: bytes, name: str, top: int = 25) -> str:
    client = flask_app.test_client()
    profiler = cProfile.Profile()
    profiler.enable()
    client.post("/api/analyze", data={"file": (io.BytesIO(payload), name)}, content_type="multipart/form-data")
    profiler.disable()
    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer)
    stats.sort_stats("tottime").print_stats(top)
    return buffer.getvalue()


# ---------------------------------------------------------------------------

def environment() -> dict:
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""

    power = sh("pmset", "-g", "batt").splitlines()
    low_power = [line.split()[-1] for line in sh("pmset", "-g").splitlines() if "lowpowermode" in line]
    return {
        "machine": platform.machine(), "cpu": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "cpus": sh("sysctl", "-n", "hw.ncpu"), "macos": platform.mac_ver()[0], "python": platform.python_version(),
        "opencv": cv2.__version__, "tesseract": str(pytesseract.get_tesseract_version()),
        "power_source": power[0] if power else "", "low_power_mode": low_power[0] if low_power else "",
        "load_average": sh("sysctl", "-n", "vm.loadavg"),
    }


def summarize(values: list[float]) -> str:
    return (f"runs {', '.join(f'{v:.2f}' for v in values)} | avg {statistics.mean(values):.2f} s | "
            f"min {min(values):.2f} s | max {max(values):.2f} s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default="test_floorplan.png")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--webview", action="store_true")
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--desktop", action="store_true", help="measure the shipped desktop app (child server) via its UI")
    parser.add_argument("--no-app-nap", action="store_true", help="experiment: disable App Nap for this process")
    parser.add_argument("--skip-in-process", action="store_true")
    args = parser.parse_args()
    if args.no_app_nap:
        _activity = disable_app_nap()  # noqa: F841 (kept alive for the process lifetime)

    image_path = (PROJECT_ROOT / args.image).resolve()
    payload = image_path.read_bytes()
    decoded = cv2.imread(str(image_path))
    out_dir = PROJECT_ROOT / "output" / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    flask_app = app_module.create_app({"UPLOAD_FOLDER": str(out_dir / "results")})
    install_instrumentation(flask_app)

    env = environment()
    print(f"Image: {image_path.name} ({decoded.shape[1]} × {decoded.shape[0]}), {len(payload) / 1024:.0f} KB")
    print(f"Environment: {env}\n")

    report = {"image": image_path.name, "size": [decoded.shape[1], decoded.shape[0]], "environment": env}
    if not args.skip_in_process:
        print("In-process requests (Flask test client → real view → real engine):")
        totals, record_sets, responses = run_in_process(flask_app, payload, image_path.name, args.runs)
        stats = aggregate(record_sets)
        print(f"\nFull request: {summarize(totals)}")
        handler = [sum(r['seconds'] for r in rs if r['path'] == ('POST /api/analyze handler',)) for rs in record_sets]
        print(f"View handler: {summarize(handler)}")
        print(f"Results identical across runs: {all(json.dumps({k: v for k, v in r.items() if 'url' not in k and k != 'result_id'}, sort_keys=True) == json.dumps({k: v for k, v in responses[0].items() if 'url' not in k and k != 'result_id'}, sort_keys=True) for r in responses)}")

        print("\nStage breakdown (mean per request; ×N = calls per request):")
        print_tree(stats, ("POST /api/analyze handler",))

        breakdown = ocr_breakdown(record_sets)
        print("\nOCR detail (mean per request):")
        print(f"  search passes: {breakdown['search_passes']:.0f} → {breakdown['search_seconds']:.3f} s; "
              f"selected-config re-run: {breakdown['rerun_passes']:.0f} → {breakdown['rerun_seconds']:.3f} s")
        print(f"  Tesseract calls total: {breakdown['tesseract_calls']:.0f}; image_to_data {breakdown['image_to_data_seconds']:.3f} s, "
              f"of which subprocess {breakdown['subprocess_seconds']:.3f} s "
              f"(rest = temp-image encode + TSV parse: {breakdown['image_to_data_seconds'] - breakdown['subprocess_seconds']:.3f} s)")
        print(f"  retry crops near labels: {breakdown['retry_calls']:.0f} calls → {breakdown['retry_seconds']:.3f} s")
        for key in ("by_variant", "by_rotation", "by_psm"):
            print(f"  {key}: " + ", ".join(f"{k}={v:.2f}s" for k, v in breakdown[key].items()))

        print("\nRepeated expensive operations (per request):")
        repeats = repeated_work(record_sets)
        for item in repeats:
            sites = ", ".join(f"{caller}: ×{v['calls']:.0f} {v['seconds']:.3f}s" for caller, v in item["sites"].items())
            print(f"  {item['operation']}: {sites or 'not called'}")

        report.update({
                  "in_process_totals": totals, "handler": handler,
                  "stages": {" > ".join(p): v for p, v in stats.items()}, "ocr": breakdown, "repeated": repeats})

    if args.http:
        print("\nReal HTTP (urllib → werkzeug socket → same instrumented app):")
        rows = run_http(flask_app, payload, image_path.name, args.runs)
        print(f"  round trip: {summarize([r['round_trip'] for r in rows])}")
        print(f"  HTTP/upload/Flask overhead outside the view: {summarize([r['overhead'] for r in rows])}")
        report["http"] = rows

    if args.webview:
        print("\nDesktop startup (desktop.start_server: spawn, import, /health):")
        startup = desktop_startup()
        print(f"  {summarize(startup)}")
        print("Desktop WKWebView (real Workbench UI → instrumented server):")
        rows, page = run_webview(flask_app, args.runs)
        print(f"  Workbench page load in WebView: {page.get('page_load_ms', float('nan')):.0f} ms")
        print(f"  WebView XHR: {summarize([r['webview_xhr'] for r in rows])}")
        print(f"  WebView/HTTP overhead outside the view: {summarize([r['overhead'] for r in rows])}")
        print(f"  UI render after response: {summarize([r['render_after_response'] for r in rows])}")
        report["webview"] = {"rows": rows, "server_startup": startup, **page}

    if args.desktop:
        print("\nShipped desktop app (desktop.run_desktop → Flask child process → engine), via the UI:")
        rows = run_real_desktop(args.runs)
        print(f"  WebView XHR: {summarize([r['webview_xhr'] for r in rows])}")
        report["desktop"] = rows

    if args.cprofile:
        text = run_cprofile(flask_app, payload, image_path.name)
        print("\ncProfile (one request, top functions by own time):")
        print(text)
        report["cprofile_top"] = text

    out = out_dir / f"profile-{image_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nSaved: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
