"""Stage-by-stage trace of the production analysis path, for diagnosing failed plans.

Measurement only: nothing in the engine is changed. The image is decoded with the
app's own upload decoder, every structural stage is re-run with the production
functions in the same order as ``engine.structure._analyze_frame``, the full
``FloorPlanAnalyzer.analyze`` is run (OCR included) with read-only wrappers that record
what each step returned, and the real Flask endpoint is called to check for swallowed
errors. Diagnostic images go to ``output/diagnostics/<name>/``.

    python -m scripts.diagnostics.trace_pipeline PATH [PATH ...] [--out output/diagnostics]
    python -m scripts.diagnostics.trace_pipeline --summary DIR      # one line per plan in DIR

The "what-if kernel" lines are counterfactual: they rebuild the wall mask with other
erosion kernels to show what the chosen kernel removed. They never feed back into the
production result.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np

import engine.analyzer as analyzer_module
import engine.structure as es
from app import _decode_upload, create_app

ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".avif", ".pdf"}


def _magic(raw: bytes) -> str:
    sigs = [(b"\x89PNG", "PNG"), (b"\xff\xd8\xff", "JPEG"), (b"%PDF", "PDF"), (b"BM", "BMP"),
            (b"II*\x00", "TIFF"), (b"MM\x00*", "TIFF"), (b"RIFF", "WEBP/RIFF"), (b"GIF8", "GIF")]
    for sig, name in sigs:
        if raw.startswith(sig):
            return name
    if raw[4:12] in (b"ftypavif", b"ftypavis"):
        return "AVIF"
    return "unknown"


def _hist_peaks(values: np.ndarray, lo=1, hi=40) -> list[list[float]]:
    """Most common stroke widths (px) with their share of measured ridge samples."""
    if values.size == 0:
        return []
    counts = np.bincount(np.clip(values.astype(int), 0, hi), minlength=hi + 1)[lo:hi + 1]
    share = counts / max(1, counts.sum())
    order = np.argsort(share)[::-1][:5]
    return [[int(i + lo), round(float(share[i]), 3)] for i in sorted(order) if share[i] > 0.02]


def _json_safe(value, path="$", problems=None):
    """Find NaN/Infinity anywhere in the API response."""
    problems = [] if problems is None else problems
    if isinstance(value, float) and not math.isfinite(value):
        problems.append(path)
    elif isinstance(value, dict):
        for k, v in value.items():
            _json_safe(v, f"{path}.{k}", problems)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _json_safe(v, f"{path}[{i}]", problems)
    return problems


# ---------------------------------------------------------------------------

def structure_trace(image: np.ndarray, out: Path | None) -> dict:
    """Re-run the structural stages with the production functions and record each one."""
    t0 = time.perf_counter()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    otsu_t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ink, symbol_ink = es._binarize(gray)
    ink_n = cv2.connectedComponents((ink > 0).astype(np.uint8))[0] - 1
    stroke = es._ridge_widths(ink)

    kmax = max(8, int(0.04 * max(gray.shape)))
    areas = [es._surviving_area_at(ink, k) for k in range(1, min(kmax, 24) + 1)]
    curve = [round(a / max(areas[0], 1), 3) for a in areas]
    drops = [round((areas[i - 1] - areas[i]) / max(areas[i - 1], 1), 3) for i in range(1, len(areas))]
    # The exact production call (multi-scale walls, line-wall fallback, deskew). Every
    # statistic and image below describes what the API actually used.
    full = es.analyze_structure(image)
    elapsed = time.perf_counter() - t0
    inference = full.walls if full.work is None else full.work.walls
    kernel, line_width, thinnest = inference.kernel, inference.line_width, inference.thinnest
    wall_mask = full.wall_mask
    wall_widths = es._ridge_widths(wall_mask)
    typical_t = full.wall_thickness
    thick_max = float(np.percentile(wall_widths, 95)) if wall_widths.size else 2 * typical_t
    wall_n = cv2.connectedComponents((wall_mask > 0).astype(np.uint8))[0] - 1
    removed = (ink > 0) & ~(wall_mask > 0)
    work = full.work if full.work is not None else full
    bands = work.bands
    axis = [b for b in bands if b.orientation != "diagonal"]
    diag = [b for b in bands if b.orientation == "diagonal"]
    candidates, cracks = full.candidates, full.cracks
    sealed, labels, spaces = full.sealed_mask, full.space_labels, full.spaces

    # Raw enclosed regions before the area / clear-width filters (same rules as segment_spaces).
    h, w = sealed.shape
    free = cv2.copyMakeBorder(cv2.bitwise_not(sealed), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    flood = free.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 128)
    exterior = (flood == 128)[1:-1, 1:-1]
    interior = ((free == 255)[1:-1, 1:-1] & ~exterior).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(interior, connectivity=4)
    min_area = max(400.0, (3.0 * typical_t) ** 2)
    raw_regions = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)), reverse=True)

    # Counterfactual: what would other kernels keep? (diagnostic only)
    what_if = {}
    for k in sorted({max(3, kernel - 3), max(3, kernel - 2), kernel - 1, kernel} - {kernel - 1 if kernel - 1 < 3 else -1}):
        if k < 3:
            continue
        m = es.build_wall_mask(ink, k)
        t_k = es._typical_thickness(m, thinnest)
        wid = es._ridge_widths(m)
        tm = float(np.percentile(wid, 95)) if wid.size else 2 * t_k
        b = es._axis_bands(m, k, tm) + es._diagonal_bands(m, t_k)
        c, cr = es.find_gaps(m, symbol_ink, b, t_k, k, tm, thinnest)
        _, _, sp = es.segment_spaces(m, c, cr, t_k)
        what_if[k] = {"wall_pct": round(100 * float((m > 0).mean()), 2), "spaces": len(sp), "candidates": len(c)}

    trace = {
        "preprocess": {
            "otsu_threshold": float(otsu_t),
            "ink_pct": round(100 * float((ink > 0).mean()), 2),
            "symbol_ink_pct": round(100 * float((symbol_ink > 0).mean()), 2),
            "ink_components": int(ink_n),
            "stroke_width_peaks_px": _hist_peaks(stroke),
        },
        "deskew": {"skew_degrees": round(float(full.skew_degrees), 2), "rotated": full.work is not None},
        "wall_kernel": {
            "surviving_area_by_kernel": curve,
            "drop_per_step": drops,
            "chosen_kernel": int(kernel), "line_width": float(line_width), "thinnest_wall_estimate": float(thinnest),
        },
        "wall_mask": {
            "wall_pct": round(100 * float((wall_mask > 0).mean()), 2),
            "wall_share_of_ink": round(float((wall_mask > 0).sum()) / max(1, int((ink > 0).sum())), 3),
            "components": int(wall_n),
            "wall_width_peaks_px": _hist_peaks(wall_widths),
            "typical_thickness_px": round(float(typical_t), 1),
            "thick_max_px": round(float(thick_max), 1),
        },
        "wall_geometry": {
            "axis_bands": len(axis), "diagonal_bands": len(diag),
            "band_length_px": int(sum(np.hypot(*(np.subtract(b.p1, b.p0))) for b in bands)) if bands and hasattr(bands[0], "p0") else None,
        },
        "openings_geometry": {
            "candidates": len(candidates), "cracks_sealed": len(cracks),
            "by_source": {s: sum(c.source == s for c in candidates) for s in ("wall_gap", "line_pair", "single_line")},
            "relations": {r: sum(c.relation == r for c in candidates) for r in ("between_rooms", "room_to_exterior", "same_space", "unknown")},
        },
        "spaces": {
            "raw_enclosed_regions": len(raw_regions),
            "raw_region_areas_px": raw_regions[:12],
            "min_area_px": round(min_area, 1),
            "min_clear_width_px": round(max(4.0, 0.75 * typical_t), 1),
            "after_filtering": len(spaces),
            "exterior_pct": round(100 * float(exterior.mean()), 1),
        },
        "wall_inference": (lambda wi: None if wi is None else {"primary_kernel": wi.kernel, "min_kernel": wi.min_kernel,
                                                              "secondary_kernel": wi.secondary_kernel, "classes": wi.classes,
                                                              "line_walls": None if wi.line_walls is None else int((wi.line_walls > 0).sum()),
                                                              "line_decisions": {d: sum(e["decision"] == d for e in wi.line_evidence)
                                                                                 for d in sorted({e["decision"] for e in wi.line_evidence})}})(
            full.work.walls if full.work is not None else full.walls),
        "production_structure_call": {
            "spaces": len(full.spaces), "candidates": len(full.candidates), "kernel": full.wall_kernel,
            "wall_thickness": round(full.wall_thickness, 1),
        },
        "what_if_kernel": what_if,
        "seconds": round(elapsed, 2),
    }

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "01_gray.png"), gray)
        cv2.imwrite(str(out / "02_ink_binary.png"), 255 - ink)
        cv2.imwrite(str(out / "03_symbol_ink.png"), 255 - symbol_ink)
        view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        view[removed] = (60, 60, 230)            # red: ink the thickness filter removed
        view[wall_mask > 0] = (40, 150, 40)      # green: kept as wall
        cv2.imwrite(str(out / "04_wall_mask_vs_removed_ink.png"), view)
        geo = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) // 2 + 127
        for b in bands:
            if hasattr(b, "p0"):
                cv2.line(geo, tuple(map(int, b.p0)), tuple(map(int, b.p1)), (200, 60, 0), 2)
        cv2.imwrite(str(out / "05_wall_geometry.png"), geo)
        sp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rng = np.random.default_rng(1)
        tint = sp.copy()
        tint[labels == es.EXTERIOR] = (235, 235, 235)
        for k in range(1, len(spaces) + 1):
            tint[labels == k] = rng.integers(80, 230, 3)
        sp = cv2.addWeighted(sp, 0.5, tint, 0.5, 0)
        sp[sealed > 0] = (30, 30, 30)
        cv2.imwrite(str(out / "06_spaces.png"), sp)
        cand = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cand[wall_mask > 0] = (150, 200, 150)
        for c in candidates:
            colour = {"wall_gap": (0, 140, 255), "line_pair": (220, 120, 0), "single_line": (200, 0, 200)}[c.source]
            cv2.line(cand, tuple(map(int, c.start)), tuple(map(int, c.end)), colour, 3)
        for c in cracks:
            cv2.line(cand, tuple(map(int, c.start)), tuple(map(int, c.end)), (0, 0, 255), 1)
        cv2.imwrite(str(out / "07_opening_candidates.png"), cand)
        # Wall-inference evidence: primary walls, accepted secondary walls, rejected candidates.
        wi = full.work.walls if full.work is not None else full.walls
        if wi is not None:
            ev = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) // 2 + 127
            ev[wi.primary > 0] = (60, 60, 60)                      # primary (thick-class) walls
            if wi.rejected is not None:
                colours = {1: (200, 160, 255), 2: (0, 165, 255), 3: (255, 120, 0)}
                for code, colour in colours.items():
                    ev[wi.rejected == code] = colour               # rejected, by reason
                ev[wi.accepted > 0] = (0, 170, 0)                  # accepted secondary walls
            y = 18
            for text, colour in (("primary wall", (60, 60, 60)), ("secondary: accepted", (0, 170, 0)),
                                 ("rejected: no straight run", (200, 160, 255)), ("rejected: not connected", (0, 165, 255)),
                                 ("rejected: fill in thicker wall", (255, 120, 0))):
                cv2.putText(ev, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
                y += 16
            cv2.imwrite(str(out / "04b_wall_evidence.png"), ev)
            if wi.line_evidence:
                lv = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) // 2 + 127
                lv[wi.mask > 0] = (60, 60, 60)
                colours = {"wall": (0, 170, 0), "free end (not anchored)": (255, 0, 255),
                           "hinged at a wall end (door leaf)": (255, 120, 0),
                           "repeated parallel lines (stairs/shelving)": (150, 150, 150),
                           "fill inside a wall gap (glazing/threshold)": (0, 165, 255)}
                for e in wi.line_evidence:
                    x, y, w, h = e["bbox"]
                    cv2.rectangle(lv, (x - 2, y - 2), (x + w + 2, y + h + 2), colours.get(e["decision"], (0, 0, 255)), 1)
                y = 18
                for text, colour in list(colours.items()) + [("other rejection", (0, 0, 255))]:
                    cv2.putText(lv, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
                    y += 15
                cv2.imwrite(str(out / "04c_line_wall_evidence.png"), lv)
        # Counterfactual wall mask at the smallest kernel tried, for comparison.
        if what_if:
            k = min(what_if)
            m = es.build_wall_mask(ink, k)
            alt = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            alt[m > 0] = (40, 150, 40)
            cv2.imwrite(str(out / f"08_whatif_wall_mask_kernel{k}.png"), alt)
    return trace


def analyzer_trace(image: np.ndarray, name: str, out: Path | None) -> dict:
    """Run FloorPlanAnalyzer.analyze exactly as the API does, recording OCR and room records."""
    seen: dict = {}
    real_ocr = analyzer_module.extract_ocr
    real_records = analyzer_module._build_room_records
    real_classify = analyzer_module.classify_openings

    def ocr_spy(img, *a, **k):
        t = time.perf_counter()
        try:
            result = real_ocr(img, *a, **k)
        except Exception as exc:          # recorded, then re-raised so behaviour is unchanged
            seen["ocr_exception"] = repr(exc)
            raise
        seen["ocr_seconds"] = round(time.perf_counter() - t, 2)
        seen["ocr"] = result
        return result

    def records_spy(room_lines, dimension_lines, spaces, wall_mask, img):
        seen["room_lines"] = [(l.text, round(l.confidence, 1)) for l in room_lines]
        seen["dimension_lines"] = [(l.text, round(l.confidence, 1)) for l in dimension_lines]
        seen["spaces_in"] = len(spaces)
        result = real_records(room_lines, dimension_lines, spaces, wall_mask, img)
        seen["records_out"] = {"rooms": len(result[0]), "unlabeled": len(result[2]), "scale": result[1]}
        return result

    def classify_spy(img, structure, scale):
        result = real_classify(img, structure, scale)
        seen["classified"] = len(result)
        return result

    analyzer_module.extract_ocr = ocr_spy
    analyzer_module._build_room_records = records_spy
    analyzer_module.classify_openings = classify_spy
    try:
        t = time.perf_counter()
        response = analyzer_module.FloorPlanAnalyzer().analyze(image, name)
        seconds = round(time.perf_counter() - t, 2)
    finally:
        analyzer_module.extract_ocr = real_ocr
        analyzer_module._build_room_records = real_records
        analyzer_module.classify_openings = real_classify

    ocr = seen.get("ocr")
    ocr_info = {"exception": seen.get("ocr_exception"), "seconds": seen.get("ocr_seconds")}
    if ocr is not None:
        ocr_info["boxes_total"] = len(ocr.boxes)
        for field in ("room_labels", "dimensions", "other_text"):
            items = getattr(ocr, field)
            ocr_info[field] = len(items)
            ocr_info[f"{field}_sample"] = [f"{i.text[:30]} ({i.confidence:.0f})" for i in list(items)[:12]]
        ocr_info["selected"] = {"variant": ocr.selected_variant, "rotation": ocr.selected_rotation, "psm": ocr.selected_psm}
    if out is not None and ocr is not None and getattr(ocr, "regions", None):
        write_ocr_evidence(image, ocr, response, out)
    if out is not None and ocr is not None:
        boxes = image.copy()
        for field, colour in (("room_labels", (0, 150, 0)), ("dimensions", (200, 80, 0)), ("other_text", (0, 0, 220))):
            for i in getattr(ocr, field, []) or []:
                x, y, bw, bh = (int(getattr(i, k, 0)) for k in ("x", "y", "width", "height"))
                cv2.rectangle(boxes, (x, y), (x + bw, y + bh), colour, 2)
        cv2.imwrite(str(out / "09_ocr_boxes.png"), boxes)

    return {
        "ocr": ocr_info,
        "room_label_lines_accepted": seen.get("room_lines"),
        "dimension_lines": seen.get("dimension_lines"),
        "room_records": {"spaces_in": seen.get("spaces_in"), **(seen.get("records_out") or {})},
        "classified_openings": seen.get("classified"),
        "final": {
            "room_count": response["room_count"],
            "unlabeled_space_count": response["unlabeled_space_count"],
            "opening_count": response["opening_count"],
            "doors": response["door_count"], "windows": response["window_count"],
            "unclassified": response["unclassified_opening_count"],
            "pixel_scale": response["pixel_scale"],
            "warnings": response["warnings"],
        },
        "seconds": seconds,
    }


ROT_COLOURS = {0: (0, 160, 0), 90: (220, 120, 0), 180: (0, 140, 255), 270: (0, 0, 220)}


def write_ocr_evidence(image: np.ndarray, ocr, response: dict, out: Path) -> None:
    """OCR aggregation evidence: every pass's observations, resolved regions, final labels."""
    regions = list(ocr.regions)
    observations = [o for r in regions for o in r.observations]
    scale = 2 if max(image.shape[:2]) < 1200 else 1

    def canvas():
        c = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) // 2 + 127
        return cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def rect(img, x, y, w, h, colour, t=1):
        cv2.rectangle(img, (x * scale, y * scale), ((x + w) * scale, (y + h) * scale), colour, t)

    def put(img, text, x, y, colour):
        cv2.putText(img, text, (x * scale, max(10, y * scale - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * scale, colour, 1, cv2.LINE_AA)

    # 10: all observations by rotation; 10a/10b: upright vs rotated passes only
    allobs = canvas()
    for o in observations:
        rect(allobs, o.x, o.y, o.width, o.height, ROT_COLOURS.get(o.rotation, (0, 0, 0)))
    cv2.imwrite(str(out / "10_ocr_all_pass_observations.png"), allobs)
    panels = []
    for title, keep in (("upright passes (0 deg)", lambda o: o.rotation == 0),
                        ("rotated passes (90/180/270 deg)", lambda o: o.rotation != 0)):
        c = canvas()
        for o in observations:
            if keep(o):
                rect(c, o.x, o.y, o.width, o.height, ROT_COLOURS.get(o.rotation, (0, 0, 0)))
                if o.confidence >= 60:
                    put(c, f"{o.text[:10]} {o.confidence:.0f}", o.x, o.y, ROT_COLOURS.get(o.rotation, (0, 0, 0)))
        cv2.putText(c, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        panels.append(c)
    # 11: resolved regions (accepted green with support, rejected red)
    reg = canvas()
    for r in regions:
        x, y, w, h = r.box
        colour = (0, 150, 0) if r.accepted else (0, 0, 220)
        rect(reg, x, y, w, h, colour)
        if r.accepted or r.confidence >= 50:
            put(reg, f"{r.text[:12]} x{r.support}", x, y, colour)
    cv2.imwrite(str(out / "11_ocr_regions_accepted_rejected.png"), reg)
    # 12: final room labels and the space each one refers to
    fin = canvas()
    for room in response.get("rooms", []):
        b = room["boundary"]
        colour = (200, 0, 200) if b and b["method"] == "open-plan-shared" else (0, 150, 0) if b else (0, 0, 220)
        if b:
            pts = np.array([[p["x"] * scale, p["y"] * scale] for p in b["polygon"]], np.int32)
            cv2.polylines(fin, [pts], True, colour, 1)
        c = room["label_center"]
        cv2.circle(fin, (c["x"] * scale, c["y"] * scale), 4, colour, -1)
        put(fin, f"{room['label_text']} -> {room['name']}{' (open plan)' if b and b['method'] == 'open-plan-shared' else ''}",
            c["x"] + 4, c["y"], colour)
    cv2.putText(fin, "final room labels", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(str(out / "12_room_labels_and_association.png"), fin)
    panels.append(fin)
    cv2.imwrite(str(out / "13_upright_vs_rotated_vs_final.png"), np.hstack(panels))
    records = [{"text": r.text, "box": list(r.box), "support": r.support, "confidence": r.confidence,
                "rotations": list(r.rotations), "accepted": r.accepted, "reason": r.reason,
                "rivals": r.rivals, "observations": len(r.observations)} for r in regions]
    (out / "ocr_regions.json").write_text(json.dumps({"passes": list(ocr.passes), "regions": records}, indent=1, default=str))


def api_trace(raw: bytes, name: str) -> dict:
    """POST the original bytes to the real Flask endpoint; capture status, logs and JSON validity."""
    log = io.StringIO()
    handler = logging.StreamHandler(log)
    handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(handler)
    try:
        app = create_app({"TESTING": True, "UPLOAD_FOLDER": str(ROOT / "output" / "diagnostics" / "_api_results")})
        client = app.test_client()
        resp = client.post("/api/analyze", data={"file": (io.BytesIO(raw), name)}, content_type="multipart/form-data")
    finally:
        logging.getLogger().removeHandler(handler)
    body = resp.get_json(silent=True)
    info = {"status": resp.status_code, "logged_warnings_or_errors": log.getvalue().strip() or None}
    if isinstance(body, dict):
        info["error"] = body.get("error")
        info["nan_or_inf_fields"] = _json_safe(body)
        info["room_count"] = body.get("room_count")
        info["unlabeled_space_count"] = body.get("unlabeled_space_count")
        info["opening_count"] = body.get("opening_count")
        bad = []
        W, H = body.get("image", {}).get("width", 0), body.get("image", {}).get("height", 0)
        for o in body.get("openings", []):
            for x, y in (o["start"], o["end"]):
                if not (-2 <= x <= W + 2 and -2 <= y <= H + 2):
                    bad.append(o["id"])
        info["openings_outside_image"] = bad
    return info


def trace_file(path: Path, out_root: Path, images: bool = True, ocr: bool = True) -> dict:
    raw = path.read_bytes()
    unchanged = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    image, load_warnings = _decode_upload(raw, path.name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    otsu_t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark = gray < otsu_t
    out = out_root / path.stem if images else None
    result = {
        "input": {
            "file": path.name, "format": _magic(raw), "bytes": len(raw),
            "width": int(image.shape[1]), "height": int(image.shape[0]),
            "source_channels": None if unchanged is None else (1 if unchanged.ndim == 2 else int(unchanged.shape[2])),
            "source_dtype": None if unchanged is None else str(unchanged.dtype),
            "decoded_channels": int(image.shape[2]), "load_warnings": load_warnings,
            "gray": {"min": int(gray.min()), "max": int(gray.max()), "mean": round(float(gray.mean()), 1),
                     "std": round(float(gray.std()), 1), "p1_p50_p99": [int(v) for v in np.percentile(gray, [1, 50, 99])],
                     "distinct_levels": int(len(np.unique(gray)))},
            "foreground": {"dark_pct": round(100 * float(dark.mean()), 2),
                           "mean_dark": round(float(gray[dark].mean()), 1) if dark.any() else None,
                           "mean_light": round(float(gray[~dark].mean()), 1)},
        },
        "structure": structure_trace(image, out),
    }
    if ocr:
        result["analyzer"] = analyzer_trace(image, path.name, out)
        result["api"] = api_trace(raw, path.name)
    if out is not None:
        (out / "trace.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--out", default=str(ROOT / "output" / "diagnostics"))
    ap.add_argument("--summary", help="directory: one summary line per plan (structure + full analysis)")
    args = ap.parse_args()
    out_root = Path(args.out)
    if args.summary:
        rows = []
        for path in sorted(Path(args.summary).iterdir(), key=lambda p: (len(p.stem), p.stem)):
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                r = trace_file(path, out_root / "_summary", images=False, ocr=True)
                s, a = r["structure"], r["analyzer"]["final"]
                rows.append([path.name, f"{r['input']['width']}x{r['input']['height']}", s["wall_kernel"]["chosen_kernel"],
                             s["wall_mask"]["wall_pct"], s["spaces"]["after_filtering"], s["openings_geometry"]["candidates"],
                             a["room_count"], a["unlabeled_space_count"], a["opening_count"]])
            except Exception as exc:
                rows.append([path.name, "-", "-", "-", "-", "-", "-", "-", f"ERROR {type(exc).__name__}: {exc}"[:60]])
            print(" | ".join(map(str, rows[-1])), flush=True)
        (out_root / "summary.json").write_text(json.dumps(rows, indent=1))
        return
    for p in args.paths:
        r = trace_file(Path(p), out_root)
        print(json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    main()
