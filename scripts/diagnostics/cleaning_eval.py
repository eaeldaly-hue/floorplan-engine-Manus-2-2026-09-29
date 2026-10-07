"""Original + existing recognition  vs  Architectural Cleaner + existing recognition.

    .venv/bin/python -m scripts.diagnostics.cleaning_eval [--plans "22.pdf:6,3.pdf:4"] [--mode on|vector]

Same render, text layer and OCR (benchmark OCR cache); only the geometry input differs. Rooms are
scored against benchmark/fixtures/pdf_plans.json (benchmark.pdf_plans.score). Writes per page to
output/cleaning/<pdf>_p<n>/: original.png, cleaned.png (skeleton), elements.png, recognition-input.png,
rooms_original.png, rooms_cleaned.png, comparison.png, cleaning.json; and output/cleaning/summary.json.

Cleaner-integrity measures (independent of the room ground truth):
  artificial_ink   share of the recognition input's ink that is on no ink of the original page,
                   excluding sealed openings and wall bodies (should be 0: the cleaner adds nothing else)
  wall_bodies_px   paper filled between wall faces (hollow walls made solid)
  sealed_openings  openings closed, by kind
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from benchmark import ocr_cache
from benchmark.holdout_inputs import load
from benchmark.pdf_plans import FIXTURE, PLANS_DIR, _analyze, score
from engine.cleaning.cleaner import clean, comparison_sheet

OUT = Path(__file__).resolve().parents[2] / "output" / "cleaning"


def artificial_ink(original, result) -> float:
    if result.recognition_image is None:
        return 0.0
    rec = result.recognition_image[..., 0] < 128
    page = cv2.dilate((cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) < 200).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    allowed = np.zeros(rec.shape, np.uint8)
    t = max(3, int(round(result.wall_thickness)) + 4)
    for o in result.openings:
        cv2.line(allowed, tuple(int(round(v)) for v in o["p0"]), tuple(int(round(v)) for v in o["p1"]), 1, t)
    if result.layer is not None and result.layer.wall_body is not None:
        allowed |= cv2.dilate(result.layer.wall_body.astype(np.uint8), np.ones((5, 5), np.uint8))
    extra = rec & ~page & ~(allowed > 0)
    return float(extra.sum() / max(1, rec.sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="22.pdf:6,3.pdf:4")
    ap.add_argument("--mode", default="on")
    args = ap.parse_args()
    ocr_cache.enable()
    doc = json.loads(FIXTURE.read_text())
    summary = {}
    for key in args.plans.split(","):
        spec = doc["plans"][key]
        pdf, page = key.split(":")
        image, evidence, _ = load(PLANS_DIR / pdf, int(page))
        name = f"{Path(pdf).stem}-page-{page}"
        d = OUT / key.replace(":", "_p").replace(".pdf", "")
        d.mkdir(parents=True, exist_ok=True)
        t = time.perf_counter()
        resp_a, st_a = _analyze(image, name, evidence)
        sec_a = time.perf_counter() - t
        t = time.perf_counter()
        cleaned = clean(image, PLANS_DIR / pdf, int(page), evidence, mode=args.mode)
        sec_clean = time.perf_counter() - t
        if cleaned.applicable:
            resp_c, st_c = _analyze(image, name, evidence, structure_image=cleaned.recognition_image, engine="legacy")
        else:
            resp_c, st_c = resp_a, st_a
        sec_c = time.perf_counter() - t
        a, c = score(spec, resp_a, st_a), score(spec, resp_c, st_c)
        a["seconds"], c["seconds"] = round(sec_a, 1), round(sec_c, 1)
        c["cleaner"] = cleaned.representation() | {"artificial_ink": round(artificial_ink(image, cleaned), 4),
                                                    "wall_bodies_px": int(cleaned.layer.wall_body.sum()) if cleaned.layer is not None and cleaned.layer.wall_body is not None else 0,
                                                    "cleaning_seconds": round(sec_clean, 1)}
        c["cleaner"].pop("openings")
        summary[key] = {"original": {k: v for k, v in a.items() if k != "per_point"},
                        "cleaned": {k: v for k, v in c.items() if k != "per_point"}}
        dec = lambda png: cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        rooms_a, rooms_c = dec(resp_a["overlay_png"]), dec(resp_c["overlay_png"])
        files = {"original.png": image, "rooms_original.png": rooms_a, "rooms_cleaned.png": rooms_c}
        if cleaned.applicable:
            files |= {"cleaned.png": cleaned.skeleton_image, "recognition-input.png": cleaned.recognition_image,
                      "elements.png": cleaned.elements_image(image), "comparison.png": comparison_sheet(image, cleaned, rooms_c)}
        for fn, im in files.items():
            cv2.imwrite(str(d / fn), im)
        (d / "cleaning.json").write_text(json.dumps(cleaned.representation(), indent=1))
        print(f"{key}: source={cleaned.source} | original sep {a['separated']}/{a['points']} merged {a['merged']} missed {a['missed']} "
              f"split {a['split_rooms']} outside {a['spaces_outside_building']} named {a['named_correct']}/{a['printed']} {a['seconds']}s"
              f" | cleaned sep {c['separated']}/{c['points']} merged {c['merged']} missed {c['missed']} split {c['split_rooms']} "
              f"outside {c['spaces_outside_building']} named {c['named_correct']}/{c['printed']} {c['seconds']}s"
              f" | artificial ink {c['cleaner']['artificial_ink']:.4f} openings {c['cleaner']['opening_counts']}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(ocr_cache.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
