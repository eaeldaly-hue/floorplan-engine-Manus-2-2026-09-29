"""A / B / C on the canonical real test cases (engine venv; research only).

  A  current engine (full analyzer: hypotheses, symbol reading, reasoning)
  B  learned perception alone (research U-Net probabilities, research/perception/infer.py)
  C  hybrid: the analyzer with the model's evidence fused into the wall layer (model walls added
     where ink exists, wall components the model sees as furniture removed) and opening types
     taken from the model where it is confident

    .venv/bin/python research/perception/eval_real.py --variant norm --save output/perception/eval_norm.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REAL = ROOT / "output/perception/real"
CLS = {"exterior": 0, "interior": 1, "wall": 2, "door": 3, "window": 4, "passage": 5, "other": 6}


def load_prob(path) -> np.ndarray:
    return np.load(path)["prob"].astype(np.float32) / 255.0


def at(mask: np.ndarray, x, y, r=3) -> bool:
    h, w = mask.shape
    x, y = int(round(x)), int(round(y))
    win = mask[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)]
    return bool(win.size and win.any())


# ---------------------------------------------------------------- B: model alone
def model_rooms(prob: np.ndarray) -> np.ndarray:
    arg = prob.argmax(0)
    barrier = np.isin(arg, [CLS["wall"], CLS["door"], CLS["window"], CLS["passage"]])
    free = (~barrier).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    h, w = free.shape
    out = np.zeros((h, w), np.int32)
    k = 0
    min_area = 0.0005 * h * w
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h or a < min_area:
            continue
        # a component that is mostly paper (exterior class) is not a room
        comp = lab == i
        if prob[CLS["exterior"]][comp].mean() > 0.5:
            continue
        k += 1
        out[comp] = k
    out[barrier] = 0
    return out


def model_openings(prob: np.ndarray, t: float) -> list[dict]:
    arg = prob.argmax(0)
    out = []
    for cls, name in ((CLS["door"], "door"), (CLS["window"], "window"), (CLS["passage"], "opening")):
        m = (arg == cls).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < max(6.0, 0.6 * t * t):
                continue
            ys, xs = np.nonzero(lab == i)
            rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            (cx, cy), (rw, rh), ang = rect
            L = max(rw, rh)
            a = np.radians(ang if rw >= rh else ang + 90)
            d = np.array([np.cos(a), np.sin(a)])
            p0, p1 = np.array([cx, cy]) - d * L / 2, np.array([cx, cy]) + d * L / 2
            out.append({"type": name, "start": p0.tolist(), "end": p1.tolist(), "center": [cx, cy], "width_pixels": L,
                        "evidence": {}})
    return out


# ---------------------------------------------------------------- C: fusion into the engine
class Fusion:
    """Patch engine.wall_inference.infer_walls so the wall layer uses the model's evidence."""

    def __init__(self, prob: np.ndarray):
        self.prob = prob

    def __enter__(self):
        import engine.structure as st
        import engine.wall_inference as wi
        self.wi, self.orig = wi, wi.infer_walls
        prob = self.prob

        def fused(ink):
            res = self.orig(ink)
            h, w = ink.shape
            P = [cv2.resize(prob[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(prob.shape[0])]
            pw, po = P[CLS["wall"]], P[CLS["other"]]
            mask = res.mask.copy()
            add = (pw > 0.5) & (ink > 0) & (mask == 0)
            mask[add] = 255
            n, lab, stt, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
            if n > 1:
                sum_o = np.bincount(lab.ravel(), weights=po.ravel(), minlength=n)
                sum_w = np.bincount(lab.ravel(), weights=pw.ravel(), minlength=n)
                area = np.maximum(stt[:, cv2.CC_STAT_AREA], 1)
                furniture = (sum_o / area > 0.5) & (sum_w / area < 0.25)
                furniture[0] = False
                mask[np.isin(lab, np.flatnonzero(furniture))] = 0
            res.mask = mask
            res.primary = mask
            return res
        wi.infer_walls = fused
        return self

    def __exit__(self, *exc):
        self.wi.infer_walls = self.orig


def retype(openings: list[dict], prob: np.ndarray) -> int:
    changed = 0
    h, w = prob.shape[1:]
    for o in openings:
        p0, p1 = np.array(o["start"], float), np.array(o["end"], float)
        t = float(o["evidence"].get("wall_thickness_px") or 6)
        band = np.zeros((h, w), np.uint8)
        cv2.line(band, tuple(int(v) for v in p0), tuple(int(v) for v in p1), 1, max(2, int(t)))
        m = band > 0
        if not m.any():
            continue
        v = {k: float(prob[CLS[k]][m].mean()) for k in ("door", "window", "passage")}
        if sum(v.values()) < 0.3:
            continue
        best = max(v, key=v.get)
        new = {"door": "door", "window": "window", "passage": "opening"}[best]
        if v[best] >= 0.25 and new != o["type"]:
            o["type"] = new
            changed += 1
    return changed


# ---------------------------------------------------------------- metrics
def opening_metrics(preds, gts):
    from benchmark.evaluate import OpeningStats
    st = OpeningStats()
    st.add(preds, gts)
    m = st.metrics()
    return st, m


def f1(p, r):
    return None if not p or not r else round(2 * p * r / (p + r), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="norm")
    ap.add_argument("--save")
    ap.add_argument("--skip-c", action="store_true")
    args = ap.parse_args()
    from benchmark import ocr_cache
    from benchmark.evaluate import OpeningStats
    from benchmark.holdout_inputs import load
    from benchmark.openings_eval import gt_openings
    from benchmark.real_plans import DEFAULT_DIR, FIXTURE, full_room_metrics, load_plan, room_metrics
    from engine.analyzer import FloorPlanAnalyzer
    from engine.structure import analyze_structure

    ocr_cache.enable()
    manifest = [m for m in json.loads((REAL / "manifest.json").read_text()) if m["variant"] == args.variant]
    dev = json.loads(FIXTURE.read_text())["plans"]
    pdf = json.loads((ROOT / "benchmark/fixtures/pdf_plans.json").read_text())["plans"]
    rows = {}
    tot = {k: OpeningStats() for k in "ABC"}
    agg: dict = {}

    def add(key, v):
        agg[key] = agg.get(key, 0) + v

    for item in manifest:
        case = item["case"]
        if not Path(item["out"]).exists():
            continue
        spec = dev.get(case) or pdf.get(case)
        if spec is None:
            continue
        if case in dev:
            img = load_plan(case, spec, DEFAULT_DIR)
            ev = None
        else:
            p, page = case.split(":")
            img, ev, _ = load(DEFAULT_DIR / p, int(page))
        prob = load_prob(item["out"])
        arg = prob.argmax(0)
        s = analyze_structure(img)
        row = {"t": item["wall_t"], "scale": item["scale"]}
        # walls / furniture at labelled points
        for kind, pts in (("protected", spec.get("protected", [])), ("nonstructural", spec.get("nonstructural", []))):
            for q in pts:
                x, y = q["point"]
                add(f"{kind}_n", 1)
                add(f"{kind}_A_wall", at(s.wall_mask > 0, x, y))
                add(f"{kind}_B_wall", at(arg == CLS["wall"], x, y))
                if kind == "nonstructural":
                    add(f"{kind}_B_other", at(arg == CLS["other"], x, y))
        rooms = spec["rooms"]
        # B rooms
        lab_b = model_rooms(prob)
        rb = room_metrics(lab_b, rooms)
        # A rooms (full analyzer)
        t0 = time.time()
        ra = FloorPlanAnalyzer().analyze(img, case, **({"text_evidence": ev} if ev else {}))
        ta = time.time() - t0
        fa = full_room_metrics(ra, rooms)
        row.update({"A_rooms": [fa["recovered"], fa["merged"], fa["missed"]], "B_rooms": [rb["separated"], rb["merged"], rb["missed"]],
                    "points": len(rooms), "A_seconds": round(ta, 1)})
        for k, v in (("A", fa["recovered"]), ("B", rb["separated"])):
            add(f"rooms_{k}", v)
        add("rooms_n", len(rooms))
        # C: hybrid
        if not args.skip_c:
            t0 = time.time()
            with Fusion(prob):
                rc = FloorPlanAnalyzer().analyze(img, case, **({"text_evidence": ev} if ev else {}))
            changed = retype(rc["openings"], prob)
            row["C_seconds"] = round(time.time() - t0, 1)
            fc = full_room_metrics(rc, rooms)
            row["C_rooms"] = [fc["recovered"], fc["merged"], fc["missed"]]
            row["C_retyped"] = changed
            add("rooms_C", fc["recovered"])
        gts = gt_openings(spec) if case in dev else None
        if gts:
            preds = {"A": ra["openings"], "B": model_openings(prob, float(s.wall_thickness or 6))}
            if not args.skip_c:
                preds["C"] = rc["openings"]
            for k, ops in preds.items():
                st, m = opening_metrics(ops, gts)
                tot[k].merge(st)
                row[f"{k}_openings"] = {x: (round(m[x], 2) if isinstance(m[x], float) else m[x]) for x in
                                        ("door_precision", "door_recall", "window_precision", "window_recall",
                                         "false_door_or_window", "missed_candidates", "passages_found")}
        rows[case] = row
        print(case, json.dumps(row), flush=True)
    summary = {"rooms": {k: round(agg.get(f"rooms_{k}", 0) / max(1, agg.get("rooms_n", 1)), 3) for k in "ABC"},
               "points": agg}
    for k, st in tot.items():
        m = st.metrics()
        summary[f"openings_{k}"] = {
            "door_P": m["door_precision"] and round(m["door_precision"], 3), "door_R": m["door_recall"] and round(m["door_recall"], 3),
            "door_F1": f1(m["door_precision"], m["door_recall"]),
            "window_P": m["window_precision"] and round(m["window_precision"], 3), "window_R": m["window_recall"] and round(m["window_recall"], 3),
            "window_F1": f1(m["window_precision"], m["window_recall"]),
            "classification": m["classification_accuracy"] and round(m["classification_accuracy"], 3),
            "false": m["false_door_or_window"], "missed": m["missed_candidates"],
            "passages_found": m["passages_found"], "passages_called_door": m["passages_called_door"]}
    print("SUMMARY", json.dumps(summary, indent=1))
    if args.save:
        Path(args.save).write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
