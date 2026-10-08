"""Open-plan zones: do object clusters separate the functional zones of one open space? (research)

For every GT open-plan group (several GT room points of different functions in ONE space of the
current engine), the objects inside that space are clustered and the free floor is split by the
nearest cluster. A group is 'resolved' when its GT points of different functions fall in different
zones; 'zones' reports how many zones that took (over-fragmentation check).

Objects:  A = drawing ink inside the space (current engine: no object typing)
          B = the research model's 'other' (furniture / fixtures) class
Text (OCR boxes) is removed from both, so labels cannot place the zones.

    .venv/bin/python research/perception/eval_zones.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REAL = ROOT / "output/perception/real"


def zones(space: np.ndarray, objects: np.ndarray, gap: float, min_area: float):
    """Cluster object components (single linkage by dilation) and split the space by nearest cluster."""
    obj = (objects & space).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(obj, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    obj = np.isin(lab, np.flatnonzero(keep)).astype(np.uint8)
    k = max(3, int(round(gap)) | 1)
    grown = cv2.dilate(obj, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    nc, clus = cv2.connectedComponents(grown, connectivity=8)
    if nc <= 1:
        return np.zeros(space.shape, np.int32), 0
    seeds = np.where(obj > 0, clus, 0).astype(np.int32)
    # nearest cluster for every free pixel (distance transform with labels)
    inv = (seeds == 0).astype(np.uint8)
    _, lbl = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(seeds)
    lut = np.zeros(lbl.max() + 1, np.int32)
    lut[lbl[ys, xs]] = seeds[ys, xs]
    z = lut[lbl]
    z[~space] = 0
    return z, nc - 1


def main() -> int:
    from benchmark import ocr_cache
    from benchmark.real_plans import DEFAULT_DIR, FIXTURE, load_plan
    from engine.analysis.ocr import extract_ocr
    from engine.analyzer import FloorPlanAnalyzer
    from engine.structure import analyze_structure

    ocr_cache.enable()
    doc = json.loads(FIXTURE.read_text())["plans"]
    manifest = {m["case"]: m for m in json.loads((REAL / "manifest.json").read_text()) if m["variant"] == "norm"}
    tot = {"groups": 0, "A": 0, "B": 0, "A_zones": 0, "B_zones": 0, "functions": 0}
    for name, spec in doc.items():
        groups: dict = {}
        for r in spec.get("rooms", []):
            if r.get("group"):
                groups.setdefault(r["group"], []).append(r)
        groups = {g: v for g, v in groups.items() if len({x["name"] for x in v}) >= 2}
        if not groups or name not in manifest or not Path(manifest[name]["out"]).exists():
            continue
        img = load_plan(name, spec, DEFAULT_DIR)
        r = FloorPlanAnalyzer().analyze(img, name)
        s = analyze_structure(img)
        t = float(s.wall_thickness or 6)
        idx = np.zeros(img.shape[:2], np.int32)
        for k, sp in enumerate(r["building"]["spaces"], 1):
            cv2.fillPoly(idx, [np.array(sp["polygon"], np.int32).reshape(-1, 1, 2)], k)
        text = np.zeros(img.shape[:2], bool)
        for b in extract_ocr(img).boxes:
            text[max(0, b.y - 3):b.y + b.height + 3, max(0, b.x - 3):b.x + b.width + 3] = True
        prob = np.load(manifest[name]["out"])["prob"]
        other_b = (prob.argmax(0) == 6) & ~text
        ink_a = (s.symbol_ink > 0) & (s.wall_mask == 0) & ~text
        for g, members in groups.items():
            ks = {int(idx[m["point"][1], m["point"][0]]) for m in members}
            if len(ks) != 1 or 0 in ks:
                continue                                        # the engine already split (or lost) it
            space = idx == ks.pop()
            tot["groups"] += 1
            tot["functions"] += len({m["name"] for m in members})
            res = {}
            for key, objects in (("A", ink_a), ("B", other_b)):
                z, nz = zones(space, objects, gap=3.0 * t, min_area=4.0 * t * t)
                labs = {}
                for m in members:
                    labs.setdefault(m["name"], set()).add(int(z[m["point"][1], m["point"][0]]))
                names = list(labs)
                ok = all(not (labs[a] & labs[b]) for i, a in enumerate(names) for b in names[i + 1:]) and all(0 not in v for v in labs.values())
                tot[key] += ok
                tot[f"{key}_zones"] += nz
                res[key] = (ok, nz)
            print(name, g, [m["name"] for m in members], "A", res["A"], "B", res["B"], flush=True)
    print("TOTAL", json.dumps(tot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
