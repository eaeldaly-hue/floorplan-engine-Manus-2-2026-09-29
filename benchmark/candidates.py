"""Candidate-level check (before classification): which GT openings have a candidate?

    .venv/bin/python -m benchmark.candidates                # sample + a few families
    .venv/bin/python -m benchmark.candidates --families all --count 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from benchmark.evaluate import OpeningStats, StructureStats, match
from benchmark.generator import FAMILIES, family_specs, generate
from engine.structure import analyze_structure

ROOT = Path(__file__).resolve().parents[1]


def as_preds(structure):
    return [{"start": list(c.start), "end": list(c.end), "type": "opening", "center": [round(v) for v in c.center]}
            for c in structure.candidates]


def sample_report(verbose=True):
    truth = json.loads((ROOT / "benchmark" / "fixtures" / "test_floorplan_openings.json").read_text())
    image = cv2.imread(str(ROOT / truth["image"]))
    for o in truth["openings"]:
        o.setdefault("wall_thickness_px", 38.0 if o["id"].startswith("W") else 30.0)
    s = analyze_structure(image)
    preds = as_preds(s)
    pairs = match(preds, truth["openings"])
    hit = {truth["openings"][j]["id"] for _, j, _ in pairs}
    missing = [o["id"] + " " + o["where"] for o in truth["openings"] if o["id"] not in hit]
    false = [preds[i]["center"] for i in range(len(preds)) if i not in {p for p, _, _ in pairs}]
    print(f"sample: {len(hit)}/{len(truth['openings'])} GT matched, {len(false)} false candidates, {len(s.spaces)} spaces")
    if verbose:
        print("  missing:", missing)
        print("  false:", false)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default="door_swing,door_no_swing_outline,window_sample_style,window_adjacent,noise,mixed,ambiguous_passages")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--errors", action="store_true")
    args = parser.parse_args()
    sample_report()
    names = list(FAMILIES) if args.families == "all" else args.families.split(",")
    total, stotal = OpeningStats(), StructureStats()
    for name in names:
        o, st = OpeningStats(), StructureStats()
        for spec in family_specs(name, args.count):
            plan = generate(spec)
            s = analyze_structure(plan.image)
            o.add(as_preds(s), plan.openings, f"{name}#{spec.seed}")
            st.add(s.space_labels, plan.room_labels, s.wall_mask, plan.wall_mask, s.adjacency_pairs(), plan.adjacency)
        m, sm = o.metrics(), st.metrics()
        total.merge(o); stotal.merge(st)
        print(f"{name:<24} cand_recall {m['candidate_recall']:.3f}  passages {m['passages_found']}/{m['passages']}  false {m['false_candidates']:3d}  "
              f"rooms {sm['room_recall_iou70']:.3f} iou {sm['mean_best_room_iou']:.3f} merged {sm['merged_spaces']} split {sm['split_rooms']} falsesp {sm['false_spaces']} walls {sm['wall_iou']:.3f} fab {sm['fabricated_wall_fraction']:.3f} adjP {sm['adjacency_precision'] or 0:.2f} adjR {sm['adjacency_recall'] or 0:.2f}")
        if args.errors:
            for e in o.errors:
                if "missed" in e or "false_candidate" in e:
                    print("     ", e)
    m, sm = total.metrics(), stotal.metrics()
    print(f"{'ALL':<24} cand_recall {m['candidate_recall']:.3f}  false {m['false_candidates']}  rooms {sm['room_recall_iou70']:.3f} iou {sm['mean_best_room_iou']:.3f} adjP {sm['adjacency_precision']:.2f} adjR {sm['adjacency_recall']:.2f}")


if __name__ == "__main__":
    main()
