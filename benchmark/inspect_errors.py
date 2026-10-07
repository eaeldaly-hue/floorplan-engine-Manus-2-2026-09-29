"""Print misclassified openings with their evidence for one family.

    .venv/bin/python -m benchmark.inspect_errors door_sliding --count 4
"""

import argparse

from benchmark.evaluate import match
from benchmark.generator import family_specs, generate
from engine.opening_detection import classify_openings
from engine.structure import analyze_structure

KEYS = ("symbol", "glazing_lines", "face_lines", "frame_marks", "arc_score", "double_arc_score", "leaf_score",
        "sliding_shift", "room_relation", "candidate_source")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--salt", type=int, default=0)
    args = parser.parse_args()
    shown = 0
    for k, spec in enumerate(family_specs(args.family, args.count, args.salt)):
        plan = generate(spec)
        s = analyze_structure(plan.image)
        ops = classify_openings(plan.image, s)
        for i, j, _ in match(ops, plan.openings):
            g, o = plan.openings[j], ops[i]
            if g.kind in ("door", "window") and o["type"] != g.kind and shown < args.limit:
                shown += 1
                ev = o["evidence"]
                print(f"[{args.family} #{k} s={spec.style.px_per_cm} lw={spec.style.line_px}] {g.id} {g.kind}/{g.style} "
                      f"w={g.width_px:.0f} t={g.wall_thickness_px:.1f} -> {o['type']} | " + " ".join(f"{key}={ev[key]}" for key in KEYS))


if __name__ == "__main__":
    main()
