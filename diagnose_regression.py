from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
IMAGE = ROOT / "test_floorplan.png"
BASELINE = Path("/tmp/floorplan-baseline")
REPORT = ROOT / "regression_report.txt"


def run(cmd, cwd=None):
    print("\n$", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    return result


def python_test(project_dir: Path, code: str):
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_dir,
        text=True,
        capture_output=True,
    )

    return result.stdout, result.stderr, result.returncode


def collect(project_dir: Path, label: str):
    code = r'''
import json
from pathlib import Path

from PIL import Image

from engine.analysis.ocr import extract_ocr, group_room_labels
from engine.walls.mask import WallMaskBuilder
from engine.walls.segments import WallSegmentExtractor
from engine.geometry_v2.segments import WallSegmentBuilder
from engine.geometry_v2.groups import WallGroupBuilder
from engine.analyzer import analyze_floor_plan

image_path = Path("test_floorplan.png")
image = Image.open(image_path).convert("RGB")

print("=== IMAGE ===")
print("size:", image.size)

# ---------------------------------------------------------
# OCR
# ---------------------------------------------------------
print("\n=== OCR ===")

ocr = extract_ocr(image)

try:
    room_lines = group_room_labels(ocr)
except TypeError:
    room_lines = group_room_labels(ocr.room_labels)

print("ocr boxes:", len(getattr(ocr, "boxes", [])))
print("ocr room labels:", len(room_lines))

for i, r in enumerate(room_lines, 1):
    text = getattr(r, "text", None) or r.get("text", "")
    center = getattr(r, "center", None)

    if center is None:
        center = r.get("center")

    print(f"ROOM_LABEL {i:02d}: {text!r} center={center}")

# ---------------------------------------------------------
# WALL MASK
# ---------------------------------------------------------
print("\n=== WALL MASK ===")

wall_mask = WallMaskBuilder(image).build()

try:
    mask_shape = wall_mask.shape
except Exception:
    mask_shape = None

print("wall mask shape:", mask_shape)

# ---------------------------------------------------------
# RAW WALL SEGMENTS
# ---------------------------------------------------------
print("\n=== WALL SEGMENTS ===")

extractor = WallSegmentExtractor(wall_mask)

try:
    raw_segments = extractor.detect(include_diagonal=True)
except TypeError:
    raw_segments = extractor.detect()

print("raw segments:", len(raw_segments))

# ---------------------------------------------------------
# V2 SEGMENTS
# ---------------------------------------------------------
print("\n=== V2 SEGMENTS ===")

try:
    builder = WallSegmentBuilder()
    v2_segments = builder.build(raw_segments)
except Exception as exc:
    print("V2 builder error:", repr(exc))
    v2_segments = raw_segments

print("v2 segments:", len(v2_segments))

# ---------------------------------------------------------
# WALL GROUPS
# ---------------------------------------------------------
print("\n=== WALL GROUPS ===")

try:
    group_builder = WallGroupBuilder()
    groups = group_builder.build(v2_segments)
except Exception as exc:
    print("WallGroupBuilder error:", repr(exc))
    groups = []

print("wall groups:", len(groups))

for g in groups:
    gid = getattr(g, "id", "?")
    orientation = getattr(g, "orientation", "?")
    thickness = getattr(g, "thickness", "?")
    segments = getattr(g, "segments", [])
    gaps = getattr(g, "gaps", [])

    print(
        f"GROUP {gid}: "
        f"orientation={orientation} "
        f"thickness={thickness} "
        f"segments={len(segments)} "
        f"gaps={len(gaps)}"
    )

# ---------------------------------------------------------
# ANALYZER
# ---------------------------------------------------------
print("\n=== ANALYZER ===")

try:
    result = analyze_floor_plan(image, source_name="test_floorplan.png")
except TypeError:
    result = analyze_floor_plan(image)

if hasattr(result, "to_dict"):
    result = result.to_dict()

print("result type:", type(result).__name__)

if isinstance(result, dict):

    print("room_count:", result.get("room_count"))
    print("unlabeled_space_count:", result.get("unlabeled_space_count"))
    print("opening_count:", result.get("opening_count"))
    print("door_count:", result.get("door_count"))
    print("window_count:", result.get("window_count"))
    print(
        "unclassified_opening_count:",
        result.get("unclassified_opening_count"),
    )

    rooms = result.get("rooms", [])

    print("\nROOM RECORDS:", len(rooms))

    linked = 0
    unlinked = 0

    for room in rooms:
        room_id = room.get("id")
        name = room.get("name")
        label = room.get("label")
        space_id = room.get("space_id")

        if space_id:
            linked += 1
        else:
            unlinked += 1

        print(
            f"{room_id}: "
            f"name={name!r} "
            f"label={label!r} "
            f"space={space_id!r}"
        )

    print("\nrooms linked to spaces:", linked)
    print("rooms without space:", unlinked)

    openings = result.get("openings", [])

    print("\nOPENINGS:", len(openings))

    for opening in openings:
        print(
            opening.get("id"),
            "|",
            opening.get("type"),
            "| confidence=",
            opening.get("confidence"),
        )

    # Save compact JSON
    Path("regression_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\nSaved: regression_result.json")

else:
    print("Analyzer did not return a dictionary.")
'''

    out, err, code_rc = python_test(project_dir, code)

    return {
        "label": label,
        "stdout": out,
        "stderr": err,
        "returncode": code_rc,
    }


def git_commit(project_dir):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_dir,
        text=True,
        capture_output=True,
    )

    return result.stdout.strip()


def main():
    print("=" * 80)
    print("FLOOR PLAN REGRESSION DIAGNOSTIC")
    print("=" * 80)

    print("\nProject:", ROOT)
    print("Image:", IMAGE)

    if not IMAGE.exists():
        print("\nERROR: test_floorplan.png not found.")
        sys.exit(1)

    if not BASELINE.exists():
        print("\nERROR: baseline worktree not found:")
        print(BASELINE)
        print("\nExpected baseline:")
        print("b34c7de")
        sys.exit(1)

    current_commit = git_commit(ROOT)
    baseline_commit = git_commit(BASELINE)

    print("\nCurrent commit :", current_commit)
    print("Baseline commit:", baseline_commit)

    # -----------------------------------------------------
    # CURRENT
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING CURRENT VERSION")
    print("=" * 80)

    current = collect(ROOT, "CURRENT")

    # -----------------------------------------------------
    # BASELINE
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING BASELINE VERSION")
    print("=" * 80)

    baseline = collect(BASELINE, "BASELINE")

    # -----------------------------------------------------
    # REPORT
    # -----------------------------------------------------
    report = []

    report.append("=" * 80)
    report.append("FLOOR PLAN REGRESSION REPORT")
    report.append("=" * 80)
    report.append("")
    report.append(f"Current commit : {current_commit}")
    report.append(f"Baseline commit: {baseline_commit}")
    report.append("")

    report.append("=" * 80)
    report.append("CURRENT OUTPUT")
    report.append("=" * 80)
    report.append(current["stdout"])

    if current["stderr"]:
        report.append("\nCURRENT STDERR\n")
        report.append(current["stderr"])

    report.append("\n" + "=" * 80)
    report.append("BASELINE OUTPUT")
    report.append("=" * 80)
    report.append(baseline["stdout"])

    if baseline["stderr"]:
        report.append("\nBASELINE STDERR\n")
        report.append(baseline["stderr"])

    report.append("\n" + "=" * 80)
    report.append("QUICK COMPARISON")
    report.append("=" * 80)

    def extract_value(text, key):
        for line in text.splitlines():
            if line.startswith(key):
                return line
        return f"{key}: NOT FOUND"

    keys = [
        "size:",
        "ocr boxes:",
        "ocr room labels:",
        "raw segments:",
        "v2 segments:",
        "wall groups:",
        "room_count:",
        "unlabeled_space_count:",
        "opening_count:",
        "door_count:",
        "window_count:",
        "unclassified_opening_count:",
        "ROOM RECORDS:",
        "rooms linked to spaces:",
        "rooms without space:",
        "OPENINGS:",
    ]

    for key in keys:
        c = extract_value(current["stdout"], key)
        b = extract_value(baseline["stdout"], key)

        report.append("")
        report.append(key)
        report.append("  CURRENT : " + c)
        report.append("  BASELINE: " + b)

    REPORT.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC FINISHED")
    print("=" * 80)

    print("\nReport saved to:")
    print(REPORT)

    print("\nCurrent JSON saved to:")
    print(ROOT / "regression_result.json")

    print("\nIMPORTANT:")
    print("No source code was modified by this diagnostic.")
    print("No git reset/checkout/restore was performed.")

    print("\nNext step:")
    print("Send me the COMPLETE terminal output starting from")
    print("FLOOR PLAN REGRESSION DIAGNOSTIC")
    print("or at minimum send regression_report.txt content.")


if __name__ == "__main__":
    main()
