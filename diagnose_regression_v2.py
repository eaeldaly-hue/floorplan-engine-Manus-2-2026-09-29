from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
BASELINE = Path("/tmp/floorplan-baseline")
IMAGE = ROOT / "test_floorplan.png"
REPORT = ROOT / "regression_report_v2.txt"


def git_head(path: Path):
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
    )
    return r.stdout.strip()


def run_analyzer(project: Path):
    code = r'''
from pathlib import Path
from PIL import Image
from engine.analyzer import analyze_floor_plan

image = Image.open("test_floorplan.png").convert("RGB")

try:
    result = analyze_floor_plan(
        image,
        source_name="test_floorplan.png",
    )
except TypeError:
    result = analyze_floor_plan(image)

if hasattr(result, "to_dict"):
    result = result.to_dict()

print("=== RESULT ===")

if not isinstance(result, dict):
    print("RESULT_TYPE:", type(result).__name__)
    raise SystemExit(0)

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

print("\n=== ROOMS ===")
print("ROOM RECORDS:", len(rooms))

linked = 0
unlinked = 0

for room in rooms:
    room_id = room.get("id")
    name = room.get("name")
    label = room.get("label")
    space = room.get("space_id")

    if space:
        linked += 1
    else:
        unlinked += 1

    print(
        f"{room_id} | "
        f"name={name!r} | "
        f"label={label!r} | "
        f"space={space!r}"
    )

print("\nrooms linked to spaces:", linked)
print("rooms without space:", unlinked)

print("\n=== OPENINGS ===")

openings = result.get("openings", [])

print("OPENINGS:", len(openings))

for opening in openings:
    print(
        opening.get("id"),
        "|",
        opening.get("type"),
        "|",
        "confidence=",
        opening.get("confidence"),
    )

print("\n=== WARNINGS ===")

for warning in result.get("warnings", []):
    print(warning)

print("\n=== RESULT KEYS ===")
print(sorted(result.keys()))
'''

    r = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project,
        text=True,
        capture_output=True,
    )

    return r


def extract(text, key):
    for line in text.splitlines():
        if line.startswith(key):
            return line
    return "NOT FOUND"


def main():

    print("=" * 80)
    print("FLOOR PLAN REGRESSION DIAGNOSTIC V2")
    print("=" * 80)

    print("\nCurrent project:")
    print(ROOT)

    print("\nBaseline:")
    print(BASELINE)

    print("\nCurrent commit:")
    print(git_head(ROOT))

    print("\nBaseline commit:")
    print(git_head(BASELINE))

    print("\n" + "=" * 80)
    print("CURRENT VERSION")
    print("=" * 80)

    current = run_analyzer(ROOT)

    print(current.stdout)

    if current.stderr:
        print("\nCURRENT ERRORS:")
        print(current.stderr)

    print("\n" + "=" * 80)
    print("BASELINE VERSION")
    print("=" * 80)

    baseline = run_analyzer(BASELINE)

    print(baseline.stdout)

    if baseline.stderr:
        print("\nBASELINE ERRORS:")
        print(baseline.stderr)

    report = []

    report.append("=" * 80)
    report.append("FLOOR PLAN REGRESSION REPORT V2")
    report.append("=" * 80)

    report.append("")
    report.append("CURRENT COMMIT:")
    report.append(git_head(ROOT))

    report.append("")
    report.append("BASELINE COMMIT:")
    report.append(git_head(BASELINE))

    report.append("")
    report.append("=" * 80)
    report.append("CURRENT")
    report.append("=" * 80)
    report.append(current.stdout)

    if current.stderr:
        report.append("\nERRORS:\n")
        report.append(current.stderr)

    report.append("")
    report.append("=" * 80)
    report.append("BASELINE")
    report.append("=" * 80)
    report.append(baseline.stdout)

    if baseline.stderr:
        report.append("\nERRORS:\n")
        report.append(baseline.stderr)

    report.append("")
    report.append("=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)

    keys = [
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
        report.append("")
        report.append(key)

        report.append(
            "CURRENT : " +
            extract(current.stdout, key)
        )

        report.append(
            "BASELINE: " +
            extract(baseline.stdout, key)
        )

    REPORT.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print("\nReport:")
    print(REPORT)

    print("\nNo source files were modified.")
    print("No git reset/checkout/restore was performed.")


if __name__ == "__main__":
    main()
