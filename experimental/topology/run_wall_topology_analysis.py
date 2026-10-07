from pathlib import Path
import json
import math

import cv2

from engine.analyzer import FloorPlanAnalyzer
from engine.geometry_v2.groups import WallGroupBuilder
from engine.geometry_v2.segments import WallSegment
from engine.walls.segments import WallSegmentExtractor


IMAGE = Path("test_floorplan.png")
OUT_DIR = Path("output_v2")
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Capture the actual wall segments used by Analyzer
# ============================================================

captured_segments = []

_original_detect = WallSegmentExtractor.detect


def capture_detect(self):
    segments = _original_detect(self)
    captured_segments.extend(segments)
    return segments


WallSegmentExtractor.detect = capture_detect


image = cv2.imread(str(IMAGE))

if image is None:
    raise SystemExit(f"Could not read {IMAGE}")


analyzer = FloorPlanAnalyzer()

analyzer.analyze(
    image,
    source_name=IMAGE.name,
)


# ============================================================
# Normalize extractor output
# ============================================================

def normalize_segment(segment):
    if isinstance(segment, WallSegment):
        return segment

    start = (
        float(segment["start"][0]),
        float(segment["start"][1]),
    )

    end = (
        float(segment["end"][0]),
        float(segment["end"][1]),
    )

    orientation = segment.get(
        "orientation",
        "horizontal",
    )

    try:
        from engine.geometry_v2.orientation import Orientation

        if hasattr(
            Orientation,
            str(orientation).upper(),
        ):
            orientation = getattr(
                Orientation,
                str(orientation).upper(),
            )
    except Exception:
        pass

    return WallSegment(
        start=start,
        end=end,
        orientation=orientation,
        length=float(
            segment.get(
                "length",
                math.hypot(
                    end[0] - start[0],
                    end[1] - start[1],
                ),
            )
        ),
        thickness=float(
            segment.get(
                "thickness",
                1.0,
            )
        ),
        confidence=float(
            segment.get(
                "confidence",
                1.0,
            )
        ),
    )


segments = [
    normalize_segment(segment)
    for segment in captured_segments
]


# ============================================================
# Helpers
# ============================================================

def orientation(segment):
    value = segment.orientation

    if hasattr(value, "value"):
        value = value.value

    return str(value).lower()


def point(segment, name):
    p = getattr(segment, name)

    return (
        float(p[0]),
        float(p[1]),
    )


def segment_length(segment):
    a = point(segment, "start")
    b = point(segment, "end")

    return math.hypot(
        b[0] - a[0],
        b[1] - a[1],
    )


def gap_length(gap):
    a = tuple(gap.start)
    b = tuple(gap.end)

    return math.hypot(
        float(b[0]) - float(a[0]),
        float(b[1]) - float(a[1]),
    )


def classify_gap(width, thickness):
    ratio = width / max(thickness, 1)

    if ratio < 1.5:
        return "small_gap"

    if ratio < 4:
        return "possible_opening"

    if ratio < 8:
        return "large_opening"

    return "structural_gap"


# ============================================================
# Basic statistics
# ============================================================

horizontal = 0
vertical = 0
other = 0

for segment in segments:
    value = orientation(segment)

    if "horizontal" in value:
        horizontal += 1

    elif "vertical" in value:
        vertical += 1

    else:
        other += 1


# ============================================================
# Wall groups
# ============================================================

groups = WallGroupBuilder().build(
    segments
)


# ============================================================
# Gap analysis
# ============================================================

gap_records = []

for group in groups:

    group_orientation = getattr(
        group.orientation,
        "value",
        group.orientation,
    )

    group_orientation = str(
        group_orientation
    )

    thickness = float(
        group.thickness
    )

    for gap_index, gap in enumerate(
        group.gaps,
        start=1,
    ):

        width = gap_length(gap)

        start = (
            float(gap.start[0]),
            float(gap.start[1]),
        )

        end = (
            float(gap.end[0]),
            float(gap.end[1]),
        )

        gap_records.append(
            {
                "group_id": group.id,
                "orientation": group_orientation,
                "gap_index": gap_index,
                "width": round(width, 2),
                "wall_thickness": round(
                    thickness,
                    2,
                ),
                "ratio": round(
                    width / max(thickness, 1),
                    2,
                ),
                "classification": classify_gap(
                    width,
                    thickness,
                ),
                "start": start,
                "end": end,
                "midpoint": (
                    (start[0] + end[0]) / 2,
                    (start[1] + end[1]) / 2,
                ),
            }
        )


# ============================================================
# H/V junction detection
# ============================================================

def intersection(a, b, tolerance=10):

    ao = orientation(a)
    bo = orientation(b)

    a0 = point(a, "start")
    a1 = point(a, "end")

    b0 = point(b, "start")
    b1 = point(b, "end")


    # Horizontal A / Vertical B

    if (
        "horizontal" in ao
        and "vertical" in bo
    ):

        y = (
            a0[1] + a1[1]
        ) / 2

        x = (
            b0[0] + b1[0]
        ) / 2

        if (
            min(a0[0], a1[0])
            - tolerance
            <= x
            <= max(a0[0], a1[0])
            + tolerance
            and
            min(b0[1], b1[1])
            - tolerance
            <= y
            <= max(b0[1], b1[1])
            + tolerance
        ):
            return x, y


    # Vertical A / Horizontal B

    if (
        "vertical" in ao
        and "horizontal" in bo
    ):

        return intersection(
            b,
            a,
            tolerance,
        )


    return None


junctions = []


for i in range(len(segments)):

    for j in range(
        i + 1,
        len(segments),
    ):

        p = intersection(
            segments[i],
            segments[j],
        )

        if p is None:
            continue


        existing = None

        for candidate in junctions:

            distance = math.hypot(
                p[0] - candidate["x"],
                p[1] - candidate["y"],
            )

            if distance <= 10:
                existing = candidate
                break


        if existing is not None:

            existing["segments"].update(
                [i, j]
            )

        else:

            junctions.append(
                {
                    "x": p[0],
                    "y": p[1],
                    "segments": {
                        i,
                        j,
                    },
                }
            )


junctions.sort(
    key=lambda item: (
        item["y"],
        item["x"],
    )
)


# ============================================================
# Terminal report
# ============================================================

print()
print("=" * 70)
print("WALL TOPOLOGY ANALYSIS v1")
print("=" * 70)

print(
    f"Image size          : "
    f"{image.shape[1]} x {image.shape[0]}"
)

print(
    f"Wall segments       : "
    f"{len(segments)}"
)

print(
    f"Wall groups         : "
    f"{len(groups)}"
)

print()
print("SEGMENT DISTRIBUTION")
print("-" * 70)

print(
    f"Horizontal          : "
    f"{horizontal}"
)

print(
    f"Vertical            : "
    f"{vertical}"
)

print(
    f"Other               : "
    f"{other}"
)


print()
print("GAP ANALYSIS")
print("-" * 70)

print(
    f"Total gaps          : "
    f"{len(gap_records)}"
)


for record in gap_records:

    print(
        f"{record['group_id']:<8} "
        f"{record['orientation']:<12} "
        f"width={record['width']:>6.1f} "
        f"thickness={record['wall_thickness']:>5.1f} "
        f"ratio={record['ratio']:>4.1f} "
        f"-> "
        f"{record['classification']}"
    )


print()
print("JUNCTION ANALYSIS")
print("-" * 70)

print(
    f"Junctions          : "
    f"{len(junctions)}"
)


for index, junction in enumerate(
    junctions,
    start=1,
):

    print(
        f"J{index:03d} "
        f"("
        f"{junction['x']:.0f}, "
        f"{junction['y']:.0f}"
        f") "
        f"segments="
        f"{len(junction['segments'])}"
    )


# ============================================================
# Visualization
# ============================================================

canvas = image.copy()


# Walls
for segment in segments:

    a = point(
        segment,
        "start",
    )

    b = point(
        segment,
        "end",
    )

    cv2.line(
        canvas,
        (
            int(round(a[0])),
            int(round(a[1])),
        ),
        (
            int(round(b[0])),
            int(round(b[1])),
        ),
        (255, 0, 0),
        3,
        cv2.LINE_AA,
    )


# Gaps
for record in gap_records:

    p1 = (
        int(round(record["start"][0])),
        int(round(record["start"][1])),
    )

    p2 = (
        int(round(record["end"][0])),
        int(round(record["end"][1])),
    )

    cv2.line(
        canvas,
        p1,
        p2,
        (0, 180, 255),
        8,
        cv2.LINE_AA,
    )


# Junctions
for junction in junctions:

    center = (
        int(round(junction["x"])),
        int(round(junction["y"])),
    )

    cv2.circle(
        canvas,
        center,
        8,
        (0, 255, 0),
        -1,
        cv2.LINE_AA,
    )


visualization_path = (
    OUT_DIR /
    "wall-topology-analysis-v1.png"
)

cv2.imwrite(
    str(visualization_path),
    canvas,
)


# ============================================================
# JSON
# ============================================================

json_data = {
    "image": {
        "width": image.shape[1],
        "height": image.shape[0],
    },
    "segments": {
        "total": len(segments),
        "horizontal": horizontal,
        "vertical": vertical,
        "other": other,
    },
    "groups": len(groups),
    "gaps": gap_records,
    "junctions": [
        {
            "x": item["x"],
            "y": item["y"],
            "segments": sorted(
                item["segments"]
            ),
        }
        for item in junctions
    ],
}


json_path = (
    OUT_DIR /
    "wall-topology-analysis-v1.json"
)

json_path.write_text(
    json.dumps(
        json_data,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    f"Visualization       : "
    f"{visualization_path}"
)

print(
    f"JSON                : "
    f"{json_path}"
)

print("=" * 70)
