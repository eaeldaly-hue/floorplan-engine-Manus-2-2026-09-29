"""Room-centric floor-plan analysis built on OpenCV and local Tesseract OCR."""

from __future__ import annotations

import math
import os
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np

from .preprocessing.text_mask import TextMasker
from .structure import EXTERIOR, analyze_structure
from .plan import adapter as plan_adapter
from .opening_detection import classify_openings, draw_openings_overlay
from .room_recovery import functional_zones, partition_space
from .analysis.room_labels import build_room_labels
from .analysis import room_lexicon
from .ocr_runtime import configure_tesseract
from .analysis.ocr import (
    OCRLine,
    configured_ocr_languages,
    extract_ocr,
    merge_document_text,
    ocr_lines,
    ocr_workers,
    pytesseract,
    run_ocr_tasks,
)
from .analysis.ocr_fusion import (
    associate_room_labels,
    group_dimensions,
    group_room_labels,
)



ROOM_TERMS = (
    ("Master bedroom", "MASTER BEDROOM"),
    ("Living room", "LIVING ROOM"),
    ("Laundry room", "LAUNDRY ROOM"),
    ("Dining room", "DINING ROOM"),
    ("Family room", "FAMILY ROOM"),
    ("Bathroom", "BATHROOM"),
    ("Bedroom", "BEDROOM"),
    ("Kitchen", "KITCHEN"),
    ("Pantry", "PANTRY"),
    ("Hallway", "HALLWAY"),
    ("Closet", "CLOSET"),
    ("Garage", "GARAGE"),
    ("Office", "OFFICE"),
    ("Porch", "PORCH"),
    ("Bath", "BATH"),
    ("Hall", "HALL"),
    ("Room", "ROOM"),
    ("Foyer", "FOYER"),
    ("Entry", "ENTRY"),
    ("Study", "STUDY"),
    ("Den", "DEN"),
    ("غرفة نوم رئيسية", "غرفة النوم الرئيسية"),
    ("غرفة معيشة", "غرفة المعيشة"),
    ("غرفة طعام", "غرفة الطعام"),
    ("غرفة عائلية", "غرفة عائلية"),
    ("غرفة غسيل", "غرفة الغسيل"),
    ("دورة مياه", "دورة المياه"),
    ("غرفة نوم", "غرفة نوم"),
    ("مطبخ", "مطبخ"),
    ("حمام", "حمام"),
    ("خزانة", "خزانة"),
    ("مخزن", "مخزن"),
    ("ممر", "ممر"),
    ("كراج", "كراج"),
    ("مرآب", "مرآب"),
    ("مكتب", "مكتب"),
    ("شرفة", "شرفة"),
    ("مدخل", "مدخل"),
    ("صالة", "صالة"),
    ("غرفة", "غرفة"),
)

IMPERIAL_PART = r"(?:\d{1,2}\s*['′’]\s*\d{0,2}\s*[\"″”]?|\d{1,2}\s*[\"″”])"
IMPERIAL_PAIR = re.compile(
    rf"(?P<w>{IMPERIAL_PART})\s*[xX×]\s*(?P<h>{IMPERIAL_PART})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
METRIC_PAIR = re.compile(
    r"(?P<w>\d+(?:[.,]\d+)?)\s*(?:m|م|متر(?:ات)?|met(?:er|re)s?)\s*[xX×]\s*"
    r"(?P<h>\d+(?:[.,]\d+)?)\s*(?:m|م|متر(?:ات)?|met(?:er|re)s?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)



def _parse_dimensions(text: str) -> dict[str, Any] | None:
    normalized = (
        text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"))
        .replace("٫", ".")
        .replace("°", "'")
        .replace("′", "'")
        .replace("’", "'")
        .replace("″", '"')
        .replace("”", '"')
        .replace("×", "x")
    )
    def expand_compact_inches(match: re.Match[str]) -> str:
        digits = match.group(1)
        choices = []
        for split in (1, 2):
            if split >= len(digits):
                continue
            feet = int(digits[:split])
            inches = int(digits[split:])
            if 1 <= feet <= 60 and 0 <= inches <= 11:
                choices.append((feet, inches))
        if not choices:
            return match.group(0)
        # Prefer a conventional room-sized value (3-30 ft): 108" is 10'8", not 1'8".
        feet, inches = min(choices, key=lambda pair: (not 3 <= pair[0] <= 30, pair[0] > 30, pair[0]))
        return f"{feet}'{inches}\""

    normalized = re.sub(r"(?<!\d)(\d{3,4})\s*[\"″”]", expand_compact_inches, normalized)

    metric = METRIC_PAIR.search(normalized)
    if metric:
        width_m = float(metric.group("w").replace(",", "."))
        height_m = float(metric.group("h").replace(",", "."))
        if width_m > 0 and height_m > 0:
            return {
                "width": round(width_m, 3),
                "height": round(height_m, 3),
                "unit": "m",
                "area": round(width_m * height_m, 2),
            }

    imperial = IMPERIAL_PAIR.search(normalized)
    if not imperial:
        return None

    def as_feet(token: str) -> float | None:
        feet = re.fullmatch(r"(\d{1,2})\s*['′’]\s*(\d{0,2})\s*[\"″”]?", token)
        if feet:
            inches = int(feet.group(2) or 0)
            if inches > 11:
                return None
            return int(feet.group(1)) + inches / 12
        inches = re.fullmatch(r"(\d{1,2})\s*[\"″”]", token)
        return int(inches.group(1)) / 12 if inches else None

    width_ft = as_feet(imperial.group("w"))
    height_ft = as_feet(imperial.group("h"))
    if width_ft is None or height_ft is None:
        return None
    if width_ft <= 0 or height_ft <= 0:
        return None
    return {
        "width": round(width_ft, 3),
        "height": round(height_ft, 3),
        "unit": "ft",
        "area": round(width_ft * height_ft, 2),
    }



def _format_dimensions(dimensions: dict[str, Any]) -> str:
    if dimensions["unit"] == "m":
        return f"{dimensions['width']:g} m × {dimensions['height']:g} m"

    def feet_inches(value: float) -> str:
        feet = int(value)
        inches = int(round((value - feet) * 12))
        if inches == 12:
            feet += 1
            inches = 0
        return f"{feet}'{inches}\""

    return f"{feet_inches(dimensions['width'])} × {feet_inches(dimensions['height'])}"



def _dimension_candidates(lines: list[OCRLine]) -> list[dict[str, Any]]:
    candidates = []
    for line in lines:
        dimensions = _parse_dimensions(line.text)
        if dimensions is None:
            continue
        candidate = {
            **dimensions,
            "text": line.text,
            "display": _format_dimensions(dimensions),
            "box": (line.x, line.y, line.width, line.height),
            "center": line.center,
            "confidence": line.confidence,
            "source": line.source,
        }
        duplicate = None
        for index, existing in enumerate(candidates):
            if (
                existing["unit"] == candidate["unit"]
                and abs(existing["center"][0] - candidate["center"][0]) < 50
                and abs(existing["center"][1] - candidate["center"][1]) < 45
            ):
                duplicate = index
                break
        if duplicate is None:
            candidates.append(candidate)
        elif candidate["confidence"] > candidates[duplicate]["confidence"]:
            candidates[duplicate] = candidate
    return candidates


def _room_name(text: str) -> str | None:
    name = room_lexicon.room_name(text)
    if name:
        return name
    normalized = re.sub(r"ـ|[\u064b-\u065f\u0670]", "", text.upper())
    normalized = normalized.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا").replace("ى", "ي")
    normalized = re.sub(r"[^A-Z0-9\s\u0600-\u06ff]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for display, term in ROOM_TERMS:
        if re.search(rf"(?<![A-Z\u0600-\u06ff]){re.escape(term)}(?![A-Z\u0600-\u06ff])", normalized):
            return display
    return None


def _associate_dimensions(room: dict[str, Any], dimensions: list[dict[str, Any]]) -> None:
    room_line: OCRLine = room["_ocr"]
    room_cx, _ = room_line.center
    eligible = []
    for index, item in enumerate(dimensions):
        x, y, width, height = item["box"]
        dy = y - room_line.bottom
        dx = abs(item["center"][0] - room_cx)
        overlap = max(0, min(room_line.right, x + width) - max(room_line.x, x))
        if dy < -8 or dy > 155:
            continue
        if dx > max(210, room_line.width * 1.4) and overlap == 0:
            continue
        score = dy / 120 + dx / max(120, room_line.width) * 0.45
        score += 0.2 * (1 - overlap / max(1, min(width, room_line.width)))
        score += max(0, 75 - item["confidence"]) / 250
        eligible.append((score, index, item))
    if eligible:
        _, index, item = min(eligible, key=lambda match: match[0])
        room["dimensions"] = {key: item[key] for key in ("width", "height", "unit", "area")}
        room["dimensions"]["text"] = item["display"]
        room["dimensions"]["raw_ocr_text"] = item["text"]
        room["dimensions"]["ocr_confidence"] = round(item["confidence"], 1)
        room["_dimension_index"] = index


def _point_in_polygon(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
    if len(polygon) < 3:
        return False
    contour = np.asarray(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def _component_map(rooms: list[dict[str, Any]], spaces: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {space["id"]: [] for space in spaces}
    for room in rooms:
        point = room["label_center"]
        containing = [space for space in spaces if _point_in_polygon(point, space["polygon"])]
        if not containing and room.get("label_box"):
            # The centre can fall on a wall line or a symbol; let points inside the label vote.
            b = room["label_box"]
            votes: dict[str, int] = {}
            for fx in (0.25, 0.5, 0.75):
                for fy in (0.3, 0.7):
                    q = (int(b["x"] + fx * b["width"]), int(b["y"] + fy * b["height"]))
                    for space in spaces:
                        if _point_in_polygon(q, space["polygon"]):
                            votes[space["id"]] = votes.get(space["id"], 0) + 1
            if votes:
                best = max(votes.values())
                containing = [space for space in spaces if votes.get(space["id"]) == best]
        if containing:
            chosen = min(containing, key=lambda space: space["area_pixels"])
            room["space_id"] = chosen["id"]
            result[chosen["id"]].append(room)
        else:
            room["space_id"] = None
    return result


def _scale_from_rooms(
    rooms_by_space: dict[str, list[dict[str, Any]]],
    spaces: list[dict[str, Any]],
) -> dict[str, float] | None:
    by_id = {space["id"]: space for space in spaces}
    estimates: dict[str, list[float]] = {"ft": [], "m": []}
    for space_id, rooms in rooms_by_space.items():
        if len(rooms) != 1:
            continue
        room = rooms[0]
        dimensions = room.get("dimensions")
        if not _plausible_dimensions(dimensions):
            continue
        space = by_id[space_id]
        bbox = space["bbox"]
        observed_area = float(space["area_pixels"])
        physical_area = float(dimensions["area"])
        if observed_area <= 0 or physical_area <= 0:
            continue
        scale = math.sqrt(observed_area / physical_area)
        if 20 <= scale <= 120:
            estimates[dimensions["unit"]].append(scale)
    best_unit = max(estimates, key=lambda unit: len(estimates[unit]))
    if not estimates[best_unit]:
        return None
    return {
        "unit": best_unit,
        "pixels_per_unit": round(statistics.median(estimates[best_unit]), 2),
        "calibration_rooms": len(estimates[best_unit]),
    }


def _line_support(mask: np.ndarray, orientation: str, position: int, start: int, end: int) -> float:
    height, width = mask.shape[:2]
    start = max(0, min(width if orientation == "horizontal" else height, start))
    end = max(0, min(width if orientation == "horizontal" else height, end))
    if end <= start:
        return 0.0
    radius = 2
    if orientation == "horizontal":
        y1, y2 = max(0, position - radius), min(height, position + radius + 1)
        patch = mask[y1:y2, start:end]
    else:
        x1, x2 = max(0, position - radius), min(width, position + radius + 1)
        patch = mask[start:end, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.count_nonzero(patch)) / float(patch.size)


def _snap_rectangle(
    center: tuple[int, int],
    width_px: float,
    height_px: float,
    wall_mask: np.ndarray,
    pixels_per_unit: float,
) -> tuple[list[tuple[int, int]], int]:
    cx, cy = center
    half_w = max(2, int(round(width_px / 2)))
    half_h = max(2, int(round(height_px / 2)))
    left, right = cx - half_w, cx + half_w
    top, bottom = cy - half_h, cy + half_h
    tolerance = max(5, int(round(pixels_per_unit * 1.2)))
    snapped = 0

    def best_edge(orientation: str, expected: int, start: int, end: int) -> tuple[int, float]:
        axis_limit = wall_mask.shape[0] if orientation == "horizontal" else wall_mask.shape[1]
        candidates = range(max(0, expected - tolerance), min(axis_limit, expected + tolerance + 1))
        scored = [(coord, _line_support(wall_mask, orientation, coord, start, end)) for coord in candidates]
        return max(scored, key=lambda pair: pair[1], default=(expected, 0.0))

    new_top, support_top = best_edge("horizontal", top, left, right)
    new_bottom, support_bottom = best_edge("horizontal", bottom, left, right)
    new_left, support_left = best_edge("vertical", left, top, bottom)
    new_right, support_right = best_edge("vertical", right, top, bottom)
    if support_top >= 0.18:
        top = new_top
        snapped += 1
    if support_bottom >= 0.18:
        bottom = new_bottom
        snapped += 1
    if support_left >= 0.18:
        left = new_left
        snapped += 1
    if support_right >= 0.18:
        right = new_right
        snapped += 1

    height, width = wall_mask.shape[:2]
    left, right = sorted((max(0, min(width - 1, left)), max(0, min(width - 1, right))))
    top, bottom = sorted((max(0, min(height - 1, top)), max(0, min(height - 1, bottom))))
    return [(left, top), (right, top), (right, bottom), (left, bottom)], snapped


def _ray_edge(
    wall_mask: np.ndarray,
    center: tuple[int, int],
    orientation: str,
    direction: int,
    max_distance: int,
) -> int | None:
    cx, cy = center
    height, width = wall_mask.shape[:2]
    origin = cx if orientation == "vertical" else cy
    limit = width if orientation == "vertical" else height
    orthogonal = cy if orientation == "vertical" else cx
    run_start = None
    run_length = 0
    band_start, band_end = orthogonal - 3, orthogonal + 4

    for distance in range(5, max_distance + 1):
        position = origin + direction * distance
        if not 0 <= position < limit:
            break
        score = _line_support(wall_mask, orientation, position, band_start, band_end)
        if score >= 0.24:
            if run_start is None:
                run_start = position
            run_length += 1
            if run_length >= 3:
                return run_start
        else:
            run_start = None
            run_length = 0
    return None


def _room_box_from_wall_rays(
    center: tuple[int, int],
    wall_mask: np.ndarray,
    pixels_per_unit: float,
) -> list[tuple[int, int]] | None:
    max_distance = max(100, int(round(pixels_per_unit * 15)))
    left = _ray_edge(wall_mask, center, "vertical", -1, max_distance)
    right = _ray_edge(wall_mask, center, "vertical", 1, max_distance)
    top = _ray_edge(wall_mask, center, "horizontal", -1, max_distance)
    bottom = _ray_edge(wall_mask, center, "horizontal", 1, max_distance)
    if None in (left, right, top, bottom):
        return None
    if right - left < pixels_per_unit * 1.5 or bottom - top < pixels_per_unit * 1.5:
        return None
    if right - left > pixels_per_unit * 60 or bottom - top > pixels_per_unit * 60:
        return None
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def _retry_dimensions_near_label(
    image: np.ndarray,
    room: dict[str, Any],
    wall_mask: np.ndarray | None = None,
) -> dict[str, Any] | None:
    if pytesseract is None:
        return None
    line: OCRLine = room["_ocr"]
    image_height, image_width = image.shape[:2]
    margin_x = max(120, line.width)
    x0 = max(0, line.x - margin_x)
    x1 = min(image_width, line.right + margin_x)
    y0 = max(0, line.y - 8)
    y1 = min(image_height, line.bottom + 170)
    crop = image[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return None

    # Thick wall strokes can cover dimension glyphs printed too close to a
    # wall. Remove only those known wall pixels in this small retry crop;
    # preserve the original image for the primary OCR pass and the overlay.
    if wall_mask is not None:
        wall_crop = wall_mask[y0:y1, x0:x1]
        crop[wall_crop > 0] = 255

    if crop.shape[1] < 900:
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    candidates = []
    for psm in (6, 7):
        try:
            lines = ocr_lines(crop, psm)
        except Exception:
            continue
        for ocr_line in lines:
            dimensions = _parse_dimensions(ocr_line.text)
            if dimensions is None:
                continue
            dimensions["text"] = _format_dimensions(dimensions)
            dimensions["raw_ocr_text"] = " ".join(ocr_line.text.split())
            dimensions["ocr_confidence"] = round(ocr_line.confidence, 1)
            dimensions["source"] = f"local-crop-psm-{psm}-wall-masked"
            candidates.append(dimensions)

    return max(candidates, key=lambda item: item["ocr_confidence"], default=None)


def _polygon_bbox(polygon: list[tuple[int, int]]) -> dict[str, int]:
    points = np.asarray(polygon, dtype=np.int32)
    x, y, width, height = cv2.boundingRect(points)
    return {"x": int(x), "y": int(y), "width": int(width), "height": int(height)}


OUTDOOR_NAMES = {"Porch", "Terrace", "Balcony", "Deck", "Patio"}

# Functions that commonly share one physical space (open plan). Any other room type sharing a
# structural space with another label means the structure merged two rooms (a missed wall or
# door), not an open plan.
OPEN_PLAN_NAMES = {"Kitchen", "Dining", "Dining room", "Living", "Living room", "Family", "Family room",
                   "Great room", "Breakfast nook", "Nook", "Foyer", "Entry", "Hall", "Hallway", "Sitting area",
                   "Den", "Study", "Library", "Playroom", "Sunroom", "Stairs", "Room"}


def _shared_space_kind(rooms: list[dict[str, Any]]) -> str:
    """'open-plan' when every label in a shared structural space is an open-plan function,
    else 'merged' (physically separate rooms the structure did not separate)."""
    return "open-plan" if all(r["name"] in OPEN_PLAN_NAMES for r in rooms) else "merged"


def _plausible_dimensions(dimensions: dict[str, Any] | None) -> bool:
    """A printed room dimension with a side under 2.5 ft / 0.75 m is a misread (no room is that
    narrow); it is kept as text but not used for geometry or scale."""
    if not dimensions:
        return False
    least = 2.5 if dimensions.get("unit") == "ft" else 0.75
    return min(float(dimensions["width"]), float(dimensions["height"])) >= least


def _labels_inside_building(room_lines: list[OCRLine], structure) -> tuple[list[OCRLine], list[str]]:
    """Room labels name places inside the building. Text that lies in the exterior and beyond
    the walls' extent (titles, legends, area tables) is not a room label; outdoor room terms
    (porch, terrace, balcony) may sit just outside the walls."""
    labels = structure.space_labels
    ys, xs = np.nonzero(structure.wall_mask)
    if xs.size == 0:
        return room_lines, []
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    kept, rejected = [], []
    h, w = labels.shape
    for line in room_lines:
        cx, cy = line.center
        inside_extent = x0 <= cx <= x1 and y0 <= cy <= y1
        win = labels[max(0, line.y):min(h, line.y + line.height), max(0, line.x):min(w, line.x + line.width)]
        exterior = win.size > 0 and float((win == EXTERIOR).mean()) > 0.5
        outdoor = (_room_name(line.text) or "") in OUTDOOR_NAMES
        if exterior and (not inside_extent or not outdoor):
            rejected.append(line.text)
        else:
            kept.append(line)
    return kept, rejected


def _split_shared_spaces(structure, room_lines: list[OCRLine], ocr_result) -> list[dict[str, Any]]:
    """Spaces holding several credible room labels are split at drawn constrictions
    (engine.room_recovery); spaces whose labels are open to each other stay whole."""
    spaces = list(structure.spaces)
    labels = [line for line in room_lines if _room_name(line.text) and line.confidence >= 30]
    if len(labels) < 2 or not spaces:
        return spaces
    by_space: dict[int, list[OCRLine]] = {}
    for line in labels:
        cx, cy = line.center
        k = int(structure.space_labels[min(max(cy, 0), structure.space_labels.shape[0] - 1),
                                       min(max(cx, 0), structure.space_labels.shape[1] - 1)])
        if k > 0:
            by_space.setdefault(k, []).append(line)
    if not any(len(v) >= 2 for v in by_space.values()):
        return spaces
    strokes = cv2.bitwise_or(structure.symbol_ink, structure.wall_mask)
    if ocr_result is not None:
        for box in ocr_result.boxes:                       # text is not a boundary
            pad = max(2, int(0.15 * box.height))
            strokes[max(0, box.y - pad):box.y + box.height + pad, max(0, box.x - pad):box.x + box.width + pad] = 0
    t = max(structure.wall_thickness, 3.0)
    out = []
    for k, space in enumerate(spaces, 1):
        lines = by_space.get(k, [])
        parts = None
        # Labels are evidence of which rooms a merged space holds, not of walls: a space whose
        # labels are all open-plan functions is one physical space with functional zones, so it
        # is not split at narrowings (only spaces holding an enclosed room type are).
        if len(lines) >= 2 and not all(_room_name(line.text) in OPEN_PLAN_NAMES for line in lines):
            parts = partition_space(structure.space_labels == k, strokes, [line.center for line in lines],
                                    min_area=max(400.0, (4.0 * t) ** 2))
        if not parts:
            out.append(space)
            continue
        for j, (mask, _members) in enumerate(parts, 1):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = max(contours, key=cv2.contourArea)
            poly = cv2.approxPolyDP(contour, max(1.5, 0.002 * cv2.arcLength(contour, True)), True)
            x, y, w, h = cv2.boundingRect(contour)
            m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
            out.append({
                "id": f"{space['id']}_{j}",
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "area_pixels": int(mask.sum()),
                "center": (int(m["m10"] / max(m["m00"], 1)), int(m["m01"] / max(m["m00"], 1))),
                "polygon": [(int(p[0][0]), int(p[0][1])) for p in poly],
                "boundary_method": "passage-partition",
            })
    return out


ZONE_METHODS = {"open-plan-shared": "open-plan-zone", "merged-space": "merged-space-zone"}

# An interior opening wider than this many single doors, with no swing arc and no double door,
# is not a door; when every room on both sides is an open-plan function it is an open connection.
OPEN_CONNECTION_DOORS = 2.0


def _plan_door_width(openings: list[dict[str, Any]]) -> float | None:
    """Typical door width of this plan: median of doors drawn with a clear swing arc."""
    widths = [o["width_pixels"] for o in openings if o["type"] == "door" and o["evidence"].get("arc_score", 0) >= 0.6]
    return float(statistics.median(widths)) if len(widths) >= 3 else None


def _join_open_connections(spaces, room_lines, openings, structure) -> list[dict[str, Any]]:
    """Interpretation layer over the physical cells. The structure seals every wall gap, so a
    kitchen open to a living room through a wide opening (a counter, a short wall stub) becomes
    two cells. Locally such an opening looks like a wide doorway; what tells them apart is what
    lies on both sides. Two cells are joined into one physical space across an opening when the
    opening is wider than OPEN_CONNECTION_DOORS of this plan's doors, carries no door symbol
    (no swing arc, no double door, no glazing), and every room label in both cells is an
    open-plan function (kitchen, dining, living, foyer ...). No wall is added or removed; cells
    holding a bedroom, bath, closet, garage etc. are never joined this way."""
    door = _plan_door_width(openings)
    if door is None or len(spaces) < 2:
        return spaces
    by_id = {sp["id"]: sp for sp in spaces}
    names: dict[str, list[str]] = {}
    for line in room_lines:
        name = _room_name(line.text)
        if not name or line.confidence < 30:
            continue
        containing = [sp for sp in spaces if _point_in_polygon(line.center, sp["polygon"])]
        if containing:
            names.setdefault(min(containing, key=lambda sp: sp["area_pixels"])["id"], []).append(name)

    def open_cell(space_id):
        n = names.get(space_id)
        return bool(n) and all(x in OPEN_PLAN_NAMES for x in n)

    parent: dict[str, str] = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    bridges: dict[tuple, list[dict[str, Any]]] = {}
    for o in openings:
        ev = o["evidence"]
        if ev.get("room_relation") != "between_rooms" or o["width_pixels"] < OPEN_CONNECTION_DOORS * door:
            continue
        if ev.get("arc_score", 0) >= 0.4 or ev.get("double_arc_score", 0) >= 0.6 or ev.get("glazing_lines", 0) or ev.get("sliding_panels"):
            continue
        ids = [x for x in ev.get("adjacent_space_ids", []) if x in by_id]
        if len(ids) == 2 and ids[0] != ids[1] and open_cell(ids[0]) and open_cell(ids[1]):
            parent[find(ids[0])] = find(ids[1])
            bridges.setdefault(tuple(sorted(ids)), []).append(o)
    groups: dict[str, list[str]] = {}
    for space_id in list(parent):
        groups.setdefault(find(space_id), []).append(space_id)
    groups = {k: v for k, v in groups.items() if len(v) > 1}
    if not groups:
        return spaces
    shape = structure.space_labels.shape
    out = [sp for sp in spaces if sp["id"] not in parent or len(groups.get(find(sp["id"]), [])) < 2]
    for members in groups.values():
        mask = np.zeros(shape, np.uint8)
        for space_id in members:
            cv2.fillPoly(mask, [np.asarray(by_id[space_id]["polygon"], np.int32)], 1)
        for pair, ops in bridges.items():
            if pair[0] in members and pair[1] in members:
                for o in ops:                                   # the opening itself belongs to the space
                    a, b = np.asarray(o["start"], float), np.asarray(o["end"], float)
                    d = (b - a) / max(1e-9, float(np.linalg.norm(b - a)))
                    n = np.array([-d[1], d[0]]) * (float(o["evidence"].get("wall_thickness_px", 4)) / 2 + 3)
                    cv2.fillPoly(mask, [np.round(np.array([a - n, b - n, b + n, a + n])).astype(np.int32)], 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        poly = cv2.approxPolyDP(contour, max(1.5, 0.002 * cv2.arcLength(contour, True)), True)
        x, y, w, h = cv2.boundingRect(contour)
        m = cv2.moments(mask, binaryImage=True)
        members = sorted(members)
        out.append({
            "id": "+".join(members),
            "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "area_pixels": int(sum(by_id[k]["area_pixels"] for k in members)),
            "center": (int(m["m10"] / max(m["m00"], 1)), int(m["m01"] / max(m["m00"], 1))),
            "polygon": [(int(q[0][0]), int(q[0][1])) for q in poly],
            "boundary_method": "wall-region",
            "joined_cells": members,
        })
    return out


def _assign_functional_zones(room_records, space_lookup, shape, scale) -> None:
    """Rooms sharing one physical space: the space stays one physical space (record
    'physical_space'); each room's own extent is its functional zone, the part of the space
    geodesically nearest its label (engine.room_recovery.functional_zones). Zone borders are
    estimates, not walls."""
    by_space: dict[str, list[dict[str, Any]]] = {}
    for record in room_records:
        b = record.get("boundary")
        if b and b["method"] in ZONE_METHODS and record.get("space_id") in space_lookup:
            by_space.setdefault(record["space_id"], []).append(record)
    for space_id, records in by_space.items():
        if len(records) < 2:
            continue
        space = space_lookup[space_id]
        region = np.zeros(shape, np.uint8)
        cv2.fillPoly(region, [np.asarray(space["polygon"], np.int32)], 1)
        zones = functional_zones(region > 0, [(r["label_center"]["x"], r["label_center"]["y"]) for r in records])
        if zones is None:
            continue
        for record, zone in zip(records, zones):
            contours, _ = cv2.findContours(zone.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            poly = cv2.approxPolyDP(contour, max(1.5, 0.002 * cv2.arcLength(contour, True)), True)
            polygon = [(int(q[0][0]), int(q[0][1])) for q in poly]
            shared_method = record["boundary"]["method"]
            record["physical_space"] = {
                "id": space_id,
                "kind": "open-plan" if shared_method == "open-plan-shared" else "merged",
                "polygon": [{"x": x, "y": y} for x, y in space["polygon"]],
                "bbox": space["bbox"],
                "area_pixels": space["area_pixels"],
            }
            record["boundary"] = {
                "polygon": [{"x": x, "y": y} for x, y in polygon],
                "bbox": _polygon_bbox(polygon),
                "method": ZONE_METHODS[shared_method],
                "confidence": 0.5 if shared_method == "open-plan-shared" else 0.35,
                "wall_edges_snapped": 0,
            }
            if record["area"].get("source") != "printed-dimensions":
                pixels = float(zone.sum())
                record["area"] = ({"value": round(pixels / (scale["pixels_per_unit"] ** 2), 2),
                                   "unit": "ft²" if scale["unit"] == "ft" else "m²", "source": "zone-estimate"}
                                  if scale else {"value": int(pixels), "unit": "px²", "source": "zone-estimate"})


def _build_room_records(
    room_lines: list[OCRLine],
    dimensions: list[OCRLine],
    spaces: list[dict[str, Any]],
    wall_mask: np.ndarray,
    image: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float] | None, list[dict[str, Any]]]:
    label_records = []
    for line in room_lines:
        name = _room_name(line.text)
        if not name or line.confidence < 30:
            continue
        label_records.append(
            {
                "name": name,
                "label_text": line.text,
                "label_confidence": round(line.confidence, 1),
                "label_center": line.center,
                "label_box": {"x": line.x, "y": line.y, "width": line.width, "height": line.height},
                "_ocr": line,
            }
        )

    # Name recognition uses sparse-text OCR; dimension lines are gathered from
    # both segmentation modes because tightly packed dimension glyphs often
    # benefit from the alternate layout hypothesis.
    dim_candidates = _dimension_candidates(dimensions)
    for room in label_records:
        _associate_dimensions(room, dim_candidates)
    # Rooms without a confident printed dimension get a local re-read near their label. The
    # re-reads depend only on their own label, so they run concurrently; the results are
    # applied in room order.
    # A page whose OCR read no dimension-like text at all does not print room dimensions: on such
    # pages (measured: 15 test plans, 427 re-reads) a local re-read never found one, while costing a
    # Tesseract run per label (378 labels on 24.pdf: 18 s). Pages with dimension text keep the re-read.
    retry = [room for room in label_records
             if not room.get("dimensions") or room["dimensions"].get("ocr_confidence", 0) < 65] if dimensions else []
    found = run_ocr_tasks(lambda room: _retry_dimensions_near_label(image, room, wall_mask), retry)
    for room, local_dimensions in zip(retry, found):
        current_dimensions = room.get("dimensions")
        if local_dimensions and (
            not current_dimensions
            or local_dimensions["ocr_confidence"] > current_dimensions.get("ocr_confidence", 0)
        ):
            room["dimensions"] = local_dimensions

    rooms_by_space = _component_map(label_records, spaces)
    scale = _scale_from_rooms(rooms_by_space, spaces)
    space_lookup = {space["id"]: space for space in spaces}
    room_records = []
    used_spaces: set[str] = set()

    for index, room in enumerate(label_records, 1):
        dimensions_record = room.get("dimensions")
        usable_dimensions = dimensions_record if _plausible_dimensions(dimensions_record) else None
        space_id = room.get("space_id")
        same_space_rooms = rooms_by_space.get(space_id, []) if space_id else []
        shared_kind = _shared_space_kind(same_space_rooms) if len(same_space_rooms) > 1 else None
        zone_extent = None
        candidate_space = space_lookup.get(space_id) if space_id else None
        polygon = None
        method = "unresolved"
        boundary_confidence = 0.0
        snapped_edges = 0

        if candidate_space and len(same_space_rooms) == 1:
            polygon = list(candidate_space["polygon"])
            method = candidate_space.get("boundary_method", "wall-region")
            boundary_confidence = 0.76
            used_spaces.add(space_id)

        if shared_kind == "open-plan" and candidate_space:
            # Functional zone of an open-plan space: the physical space is the shared one. A
            # printed size only gives the zone's approximate extent (metadata), never a boundary.
            polygon = list(candidate_space["polygon"])
            method = "open-plan-shared"
            boundary_confidence = 0.6
            used_spaces.add(space_id)
            if usable_dimensions and scale and usable_dimensions["unit"] == scale["unit"]:
                extent, _ = _snap_rectangle(room["label_center"], usable_dimensions["width"] * scale["pixels_per_unit"],
                                            usable_dimensions["height"] * scale["pixels_per_unit"], wall_mask, scale["pixels_per_unit"])
                zone_extent = {"polygon": [{"x": x, "y": y} for x, y in extent], "method": "dimension-estimate"}
        elif usable_dimensions and scale and usable_dimensions["unit"] == scale["unit"]:
            expected_width = usable_dimensions["width"] * scale["pixels_per_unit"]
            expected_height = usable_dimensions["height"] * scale["pixels_per_unit"]
            expected_area = max(1.0, expected_width * expected_height)
            region_area = float(candidate_space["area_pixels"]) if candidate_space else 0.0
            region_ratio = region_area / expected_area if region_area else 0.0
            if candidate_space and (len(same_space_rooms) > 1 or not 0.48 <= region_ratio <= 2.2):
                polygon, snapped_edges = _snap_rectangle(
                    room["label_center"],
                    expected_width,
                    expected_height,
                    wall_mask,
                    scale["pixels_per_unit"],
                )
                method = "dimension-box-wall-snapped" if snapped_edges else "dimension-box-estimate"
                boundary_confidence = min(0.88, 0.46 + 0.09 * snapped_edges)
                used_spaces.add(space_id)
            elif not polygon:
                polygon, snapped_edges = _snap_rectangle(
                    room["label_center"],
                    expected_width,
                    expected_height,
                    wall_mask,
                    scale["pixels_per_unit"],
                )
                method = "dimension-box-wall-snapped" if snapped_edges else "dimension-box-estimate"
                boundary_confidence = min(0.88, 0.46 + 0.09 * snapped_edges)

        if not polygon and candidate_space and len(same_space_rooms) > 1:
            # Several named rooms in one structural space: each label refers to the shared space;
            # no boundary is invented. Rooms that are not open-plan functions (bedroom, bath,
            # closet, garage ...) sharing a space are a structural merge, reported as such.
            polygon = list(candidate_space["polygon"])
            method = "open-plan-shared" if shared_kind == "open-plan" else "merged-space"
            boundary_confidence = 0.6 if shared_kind == "open-plan" else 0.4
            used_spaces.add(space_id)

        if not polygon and scale:
            polygon = _room_box_from_wall_rays(
                room["label_center"],
                wall_mask,
                scale["pixels_per_unit"],
            )
            if polygon:
                method = "wall-ray-estimate"
                boundary_confidence = 0.66

        bbox = _polygon_bbox(polygon) if polygon else None
        polygon_area = round(abs(cv2.contourArea(np.asarray(polygon, dtype=np.int32)))) if polygon else None
        if usable_dimensions:
            area = {
                "value": usable_dimensions["area"],
                "unit": "ft²" if usable_dimensions["unit"] == "ft" else "m²",
                "source": "printed-dimensions",
            }
        elif polygon_area is not None and scale:
            area = {
                "value": round(polygon_area / (scale["pixels_per_unit"] ** 2), 2),
                "unit": "ft²" if scale["unit"] == "ft" else "m²",
                "source": "pixel-scale-estimate",
            }
        else:
            area = {
                "value": polygon_area,
                "unit": "px²",
                "source": "polygon-pixels" if polygon_area is not None else "unavailable",
            }

        room_records.append(
            {
                "id": f"room-{index:02d}",
                "name": room["name"],
                "label_text": room["label_text"],
                "label_confidence": room["label_confidence"],
                **({"label_source": "pdf-text"} if getattr(room.get("_ocr"), "source", "") == "pdf-text" else {}),
                "label_center": {"x": room["label_center"][0], "y": room["label_center"][1]},
                "dimensions": dimensions_record,
                "area": area,
                "boundary": {
                    "polygon": [{"x": x, "y": y} for x, y in polygon],
                    "bbox": bbox,
                    "method": method,
                    "confidence": round(boundary_confidence, 2),
                    "wall_edges_snapped": snapped_edges,
                } if polygon else None,
                "space_id": space_id,
                "zone_extent": zone_extent,
            }
        )

    # Consistency: an estimated boundary (dimension box / wall rays) must not contain another
    # named room's label. Such a box is wrong (scale or dimension misread); the room falls back
    # to its own structural space instead.
    centers = [(r["id"], r["label_center"]["x"], r["label_center"]["y"]) for r in room_records]
    for record in room_records:
        b = record["boundary"]
        if not b or b["method"] in ("wall-region", "open-plan-shared", "merged-space", "passage-partition"):
            continue
        poly = [(q["x"], q["y"]) for q in b["polygon"]]
        if not any(i != record["id"] and _point_in_polygon((x, y), poly) for i, x, y in centers):
            continue
        space = space_lookup.get(record["space_id"]) if record["space_id"] else None
        if space is None:
            record["boundary"] = None
            continue
        sharers = rooms_by_space.get(record["space_id"], [])
        sharing = len(sharers) > 1
        record["boundary"] = {
            "polygon": [{"x": x, "y": y} for x, y in space["polygon"]],
            "bbox": space["bbox"],
            "method": (("open-plan-shared" if _shared_space_kind(sharers) == "open-plan" else "merged-space") if sharing
                       else space.get("boundary_method", "wall-region")),
            "confidence": 0.6 if sharing else 0.76,
            "wall_edges_snapped": 0,
        }

    shared: dict[str, list[str]] = {}
    for record in room_records:
        if record["boundary"] and record["boundary"]["method"] in ("open-plan-shared", "merged-space"):
            shared.setdefault(record["space_id"], []).append(record["id"])
    for record in room_records:
        ids = shared.get(record["space_id"]) if record["boundary"] and record["boundary"]["method"] in ("open-plan-shared", "merged-space") else None
        record["shares_space_with"] = [i for i in ids if i != record["id"]] if ids else []

    _assign_functional_zones(room_records, space_lookup, wall_mask.shape, scale)

    unlabeled = []
    for space in spaces:
        if space["id"] in used_spaces or rooms_by_space.get(space["id"]):
            continue
        unit = scale["unit"] if scale else None
        area = (
            {
                "value": round(space["area_pixels"] / (scale["pixels_per_unit"] ** 2), 2),
                "unit": "ft²" if unit == "ft" else "m²",
                "source": "pixel-scale-estimate",
            }
            if scale
            else {"value": space["area_pixels"], "unit": "px²", "source": "polygon-pixels"}
        )
        unlabeled.append(
            {
                "id": space["id"],
                "name": "Unlabeled space",
                "area": area,
                "area_pixels": space["area_pixels"],
                "boundary": {
                    "polygon": [{"x": x, "y": y} for x, y in space["polygon"]],
                    "bbox": space["bbox"],
                    "method": space.get("boundary_method", "wall-region"),
                    "confidence": 0.62,
                    "wall_edges_snapped": 0,
                },
            }
        )

    return room_records, scale, unlabeled


def _infer_anonymous_dimensions(
    image: np.ndarray,
    dimension_lines: list[OCRLine],
    wall_mask: np.ndarray,
    unlabeled: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float] | None, int]:
    """Use printed measurements as room cues when names are absent.

    Measurements often remain legible even when room names do not. Regions
    containing one measurement can be calibrated from their pixel area;
    regions containing several measurements are likely merged and are
    replaced by separately wall-snapped dimension estimates.
    """
    raw = [item for item in _dimension_candidates(dimension_lines) if item["confidence"] >= 72]
    candidates: list[dict[str, Any]] = []
    for item in sorted(raw, key=lambda value: value["confidence"], reverse=True):
        if min(item["width"], item["height"]) < 2.3 or max(item["width"], item["height"]) > 60:
            continue
        duplicate = any(
            old["unit"] == item["unit"]
            and abs(old["width"] - item["width"]) < 0.15
            and abs(old["height"] - item["height"]) < 0.15
            and abs(old["center"][0] - item["center"][0]) <= 110
            and abs(old["center"][1] - item["center"][1]) <= 75
            for old in candidates
        )
        if not duplicate:
            candidates.append(item)

    polygons = [
        [(point["x"], point["y"]) for point in space["boundary"]["polygon"]]
        for space in unlabeled
    ]
    matches: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(unlabeled))}
    unmatched = []
    for item in candidates:
        containing = [
            index for index, polygon in enumerate(polygons)
            if _point_in_polygon(item["center"], polygon)
        ]
        if containing:
            index = min(containing, key=lambda value: unlabeled[value]["area_pixels"])
            matches[index].append(item)
        else:
            unmatched.append(item)

    for index, items in matches.items():
        box = unlabeled[index]["boundary"]["bbox"]
        box_aspect = max(box["width"], box["height"]) / max(1, min(box["width"], box["height"]))
        plausible = []
        for item in items:
            dim_aspect = max(item["width"], item["height"]) / min(item["width"], item["height"])
            if max(box_aspect, dim_aspect) / min(box_aspect, dim_aspect) <= 2.0:
                plausible.append(item)
        matches[index] = plausible
    # A measurement spatially inside a candidate but strongly inconsistent
    # with its shape is more likely an OCR misread than a usable room cue.

    # OCR can misplace a measurement into a neighboring connected component.
    # Prefer measurements near the center of their candidate region; this
    # also prevents one merged polygon from swallowing measurements in a
    # distant room elsewhere on the page.
    for index, items in matches.items():
        if len(items) < 2:
            continue
        box = unlabeled[index]["boundary"]["bbox"]
        center = (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        limit = max(90.0, math.hypot(box["width"], box["height"]) * 0.22)
        nearby = [
            item for item in items
            if math.dist(item["center"], center) <= limit
        ]
        if nearby:
            matches[index] = nearby
        else:
            matches[index] = [min(items, key=lambda item: math.dist(item["center"], center))]

    # A consensus window rejects OCR/geometry pairings whose implied scale
    # differs sharply from the repeated scale supported by other rooms.
    scale_votes = []
    for index, items in matches.items():
        pixel_area = float(unlabeled[index]["area_pixels"])
        for item in items:
            scale = math.sqrt(pixel_area / max(item["area"], 0.01))
            if 20 <= scale <= 120:
                scale_votes.append((scale, item["unit"]))

    pixel_scale = None
    if scale_votes:
        by_unit: dict[str, list[float]] = {}
        for value, unit in scale_votes:
            by_unit.setdefault(unit, []).append(value)
        unit = max(by_unit, key=lambda key: len(by_unit[key]))
        values = by_unit[unit]
        clusters = [
            [value for value in values if abs(value - anchor) / anchor <= 0.18]
            for anchor in values
        ]
        inliers = max(clusters, key=len)
        if len(inliers) >= 3:
            pixel_scale = {
                "unit": unit,
                "pixels_per_unit": round(float(statistics.median(inliers)), 2),
                "calibration_rooms": len(inliers),
            }

    scale_value = pixel_scale["pixels_per_unit"] if pixel_scale else None
    scale_unit = pixel_scale["unit"] if pixel_scale else None
    replaced = set()
    estimates: list[dict[str, Any]] = []
    kept = list(unlabeled)

    def dimension_record(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "width": item["width"], "height": item["height"], "unit": item["unit"],
            "area": item["area"], "display": item["display"], "text": item["display"],
            "ocr_confidence": round(item["confidence"], 1),
            "source": item["source"],
        }

    def add_estimate(item: dict[str, Any]) -> None:
        if scale_value is None or item["unit"] != scale_unit:
            return
        width_px = item["width"] * scale_value
        height_px = item["height"] * scale_value
        polygon, snapped = _snap_rectangle(
            item["center"], width_px, height_px, wall_mask, scale_value
        )
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        pixels = mask > 0
        area_pixels = int(pixels.sum())
        if area_pixels == 0:
            return
        # Avoid adding a second card when this estimate is already represented
        # by an existing wall region or another nearby dimension line.
        for existing in kept + estimates:
            existing_points = np.asarray(
                [[point["x"], point["y"]] for point in existing["boundary"]["polygon"]],
                dtype=np.int32,
            )
            existing_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(existing_mask, [existing_points], 255)
            other = existing_mask > 0
            overlap = int(np.logical_and(pixels, other).sum())
            if overlap / max(1, min(area_pixels, int(other.sum()))) >= 0.72:
                return

        x, y, box_width, box_height = cv2.boundingRect(np.asarray(polygon, dtype=np.int32))
        estimates.append({
            "id": f"dimension-space-{len(estimates) + 1:02d}",
            "name": "Unlabeled space",
            "dimensions": dimension_record(item),
            "area": {
                "value": item["area"],
                "unit": "ft²" if scale_unit == "ft" else "m²",
                "source": "printed-dimensions",
            },
            "area_pixels": area_pixels,
            "boundary": {
                "polygon": [{"x": px, "y": py} for px, py in polygon],
                "bbox": {"x": x, "y": y, "width": box_width, "height": box_height},
                "method": "dimension-only-estimate",
                "confidence": round(min(0.82, 0.46 + 0.09 * snapped), 2),
                "wall_edges_snapped": snapped,
            },
        })

    for index, items in matches.items():
        if not items:
            continue
        if scale_value is None:
            # Still expose a clear OCR measurement when there is no reliable
            # calibration for pixel-derived boundaries.
            selected = max(items, key=lambda value: value["confidence"])
            unlabeled[index]["dimensions"] = dimension_record(selected)
            continue

        compatible = [item for item in items if item["unit"] == scale_unit]
        if not compatible:
            continue
        if len(compatible) == 1:
            item = compatible[0]
            expected = item["area"] * scale_value * scale_value
            ratio = unlabeled[index]["area_pixels"] / max(1, expected)
            if 0.48 <= ratio <= 2.2:
                unlabeled[index]["dimensions"] = dimension_record(item)
                unlabeled[index]["area"] = {
                    "value": item["area"],
                    "unit": "ft²" if scale_unit == "ft" else "m²",
                    "source": "printed-dimensions",
                }
                continue

        # Several measurement centers inside one connected region indicate
        # that the wall pass merged rooms; draw separate, explicitly estimated
        # boxes instead of presenting the merged blob as one room.
        estimate_count_before = len(estimates)
        for item in compatible:
            add_estimate(item)
        if len(estimates) > estimate_count_before:
            replaced.add(index)
        else:
            # If the snapped box is effectively the same region, retain the
            # observed wall polygon and report the printed area as a cue.
            selected = max(compatible, key=lambda value: value["confidence"])
            unlabeled[index]["dimensions"] = dimension_record(selected)
            unlabeled[index]["area"] = {
                "value": selected["area"],
                "unit": "ft²" if selected["unit"] == "ft" else "m²",
                "source": "printed-dimensions",
            }

    for index in sorted(replaced, reverse=True):
        kept.pop(index)
    for item in unmatched:
        add_estimate(item)

    return kept + estimates, pixel_scale, len(estimates)


def _draw_overlay(image: np.ndarray, rooms: list[dict[str, Any]], unlabeled: list[dict[str, Any]], zones=()) -> bytes:
    base = image.copy()
    fills = base.copy()
    for room in rooms:
        boundary = room.get("boundary")
        if not boundary:
            continue
        points = np.asarray([[point["x"], point["y"]] for point in boundary["polygon"]], dtype=np.int32)
        color = (66, 160, 88) if boundary["method"] == "wall-region" else (0, 166, 245)
        cv2.fillPoly(fills, [points], color)
    for index, item in enumerate(unlabeled, 1):
        points = np.asarray([[point["x"], point["y"]] for point in item["boundary"]["polygon"]], dtype=np.int32)
        color = (0, 166, 245) if item["boundary"]["method"] == "dimension-only-estimate" else (170, 120, 70)
        cv2.fillPoly(fills, [points], color)
    overlay = cv2.addWeighted(fills, 0.20, base, 0.80, 0)

    for room in rooms:
        boundary = room.get("boundary")
        if not boundary:
            continue
        points = np.asarray([[point["x"], point["y"]] for point in boundary["polygon"]], dtype=np.int32)
        color = (45, 130, 65) if boundary["method"] == "wall-region" else (0, 120, 220)
        cv2.polylines(overlay, [points], True, color, max(2, image.shape[1] // 900), cv2.LINE_AA)
        box = boundary["bbox"]
        badge_x = min(image.shape[1] - 30, box["x"] + 6)
        badge_y = min(image.shape[0] - 5, box["y"] + 20)
        cv2.rectangle(overlay, (badge_x - 4, badge_y - 16), (badge_x + 30, badge_y + 4), (255, 255, 255), -1)
        cv2.putText(overlay, room["id"][-2:], (badge_x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 55, 150), 1, cv2.LINE_AA)
    for index, item in enumerate(unlabeled, 1):
        points = np.asarray([[point["x"], point["y"]] for point in item["boundary"]["polygon"]], dtype=np.int32)
        color = (0, 120, 220) if item["boundary"]["method"] == "dimension-only-estimate" else (170, 120, 70)
        cv2.polylines(overlay, [points], True, color, max(2, image.shape[1] // 900), cv2.LINE_AA)
        box = item["boundary"]["bbox"]
        badge_x = min(image.shape[1] - 34, box["x"] + 5)
        badge_y = min(image.shape[0] - 5, box["y"] + 20)
        cv2.rectangle(overlay, (badge_x - 4, badge_y - 16), (badge_x + 31, badge_y + 4), (255, 255, 255), -1)
        cv2.putText(overlay, f"U{index:02d}", (badge_x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    for zone in zones:                         # functional zones not drawn as a room: dashed, never a wall
        if zone.get("room_ids"):
            continue
        from .arch.building_view import ZONE_COL, _dashed
        color = ZONE_COL.get(zone["function"], (90, 90, 90))
        _dashed(overlay, zone["polygon"], color, max(2, image.shape[1] // 900))
        cx, cy = (int(v) for v in zone["center"])
        cv2.putText(overlay, f"{zone['function']} zone", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("Could not encode analysis overlay")
    return encoded.tobytes()


def _topology(structure, openings: list[dict[str, Any]], rooms: list[dict[str, Any]]) -> dict[str, Any]:
    """Room ↔ room / room ↔ exterior relationships through openings, and shared walls."""
    names: dict[str, list[str]] = {}
    for room in rooms:
        if room.get("space_id"):
            names.setdefault(room["space_id"], []).append(room["id"])

    def side(space_id):
        return space_id if space_id else "exterior"

    connections = []
    for opening in openings:
        evidence = opening["evidence"]
        ids = evidence.get("adjacent_space_ids") or []
        relation = evidence.get("room_relation")
        if relation == "between_rooms" and len(ids) == 2:
            between = ids
        elif relation == "room_to_exterior" and ids:
            between = [ids[0], "exterior"]
        else:
            continue
        connections.append({"opening_id": opening["id"], "type": opening["type"], "between": between})
    return {
        "spaces": [
            {"id": space["id"], "area_pixels": space["area_pixels"], "room_ids": names.get(space["id"], [])}
            for space in structure.spaces
        ],
        "connections": connections,
        "wall_adjacency": [
            [structure.space_id(a), structure.space_id(b)] for a, b in sorted(structure.wall_adjacency)
        ],
        "wall_thickness_px": round(structure.wall_thickness, 1),
    }


def _read_objects(vector, ocr_result, structure, ppc, work_scale) -> list:
    """Typed objects of the page's vector layer (page frame) mapped into the analysed frame."""
    from .arch.objects import vector_objects

    if vector is None or not ppc:
        return []
    boxes = getattr(ocr_result, "boxes", ()) or ()
    if work_scale != 1.0:
        from types import SimpleNamespace
        boxes = [SimpleNamespace(text=b.text, x=b.x / work_scale, y=b.y / work_scale, width=b.width / work_scale,
                                 height=b.height / work_scale, confidence=getattr(b, "confidence", 60)) for b in boxes]
    mask = structure.wall_mask
    if mask is not None and mask.shape[:2] != vector.layer.shape[:2]:
        mask = cv2.resize(mask, (vector.layer.shape[1], vector.layer.shape[0]), interpolation=cv2.INTER_NEAREST)
    objects = vector_objects(vector, boxes, ppc / work_scale, wall_mask=mask,
                             wall_thickness=structure.wall_thickness / work_scale)
    for o in objects:
        o.bbox = tuple(v * work_scale for v in o.bbox)
    return objects


def _functional_zones(building, structure, cleaned, ocr_result, rooms, openings, scale, shape, work_scale=1.0) -> None:
    """Typed objects (engine.arch.objects) and functional zones (engine.arch.zones) in the
    structured plan: one structural space can hold kitchen / dining / living zones."""
    from .arch.objects import DOOR_CM, px_per_cm, within_walls
    from .arch.zones import infer_zones

    # The vector layer is in the page frame; the analysis may run on a rescaled reading (work_scale):
    # the scale is kept in both frames (px per cm on the page, and in the analysed image).
    vector = cleaned if getattr(cleaned, "layer", None) is not None else None
    ppc, how = px_per_cm(None, scale)                      # printed dimensions: analysed frame
    if ppc is None and vector is not None and vector.applicable:
        page_ppc, how = px_per_cm(vector, None)            # the page's own doors: page frame
        ppc = page_ppc * work_scale if page_ppc else None
    if ppc is None:
        doors = [o["width_pixels"] for o in openings if o["type"] == "door" and o.get("width_pixels")]
        if len(doors) >= 3:
            ppc, how = float(np.median(doors)) / DOOR_CM, f"median of {len(doors)} door widths"
    objects = _read_objects(vector, ocr_result, structure, ppc, work_scale)
    objects = within_walls(objects, building)
    labels = [(r["name"], (r["label_center"]["x"], r["label_center"]["y"])) for r in rooms
              if (r.get("label_center") or {}).get("x") is not None]
    summary = infer_zones(building["spaces"], shape, objects, labels, ppc, structure.wall_mask)
    summary["scale_source"] = how
    building["objects"] = [o.to_dict() for o in objects]
    building["zones_summary"] = summary
    building["summary"]["objects"] = len(objects)
    building["summary"]["zones"] = summary["zones"]


ABBREVIATION_CONFIDENCE = 75.0


def _abbreviation_only(text: str) -> bool:
    """The text is a room name only through the drafting-abbreviation lexicon (BA, BR, BED, KIT...)."""
    if not room_lexicon.abbreviations_active():
        return False
    with room_lexicon.abbreviations(False):
        return _room_name(text) is None


def _line_source(group) -> str:
    """'pdf-text' when every word of the label comes from the document's own text."""
    boxes = getattr(group, "boxes", ())
    return "pdf-text" if boxes and all(box.source == "pdf-text" for box in boxes) else "robust-ocr"


class FloorPlanAnalyzer:
    """Analyze an OpenCV BGR image and return room records plus a PNG overlay."""

    def analyze(self, image: np.ndarray, source_name: str = "floor-plan", text_evidence=None,
                structure_image: np.ndarray | None = None, structure_engine: str = "legacy",
                abbreviations: bool | None = None, cleaner=None, cleaner_candidate=None) -> dict[str, Any]:
        """See `_analyze`. `abbreviations` (experimental): drafting abbreviations (BDRM, BA-1, KIT,
        W/D...) count as room names; default: only with a structural image or a cleaner.
        `cleaner`: callable returning an engine.cleaning CleanResult; it runs in the structural
        job (concurrently with OCR) and, when applicable, its recognition image is the geometry
        input. The CleanResult is returned under the private key "_clean_result".
        `cleaner_candidate`: like `cleaner`, but only a hypothesis (engine.reconstruction): it runs
        when the default reading of the page is broken, and its cleaned reading is used only if it
        explains the page's labels and text better (a vector PDF analysed without the cleaning
        toggle)."""
        use = (structure_image is not None or cleaner is not None or cleaner_candidate is not None) \
            if abbreviations is None else abbreviations
        with room_lexicon.abbreviations(use):
            return self._analyze(image, source_name, text_evidence, structure_image, structure_engine, cleaner,
                                 cleaner_candidate)

    def _analyze(self, image: np.ndarray, source_name: str = "floor-plan", text_evidence=None,
                 structure_image: np.ndarray | None = None, structure_engine: str = "legacy",
                 cleaner=None, cleaner_candidate=None) -> dict[str, Any]:
        """`text_evidence`: optional OCR-style boxes of the document's own text in this image's
        pixel frame (a PDF text layer, ingest.pdf_text). It is merged with OCR of the page; when
        it is None (images, scans) the analysis is exactly the OCR-only analysis.

        `structure_image` (experimental): a structural rendering of the same page in the same
        frame (engine.semantic: walls, columns, doors, windows only). Walls, spaces, openings and
        the Plan Model then run on it; text, labels, dimensions and overlays keep `image`. None:
        everything runs on `image` (unchanged behaviour). With a structural image,
        `structure_engine` "legacy" (default) keeps the legacy structure when it finds spaces
        (measured best, benchmark.pdf_plans variant E); "plan" reconstructs with the Plan Model,
        reading opening symbols from `image` (variant C)."""
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a decoded three-channel color image")
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            raise ValueError("The image is too small to analyze (minimum 100 × 100 pixels)")
        if height * width > 25_000_000:
            raise ValueError("The image is too large (maximum 25 megapixels)")

        if structure_image is None:
            geometry = image
        elif structure_image.shape != image.shape:
            raise ValueError("The structural image must have the same size as the image")
        else:
            geometry = structure_image
        warnings = []
        ocr_result = None
        room_lines: list[OCRLine] = []
        dimension_lines: list[OCRLine] = []

        # Everything structural needs only the image, so it runs while OCR reads the text:
        # [cleaning ->] walls / gaps / spaces -> Plan Model reconstruction (when used). Text-
        # dependent steps (label attachment) follow after OCR. Results, or the exception, are taken
        # where they were computed before.
        # The page's vector geometry, cleaned, is needed on every PDF page (an alternative reading,
        # and the furniture): it depends only on the page, so it starts now, alongside the default
        # reading and OCR, and is collected where it was computed before.
        cand_job = None
        if cleaner_candidate is not None and cleaner is None and structure_image is None \
                and os.environ.get("FLOORPLAN_VECTORS_EARLY", "1") != "0":
            cand_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="floorplan-vectors")
            cand_job = cand_pool.submit(cleaner_candidate)
            cand_pool.shutdown(wait=False)

        def candidate():
            return cand_job.result() if cand_job is not None else cleaner_candidate()

        def structural():
            cleaned, geom = None, geometry
            if cleaner is not None:
                cleaned = cleaner()
                if cleaned is not None and cleaned.applicable:
                    geom = cleaned.recognition_image
            st = analyze_structure(geom)
            prefer = geom is not image and structure_engine == "plan"
            hyps, blocks, vector, cand = None, [], None, None
            if geom is image:
                # ambiguous plans get alternative readings (engine.reconstruction); easy ones run once
                from . import reconstruction
                if cleaner_candidate is not None and (plan_adapter.legacy_failed(st) or reconstruction._suspicious(st)):
                    # the page's vector geometry, cleaned: the first alternative for a broken page
                    cand = candidate()
                    if cand is not None and cand.applicable:
                        vector = reconstruction.Hypothesis("vector-cleaned", 1.0, cand.recognition_image,
                                                           analyze_structure(cand.recognition_image),
                                                           f"the page's vector geometry, cleaned ({cand.source})")
                        vector.cleaned = cand
                hyps = reconstruction.generate(geom, st, vector_candidate=vector is not None)
                if vector is not None:
                    hyps.append(vector)
                if len(hyps) > 1:
                    blocks = reconstruction.text_blocks(image)
            # the Plan Model fallback is not needed when the page's own vectors give the reading
            pending = None if vector is not None else plan_adapter.prepare(
                geom, st, evidence_image=image if prefer else None, prefer=prefer)
            # Furniture and fixtures are read from the page's own vectors whichever wall reading
            # wins (engine.arch.objects); the cleaned layer is computed here, alongside OCR.
            layer = cleaned if cleaned is not None and cleaned.layer is not None else cand
            if layer is None and cleaner_candidate is not None:
                try:
                    layer = candidate()
                except Exception:
                    layer = None
            return cleaned, geom, st, pending, (hyps, blocks, layer)

        background = ThreadPoolExecutor(max_workers=1, thread_name_prefix="floorplan-structure") \
            if ocr_workers() > 1 else None
        structure_job = background.submit(structural) if background else None

        if pytesseract is not None or text_evidence:
            try:
                ocr_result = extract_ocr(image) if pytesseract is not None else None
                if text_evidence:
                    ocr_result = merge_document_text(text_evidence, ocr_result)

                # Convert grouped OCR evidence back to the legacy OCRLine
                # interface used by the existing room/dimension pipeline.
                room_groups = build_room_labels(ocr_result.boxes)
                dimension_groups = group_dimensions(ocr_result.dimensions)

                for group in room_groups:
                    if not _room_name(group.text):
                        continue
                    if _abbreviation_only(group.text) and _line_source(group) != "pdf-text" \
                            and group.confidence < ABBREVIATION_CONFIDENCE:
                        continue    # a drafting tag (BA, BDRM...) read weakly by OCR is usually a symbol

                    room_lines.append(
                        OCRLine(
                            text=group.text,
                            x=group.x,
                            y=group.y,
                            width=group.width,
                            height=group.height,
                            confidence=group.confidence,
                            source=_line_source(group),
                        )
                    )

                for group in dimension_groups:
                    dimension_lines.append(
                        OCRLine(
                            text=group.text,
                            x=group.x,
                            y=group.y,
                            width=group.width,
                            height=group.height,
                            confidence=group.confidence,
                            source=_line_source(group),
                        )
                    )

            except Exception as exc:
                warnings.append(f"OCR could not read text: {type(exc).__name__}")
        else:
            warnings.append("Tesseract OCR is unavailable; room names and dimensions were not read")

        # One structural pass: walls, wall gaps, sealed spaces and their topology.
        if structure_job is not None:
            try:
                cleaned, geometry, structure, pending_plan, (hyps, blocks, object_layer) = structure_job.result()
            finally:
                background.shutdown(wait=True)
        else:
            cleaned, geometry, structure, pending_plan, (hyps, blocks, object_layer) = structural()
        work_scale, reconstruction_report = 1.0, None
        plan_model, plan_notes = None, []
        if hyps and len(hyps) > 1:
            from . import reconstruction
            evidence = [(_room_name(line.text), (line.center[0], line.center[1])) for line in room_lines
                        if _room_name(line.text)]
            if pending_plan is not None:
                # the Plan Model reading competes with the other readings on the same evidence
                pm, pm_notes, pm_structure = plan_adapter.finish(pending_plan, room_lines, structure)
                if pm_structure is not structure:
                    hyps.insert(1, reconstruction.Hypothesis("plan-model", 1.0, image, pm_structure,
                                                             "reconstructed by the Plan Model (default found no space)"))
                pending_plan = None
            best, reconstruction_report = reconstruction.choose(hyps, evidence, blocks)
            structure = best.structure
            if best.name == "vector-cleaned":
                cleaned, geometry = best.cleaned, best.image
                pending_plan = None
            if best.name == "plan-model":
                plan_model, plan_notes = pm, pm_notes
            if best.scale != 1.0:
                # the whole analysis continues on the normalised reading; mapped back at the end
                work_scale = best.scale
                image = geometry = best.image
                room_lines = reconstruction.scale_lines(room_lines, work_scale)
                dimension_lines = reconstruction.scale_lines(dimension_lines, work_scale)
                ocr_result = reconstruction.scale_ocr(ocr_result, work_scale)
            if best.name != "default":
                warnings.append(f"Reconstruction: {best.reason} - chosen over the default reading because it "
                                f"explains the room labels and text blocks better.")
        # Plan Model: only where the legacy structure found nothing (or in shadow mode); in
        # fallback its spaces replace the empty legacy ones
        if pending_plan is not None:
            plan_model, plan_notes, structure = plan_adapter.finish(pending_plan, room_lines, structure)
        warnings.extend(plan_notes)
        wall_mask = structure.wall_mask
        room_lines, rejected_labels = _labels_inside_building(room_lines, structure)
        if rejected_labels:
            warnings.append(
                "Room-like text outside the building was not used as a room label: "
                + ", ".join(f"'{text}'" for text in rejected_labels[:4])
            )
        if ocr_result is not None and pytesseract is not None:
            # context-aware reading: the text lines inside unnamed spaces, read one at a time
            from .analysis.context_ocr import reread
            room_lines, context_notes = reread(image, structure, room_lines)
            room_lines, _ = _labels_inside_building(room_lines, structure)
        spaces = _split_shared_spaces(structure, room_lines, ocr_result)
        if cleaned is not None and cleaned.applicable:
            # the cleaner typed (and sealed) the doors and windows: report those, related to the
            # final spaces, instead of re-detecting gaps on the sealed plan (engine.arch.openings)
            from .arch.openings import typed_openings
            openings = typed_openings(cleaned, structure)
        else:
            openings = classify_openings(geometry, structure)
            legible = next((h for h in (hyps or []) if h.name == "normalised-scale"), None) if work_scale == 1.0 else None
            if legible is not None and openings:
                # under-resolved plan: read the opening symbols on the normalised reading
                from .arch.symbol_scale import renumber, transfer_types
                transfer_types(openings, classify_openings(legible.image, legible.structure), legible.scale,
                               structure.wall_thickness)
                renumber(openings)
        # the alternative readings are decided and used: release their images and structures
        hyps = blocks = legible = best = None
        spaces = _join_open_connections(spaces, room_lines, openings, structure)

        rooms, scale, unlabeled = _build_room_records(
            room_lines,
            dimension_lines,
            spaces,
            wall_mask,
            image,
        )
        dimension_estimate_count = 0
        if not rooms:
            unlabeled, anonymous_scale, dimension_estimate_count = _infer_anonymous_dimensions(
                image, dimension_lines, wall_mask, unlabeled
            )
            if scale is None:
                scale = anonymous_scale
        if not rooms:
            warnings.append("No room labels were identified; geometric regions are shown as unlabeled candidates")
            if dimension_estimate_count:
                warnings.append(
                    f"Printed dimensions and a scale calibrated from {scale['calibration_rooms']} geometric region(s) produced {dimension_estimate_count} additional estimated room boundary/boundaries. Review these estimates against the walls."
                )
        if rooms and not any(room.get("dimensions") for room in rooms):
            warnings.append("No room dimensions were parsed; only pixel-based area is available")
        if scale:
            calibration_source = "geometric regions" if not rooms else "labeled regions"
            warnings.append(
                f"Pixel scale estimated at {scale['pixels_per_unit']} px/{scale['unit']} from "
                f"{scale['calibration_rooms']} {calibration_source}; dimension-derived room areas use the printed plan text."
            )
        if any(room["boundary"] and room["boundary"]["method"] not in ("wall-region", "open-plan-zone", "merged-space-zone") for room in rooms):
            warnings.append("Orange boundaries are dimension-based estimates; green boundaries follow extracted wall regions.")
        if any(room.get("physical_space", {}).get("kind") == "open-plan" for room in rooms):
            warnings.append("Open-plan spaces (blue outline) hold several functional zones with no wall between them; "
                            "the dashed zone extents are estimates, not walls.")
        if any(room.get("physical_space", {}).get("kind") == "merged" for room in rooms):
            warnings.append("Some rooms that are normally walled off (e.g. bedroom, bath, closet) share one detected space "
                            "(red outline): a wall or door between them was probably not detected.")

        if scale and scale.get("pixels_per_unit"):
            for opening in openings:
                opening["width_display"] = f"{opening['width_pixels'] / scale['pixels_per_unit']:.1f} {scale['unit']}"
        opening_counts = {
            kind: sum(opening["type"] == kind for opening in openings)
            for kind in ("door", "window", "opening")
        }
        if not openings:
            warnings.append("No likely door/window wall gaps were detected; inspect image clarity and wall continuity.")

        openings_overlay_png = draw_openings_overlay(image, openings)
        result = {
            "source_name": source_name,
            "image": {"width": width, "height": height},
            "room_count": len(rooms),
            "unlabeled_space_count": len(unlabeled),
            "pixel_scale": scale,
            "rooms": rooms,
            "unlabeled_spaces": unlabeled,
            "openings": openings,
            "opening_count": len(openings),
            "door_count": opening_counts["door"],
            "window_count": opening_counts["window"],
            "unclassified_opening_count": opening_counts["opening"],
            "topology": _topology(structure, openings, rooms),
            "warnings": warnings,
            "openings_overlay_png": openings_overlay_png,
        }
        try:
            from .arch.building import build_building
            result["building"] = build_building(structure, spaces, rooms, unlabeled, openings, scale, plan_model)
            _functional_zones(result["building"], structure, cleaned if cleaned is not None else object_layer,
                              ocr_result, rooms, openings, scale, image.shape, work_scale)
            from .arch.projection import project
            project(result)                              # zones / objects into the flat records
            from .arch.building_view import render_png
            result["building_overlay_png"] = render_png(image, result["building"], reconstruction_report)
        except Exception as exc:                       # the structured plan never breaks the analysis
            warnings.append(f"Structured building model unavailable: {type(exc).__name__}")
        result["overlay_png"] = _draw_overlay(image, rooms, unlabeled, result.get("zones", ()))
        if plan_model is not None:
            result["plan_model"] = plan_model
        if cleaner is not None or (cleaned is not None and cleaner_candidate is not None):
            result["_clean_result"] = cleaned
        elif object_layer is not None and getattr(object_layer, "layer", None) is not None:
            result["_object_layer"] = object_layer       # read for furniture only (Elements view)
        if reconstruction_report is not None:
            result["reconstruction"] = reconstruction_report
        if work_scale != 1.0:
            from .rescale import to_original
            result = to_original(result, work_scale, width, height)
        return result
