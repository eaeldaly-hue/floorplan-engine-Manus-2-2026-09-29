"""Room-centric floor-plan analysis built on OpenCV and local Tesseract OCR."""

from __future__ import annotations

import math
import os
import re
import statistics
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from .preprocessing.text_mask import TextMasker
from .rooms.space_segmentation import SpaceSegmenter
from .structural.wall_region_detector import WallRegionDetector

try:
    import pytesseract
    from pytesseract import Output
except ImportError:  # The geometric pipeline remains usable without OCR.
    pytesseract = None
    Output = None


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


@dataclass
class OCRLine:
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    source: str

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@lru_cache(maxsize=1)
def configured_ocr_languages() -> str:
    requested = os.environ.get("FLOORPLAN_OCR_LANG", "eng").strip() or "eng"
    if pytesseract is None:
        return ""
    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception:
        return "eng"
    requested_parts = requested.split("+")
    if all(language in available for language in requested_parts):
        return requested
    if "eng" in available:
        return "eng"
    return next(iter(sorted(available)), "eng")


def _ocr_lines(image: np.ndarray, psm: int) -> list[OCRLine]:
    if pytesseract is None:
        return []
    data = pytesseract.image_to_data(
        image,
        config=f"--oem 3 --psm {psm}",
        output_type=Output.DICT,
        lang=configured_ocr_languages(),
    )
    grouped: dict[tuple[int, int, int, int], list[tuple[int, int, int, int, str, float]]] = {}
    keys = ("page_num", "block_num", "par_num", "line_num")
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        if confidence < 18:
            continue
        key = tuple(int(data[field][index]) for field in keys)
        grouped.setdefault(key, []).append(
            (
                int(data["left"][index]),
                int(data["top"][index]),
                int(data["width"][index]),
                int(data["height"][index]),
                text,
                confidence,
            )
        )

    lines = []
    for tokens in grouped.values():
        tokens.sort(key=lambda token: token[0])
        left = min(token[0] for token in tokens)
        top = min(token[1] for token in tokens)
        right = max(token[0] + token[2] for token in tokens)
        bottom = max(token[1] + token[3] for token in tokens)
        lines.append(
            OCRLine(
                text=" ".join(token[4] for token in tokens),
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
                confidence=sum(token[5] for token in tokens) / len(tokens),
                source=f"tesseract-psm-{psm}",
            )
        )
    return lines


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
        # Prefer a conventional room-sized value over a very large dimension.
        feet, inches = min(choices, key=lambda pair: (pair[0] > 30, pair[0]))
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
        if not dimensions:
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


def _retry_dimensions_near_label(image: np.ndarray, room: dict[str, Any]) -> dict[str, Any] | None:
    if pytesseract is None:
        return None
    line: OCRLine = room["_ocr"]
    image_height, image_width = image.shape[:2]
    margin_x = max(120, line.width)
    x0 = max(0, line.x - margin_x)
    x1 = min(image_width, line.right + margin_x)
    y0 = max(0, line.y - 8)
    y1 = min(image_height, line.bottom + 170)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    if crop.shape[1] < 900:
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for psm in (6, 7):
        try:
            text = pytesseract.image_to_string(
                crop,
                config=f"--oem 3 --psm {psm}",
                lang=configured_ocr_languages(),
            )
        except Exception:
            continue
        dimensions = _parse_dimensions(text)
        if dimensions:
            dimensions["text"] = _format_dimensions(dimensions)
            dimensions["raw_ocr_text"] = " ".join(text.split())
            dimensions["ocr_confidence"] = 45.0
            dimensions["source"] = f"local-crop-psm-{psm}"
            return dimensions
    return None


def _polygon_bbox(polygon: list[tuple[int, int]]) -> dict[str, int]:
    points = np.asarray(polygon, dtype=np.int32)
    x, y, width, height = cv2.boundingRect(points)
    return {"x": int(x), "y": int(y), "width": int(width), "height": int(height)}


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
        if not room.get("dimensions"):
            local_dimensions = _retry_dimensions_near_label(image, room)
            if local_dimensions:
                room["dimensions"] = local_dimensions

    rooms_by_space = _component_map(label_records, spaces)
    scale = _scale_from_rooms(rooms_by_space, spaces)
    space_lookup = {space["id"]: space for space in spaces}
    room_records = []
    used_spaces: set[str] = set()

    for index, room in enumerate(label_records, 1):
        dimensions_record = room.get("dimensions")
        space_id = room.get("space_id")
        same_space_rooms = rooms_by_space.get(space_id, []) if space_id else []
        candidate_space = space_lookup.get(space_id) if space_id else None
        polygon = None
        method = "unresolved"
        boundary_confidence = 0.0
        snapped_edges = 0

        if candidate_space and len(same_space_rooms) == 1:
            polygon = list(candidate_space["polygon"])
            method = "wall-region"
            boundary_confidence = 0.76
            used_spaces.add(space_id)

        if dimensions_record and scale and dimensions_record["unit"] == scale["unit"]:
            expected_width = dimensions_record["width"] * scale["pixels_per_unit"]
            expected_height = dimensions_record["height"] * scale["pixels_per_unit"]
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
        if dimensions_record:
            area = {
                "value": dimensions_record["area"],
                "unit": "ft²" if dimensions_record["unit"] == "ft" else "m²",
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
            }
        )

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
                    "method": "wall-region",
                    "confidence": 0.62,
                    "wall_edges_snapped": 0,
                },
            }
        )

    return room_records, scale, unlabeled


def _draw_overlay(image: np.ndarray, rooms: list[dict[str, Any]], unlabeled: list[dict[str, Any]]) -> bytes:
    base = image.copy()
    fills = base.copy()
    for room in rooms:
        boundary = room.get("boundary")
        if not boundary:
            continue
        points = np.asarray([[point["x"], point["y"]] for point in boundary["polygon"]], dtype=np.int32)
        color = (66, 160, 88) if boundary["method"] == "wall-region" else (0, 166, 245)
        cv2.fillPoly(fills, [points], color)
    for item in unlabeled:
        points = np.asarray([[point["x"], point["y"]] for point in item["boundary"]["polygon"]], dtype=np.int32)
        cv2.fillPoly(fills, [points], (170, 120, 70))
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
    for item in unlabeled:
        points = np.asarray([[point["x"], point["y"]] for point in item["boundary"]["polygon"]], dtype=np.int32)
        cv2.polylines(overlay, [points], True, (170, 120, 70), max(2, image.shape[1] // 900), cv2.LINE_AA)

    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("Could not encode analysis overlay")
    return encoded.tobytes()


class FloorPlanAnalyzer:
    """Analyze an OpenCV BGR image and return room records plus a PNG overlay."""

    def analyze(self, image: np.ndarray, source_name: str = "floor-plan") -> dict[str, Any]:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a decoded three-channel color image")
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            raise ValueError("The image is too small to analyze (minimum 100 × 100 pixels)")
        if height * width > 25_000_000:
            raise ValueError("The image is too large (maximum 25 megapixels)")

        warnings = []
        ocr_primary: list[OCRLine] = []
        ocr_secondary: list[OCRLine] = []
        if pytesseract is not None:
            try:
                ocr_primary = _ocr_lines(image, 11)
                ocr_secondary = _ocr_lines(image, 6)
            except Exception as exc:
                warnings.append(f"OCR could not read text: {type(exc).__name__}")
        else:
            warnings.append("Tesseract OCR is unavailable; room names and dimensions were not read")

        room_lines = [line for line in ocr_primary if _room_name(line.text)]
        dimension_lines = ocr_primary + ocr_secondary

        clean_image, _ = TextMasker(image).remove_text()
        wall_mask = WallRegionDetector(clean_image).detect()
        spaces = SpaceSegmenter(
            clean_image,
            wall_mask=wall_mask,
            gap_close_ratio=0.0625,
            min_area=max(1000, round(height * width * 0.0015)),
            max_image_area_ratio=0.80,
        ).detect_spaces()

        rooms, scale, unlabeled = _build_room_records(
            room_lines,
            dimension_lines,
            spaces,
            wall_mask,
            image,
        )
        if not rooms:
            warnings.append("No room labels were identified; geometric regions are shown as unlabeled candidates")
        if rooms and not any(room.get("dimensions") for room in rooms):
            warnings.append("No room dimensions were parsed; only pixel-based area is available")
        if scale:
            warnings.append(
                f"Pixel scale estimated at {scale['pixels_per_unit']} px/{scale['unit']} from "
                f"{scale['calibration_rooms']} labeled regions; dimension-derived room areas use the printed plan text."
            )
        if any(room["boundary"] and room["boundary"]["method"] != "wall-region" for room in rooms):
            warnings.append("Orange boundaries are dimension-based estimates; green boundaries follow extracted wall regions.")

        overlay_png = _draw_overlay(image, rooms, unlabeled)
        return {
            "source_name": source_name,
            "image": {"width": width, "height": height},
            "room_count": len(rooms),
            "unlabeled_space_count": len(unlabeled),
            "pixel_scale": scale,
            "rooms": rooms,
            "unlabeled_spaces": unlabeled,
            "warnings": warnings,
            "overlay_png": overlay_png,
        }
