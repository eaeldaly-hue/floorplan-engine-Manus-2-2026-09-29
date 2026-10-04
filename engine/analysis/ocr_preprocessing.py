"""OCR preprocessing and text-mask helpers for floor-plan images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class OCRVariant:
    """An OCR-ready image variant."""

    name: str
    image: np.ndarray


def prepare_ocr_variants(
    image: np.ndarray,
    *,
    max_side: int = 5000,
) -> tuple[list[OCRVariant], float, float]:
    """
    Prepare several OCR-friendly versions of a floor-plan image.

    Returns:
        variants:
            OCR-ready image variants.
        scale_x:
            Horizontal scale applied to the source image.
        scale_y:
            Vertical scale applied to the source image.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    height, width = image.shape[:2]

    longest = max(height, width)

    scale = min(2.0, 2200.0 / longest) if longest < 2200 else 1.0

    if longest > max_side:
        scale = max_side / longest

    if abs(scale - 1.0) > 0.01:
        image = cv2.resize(
            image,
            (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            ),
            interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
        )

    scale_x = image.shape[1] / width
    scale_y = image.shape[0] / height

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=7,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    _, otsu = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    min_dimension = min(denoised.shape[:2])

    block = min(35, min_dimension if min_dimension % 2 else min_dimension - 1)
    block = max(3, block)

    adaptive_c7 = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        7,
    )

    adaptive_c13 = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        13,
    )

    variants = [
        OCRVariant("enhanced_gray", denoised),
        OCRVariant("otsu", otsu),
        OCRVariant("adaptive_c7", adaptive_c7),
        OCRVariant("adaptive_c13", adaptive_c13),
    ]

    return variants, scale_x, scale_y


def _extract_text_pixels(
    gray_crop: np.ndarray,
) -> np.ndarray:
    """
    Extract dark foreground pixels that may belong to text.

    The result is a binary mask where white pixels are candidate
    foreground/text pixels.
    """
    if gray_crop.size == 0:
        return np.zeros_like(gray_crop, dtype=np.uint8)

    # Otsu gives us a robust threshold for black-on-white drawings.
    _, binary = cv2.threshold(
        gray_crop,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Remove tiny isolated noise while keeping character strokes.
    kernel = np.ones((2, 2), dtype=np.uint8)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    return binary


def _detect_structural_lines(
    foreground: np.ndarray,
    *,
    min_horizontal_length: int,
    min_vertical_length: int,
) -> np.ndarray:
    """
    Detect long horizontal/vertical structures inside an OCR box.

    These structures are usually walls, dimension lines, or other
    drawing geometry rather than text.
    """
    if foreground.size == 0:
        return np.zeros_like(foreground, dtype=np.uint8)

    height, width = foreground.shape[:2]

    line_mask = np.zeros_like(foreground, dtype=np.uint8)

    horizontal_length = max(
        5,
        min(width, min_horizontal_length),
    )

    vertical_length = max(
        5,
        min(height, min_vertical_length),
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_length, 1),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_length),
    )

    horizontal = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    vertical = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    line_mask = cv2.bitwise_or(horizontal, vertical)

    return line_mask


def build_text_mask(
    image: np.ndarray,
    boxes: list[dict],
    *,
    padding: int = 2,
    line_length_ratio: float = 0.70,
) -> np.ndarray:
    """
    Build a refined text-pixel mask from OCR bounding boxes.

    OCR bounding boxes are used only as regions of interest. The entire
    rectangle is NOT marked as text.

    Inside each box:
        1. Detect dark foreground pixels.
        2. Detect long horizontal/vertical structures.
        3. Remove those structural lines from the text mask.

    White pixels represent candidate text pixels.
    Black pixels represent everything else.

    Args:
        image:
            Original floor-plan image.
        boxes:
            OCR boxes containing x/y/width/height.
        padding:
            Extra pixels around each OCR box.
        line_length_ratio:
            Minimum fraction of the OCR-box dimension used when
            detecting long structural lines.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    height, width = image.shape[:2]

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Image must be grayscale or BGR.")

    mask = np.zeros((height, width), dtype=np.uint8)

    for box in boxes:
        try:
            x = int(box["x"])
            y = int(box["y"])
            box_width = int(box["width"])
            box_height = int(box["height"])
        except (KeyError, TypeError, ValueError):
            continue

        if box_width <= 0 or box_height <= 0:
            continue

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + box_width + padding)
        y2 = min(height, y + box_height + padding)

        if x1 >= x2 or y1 >= y2:
            continue

        crop = gray[y1:y2, x1:x2]

        foreground = _extract_text_pixels(crop)

        crop_height, crop_width = foreground.shape[:2]

        min_horizontal_length = max(
            8,
            int(crop_width * line_length_ratio),
        )

        min_vertical_length = max(
            8,
            int(crop_height * line_length_ratio),
        )

        structural_lines = _detect_structural_lines(
            foreground,
            min_horizontal_length=min_horizontal_length,
            min_vertical_length=min_vertical_length,
        )

        # Candidate text = foreground pixels that are not long
        # horizontal/vertical structural lines.
        text_pixels = cv2.bitwise_and(
            foreground,
            cv2.bitwise_not(structural_lines),
        )

        # A small dilation reconnects broken character strokes without
        # turning the whole OCR rectangle into a mask.
        reconnect_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (2, 2),
        )

        text_pixels = cv2.dilate(
            text_pixels,
            reconnect_kernel,
            iterations=1,
        )

        mask[y1:y2, x1:x2] = cv2.bitwise_or(
            mask[y1:y2, x1:x2],
            text_pixels,
        )

    return mask


def remove_text_from_image(
    image: np.ndarray,
    text_mask: np.ndarray,
    *,
    inpaint_radius: float = 3.0,
) -> np.ndarray:
    """
    Remove detected text regions using OpenCV inpainting.

    This is intended for a cleaned structural-analysis copy,
    not for replacing the original floor-plan image.
    """
    if image.shape[:2] != text_mask.shape[:2]:
        raise ValueError("Image and text mask dimensions must match.")

    return cv2.inpaint(
        image,
        text_mask,
        inpaint_radius,
        cv2.INPAINT_TELEA,
    )