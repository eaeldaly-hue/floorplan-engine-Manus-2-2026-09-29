from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .analyzer import FloorPlanAnalyzer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract room names, dimensions, areas, and boundaries from a floor plan.")
    parser.add_argument("image", nargs="?", default="test_floorplan.png", help="Input image (PNG/JPG/TIFF/BMP).")
    parser.add_argument("--out", default="output/analysis", help="Output directory for overlay.png and rooms.json.")
    args = parser.parse_args(argv)

    source = Path(args.image)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Could not read image: {source}")
    result = FloorPlanAnalyzer().analyze(image, source.name)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overlay.png").write_bytes(result.pop("overlay_png"))
    result["overlay_file"] = str(output_dir / "overlay.png")
    (output_dir / "rooms.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Rooms detected: {result['room_count']}")
    print(f"Unlabeled enclosed spaces: {result['unlabeled_space_count']}")
    if result["pixel_scale"]:
        scale = result["pixel_scale"]
        print(f"Estimated scale: {scale['pixels_per_unit']} px/{scale['unit']}")
    print(f"Overlay: {output_dir / 'overlay.png'}")
    print(f"Room data: {output_dir / 'rooms.json'}")
    for warning in result["warnings"]:
        print(f"Note: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
