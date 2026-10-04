"""Project-owned door/window symbol classifier trained only on permissive data."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "opening_symbol_classifier.pt"
INPUT_SIZE = 96


def build_model():
    """Create the small CNN without importing torch for geometry-only users."""
    import torch
    from torch import nn

    class OpeningSymbolNet(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            in_channels = 1
            for out_channels in (16, 32, 48, 64):
                blocks.extend([
                    nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                ])
                in_channels = out_channels
            self.features = nn.Sequential(*blocks)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.15), nn.Linear(64, 2)
            )

        def forward(self, value):
            return self.head(self.features(value))

    return OpeningSymbolNet()


def opening_crop(image: np.ndarray, opening: Any) -> np.ndarray:
    """Normalize a gap and its nearby symbol into a direction-aligned patch."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    cx = (opening.start[0] + opening.end[0]) / 2.0
    cy = (opening.start[1] + opening.end[1]) / 2.0
    tx = (opening.end[0] - opening.start[0]) / max(1.0, float(opening.width))
    ty = (opening.end[1] - opening.start[1]) / max(1.0, float(opening.width))
    nx, ny = -ty, tx
    physical_span = max(float(opening.width) * 2.6, float(opening.wall_thickness) * 9.0, 64.0)
    scale = INPUT_SIZE / physical_span
    center_x = (INPUT_SIZE - 1) / 2.0
    center_y = (INPUT_SIZE - 1) / 2.0
    transform = np.asarray([
        [scale * tx, scale * ty, center_x - scale * (tx * cx + ty * cy)],
        [scale * nx, scale * ny, center_y - scale * (nx * cx + ny * cy)],
    ], dtype=np.float32)
    patch = cv2.warpAffine(
        gray, transform, (INPUT_SIZE, INPUT_SIZE), flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    return patch


def model_status() -> dict[str, Any]:
    weights = Path(os.environ.get("FLOORPLAN_OPENING_MODEL", DEFAULT_WEIGHTS)).expanduser()
    if not weights.is_file():
        return {"available": False, "reason": "commercial_model_not_trained", "weights": str(weights)}
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return {"available": False, "reason": "torch_missing", "detail": type(exc).__name__, "weights": str(weights)}
    return {"available": True, "reason": "ready", "weights": str(weights)}


class OpeningSymbolClassifier:
    def __init__(self, weights: str | Path | None = None):
        self.weights = Path(weights or os.environ.get("FLOORPLAN_OPENING_MODEL", DEFAULT_WEIGHTS)).expanduser()
        self.model = None
        self.torch = None

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        checkpoint = torch.load(self.weights, map_location="cpu", weights_only=True)
        model = build_model()
        model.load_state_dict(checkpoint["state_dict"])
        self.torch, self.model = torch, model.to(device).eval()
        self.device = device

    def classify(self, image: np.ndarray, opening: Any) -> dict[str, Any] | None:
        if not self.weights.is_file():
            return None
        self._load()
        patch = opening_crop(image, opening)
        tensor = self.torch.from_numpy(patch.astype(np.float32)[None, None] / 255.0)
        with self.torch.inference_mode():
            probabilities = self.model(tensor.to(self.device)).softmax(1)[0].cpu().numpy()
        class_index = int(np.argmax(probabilities))
        return {
            "type": ("door", "window")[class_index],
            "confidence": float(probabilities[class_index]),
            "door_probability": float(probabilities[0]),
            "window_probability": float(probabilities[1]),
        }


@lru_cache(maxsize=1)
def get_classifier() -> OpeningSymbolClassifier | None:
    status = model_status()
    return OpeningSymbolClassifier(status["weights"]) if status["available"] else None


def classify_opening_symbol(image: np.ndarray, opening: Any) -> dict[str, Any] | None:
    classifier = get_classifier()
    if classifier is None:
        return None
    try:
        return classifier.classify(image, opening)
    except Exception:
        return None
