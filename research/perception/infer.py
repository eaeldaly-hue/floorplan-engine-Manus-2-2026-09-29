"""Run the research U-Net on images listed in a manifest; write class probabilities (uint8, x255)
at the original resolution.

    python research/perception/infer.py --weights output/perception/unet.pt --manifest output/perception/real/manifest.json

Manifest: [{"name": ..., "image": path to a grey PNG, "scale": working scale, "out": .npz path}, ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import N_CLASSES, UNet  # noqa: E402

TILE, OVERLAP = 1024, 96


def predict(model, gray: np.ndarray, scale: float, device) -> np.ndarray:
    h0, w0 = gray.shape
    img = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC) \
        if abs(scale - 1) > 1e-3 else gray
    h, w = img.shape
    x = 1.0 - img.astype(np.float32) / 255.0
    acc = np.zeros((N_CLASSES, h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    step = TILE - OVERLAP
    ys = list(range(0, max(1, h - OVERLAP), step)) or [0]
    xs = list(range(0, max(1, w - OVERLAP), step)) or [0]
    with torch.no_grad():
        for y in ys:
            for xx in xs:
                tile = x[y:y + TILE, xx:xx + TILE]
                th, tw = tile.shape
                H, W = (th + 15) // 16 * 16, (tw + 15) // 16 * 16
                pad = np.zeros((H, W), np.float32)
                pad[:th, :tw] = tile
                p = torch.softmax(model(torch.from_numpy(pad)[None, None].to(device)), 1)[0, :, :th, :tw].cpu().numpy()
                acc[:, y:y + th, xx:xx + tw] += p
                cnt[y:y + th, xx:xx + tw] += 1
    prob = acc / np.maximum(cnt, 1)
    if (h, w) != (h0, w0):
        prob = np.stack([cv2.resize(c, (w0, h0), interpolation=cv2.INTER_LINEAR) for c in prob])
    return prob


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="output/perception/unet.pt")
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(args.weights, map_location="cpu", weights_only=True)
    model = UNet(ck["base"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    for item in json.loads(Path(args.manifest).read_text()):
        gray = cv2.imread(item["image"], cv2.IMREAD_GRAYSCALE)
        t = time.time()
        prob = predict(model, gray, float(item["scale"]), device)
        np.savez_compressed(item["out"], prob=(prob * 255).round().astype(np.uint8))
        print(item["name"], gray.shape, f"x{item['scale']:.2f}", f"{time.time() - t:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
