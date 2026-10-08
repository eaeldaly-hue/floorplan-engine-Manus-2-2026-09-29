"""Train the research U-Net on synthetic plans (research/perception/synth_data.py).

    python research/perception/train.py --data output/perception/synth --iters 6000 --out output/perception/unet.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

cv2.setNumThreads(0)                       # OpenCV threads deadlock inside DataLoader workers on macOS
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import N_CLASSES, UNet  # noqa: E402

WEIGHTS = [0.5, 1.0, 2.0, 4.0, 4.0, 4.0, 1.5]


class Crops(Dataset):
    def __init__(self, files, size=320, length=10**7, seed=0):
        self.files, self.size, self.length, self.seed = files, size, length, seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = random.Random(self.seed * 1_000_003 + idx)
        f = self.files[rng.randrange(len(self.files))]
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(f.replace(".png", "_m.png"), cv2.IMREAD_GRAYSCALE)
        s = math.exp(rng.uniform(math.log(0.6), math.log(1.6)))             # scale jitter
        src = int(round(self.size / s))
        h, w = img.shape
        if h < src or w < src:
            pad_h, pad_w = max(0, src - h), max(0, src - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=int(np.percentile(img, 90)))
            msk = cv2.copyMakeBorder(msk, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            h, w = img.shape
        # bias crops toward the building, not the paper margin
        for _ in range(5):
            y, x = rng.randint(0, h - src), rng.randint(0, w - src)
            if (msk[y:y + src, x:x + src] > 0).mean() > 0.3:
                break
        img, msk = img[y:y + src, x:x + src], msk[y:y + src, x:x + src]
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        k = rng.randrange(4)
        img, msk = np.rot90(img, k), np.rot90(msk, k)
        if rng.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        x = 1.0 - img.astype(np.float32) / 255.0
        g = rng.uniform(0.7, 1.4)
        x = np.clip(x, 0, 1) ** g * rng.uniform(0.75, 1.0)
        return torch.from_numpy(np.ascontiguousarray(x))[None], torch.from_numpy(np.ascontiguousarray(msk)).long()


def evaluate(model, files, device, size=640):
    model.eval()
    inter = np.zeros(N_CLASSES)
    union = np.zeros(N_CLASSES)
    with torch.no_grad():
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(f.replace(".png", "_m.png"), cv2.IMREAD_GRAYSCALE)
            h, w = img.shape
            H, W = (h + 15) // 16 * 16, (w + 15) // 16 * 16
            x = np.zeros((H, W), np.float32)
            x[:h, :w] = 1.0 - img / 255.0
            pred = model(torch.from_numpy(x)[None, None].to(device)).argmax(1)[0, :h, :w].cpu().numpy()
            ok = msk != 255
            for c in range(N_CLASSES):
                a, b = (pred == c) & ok, (msk == c) & ok
                inter[c] += (a & b).sum()
                union[c] += (a | b).sum()
    model.train()
    return (inter / np.maximum(union, 1)).round(3).tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="output/perception/synth")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="output/perception/unet.pt")
    args = ap.parse_args()
    files = sorted(f for f in glob.glob(f"{args.data}/*.png") if not f.endswith("_m.png"))
    val, train = files[-100:], files[:-100]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet(args.base).to(device)
    print("params", sum(p.numel() for p in model.parameters()), "train", len(train), "val", len(val), device, flush=True)
    loader = DataLoader(Crops(train, length=args.iters * args.batch), batch_size=args.batch, num_workers=6,
                        persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.1)
    weights = torch.tensor(WEIGHTS, device=device)
    t0 = time.time()
    run = 0.0
    for it, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y, weight=weights, ignore_index=255)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * float(loss.detach()) if it > 1 else float(loss.detach())
        if it % 200 == 0 or it in (10, 50):
            print(f"it {it} loss {run:.3f} {time.time() - t0:.0f}s", flush=True)
        if it % 2000 == 0 or it == args.iters:
            iou = evaluate(model, val[:40], device)
            print("val IoU", dict(zip(["ext", "int", "wall", "door", "win", "pass", "other"], iou)), flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "base": args.base, "iters": it, "val_iou": iou}, args.out)
        if it >= args.iters:
            break
    Path(args.out).with_suffix(".json").write_text(json.dumps({"iters": args.iters, "base": args.base}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
