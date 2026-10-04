"""Train a commercial-use door/window patch classifier from PERDAW plan views.

The dataset is licensed MIT. Validation holds out complete object families to
reduce leakage between augmented versions of the same source symbol.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import re

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from engine.opening_symbol_model import INPUT_SIZE, build_model


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
HOLDOUT_OBJECTS = {"door_8", "door_9", "window_8", "window_9"}


def discover_images(root: Path) -> list[tuple[Path, int, str]]:
    examples = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parent_and_file = f"{path.parent.name}/{path.name}".lower()
        if "_plan" not in parent_and_file and "plan" not in path.parent.name.lower():
            continue
        text = f"{path.parent.name}/{path.name}".lower()
        if "door" in text:
            label = 0
        elif "window" in text:
            label = 1
        else:
            continue
        match = re.search(r"(door|window)[_-]?(\d+)", text)
        if not match:
            continue
        group = f"{match.group(1)}_{match.group(2)}"
        examples.append((path, label, group))
    return sorted(examples, key=lambda item: str(item[0]))


class SymbolDataset(Dataset):
    def __init__(self, examples, training: bool):
        self.examples = examples
        self.training = training

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        path, label, _ = self.examples[index]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Could not read {path}")
        height, width = gray.shape
        scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
        resized = cv2.resize(gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        canvas = np.full((INPUT_SIZE, INPUT_SIZE), 255, dtype=np.uint8)
        y = (INPUT_SIZE - resized.shape[0]) // 2
        x = (INPUT_SIZE - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized

        if self.training:
            angle = random.uniform(-180, 180)
            matrix = cv2.getRotationMatrix2D(((INPUT_SIZE - 1) / 2,) * 2, angle, random.uniform(0.78, 1.22))
            matrix[0, 2] += random.uniform(-8, 8)
            matrix[1, 2] += random.uniform(-8, 8)
            canvas = cv2.warpAffine(canvas, matrix, (INPUT_SIZE, INPUT_SIZE), borderValue=255)
            if random.random() < 0.35:
                canvas = cv2.GaussianBlur(canvas, (3, 3), random.uniform(0.1, 1.0))
            contrast = random.uniform(0.65, 1.35)
            brightness = random.uniform(-20, 20)
            canvas = np.clip(canvas.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
            if random.random() < 0.20:
                kernel = np.ones((2, 2), np.uint8)
                canvas = cv2.dilate(canvas, kernel, iterations=1) if random.random() < 0.5 else cv2.erode(canvas, kernel, iterations=1)

        tensor = torch.from_numpy(canvas.astype(np.float32)[None] / 255.0)
        return tensor, torch.tensor(label, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Extracted PERDAW dataset directory")
    parser.add_argument("--output", type=Path, default=Path("models/opening_symbol_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = discover_images(args.dataset)
    if not examples:
        raise SystemExit("No PERDAW plan-view door/window images were found")
    train_examples = [sample for sample in examples if sample[2] not in HOLDOUT_OBJECTS]
    val_examples = [sample for sample in examples if sample[2] in HOLDOUT_OBJECTS]
    if not train_examples or not val_examples:
        raise SystemExit("Expected plan-view folders with door_N/window_N names for grouped validation")
    print(f"Found {len(examples)} plan views: train={len(train_examples)}, held-out={len(val_examples)}")

    train_loader = DataLoader(SymbolDataset(train_examples, True), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(SymbolDataset(val_examples, False), batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(labels)
            train_correct += int((logits.argmax(1) == labels).sum().item())
            train_total += len(labels)

        model.eval()
        val_correct = val_total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                logits = model(images.to(device))
                val_correct += int((logits.argmax(1).cpu() == labels).sum().item())
                val_total += len(labels)
        val_accuracy = val_correct / max(1, val_total)
        print(
            f"epoch={epoch:02d} loss={train_loss / max(1, train_total):.4f} "
            f"train_acc={train_correct / max(1, train_total):.3f} val_acc={val_accuracy:.3f}",
            flush=True,
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                "state_dict": model.cpu().state_dict(),
                "class_names": ["door", "window"],
                "input_size": INPUT_SIZE,
                "training_data": "PERDAW plan views, MIT license; object-family holdout",
                "validation_accuracy": best_accuracy,
            }, args.output)
            model.to(device)
    print(f"saved {args.output} with held-out accuracy {best_accuracy:.3f}")


if __name__ == "__main__":
    main()
