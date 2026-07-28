#!/usr/bin/env python3
"""Train a compact HM01B0-monochrome semantic segmenter from successful shards."""

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


CLASSES = ["background", "course", "boundary", "obstacle", "gate", "lab_clutter"]


class CourseDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        for shard in sorted(root.glob("shard_*")):
            if not (shard / "_SUCCESS").is_file():
                continue
            for mono in sorted(shard.glob("hm01b0_mono_*.png")):
                suffix = mono.stem.removeprefix("hm01b0_mono_")
                semantic = shard / f"semantic_segmentation_{suffix}.png"
                labels = shard / f"semantic_segmentation_labels_{suffix}.json"
                if semantic.is_file() and labels.is_file():
                    self.samples.append((mono, semantic, labels))
        if not self.samples:
            raise RuntimeError(f"no successful HM01B0 samples found below {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        mono_path, semantic_path, labels_path = self.samples[index]
        mono = cv2.imread(str(mono_path), cv2.IMREAD_UNCHANGED)
        semantic = cv2.imread(str(semantic_path), cv2.IMREAD_UNCHANGED)
        labels = json.loads(labels_path.read_text())
        target = np.zeros(semantic.shape, dtype=np.int64)
        for raw_id, fields in labels.items():
            name = str(fields.get("class", "background")).lower()
            class_index = CLASSES.index(name) if name in CLASSES else 0
            target[semantic == int(raw_id)] = class_index
        image = torch.from_numpy(mono.copy()).unsqueeze(0).float().div_(255.0)
        return image, torch.from_numpy(target), str(mono_path)


class TinySegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 24, 2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, len(CLASSES), 2, stride=2),
        )

    def forward(self, inputs):
        return self.net(inputs)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=2026)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
dataset = CourseDataset(args.dataset)
indices = list(range(len(dataset)))
random.shuffle(indices)
split = max(1, int(len(indices) * 0.9))
train_set = Subset(dataset, indices[:split])
val_set = Subset(dataset, indices[split:])
train_loader = DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
)
val_loader = DataLoader(
    val_set, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinySegmenter().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
history, best_loss, best_state = [], float("inf"), None
for epoch in range(1, args.epochs + 1):
    model.train()
    train_total, train_items = 0.0, 0
    for images, targets, _ in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(images), targets)
        loss.backward()
        optimizer.step()
        train_total += float(loss.item()) * images.shape[0]
        train_items += images.shape[0]

    model.eval()
    val_total, val_items = 0.0, 0
    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = nn.functional.cross_entropy(model(images), targets)
            val_total += float(loss.item()) * images.shape[0]
            val_items += images.shape[0]
    train_loss = train_total / train_items
    val_loss = val_total / max(1, val_items)
    history.append((epoch, train_loss, val_loss))
    print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
    if val_loss < best_loss:
        best_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
checkpoint = args.output_dir / "hm01b0_segmenter_best.pt"
torch.save(
    {
        "model": best_state,
        "classes": CLASSES,
        "input": {"channels": 1, "height": 160, "width": 160},
        "samples": len(dataset),
        "best_val_loss": best_loss,
    },
    checkpoint,
)

palette = np.asarray(
    [[0, 0, 0], [40, 190, 40], [255, 160, 0], [40, 40, 230],
     [0, 165, 255], [180, 80, 180]],
    dtype=np.uint8,
)
model.eval()
with torch.no_grad():
    for output_index, dataset_index in enumerate(indices[split : split + 8]):
        image, target, source = dataset[dataset_index]
        prediction = model(image.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
        panel = np.hstack(
            [
                cv2.cvtColor(image[0].mul(255).byte().numpy(), cv2.COLOR_GRAY2BGR),
                palette[target.numpy()],
                palette[prediction],
            ]
        )
        cv2.imwrite(str(args.output_dir / f"prediction_{output_index:04d}.png"), panel)

with (args.output_dir / "log.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["epoch", "train_loss", "val_loss"])
    writer.writerows(history)
print(f"saved {checkpoint} ({checkpoint.stat().st_size / 2**20:.2f} MiB)")
