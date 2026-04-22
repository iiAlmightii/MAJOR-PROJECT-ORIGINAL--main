"""
Action Recognition Training Script
─────────────────────────────────────
Classifies volleyball actions from 16-frame clips using a 3D CNN (R3D-18).

Actions: serve · reception · set · attack/spike · block · dig · free_ball

Architecture
------------
  torchvision.models.video.r3d_18 (3D ResNet-18)
  → Replace final FC with a 7-class head
  → Fine-tune from Kinetics-400 pretrained weights

Inference integration
----------------------
At inference time, the action recogniser receives:
  - A 16-frame temporal window centred on the detected ball contact frame
  - Optionally cropped to the player bounding box ± 50%
Returns the action class + confidence score.

Usage
-----
# Install torchvision:
  pip install torch torchvision

# Generate dataset structure first:
  python ../../training/datasets/download_datasets.py --task action

# Train:
  python train_action.py --data ../../training/datasets/action_recognition \
                         --epochs 50 --batch 8

# Evaluate:
  python train_action.py --eval --weights ../../models/weights/action_recognition.pt
"""

import argparse
import os
import time
import shutil
from pathlib import Path
from typing import Tuple

ACTION_CLASSES = ["serve", "reception", "set", "attack", "block", "dig", "free_ball"]
NUM_CLASSES    = len(ACTION_CLASSES)
CLIP_FRAMES    = 16
CLIP_SIZE      = 112     # spatial size after crop

PROJECT_ROOT = Path(__file__).parent.parent.parent
WEIGHTS_OUT  = PROJECT_ROOT / "models" / "weights"
WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)


def build_model():
    import torch
    import torchvision.models.video as vm

    model = vm.r3d_18(weights=vm.R3D_18_Weights.KINETICS400_V1)
    # Replace classifier
    in_features = model.fc.in_features
    model.fc = __import__('torch').nn.Linear(in_features, NUM_CLASSES)
    return model


def build_dataloader(data_dir: str, split: str, batch_size: int):
    """Build a DataLoader from a directory of action class folders."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    import torchvision.transforms as T

    # We use a simple frame-sampler dataset
    dataset = ActionClipDataset(
        root=os.path.join(data_dir, split),
        n_frames=CLIP_FRAMES,
        spatial_size=CLIP_SIZE,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
    )


class ActionClipDataset:
    """
    Loads N consecutive frames from mp4/avi clips organised as:
      root/
        serve/    clip1.mp4 ...
        spike/    ...
    """
    def __init__(self, root: str, n_frames: int = 16, spatial_size: int = 112):
        import glob
        self.clips   = []
        self.labels  = []
        self.n_frames = n_frames
        self.size     = spatial_size

        for label_idx, action in enumerate(ACTION_CLASSES):
            pattern = os.path.join(root, action, "*")
            for f in glob.glob(pattern):
                if f.lower().endswith((".mp4",".avi",".mov")):
                    self.clips.append(f)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        import cv2, torch, numpy as np

        cap   = cv2.VideoCapture(self.clips[idx])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        step   = max(1, total // self.n_frames)
        for i in range(self.n_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(i * step, total - 1))
            ret, fr = cap.read()
            if not ret:
                fr = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            fr = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (self.size, self.size))
            frames.append(fr)
        cap.release()

        # Shape: T × H × W × C → C × T × H × W (float32, normalised)
        clip = np.stack(frames, axis=0).astype(np.float32) / 255.0
        # Normalise with Kinetics mean/std
        mean = np.array([0.43216, 0.394666, 0.37645])
        std  = np.array([0.22803, 0.22145,  0.216989])
        clip = (clip - mean) / std
        clip = clip.transpose(3, 0, 1, 2)   # C T H W
        return torch.tensor(clip, dtype=torch.float32), self.labels[idx]


def train(args):
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"  Device: {device}")

    model = build_model().to(device)
    train_loader = build_dataloader(args.data, "train", args.batch)
    val_loader   = build_dataloader(args.data, "val",   max(1, args.batch // 2))

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_acc  = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for clips, labels in train_loader:
            clips  = clips.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(clips)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * clips.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

        train_acc  = correct / total if total > 0 else 0
        avg_loss   = running_loss / total if total > 0 else 0

        # Validation
        val_acc = _evaluate_model(model, val_loader, device)
        scheduler.step()

        print(f"  Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"Train: {train_acc:.3f} | Val: {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            dest = WEIGHTS_OUT / "action_recognition.pt"
            torch.save(model.state_dict(), str(dest))
            print(f"    → Best model saved ({val_acc:.3f})")

    print(f"\n  Training complete. Best val accuracy: {best_acc:.4f}")
    print(f"  Weights: {WEIGHTS_OUT / 'action_recognition.pt'}")


def _evaluate_model(model, loader, device) -> float:
    import torch
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="training/datasets/action_recognition")
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--device",  default="0")
    parser.add_argument("--eval",    action="store_true")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  Action Recognition Training  — R3D-18")
    print(f"  Classes: {ACTION_CLASSES}")
    print("=" * 60)

    if args.eval:
        import torch
        w = args.weights or str(WEIGHTS_OUT / "action_recognition.pt")
        model = build_model()
        model.load_state_dict(torch.load(w, map_location="cpu"))
        device = torch.device("cpu")
        val_loader = build_dataloader(args.data, "val", 4)
        acc = _evaluate_model(model, val_loader, device)
        print(f"  Validation accuracy: {acc:.4f}")
    else:
        train(args)


if __name__ == "__main__":
    main()
