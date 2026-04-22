"""
Pose-LSTM Action Recognition Training
───────────────────────────────────────
Trains a bidirectional LSTM on pose sequences extracted by extract_poses.py.

Architecture (from the research paper approach):
    Input:  (batch, 30 frames, 34 features)  ← 17 keypoints × 2 coords
    → BiLSTM(hidden=128, layers=2, dropout=0.3)
    → LayerNorm
    → FC(256) → ReLU → Dropout(0.4)
    → FC(num_classes)
    → Softmax

Training strategy:
  - Phase 1 (spike only): binary classifier spike vs background
    → Quick validation that the pose approach works
  - Phase 2 (all 5 classes): add serve, block, defense, set
    → Final multi-class model

Usage
─────
# Phase 1 — train on Spike.mp4 only (validation mode):
  python train_lstm.py --phase 1 --epochs 100

# Phase 2 — all classes:
  python train_lstm.py --phase 2 --epochs 200

# Evaluate:
  python train_lstm.py --eval --weights models/weights/action_lstm.pt

Requirements
────────────
  pip install torch numpy scikit-learn matplotlib
"""

import argparse
import json
import os
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

ROOT         = Path(__file__).resolve().parent.parent.parent
POSE_DATA    = ROOT / "training" / "action_recognition" / "pose_data"
MODELS_OUT   = ROOT / "training" / "action_recognition" / "models"
WEIGHTS_OUT  = ROOT / "models" / "weights"
MODELS_OUT.mkdir(parents=True, exist_ok=True)
WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)

# All action classes (grow this as you add more videos)
ALL_CLASSES     = ["spike", "serve", "block", "defense", "set"]
PHASE1_CLASSES  = ["spike"]    # binary: spike vs background

NUM_FRAMES  = 30
NUM_FEATURES = 34   # 17 kps × 2 coords


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_pose_dataset(
    data_dir: Path,
    classes: List[str],
    val_ratio: float = 0.2,
    augment: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all .npy clips from data_dir/class_name/*.npy
    Returns X_train, y_train, X_val, y_val
    """
    X_all, y_all = [], []
    label2idx = {c: i for i, c in enumerate(classes)}

    # Load positive class samples
    for label in classes:
        class_dir = data_dir / label
        if not class_dir.exists():
            print(f"  WARNING: No data for class '{label}' at {class_dir}")
            continue
        clips = sorted(class_dir.glob("*.npy"))
        if not clips:
            print(f"  WARNING: No .npy files in {class_dir}")
            continue
        print(f"  Class '{label}': {len(clips)} clips")
        for clip_path in clips:
            seq = np.load(str(clip_path))   # (30, 34)
            if seq.shape != (NUM_FRAMES, NUM_FEATURES):
                # Resize temporal dim
                seq = _resize_sequence(seq, NUM_FRAMES)
            X_all.append(seq)
            y_all.append(label2idx[label])

            # Augmentation for small dataset
            if augment:
                for aug in _augment(seq):
                    X_all.append(aug)
                    y_all.append(label2idx[label])

    # If only spike class exists (Phase 1), add background samples
    if len(classes) == 1 and "spike" in classes:
        bg_samples = _generate_background_samples(n=len(X_all) * 2)
        X_all.extend(bg_samples)
        y_all.extend([1] * len(bg_samples))   # class 1 = background
        # Update classes for binary mode
        classes = ["spike", "background"]
        label2idx = {c: i for i, c in enumerate(classes)}

    if not X_all:
        raise RuntimeError(f"No pose data found in {data_dir}. Run extract_poses.py first.")

    X = np.array(X_all, dtype=np.float32)   # (N, 30, 34)
    y = np.array(y_all, dtype=np.int64)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split
    n_val  = max(1, int(len(X) * val_ratio))
    X_val, y_val     = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]

    print(f"\n  Dataset: {len(X_train)} train, {len(X_val)} val")
    print(f"  Classes: {classes}")
    for i, c in enumerate(classes):
        n = (y == i).sum()
        print(f"    [{i}] {c}: {n} samples")

    return X_train, y_train, X_val, y_val, classes


def _resize_sequence(seq: np.ndarray, target: int) -> np.ndarray:
    """Linearly interpolate sequence to target length."""
    orig_len = seq.shape[0]
    if orig_len == target:
        return seq
    idx = np.linspace(0, orig_len - 1, target)
    return np.array([seq[int(i)] for i in idx], dtype=np.float32)


def _augment(seq: np.ndarray) -> List[np.ndarray]:
    """Data augmentation for small pose datasets."""
    augmented = []
    # 1. Temporal jitter (shift by ±2 frames)
    for shift in [-2, 2]:
        shifted = np.roll(seq, shift, axis=0)
        augmented.append(shifted)
    # 2. Gaussian noise
    noisy = seq + np.random.normal(0, 0.02, seq.shape).astype(np.float32)
    augmented.append(noisy)
    # 3. Horizontal flip (negate x coords, which are at even indices)
    flipped = seq.copy()
    flipped[:, 0::2] = -flipped[:, 0::2]
    augmented.append(flipped)
    # 4. Speed variation (subsample every other frame, pad end)
    fast = seq[::2]
    fast = np.concatenate([fast, fast[-1:].repeat(len(seq) - len(fast), axis=0)])
    augmented.append(fast[:len(seq)])
    return augmented


def _generate_background_samples(n: int = 50) -> List[np.ndarray]:
    """Generate random 'non-action' background pose sequences."""
    samples = []
    for _ in range(n):
        # Simulate random walk around neutral pose
        seq = np.cumsum(np.random.normal(0, 0.05, (NUM_FRAMES, NUM_FEATURES)), axis=0)
        seq = (seq - seq.mean()) / (seq.std() + 1e-6)
        samples.append(seq.astype(np.float32))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, input_size: int = NUM_FEATURES):
    import torch
    import torch.nn as nn

    class PoseLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size   = input_size,
                hidden_size  = 128,
                num_layers   = 2,
                batch_first  = True,
                bidirectional= True,
                dropout      = 0.3,
            )
            self.norm = nn.LayerNorm(256)   # 128 × 2 directions
            self.head = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            # x: (batch, seq_len, features)
            out, _ = self.lstm(x)
            # Use last hidden state from both directions
            last = out[:, -1, :]          # (batch, 256)
            last = self.norm(last)
            return self.head(last)

    return PoseLSTM()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import TensorDataset, DataLoader

    classes_to_use = PHASE1_CLASSES if args.phase == 1 else ALL_CLASSES
    print(f"\n  Phase {args.phase} training — classes: {classes_to_use}")

    # Load data
    X_train, y_train, X_val, y_val, final_classes = load_pose_dataset(
        POSE_DATA, classes_to_use
    )

    # Save class mapping
    class_map = {i: c for i, c in enumerate(final_classes)}
    with open(MODELS_OUT / "class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)

    num_classes = len(final_classes)
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    print(f"  Device: {device}")

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=(str(device) != "cpu"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    model     = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=_compute_class_weights(y_train, num_classes, device)
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n  Training for {args.epochs} epochs...\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            preds   = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += y_batch.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc  = correct / total

        val_acc, val_report = evaluate_model(model, val_loader, device, final_classes)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            weights_path = WEIGHTS_OUT / f"action_lstm_phase{args.phase}.pt"
            torch.save({
                "model_state":  model.state_dict(),
                "classes":      final_classes,
                "num_classes":  num_classes,
                "input_size":   NUM_FEATURES,
                "seq_len":      NUM_FRAMES,
                "phase":        args.phase,
                "epoch":        epoch,
                "val_acc":      val_acc,
            }, str(weights_path))
            marker = " ← BEST"
        else:
            marker = ""

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:4d}/{args.epochs}  "
                  f"Loss={train_loss:.4f}  Train={train_acc:.3f}  Val={val_acc:.3f}{marker}")

    # Save history
    with open(MODELS_OUT / f"history_phase{args.phase}.json", "w") as f:
        json.dump(history, f)

    print(f"\n{'='*55}")
    print(f"  Training complete  |  Best val acc: {best_val_acc:.4f}")
    print(f"  Weights: {WEIGHTS_OUT / f'action_lstm_phase{args.phase}.pt'}")
    print(f"  Classes: {final_classes}")
    print("=" * 55)

    # Print final classification report
    print(f"\n  Final validation report:\n{val_report}")


def evaluate_model(model, loader, device, classes):
    import torch
    from collections import defaultdict

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds   = model(X_batch).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()

    # Simple per-class report
    lines = []
    for i, cls in enumerate(classes):
        mask = all_labels == i
        if mask.sum() == 0:
            continue
        cls_acc = (all_preds[mask] == i).mean()
        lines.append(f"    {cls:<12} acc={cls_acc:.3f}  n={mask.sum()}")

    return acc, "\n".join(lines)


def _compute_class_weights(y: np.ndarray, num_classes: int, device):
    """Inverse frequency weighting for imbalanced classes."""
    import torch
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights /= weights.sum()
    return torch.tensor(weights * num_classes, dtype=torch.float32).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase",   type=int, default=1, choices=[1, 2],
                   help="1=spike only validation, 2=all classes")
    p.add_argument("--epochs",  type=int, default=100)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--device",  default="0")
    p.add_argument("--eval",    action="store_true")
    p.add_argument("--weights", default=None)
    args = p.parse_args()

    print("=" * 55)
    print("  Pose + LSTM  Action Recognition Training")
    print("=" * 55)

    if args.eval:
        import torch
        w = args.weights or str(WEIGHTS_OUT / "action_lstm_phase1.pt")
        ckpt     = torch.load(w, map_location="cpu")
        classes  = ckpt["classes"]
        model    = build_model(len(classes))
        model.load_state_dict(ckpt["model_state"])
        device   = torch.device("cpu")
        from torch.utils.data import TensorDataset, DataLoader
        X_tr, y_tr, X_v, y_v, _ = load_pose_dataset(POSE_DATA, PHASE1_CLASSES)
        ds  = TensorDataset(torch.tensor(X_v), torch.tensor(y_v))
        ldr = DataLoader(ds, batch_size=16)
        acc, report = evaluate_model(model, ldr, device, classes)
        print(f"\n  Val accuracy: {acc:.4f}\n{report}")
    else:
        train(args)


if __name__ == "__main__":
    main()
