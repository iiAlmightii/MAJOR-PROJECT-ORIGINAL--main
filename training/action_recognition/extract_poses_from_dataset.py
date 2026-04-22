"""
Pose Extraction from Labeled Image Dataset
──────────────────────────────────────────
Bridges the gap between the YOLO-format Action recognition dataset and
the Pose+LSTM pipeline — without needing to collect/annotate new videos.

How it works:
  1. Reads images from Dataset/Action recognition/train/images/
  2. Reads YOLO-format labels (class_id cx cy w h per person)
  3. For each labeled bounding box → crop person → run MediaPipe pose
  4. Groups consecutive same-class frames into 30-frame sequences
  5. Saves .npy files to training/action_recognition/pose_data/{class}/

Classes (from data.yaml):
  0 = block   1 = defense   2 = serve   3 = set   4 = spike

Output goes directly into the pose_data/ directory that train_lstm.py reads.
After running this, you can immediately train the LSTM:
  python train_lstm.py --phase 2 --epochs 80 --device 0

Usage
─────
  # Full extraction (takes ~20-30 min for 11k images):
  python extract_poses_from_dataset.py

  # Quick test with just 200 images:
  python extract_poses_from_dataset.py --limit 200

  # Use test split as well:
  python extract_poses_from_dataset.py --also-test

  # Force re-extract (overwrite existing .npy files):
  python extract_poses_from_dataset.py --force

Requirements
────────────
  pip install mediapipe opencv-python numpy
  (mediapipe is much lighter than rtmlib — fine for 4GB RAM)
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Dict
from collections import defaultdict

ROOT         = Path(__file__).resolve().parent.parent.parent
DATASET_DIR  = ROOT / "Dataset" / "Action recognition"
POSE_DATA    = ROOT / "training" / "action_recognition" / "pose_data"

# YOLO class index → LSTM class name
CLASS_NAMES = {
    0: "block",
    1: "defense",
    2: "serve",
    3: "set",
    4: "spike",
}

NUM_FRAMES   = 30
NUM_KPS      = 17
MIN_CONF     = 0.3
MIN_PERSON_H = 40   # px — skip tiny crops (distant audience)


# ─────────────────────────────────────────────────────────────────────────────
# Pose extraction (MediaPipe — lightweight, works on 4GB RAM)
# ─────────────────────────────────────────────────────────────────────────────

def load_mediapipe():
    try:
        import mediapipe as mp
        pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.4,
        )
        print("  [Pose] MediaPipe loaded (static image mode)")
        return pose
    except ImportError:
        print("  ERROR: mediapipe not installed.")
        print("  Install: pip install mediapipe")
        sys.exit(1)


def extract_pose_mediapipe(model, crop: np.ndarray) -> Optional[np.ndarray]:
    """Extract 17-keypoint pose from a person crop. Returns (17, 3) or None."""
    import mediapipe as mp
    if crop.shape[0] < MIN_PERSON_H or crop.shape[1] < 20:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    res = model.process(rgb)
    if not res.pose_landmarks:
        return None
    h, w = crop.shape[:2]
    # MediaPipe 33 landmarks → 17 COCO keypoints
    MP_TO_COCO = [0, 12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27, 2, 1, 4, 3]
    lm = res.pose_landmarks.landmark
    kps = np.zeros((17, 3), dtype=np.float32)
    for i, idx in enumerate(MP_TO_COCO):
        l = lm[idx]
        kps[i] = [l.x * w, l.y * h, l.visibility]
    return kps


def normalise_keypoints(kps: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Normalise (17, 3) → (34,) relative to torso, scale-invariant."""
    valid = kps[:, 2] > MIN_CONF
    if valid.sum() < 4:
        out = kps[:, :2].copy()
        out[:, 0] /= max(img_w, 1)
        out[:, 1] /= max(img_h, 1)
        return out.flatten()
    torso_pts = kps[[5, 6, 11, 12], :2]
    centre    = torso_pts.mean(axis=0)
    hip_mid   = kps[[11, 12], :2].mean(axis=0)
    shldr_mid = kps[[5, 6],   :2].mean(axis=0)
    scale     = np.linalg.norm(shldr_mid - hip_mid) + 1e-6
    normed    = (kps[:, :2] - centre) / scale
    normed[~valid] = 0.0
    return normed.flatten()   # (34,)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset reading (YOLO format)
# ─────────────────────────────────────────────────────────────────────────────

def read_labels(label_path: Path) -> List[Dict]:
    """Parse YOLO label file → list of {class_id, cx, cy, w, h}."""
    entries = []
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
                if cid in CLASS_NAMES:
                    entries.append({"class_id": cid, "cx": cx, "cy": cy, "w": w, "h": h})
    except Exception:
        pass
    return entries


def crop_person(img: np.ndarray, cx: float, cy: float, w: float, h: float,
                pad: float = 0.1) -> np.ndarray:
    """Crop bounding box from image with padding."""
    ih, iw = img.shape[:2]
    x1 = int((cx - w / 2 - w * pad) * iw)
    y1 = int((cy - h / 2 - h * pad) * ih)
    x2 = int((cx + w / 2 + w * pad) * iw)
    y2 = int((cy + h / 2 + h * pad) * ih)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x2), min(ih, y2)
    return img[y1:y2, x1:x2]


# ─────────────────────────────────────────────────────────────────────────────
# Sequence building from per-frame pose vectors
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(pose_vectors: List[np.ndarray], target: int = NUM_FRAMES,
                    stride: int = 15) -> List[np.ndarray]:
    """
    Convert a list of per-frame pose vectors into fixed-length sequences.
    Uses a sliding window with stride to generate multiple sequences.
    For short lists, repeats/interpolates to fill the window.
    """
    n = len(pose_vectors)
    if n == 0:
        return []

    sequences = []

    if n < target:
        # Repeat + interpolate to fill one sequence
        indices = np.linspace(0, n - 1, target)
        seq = np.array([pose_vectors[int(i)] for i in indices], dtype=np.float32)
        sequences.append(seq)
    else:
        # Sliding window
        for start in range(0, n - target + 1, stride):
            seq = np.array(pose_vectors[start:start + target], dtype=np.float32)
            sequences.append(seq)

    return sequences


def augment_sequence(seq: np.ndarray) -> List[np.ndarray]:
    """Light augmentation for pose sequences."""
    augmented = []
    # Temporal jitter ±2
    for shift in [-2, 2]:
        shifted = np.roll(seq, shift, axis=0)
        augmented.append(shifted)
    # Gaussian noise
    noisy = seq + np.random.normal(0, 0.015, seq.shape).astype(np.float32)
    augmented.append(noisy)
    # Horizontal flip
    flipped = seq.copy()
    flipped[:, 0::2] = -flipped[:, 0::2]
    augmented.append(flipped)
    return augmented


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_split(split_dir: Path, pose_model, args) -> Dict[str, int]:
    """Extract poses from one dataset split (train/test). Returns per-class counts."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():
        print(f"  WARNING: {images_dir} not found, skipping")
        return {}

    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    print(f"\n  Processing {len(image_paths)} images from {split_dir.name}/")

    # Collect per-class pose vectors (grouped by approximate video source)
    # Filenames often share a prefix (video ID), so sort by name = temporal order
    per_class_poses: Dict[int, List[np.ndarray]] = defaultdict(list)

    t0 = time.time()
    n_pose_found = 0
    n_pose_miss  = 0

    for i, img_path in enumerate(image_paths):
        if i % 500 == 0:
            elapsed = time.time() - t0
            print(f"    [{i}/{len(image_paths)}]  poses_found={n_pose_found}  "
                  f"elapsed={elapsed:.0f}s", end="\r")

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        labels = read_labels(label_path)
        if not labels:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for lbl in labels:
            crop = crop_person(img, lbl["cx"], lbl["cy"], lbl["w"], lbl["h"])
            if crop.shape[0] < MIN_PERSON_H:
                continue

            kps = extract_pose_mediapipe(pose_model, crop)
            if kps is None:
                n_pose_miss += 1
                continue

            pose_vec = normalise_keypoints(kps, crop.shape[1], crop.shape[0])
            per_class_poses[lbl["class_id"]].append(pose_vec)
            n_pose_found += 1

    print(f"\n  Poses extracted: {n_pose_found}  |  Failed: {n_pose_miss}")

    # Convert per-class pose lists → 30-frame sequences → .npy files
    counts: Dict[str, int] = {}
    for class_id, poses in per_class_poses.items():
        class_name = CLASS_NAMES[class_id]
        out_dir = POSE_DATA / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Count existing clips to avoid overwrite conflicts
        existing = len(list(out_dir.glob("*.npy")))
        clip_idx = existing if not args.force else 0

        sequences = build_sequences(poses, target=NUM_FRAMES, stride=10)
        saved = 0
        for seq in sequences:
            out_path = out_dir / f"clip_{clip_idx:05d}.npy"
            if out_path.exists() and not args.force:
                clip_idx += 1
                continue
            np.save(str(out_path), seq)
            clip_idx += 1
            saved += 1

            # Save augmented versions too (only for small classes)
            if len(sequences) < 50:
                for aug_seq in augment_sequence(seq):
                    out_path = out_dir / f"clip_{clip_idx:05d}.npy"
                    np.save(str(out_path), aug_seq)
                    clip_idx += 1
                    saved += 1

        counts[class_name] = saved
        print(f"  {class_name:<10}: {len(poses):>5} poses  →  {len(sequences):>4} sequences  "
              f"→  {saved:>4} clips saved")

    return counts


def main():
    p = argparse.ArgumentParser(
        description="Extract MediaPipe poses from Action recognition dataset → LSTM training data"
    )
    p.add_argument("--dataset", default=str(DATASET_DIR),
                   help="Path to Dataset/Action recognition/ directory")
    p.add_argument("--output",  default=str(POSE_DATA),
                   help="Output directory for pose_data/")
    p.add_argument("--limit",   type=int, default=0,
                   help="Limit images per split (0 = all). Use 200 for quick test.")
    p.add_argument("--also-test", action="store_true",
                   help="Also extract from test/ split (in addition to train/)")
    p.add_argument("--force",   action="store_true",
                   help="Re-extract and overwrite existing .npy files")
    args = p.parse_args()

    global POSE_DATA
    POSE_DATA = Path(args.output)

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"ERROR: Dataset not found at {dataset_dir}")
        sys.exit(1)

    print("=" * 65)
    print("  Pose Extraction from Volleyball Actions Dataset")
    print(f"  Dataset : {dataset_dir}")
    print(f"  Output  : {POSE_DATA}")
    print(f"  Classes : {list(CLASS_NAMES.values())}")
    if args.limit:
        print(f"  Limit   : {args.limit} images per split (test run)")
    print("=" * 65)
    print("\n  NOTE: Using MediaPipe (static_image_mode) — works on 4GB RAM")
    print("  Estimated time: 20-35 min for full 11k train split\n")

    pose_model = load_mediapipe()

    t_start = time.time()
    all_counts: Dict[str, int] = defaultdict(int)

    # Always extract train split
    counts = extract_from_split(dataset_dir / "train", pose_model, args)
    for k, v in counts.items():
        all_counts[k] += v

    # Optionally extract test split
    if args.also_test:
        counts = extract_from_split(dataset_dir / "test", pose_model, args)
        for k, v in counts.items():
            all_counts[k] += v

    elapsed = time.time() - t_start

    print(f"\n{'=' * 65}")
    print(f"  Extraction complete in {elapsed / 60:.1f} minutes")
    print(f"\n  Total clips saved:")
    total = 0
    for cls, n in sorted(all_counts.items()):
        print(f"    {cls:<12}: {n:>5} clips")
        total += n
    print(f"    {'TOTAL':<12}: {total:>5} clips")
    print(f"\n  Output: {POSE_DATA}")

    # Save extraction summary
    summary = {"elapsed_sec": round(elapsed), "clips": dict(all_counts)}
    summary_path = POSE_DATA / "dataset_extraction_summary.json"
    POSE_DATA.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")

    print(f"\n{'=' * 65}")
    print("  NEXT STEP: Train the LSTM on extracted poses:")
    print("    python train_lstm.py --phase 2 --epochs 80 --device 0")
    print()
    print("  This trains on ALL 5 classes at once (since we have data for all).")
    print("  Expected training time: ~5-10 min for the LSTM (small model)")
    print("=" * 65)


if __name__ == "__main__":
    main()
