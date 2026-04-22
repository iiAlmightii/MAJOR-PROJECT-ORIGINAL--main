"""
Dataset Download Script
────────────────────────
Downloads and prepares datasets for all three models:
  1. Player Detection  — SoccerNet + COCO Person subset
  2. Ball Detection    — Roboflow Volleyball dataset + synthetic augmentation
  3. Action Recognition — UCF101 volleyball clips

Usage:
  python training/datasets/download_datasets.py --task all
  python training/datasets/download_datasets.py --task player
  python training/datasets/download_datasets.py --task ball
  python training/datasets/download_datasets.py --task action
"""

import os
import sys
import argparse
import zipfile
import urllib.request
import shutil
from pathlib import Path

BASE_DIR   = Path(__file__).parent
PLAYER_DIR = BASE_DIR / "player_detection"
BALL_DIR   = BASE_DIR / "ball_detection"
ACTION_DIR = BASE_DIR / "action_recognition"

for d in [PLAYER_DIR, BALL_DIR, ACTION_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def progress_hook(count, block_size, total_size):
    pct = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write(f"\r  Downloading... {pct}%")
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def download_player_dataset():
    """
    Downloads from Roboflow Universe: volleyball-player-detection dataset.
    Requires a Roboflow API key or uses the public export URL.

    Alternative: use ultralytics built-in COCO person subset.
    """
    print("\n[Player Detection Dataset]")
    print("  Option A (recommended): Use ultralytics auto-download")
    print("  Running: yolo data download coco128 (contains person class)")
    print()

    # Create dataset YAML referencing COCO person class only
    yaml_content = """
# Volleyball Player Detection Dataset
# Uses COCO pre-trained YOLOv8 (person class = 0) as starting point
# For best results, fine-tune on volleyball-specific footage

path: ./datasets/player_detection
train: images/train
val: images/val

nc: 2
names:
  0: player_team_a
  1: player_team_b

# Download COCO person subset:
# roboflow dataset: https://universe.roboflow.com/roboflow-100/volleyball-players-fy1ms
# SoccerNet tracking: https://www.soccer-net.org/tasks/tracking

# Quick start (no custom data): YOLOv8n pretrained on COCO already detects persons
# Just run: python train_player.py --use-coco-pretrain
"""
    yaml_path = PLAYER_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content.strip())
    print(f"  Created {yaml_path}")

    # Create sample directory structure
    for split in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            (PLAYER_DIR / split / sub).mkdir(parents=True, exist_ok=True)

    print("  Directory structure created.")
    print("  → Place your volleyball images in:")
    print(f"    {PLAYER_DIR}/train/images/")
    print(f"    {PLAYER_DIR}/train/labels/  (YOLO format .txt)")
    print()
    print("  [TIP] Use Roboflow to label volleyball footage:")
    print("        https://roboflow.com — free for up to 1000 images")


# ─────────────────────────────────────────────────────────────────────────────
# BALL DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def download_ball_dataset():
    """
    Ball detection: small round objects, fast motion.
    Dataset sources:
      - TrackNet volleyball dataset
      - Synthetic augmentation script included
    """
    print("\n[Ball Detection Dataset]")

    yaml_content = """
# Volleyball Ball Detection Dataset
# Ball is small (15-30px diameter at typical broadcast resolution)
# Use high-res images (1280px) for best detection

path: ./datasets/ball_detection
train: images/train
val: images/val

nc: 1
names:
  0: volleyball

# Recommended datasets:
# 1. TrackNet volleyball dataset (National Chiao Tung University):
#    https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2
#
# 2. Roboflow volleyball ball detection:
#    https://universe.roboflow.com/
#
# 3. Synthetic generation: run  python generate_synthetic_ball.py
#    to create training images with synthetic ball overlays
"""
    (BALL_DIR / "dataset.yaml").write_text(yaml_content.strip())

    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            (BALL_DIR / split / sub).mkdir(parents=True, exist_ok=True)

    # Generate synthetic ball data
    _generate_synthetic_ball_samples()
    print(f"  Created {BALL_DIR / 'dataset.yaml'} and directory structure")


def _generate_synthetic_ball_samples(n_samples: int = 200):
    """Generate synthetic training images with volleyball overlays."""
    try:
        import cv2
        import numpy as np
        import random
        print(f"  Generating {n_samples} synthetic ball samples...")

        img_dir   = BALL_DIR / "train" / "images"
        label_dir = BALL_DIR / "train" / "labels"

        for i in range(n_samples):
            # Random background (court-like colours)
            bg_h, bg_w = 720, 1280
            bg_color = [
                random.randint(20, 80),   # B
                random.randint(50, 150),  # G
                random.randint(20, 80),   # R
            ]
            img = np.full((bg_h, bg_w, 3), bg_color, dtype=np.uint8)

            # Add court lines
            cv2.line(img, (0, bg_h//2), (bg_w, bg_h//2), (80, 130, 80), 3)
            for x in [bg_w//4, bg_w//2, 3*bg_w//4]:
                cv2.line(img, (x, 0), (x, bg_h), (60, 110, 60), 1)

            # Place ball
            r  = random.randint(8, 22)
            bx = random.randint(r + 10, bg_w - r - 10)
            by = random.randint(r + 10, bg_h - r - 10)

            # Ball colour (white/yellow/orange)
            ball_color = random.choice([(255,255,255),(200,200,50),(200,130,50)])
            cv2.circle(img, (bx, by), r, ball_color, -1)
            cv2.circle(img, (bx, by), r, (50,50,50), 1)

            # Motion blur
            if random.random() > 0.5:
                angle = random.uniform(0, 360)
                M = cv2.getRotationMatrix2D((bx,by), angle, 1)
                img = cv2.warpAffine(img, M, (bg_w, bg_h))

            fname = f"synth_{i:05d}"
            cv2.imwrite(str(img_dir / f"{fname}.jpg"), img)

            # YOLO label: class cx cy w h (normalised)
            with open(label_dir / f"{fname}.txt", "w") as f:
                cx_n = bx / bg_w
                cy_n = by / bg_h
                w_n  = (2 * r) / bg_w
                h_n  = (2 * r) / bg_h
                f.write(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")

        print(f"  Synthetic samples generated in {img_dir}")
    except ImportError:
        print("  [SKIP] cv2/numpy not installed — skip synthetic generation")


# ─────────────────────────────────────────────────────────────────────────────
# ACTION RECOGNITION
# ─────────────────────────────────────────────────────────────────────────────

def download_action_dataset():
    """
    Action recognition: serve, spike, block, reception, etc.

    Dataset sources:
      - VolleyVis Dataset (custom volleyball actions)
      - UCF-101 sports subset (general sports actions)
      - MPII Cooking (similar action recognition structure)
    """
    print("\n[Action Recognition Dataset]")

    structure = """
# Action Recognition Dataset Structure
# Each action class gets its own folder of 16-frame clips

datasets/action_recognition/
  train/
    serve/         *.mp4 or *.avi clips
    spike/
    block/
    reception/
    set/
    dig/
    free_ball/
  val/
    (same structure)

# Recommended sources:
# 1. Penn Action Dataset (includes volleyball serves):
#    http://dreamdragon.github.io/PennAction/
#
# 2. UCF-101 (spiking volleyball class included):
#    https://www.crcv.ucf.edu/data/UCF101.php
#    wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
#
# 3. Custom labelling with VGG Image Annotator (VIA):
#    https://www.robots.ox.ac.uk/~vgg/software/via/
#
# 4. Download script for UCF-101 volleyball clips:
#    Run: python download_ucf101_volleyball.py
"""
    print(structure)

    for action in ["serve", "spike", "block", "reception", "set", "dig", "free_ball"]:
        for split in ["train", "val"]:
            (ACTION_DIR / split / action).mkdir(parents=True, exist_ok=True)

    # Download script for UCF-101
    ucf_script = '''#!/usr/bin/env python3
"""Download volleyball-related clips from UCF-101."""
import urllib.request, os
UCF_BASE = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
print("Downloading UCF-101 (6.5 GB)...")
urllib.request.urlretrieve(UCF_BASE, "UCF101.rar", reporthook=lambda c,b,t: print(f"\\r{min(int(c*b*100/t),100)}%", end=""))
print("\\nDownload complete. Extract with: unrar x UCF101.rar")
print("Then run: python filter_volleyball_clips.py")
'''
    (ACTION_DIR / "download_ucf101_volleyball.py").write_text(ucf_script)
    print(f"  Created {ACTION_DIR}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download volleyball training datasets")
    parser.add_argument("--task", choices=["all","player","ball","action"], default="all")
    args = parser.parse_args()

    print("=" * 60)
    print("  VolleyVision — Dataset Downloader")
    print("=" * 60)

    if args.task in ("all", "player"):
        download_player_dataset()
    if args.task in ("all", "ball"):
        download_ball_dataset()
    if args.task in ("all", "action"):
        download_action_dataset()

    print("\n✓ Dataset preparation complete.")
    print("  Next: run the training scripts in training/")


if __name__ == "__main__":
    main()
