"""
Ball Detection — COCO Streaming via HuggingFace
─────────────────────────────────────────────────
Streams COCO images from HuggingFace without downloading the full dataset.
Filters for class 32 (sports ball) and converts to YOLO format on-the-fly.

Strategy
─────────
1. Stream COCO-2017 detection from HuggingFace datasets
2. Filter images that contain category_id 33 (sports ball in COCO = 1-indexed = class 32 in 0-indexed)
3. Save images + YOLO labels to local staging directory
4. Mix with synthetic volleyball ball samples (from Phase 2 generator)
5. Fine-tune YOLOv8 on the combined dataset

COCO category 33 = "sports ball" (volleyball, soccer ball, etc.)

Usage
─────
# Step 1: Collect COCO sports ball samples (streams, no full download)
  python stream_coco_ball.py --collect --max-samples 500

# Step 2: Train
  python stream_coco_ball.py --train --epochs 150

# Step 3: Evaluate
  python stream_coco_ball.py --eval

Requirements
────────────
  pip install datasets huggingface_hub pillow ultralytics
"""

import argparse
import os
import json
import shutil
from pathlib import Path
from typing import Optional

ROOT        = Path(__file__).resolve().parent.parent.parent
STAGE_DIR   = ROOT / "training" / "datasets" / "ball_detection_coco"
WEIGHTS_OUT = ROOT / "models" / "weights"
WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)

for d in [STAGE_DIR / "train" / "images",
          STAGE_DIR / "train" / "labels",
          STAGE_DIR / "val"   / "images",
          STAGE_DIR / "val"   / "labels"]:
    d.mkdir(parents=True, exist_ok=True)

COCO_SPORTS_BALL_ID = 33   # COCO 1-indexed category id for "sports ball"
YOLO_CLASS_ID       = 0    # Single class: volleyball/ball


def collect_coco_samples(max_samples: int = 500, val_ratio: float = 0.15):
    """
    Stream COCO 2017 from HuggingFace and extract sports ball images.
    No full dataset download — uses HF streaming mode.
    """
    print(f"  Streaming COCO 2017 from HuggingFace (max {max_samples} ball images)...")
    print("  This may take a few minutes on first run (streaming, not full download)\n")

    try:
        from datasets import load_dataset
        from PIL import Image
        import io
        import requests
    except ImportError:
        print("  ERROR: Install required packages:")
        print("  pip install datasets huggingface_hub pillow requests")
        return 0

    # Stream the dataset — no local download of the full 20GB
    try:
        ds = load_dataset(
            "detection-datasets/coco",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Could not load COCO from HF: {e}")
        print("  Trying alternate source...")
        try:
            ds = load_dataset(
                "rafaelpadilla/coco2017",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as e2:
            print(f"  Alternate also failed: {e2}")
            print("  Falling back to synthetic ball generation only.")
            _generate_synthetic_volleyball_balls(max_samples)
            return max_samples

    collected = 0
    val_count = 0
    skipped   = 0

    for sample in ds:
        if collected >= max_samples:
            break

        # Check if this image contains a sports ball
        objects = sample.get("objects", {})
        categories = objects.get("category", []) if isinstance(objects, dict) else []

        # HF COCO format varies — try both common schemas
        if not categories:
            anns = sample.get("annotations", [])
            categories = [a.get("category_id", 0) for a in anns] if anns else []

        if COCO_SPORTS_BALL_ID not in categories:
            skipped += 1
            continue

        # Determine split
        is_val = collected < int(max_samples * val_ratio)
        split  = "val" if is_val else "train"

        try:
            # Get image
            img = sample.get("image")
            if img is None:
                continue

            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)

            img_w, img_h = img.size
            fname = f"coco_ball_{collected:05d}"

            # Save image
            img_path = STAGE_DIR / split / "images" / f"{fname}.jpg"
            img.convert("RGB").save(str(img_path), quality=90)

            # Build YOLO labels for sports ball annotations
            label_lines = []
            if isinstance(objects, dict):
                bboxes     = objects.get("bbox", [])
                cat_ids    = objects.get("category", [])
                for bbox, cat in zip(bboxes, cat_ids):
                    if cat != COCO_SPORTS_BALL_ID:
                        continue
                    # COCO bbox: [x, y, w, h] in pixels
                    x, y, w, h = bbox
                    cx = (x + w / 2) / img_w
                    cy = (y + h / 2) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    cx, cy, nw, nh = [max(0, min(1, v)) for v in [cx, cy, nw, nh]]
                    label_lines.append(f"{YOLO_CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if not label_lines:
                img_path.unlink(missing_ok=True)
                continue

            label_path = STAGE_DIR / split / "labels" / f"{fname}.txt"
            label_path.write_text("\n".join(label_lines))

            collected += 1
            if is_val:
                val_count += 1

            if collected % 50 == 0:
                print(f"  Collected: {collected}/{max_samples} (val: {val_count})")

        except Exception as e:
            skipped += 1
            continue

    print(f"\n  ✓ Collected {collected} sports ball images from COCO")
    print(f"  Train: {collected - val_count}  Val: {val_count}  Skipped: {skipped}")
    return collected


def _generate_synthetic_volleyball_balls(n: int = 300):
    """Generate synthetic volleyball images as additional training data."""
    import cv2
    import numpy as np
    import random

    print(f"  Generating {n} synthetic volleyball ball samples...")

    court_colors = [
        [30, 100, 50], [20, 80, 120], [80, 60, 30],
        [40, 40, 120], [100, 100, 50],
    ]

    for i in range(n):
        split = "val" if i < int(n * 0.15) else "train"
        bg_h, bg_w = 720, 1280
        bg = np.array(random.choice(court_colors), dtype=np.uint8)
        img = np.full((bg_h, bg_w, 3), bg + np.random.randint(-20, 20, 3), dtype=np.uint8)
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Horizontal line (net)
        cv2.line(img, (0, bg_h // 2), (bg_w, bg_h // 2), (200, 200, 200), 3)
        # Court lines
        for x in [bg_w // 4, bg_w * 3 // 4]:
            cv2.line(img, (x, 0), (x, bg_h), (180, 180, 180), 1)

        r  = random.randint(6, 20)
        bx = random.randint(r + 5, bg_w - r - 5)
        by = random.randint(r + 5, bg_h - r - 5)

        # Volleyball pattern
        ball_base = random.choice([(255,255,255),(230,220,200),(200,200,60)])
        cv2.circle(img, (bx, by), r,     ball_base,       -1)
        cv2.circle(img, (bx, by), r,     (80, 80, 80),     1)
        cv2.circle(img, (bx, by), r // 2,(100, 100, 100),  1)

        # Motion blur sometimes
        if random.random() > 0.6:
            ksize = random.choice([3, 5])
            angle = random.uniform(0, 360)
            M = cv2.getRotationMatrix2D((bx, by), angle, 1)
            img = cv2.warpAffine(img, M, (bg_w, bg_h))

        fname = f"synth_{i:05d}"
        cv2.imwrite(str(STAGE_DIR / split / "images" / f"{fname}.jpg"), img)

        cx_n = bx / bg_w
        cy_n = by / bg_h
        w_n  = (2 * r) / bg_w
        h_n  = (2 * r) / bg_h
        with open(STAGE_DIR / split / "labels" / f"{fname}.txt", "w") as f:
            f.write(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")

    print(f"  ✓ Synthetic samples generated in {STAGE_DIR}")


def write_yaml():
    yaml = f"""# Ball Detection — COCO Streaming + Synthetic
path: {STAGE_DIR}
train: train/images
val:   val/images
nc: 1
names:
  0: volleyball
"""
    yaml_path = STAGE_DIR / "dataset.yaml"
    yaml_path.write_text(yaml)
    return yaml_path


def train(args):
    from ultralytics import YOLO

    yaml_path = write_yaml()
    print("=" * 60)
    print("  Ball Detection Training")
    print(f"  Data: {yaml_path}")
    print(f"  Model: {args.model}  Epochs: {args.epochs}")
    print("=" * 60)

    train_imgs = list((STAGE_DIR / "train" / "images").glob("*.jpg"))
    val_imgs   = list((STAGE_DIR / "val"   / "images").glob("*.jpg"))
    print(f"  Train images: {len(train_imgs)}  Val images: {len(val_imgs)}")

    if len(train_imgs) < 10:
        print("  WARNING: Very few training images. Run --collect first.")

    model   = YOLO(f"{args.model}.pt")
    results = model.train(
        data    = str(yaml_path),
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device,
        project = str(ROOT / "runs" / "ball_detection"),
        name    = "coco_exp",
        lr0     = 0.005,
        box     = 10.0,
        cls     = 0.3,
        mosaic  = 1.0,
        mixup   = 0.2,
        copy_paste=0.2,
        patience= 40,
        save_period=25,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        dest = WEIGHTS_OUT / "ball_detection.pt"
        shutil.copy(str(best), str(dest))
        print(f"\n  ✓ Best weights → {dest}")


def evaluate(args):
    from ultralytics import YOLO
    w = args.weights or str(WEIGHTS_OUT / "ball_detection.pt")
    model   = YOLO(w)
    metrics = model.val(data=str(write_yaml()), imgsz=args.imgsz,
                        conf=0.3, iou=0.3, device=args.device)
    print(f"\n  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collect",      action="store_true", help="Stream COCO and collect ball images")
    p.add_argument("--max-samples",  type=int, default=500)
    p.add_argument("--synthetic",    action="store_true", help="Add synthetic ball samples")
    p.add_argument("--train",        action="store_true")
    p.add_argument("--eval",         action="store_true")
    p.add_argument("--model",        default="yolov8n")
    p.add_argument("--epochs",       type=int, default=150)
    p.add_argument("--imgsz",        type=int, default=1280)
    p.add_argument("--batch",        type=int, default=8)
    p.add_argument("--device",       default="0")
    p.add_argument("--weights",      default=None)
    args = p.parse_args()

    if args.collect:
        n = collect_coco_samples(args.max_samples)
        if n < 50:
            print("  Supplementing with synthetic samples...")
            _generate_synthetic_volleyball_balls(300)
    elif args.synthetic:
        _generate_synthetic_volleyball_balls(300)
    elif args.train:
        if not args.collect:
            # Auto-add synthetics if no COCO images
            train_imgs = list((STAGE_DIR / "train" / "images").glob("*.jpg"))
            if len(train_imgs) < 20:
                print("  No collected images — generating synthetic data first...")
                _generate_synthetic_volleyball_balls(300)
        train(args)
    elif args.eval:
        evaluate(args)
    else:
        print("Usage: python stream_coco_ball.py [--collect] [--train] [--eval]")


if __name__ == "__main__":
    main()
