"""
Player Detection Training Script
──────────────────────────────────
Trains YOLOv8 to detect volleyball players with consistent bounding boxes.

Strategy
--------
1. Start from YOLOv8n/s/m pre-trained on COCO (person detection already works)
2. Fine-tune on volleyball-specific images for better precision on court
3. Track two classes: team_a (blue) and team_b (red) for automatic team detection

Usage
-----
# Quickstart (no custom data — COCO pretrain works out of the box):
python train_player.py --use-coco-pretrain --model yolov8n

# With custom dataset:
python train_player.py --data datasets/player_detection/dataset.yaml \
                       --model yolov8s --epochs 100 --imgsz 1280

# Resume interrupted training:
python train_player.py --resume runs/player_detection/exp/weights/last.pt
"""

import argparse
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
WEIGHTS_OUT  = PROJECT_ROOT / "models" / "weights"
WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)


def train(args):
    from ultralytics import YOLO

    print("=" * 60)
    print("  Player Detection Training  — YOLOv8")
    print("=" * 60)

    if args.resume:
        print(f"  Resuming from: {args.resume}")
        model = YOLO(args.resume)
        results = model.train(resume=True)
    elif args.use_coco_pretrain:
        print(f"  Using COCO pretrained {args.model}.pt (person class)")
        print("  TIP: This already detects players well without fine-tuning!")
        model = YOLO(f"{args.model}.pt")
        print(f"  Model loaded: {args.model}.pt")
        print("  Saving as player_detection.pt ...")
        shutil.copy(
            str(Path.home() / ".ultralytics" / "assets" / f"{args.model}.pt"),
            str(WEIGHTS_OUT / "player_detection.pt")
        )
        print(f"  Saved to {WEIGHTS_OUT / 'player_detection.pt'}")
        return
    else:
        print(f"  Training from {args.model}.pt on {args.data}")
        model = YOLO(f"{args.model}.pt")

        results = model.train(
            data    = args.data,
            epochs  = args.epochs,
            imgsz   = args.imgsz,
            batch   = args.batch,
            device  = args.device,
            project = "runs/player_detection",
            name    = "exp",
            # Hyperparameters
            lr0         = 0.01,
            lrf         = 0.01,
            momentum    = 0.937,
            weight_decay= 0.0005,
            warmup_epochs=3.0,
            box         = 7.5,
            cls         = 0.5,
            dfl         = 1.5,
            # Augmentation
            hsv_h   = 0.015,
            hsv_s   = 0.7,
            hsv_v   = 0.4,
            degrees = 10.0,
            translate=0.1,
            scale   = 0.5,
            flipud  = 0.0,
            fliplr  = 0.5,
            mosaic  = 1.0,
            mixup   = 0.1,
            # Save best
            save_period = 10,
        )

        # Copy best weights
        best = Path(results.save_dir) / "weights" / "best.pt"
        if best.exists():
            dest = WEIGHTS_OUT / "player_detection.pt"
            shutil.copy(str(best), str(dest))
            print(f"\n  Best weights saved → {dest}")


def evaluate(args):
    from ultralytics import YOLO
    weights = args.weights or str(WEIGHTS_OUT / "player_detection.pt")
    print(f"\n  Evaluating {weights} on {args.data}")
    model   = YOLO(weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, conf=0.45, iou=0.45)
    print(f"\n  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision:{metrics.box.mp:.4f}")
    print(f"  Recall:   {metrics.box.mr:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train volleyball player detector")
    parser.add_argument("--data",       default="Dataset/Refree detector/data.yaml")
    parser.add_argument("--model",      default="yolov8n",
                        choices=["yolov8n","yolov8s","yolov8m","yolov8l","yolov8x"])
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--imgsz",      type=int, default=1280)
    parser.add_argument("--batch",      type=int, default=16)
    parser.add_argument("--device",     default="0",
                        help="GPU id (0/1) or 'cpu'")
    parser.add_argument("--use-coco-pretrain", action="store_true",
                        help="Skip training; use COCO pretrain directly")
    parser.add_argument("--resume",     default=None,
                        help="Resume from checkpoint .pt")
    parser.add_argument("--eval",       action="store_true",
                        help="Evaluate existing weights instead of training")
    parser.add_argument("--weights",    default=None,
                        help="Weights to evaluate (with --eval)")
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
