"""
Ball Detection Training Script
────────────────────────────────
Trains a small YOLOv8 specifically for volleyball ball detection.

Key challenges:
  - Ball is very small (8-25px radius in broadcast footage)
  - Extremely fast motion → motion blur
  - Occlusion by players, net, hands

Solutions applied:
  - High input resolution (1280px)
  - YOLOv8n-p2 (adds P2 head for small objects)
  - Heavy mosaic + copy-paste augmentation
  - Synthetic training data included

Usage
-----
# Generate synthetic data first:
python ../../training/datasets/download_datasets.py --task ball

# Train:
python train_ball.py --epochs 200 --imgsz 1280

# With real data mixed in:
python train_ball.py --data dataset.yaml --epochs 200 --imgsz 1280 --batch 8
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
WEIGHTS_OUT  = PROJECT_ROOT / "models" / "weights"
WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)


def train(args):
    from ultralytics import YOLO

    print("=" * 60)
    print("  Ball Detection Training  — YOLOv8")
    print("=" * 60)
    print(f"  Data:   {args.data}")
    print(f"  Model:  {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  ImgSz:  {args.imgsz}")

    model = YOLO(f"{args.model}.pt")

    results = model.train(
        data    = args.data,
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device,
        project = "runs/ball_detection",
        name    = "exp",
        # Tuned for small object detection
        lr0         = 0.005,
        lrf         = 0.005,
        momentum    = 0.937,
        weight_decay= 0.0005,
        warmup_epochs=5.0,
        box         = 10.0,     # higher box loss weight for small objects
        cls         = 0.3,
        dfl         = 1.5,
        # Augmentation — aggressive for ball
        hsv_h    = 0.02,
        hsv_s    = 0.9,
        hsv_v    = 0.5,
        degrees  = 15.0,
        translate= 0.2,
        scale    = 0.7,
        flipud   = 0.3,
        fliplr   = 0.5,
        mosaic   = 1.0,
        mixup    = 0.2,
        copy_paste=0.3,
        # Save
        save_period=20,
        patience   =50,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        dest = WEIGHTS_OUT / "ball_detection.pt"
        shutil.copy(str(best), str(dest))
        print(f"\n  Best weights saved → {dest}")


def evaluate(weights: str, data: str, imgsz: int):
    from ultralytics import YOLO
    model   = YOLO(weights)
    metrics = model.val(data=data, imgsz=imgsz, conf=0.3, iou=0.3)
    print(f"\n  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train volleyball ball detector")
    parser.add_argument("--data",   default="Dataset/Ball detector/data.yaml")
    parser.add_argument("--model",  default="yolov8n",
                        choices=["yolov8n","yolov8s","yolov8m"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz",  type=int, default=1280)
    parser.add_argument("--batch",  type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--eval",   action="store_true")
    parser.add_argument("--weights",default=None)
    args = parser.parse_args()

    if args.eval:
        w = args.weights or str(WEIGHTS_OUT / "ball_detection.pt")
        evaluate(w, args.data, args.imgsz)
    else:
        train(args)


if __name__ == "__main__":
    main()
