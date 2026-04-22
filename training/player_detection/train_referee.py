"""
Referee Detector Training
──────────────────────────
Trains YOLOv8 to detect volleyball referees so they can be excluded
from the player tracking overlay in the CV pipeline.

Dataset: Dataset/Refree detector/  (665 train, 100 val images, 1 class)
Output:  models/weights/referee_detection.pt

The CV pipeline loads this model and suppresses any player detection that
overlaps significantly with a referee bounding box (IoU > 0.3).

Usage
─────
  python train_referee.py --epochs 60
  python train_referee.py --eval
"""

import argparse
import shutil
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent.parent
DATA_YAML   = ROOT / "Dataset" / "Refree detector" / "data.yaml"
WEIGHTS_OUT = ROOT / "models" / "weights"
WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)


def train(args):
    from ultralytics import YOLO

    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} not found")
        return

    print("=" * 60)
    print("  Referee Detector Training  — YOLOv8")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  Model:   {args.model}  Epochs: {args.epochs}")
    print("=" * 60)

    model = YOLO(f"{args.model}.pt")

    results = model.train(
        data    = str(DATA_YAML),
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device,
        project = str(ROOT / "runs" / "referee_detection"),
        name    = "exp",

        lr0          = 0.005,
        lrf          = 0.01,
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs= 3,
        box          = 7.5,
        cls          = 0.5,
        dfl          = 1.5,
        patience     = 25,

        hsv_h    = 0.02,
        hsv_s    = 0.7,
        hsv_v    = 0.4,
        degrees  = 5.0,
        translate= 0.1,
        scale    = 0.5,
        flipud   = 0.0,
        fliplr   = 0.5,
        mosaic   = 1.0,
        mixup    = 0.1,

        save_period = 10,
        plots       = True,
        verbose     = True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        dest = WEIGHTS_OUT / "referee_detection.pt"
        shutil.copy(str(best), str(dest))
        print(f"\n  Best weights saved → {dest}")
    else:
        print("  best.pt not found — check training output")


def evaluate(args):
    from ultralytics import YOLO
    weights = args.weights or str(WEIGHTS_OUT / "referee_detection.pt")
    if not Path(weights).exists():
        print(f"ERROR: weights not found at {weights}")
        return
    model = YOLO(weights)
    m = model.val(data=str(DATA_YAML), imgsz=args.imgsz, conf=0.45, device=args.device)
    print(f"\n  mAP50: {m.box.map50:.4f}  mAP50-95: {m.box.map:.4f}")
    print(f"  Precision: {m.box.mp:.4f}  Recall: {m.box.mr:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train volleyball referee detector")
    parser.add_argument("--model",   default="yolov8n",
                        choices=["yolov8n", "yolov8s", "yolov8m"])
    parser.add_argument("--epochs",  type=int, default=60)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--device",  default="0")
    parser.add_argument("--eval",    action="store_true")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
