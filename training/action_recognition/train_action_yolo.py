"""
Action Detection Training — YOLOv8 on Volleyball Activity Dataset
──────────────────────────────────────────────────────────────────
Trains a YOLOv8 detector to recognise volleyball actions directly
from full-frame images without needing pose estimation.

Dataset: Dataset/Volleyball Activity Dataset.v1i.yolov8 (1)/
  17 495 training images, 7 classes:
    Defense-Move, attack, block, reception, service, setting, stand

Output: models/weights/action_detection.pt
  Used by ActionService as a fallback when LSTM weights are absent.

Usage
─────
# Recommended (quick, good accuracy):
  python train_action_yolo.py --model yolov8s --epochs 50

# More accurate (slower, needs more VRAM):
  python train_action_yolo.py --model yolov8m --epochs 100 --imgsz 1280

# Evaluate only:
  python train_action_yolo.py --eval
"""

import argparse
import shutil
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = ROOT / "Dataset" / "Volleyball Activity Dataset.v1i.yolov8 (1)"
DATA_YAML   = DATASET_DIR / "data.yaml"
WEIGHTS_OUT = ROOT / "models" / "weights"
WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)


def train(args):
    from ultralytics import YOLO

    if not DATA_YAML.exists():
        print(f"ERROR: Dataset not found at {DATA_YAML}")
        print("Make sure the Volleyball Activity Dataset is in Dataset/ folder.")
        return

    print("=" * 60)
    print("  Action Detection Training  — YOLOv8")
    print(f"  Dataset: {DATA_YAML}")
    print(f"  Model:   {args.model}")
    print(f"  Epochs:  {args.epochs}  |  ImgSz: {args.imgsz}")
    print("=" * 60)

    model = YOLO(f"{args.model}.pt")

    results = model.train(
        data    = str(DATA_YAML),
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device,
        project = str(ROOT / "runs" / "action_detection"),
        name    = "exp",

        # Hyperparameters
        lr0          = 0.005,
        lrf          = 0.01,
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs= 3,
        box          = 7.5,
        cls          = 0.8,       # slightly higher cls weight for action classes
        dfl          = 1.5,
        patience     = 20,

        # Augmentation — important for action generalisation
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
        dest = WEIGHTS_OUT / "action_detection.pt"
        shutil.copy(str(best), str(dest))
        print(f"\n  Best weights saved → {dest}")
        print("  The ActionService will auto-load this on next analysis run.")
    else:
        print("  best.pt not found — check training output")


def evaluate(args):
    from ultralytics import YOLO

    weights = args.weights or str(WEIGHTS_OUT / "action_detection.pt")
    if not Path(weights).exists():
        print(f"ERROR: weights not found at {weights}")
        return

    print(f"\n  Evaluating: {weights}")
    model   = YOLO(weights)
    metrics = model.val(
        data   = str(DATA_YAML),
        imgsz  = args.imgsz,
        conf   = 0.45,
        device = args.device,
        plots  = True,
    )
    print(f"\n  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    print("\n  Per-class mAP50:")
    for i, name in model.names.items():
        try:
            print(f"    {name:<16}: {metrics.box.ap50[i]:.4f}")
        except (IndexError, AttributeError):
            pass


def main():
    parser = argparse.ArgumentParser(description="Train YOLO action detector for volleyball")
    parser.add_argument("--model",   default="yolov8s",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l"])
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--device",  default="0",
                        help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--eval",    action="store_true")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
