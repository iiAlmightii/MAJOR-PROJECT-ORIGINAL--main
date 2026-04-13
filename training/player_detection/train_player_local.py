"""
Player / Referee Detection Training — Local Dataset
────────────────────────────────────────────────────
Uses the local referee detection dataset at:
  Dataset/Refree detector/
  - Classes defined in training/player_detection/dataset_local.yaml

This trains YOLOv8 to detect referees and players on court.

Usage
─────
# Quick fine-tune (recommended — fast, good results on 177 images):
  python train_player_local.py --model yolov8n --epochs 80

# More powerful model:
  python train_player_local.py --model yolov8s --epochs 150 --imgsz 1280

# Evaluation only:
  python train_player_local.py --eval
"""

import argparse
import shutil
import os
from pathlib import Path

ROOT         = Path(__file__).resolve().parent.parent.parent
DATA_YAML    = ROOT / "training" / "player_detection" / "dataset_local.yaml"
WEIGHTS_OUT  = ROOT / "models" / "weights"
WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)


def train(args):
    from ultralytics import YOLO

    print("=" * 60)
    print("  Player Detection Training  (Local Dataset)")
    print(f"  Data:   {DATA_YAML}")
    print(f"  Model:  {args.model}")
    print(f"  Epochs: {args.epochs}  |  ImgSz: {args.imgsz}  |  Batch: {args.batch}")
    print("=" * 60)

    model = YOLO(f"{args.model}.pt")   # downloads pretrained weights if needed

    results = model.train(
        data    = str(DATA_YAML),
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device,
        project = str(ROOT / "runs" / "player_detection"),
        name    = "local_exp",

        # ── Hyperparameters tuned for small dataset (177 images) ──
        lr0          = 0.005,        # lower LR to avoid overfitting on small data
        lrf          = 0.01,
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs= 5,
        box          = 7.5,
        cls          = 0.5,
        dfl          = 1.5,
        patience     = 30,           # early stop if no improvement

        # ── Augmentation — heavy, compensates for small dataset ──
        hsv_h    = 0.02,
        hsv_s    = 0.8,
        hsv_v    = 0.5,
        degrees  = 10.0,
        translate= 0.15,
        scale    = 0.6,
        flipud   = 0.1,
        fliplr   = 0.5,
        mosaic   = 1.0,
        mixup    = 0.15,
        copy_paste=0.1,

        # ── Logging ──
        save_period  = 20,
        plots        = True,
        verbose      = True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        dest = WEIGHTS_OUT / "player_detection.pt"
        shutil.copy(str(best), str(dest))
        print(f"\n  ✓ Best weights → {dest}")
        print(f"  mAP50 achieved: check runs/player_detection/local_exp/")
    else:
        print("  ✗ best.pt not found — check training output")


def evaluate(args):
    from ultralytics import YOLO

    weights = args.weights or str(WEIGHTS_OUT / "player_detection.pt")
    if not os.path.exists(weights):
        print(f"  ERROR: weights not found at {weights}")
        print("  Run training first: python train_player_local.py")
        return

    print(f"\n  Evaluating: {weights}")
    model   = YOLO(weights)
    metrics = model.val(
        data   = str(DATA_YAML),
        imgsz  = args.imgsz,
        conf   = 0.45,
        iou    = 0.45,
        device = args.device,
        plots  = True,
    )

    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    print()
    print("  Per-class mAP50:")
    model_classes = YOLO(weights).names
    for i, name in model_classes.items():
        try:
            print(f"    {name:<12}: {metrics.box.ap50[i]:.4f}")
        except (IndexError, AttributeError):
            pass
    print("=" * 50)


def infer_sample(args):
    """Run inference on a sample image and display results."""
    from ultralytics import YOLO
    import glob

    weights = args.weights or str(WEIGHTS_OUT / "player_detection.pt")
    model   = YOLO(weights)
    class_names = model.names  # read from model, not hardcoded

    import yaml
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    dataset_root = Path(DATA_YAML).parent / Path(cfg.get("path", ""))
    val_rel      = cfg.get("val", "valid/images")
    val_dir      = dataset_root / val_rel

    val_images = glob.glob(str(val_dir / "*.jpg"))
    if not val_images:
        print(f"  No validation images found under {val_dir}")
        return

    print(f"  Running inference on {len(val_images)} validation images...")
    results = model.predict(
        source    = val_images[:5],
        conf      = 0.45,
        save      = True,
        save_dir  = str(ROOT / "runs" / "player_detection" / "inference"),
        line_width= 2,
    )
    for r in results:
        boxes = r.boxes
        print(f"  {Path(r.path).name}: {len(boxes)} detections")
        for box in boxes:
            cls  = int(box.cls)
            conf = float(box.conf)
            name = class_names.get(cls, f"cls{cls}")
            print(f"    [{name}] conf={conf:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate volleyball player detector on local dataset")
    parser.add_argument("--model",   default="yolov8n",
                        choices=["yolov8n","yolov8s","yolov8m","yolov8l"])
    parser.add_argument("--epochs",  type=int, default=80)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--device",  default="0",
                        help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--eval",    action="store_true",
                        help="Run evaluation on val set")
    parser.add_argument("--infer",   action="store_true",
                        help="Run inference on sample validation images")
    parser.add_argument("--weights", default=None,
                        help="Path to weights for --eval or --infer")
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    elif args.infer:
        infer_sample(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
