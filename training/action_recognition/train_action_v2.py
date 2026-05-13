"""
Action Detection Training v2 — YOLOv8 on Volleyball Actions Dataset
──────────────────────────────────────────────────────────────────────
Trains YOLOv8n on the correct volleyball-specific actions dataset.

Dataset: Dataset/Action recognition/
  11,019 train images + 1,452 test images, 5 classes:
    block, defense, serve, set, spike

This is the RIGHT dataset for volleyball action detection:
  - Purpose-built volleyball action bounding box annotations
  - No noise classes (unlike Volleyball Activity Dataset which has "stand")
  - Classes map directly to volleyball game events

Why NOT use Volleyball Activity Dataset (17k images)?
  - Includes "stand" (dominant class, adds noise)
  - General volleyball player positions, not action-specific
  - Took hours to train with worse accuracy for action detection

Output: models/weights/action_detection.pt
  ActionService auto-loads this as YOLO fallback when LSTM weights absent.

YOLO class → DB ActionType mapping (in action_service.py):
  block   → ActionType.block
  defense → ActionType.dig
  serve   → ActionType.serve
  set     → ActionType.set
  spike   → ActionType.attack

Usage
─────
# Recommended (fast, 4GB VRAM, ~30-40 min):
  python train_action_v2.py

# More epochs for higher accuracy:
  python train_action_v2.py --epochs 60

# Evaluate existing weights:
  python train_action_v2.py --eval

For even better accuracy later (when you have match video clips):
  Use the LSTM pipeline: training/action_recognition/run_phase3_pipeline.py
  LSTM on pose keypoints is temporal (30-frame sequences) and much more
  accurate for "who performed which action". See CLAUDE.md for details.
"""

import argparse
import shutil
import yaml
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = ROOT / "Dataset" / "Action recognition"
WEIGHTS_OUT = ROOT / "models" / "weights"
WEIGHTS_OUT.mkdir(parents=True, exist_ok=True)


def fix_val_path(data_yaml: Path) -> Path:
    """
    Many Roboflow exports set val: ../valid/images but never include a valid/ split.
    This function patches the yaml in-place to use test/ if it exists, else train/.
    Returns the (possibly modified) yaml path.
    """
    dataset_dir = data_yaml.parent
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    val_path = cfg.get("val", "")
    # Resolve relative to dataset directory
    val_abs = (dataset_dir / val_path.lstrip("../")).resolve() if val_path else None

    if val_abs and val_abs.exists():
        return data_yaml   # already valid, no fix needed

    # Find a working split: prefer test/, fallback to train/
    for candidate in ["../test/images", "../train/images"]:
        candidate_abs = (dataset_dir / candidate.lstrip("../")).resolve()
        if candidate_abs.exists():
            cfg["val"] = candidate
            with open(data_yaml, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            print(f"  [fix] val path updated: {val_path} → {candidate}")
            return data_yaml

    print(f"  WARNING: Could not find a valid val split for {data_yaml}")
    return data_yaml


def train(args):
    from ultralytics import YOLO

    data_yaml = Path(args.data) if args.data else DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"ERROR: Dataset yaml not found at {data_yaml}")
        print("Use --data to point to a different dataset, e.g.:")
        print("  --data 'Dataset/Action Recognition 2/data.yaml'")
        return

    # Auto-fix broken val path (common Roboflow export issue)
    data_yaml = fix_val_path(data_yaml)

    # Read class names for display
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    class_names = cfg.get("names", [])
    nc = cfg.get("nc", len(class_names))

    print("=" * 65)
    print("  Action Detection Training v2  — YOLOv8")
    print(f"  Dataset : {data_yaml}")
    print(f"  Classes : {class_names}  ({nc} classes)")
    print(f"  Model   : {args.model}")
    print(f"  Epochs  : {args.epochs}  |  ImgSz: {args.imgsz}  |  Batch: {args.batch}")
    print(f"  FP16    : {not args.no_half} (saves VRAM)")
    print("=" * 65)

    model = YOLO(f"{args.model}.pt")   # always starts from COCO-pretrained backbone

    results = model.train(
        data    = str(data_yaml),
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device,
        half    = not args.no_half,        # FP16 — cuts VRAM usage by ~40%
        project = str(ROOT / "runs" / "action_detection_v2"),
        name    = "exp",

        # Hyperparameters tuned for volleyball actions
        lr0          = 0.01,
        lrf          = 0.01,
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs= 3,
        box          = 7.5,
        cls          = 1.0,               # higher cls weight — action class matters more than box accuracy
        dfl          = 1.5,
        patience     = 15,                # early stopping

        # Augmentation — important for generalising to different match footage
        hsv_h    = 0.015,
        hsv_s    = 0.7,
        hsv_v    = 0.4,
        degrees  = 3.0,                   # small rotation (volleyball courts are level)
        translate= 0.1,
        scale    = 0.5,
        flipud   = 0.0,                   # no vertical flip (players don't play upside down)
        fliplr   = 0.5,
        mosaic   = 0.8,
        mixup    = 0.0,                   # disabled — mosaic is enough for 11k images

        save_period = 10,
        plots       = True,
        verbose     = True,
        workers     = 4,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        dest = WEIGHTS_OUT / "action_detection.pt"
        shutil.copy(str(best), str(dest))
        print(f"\n  Best weights saved → {dest}")
        print("  The ActionService will auto-load this on next backend restart.")
        print("\n  Class → DB mapping:")
        print("    block   → ActionType.block")
        print("    defense → ActionType.dig")
        print("    serve   → ActionType.serve")
        print("    set     → ActionType.set")
        print("    spike   → ActionType.attack")
    else:
        print("  WARNING: best.pt not found — check training output")


def evaluate(args):
    from ultralytics import YOLO
    import yaml

    weights = args.weights or str(WEIGHTS_OUT / "action_detection.pt")
    if not Path(weights).exists():
        print(f"ERROR: weights not found at {weights}")
        return

    data_yaml = Path(args.data) if args.data else DATASET_DIR / "data.yaml"
    data_yaml = fix_val_path(data_yaml)

    print(f"\n  Evaluating: {weights}")
    model   = YOLO(weights)
    metrics = model.val(
        data   = str(data_yaml),
        imgsz  = args.imgsz,
        conf   = 0.4,
        device = args.device,
        plots  = True,
    )
    print(f"\n  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")
    print("\n  Per-class mAP50:")
    for i, name in model.names.items():
        try:
            print(f"    {name:<12}: {metrics.box.ap50[i]:.4f}")
        except (IndexError, AttributeError):
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO action detector on Volleyball Actions dataset (5 classes)"
    )
    parser.add_argument("--data",     default=None,
                        help="Path to data.yaml (default: Dataset/Action recognition/data.yaml). "
                             "Examples: 'Dataset/Action Recognition 2/data.yaml', "
                             "'Dataset/Action recognition 3/data.yaml'")
    parser.add_argument("--model",    default="yolov8n",
                        choices=["yolov8n", "yolov8s", "yolov8m"],
                        help="yolov8n = fastest (4GB VRAM), yolov8s = better accuracy")
    parser.add_argument("--epochs",   type=int,   default=40,
                        help="Number of epochs (40 is enough for 11k images)")
    parser.add_argument("--imgsz",    type=int,   default=640)
    parser.add_argument("--batch",    type=int,   default=12,
                        help="Batch size (12 fits in 4GB VRAM with FP16)")
    parser.add_argument("--device",   default="0",
                        help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--no-half",  action="store_true",
                        help="Disable FP16 (uses more VRAM but may be more stable)")
    parser.add_argument("--eval",     action="store_true",
                        help="Evaluate existing weights instead of training")
    parser.add_argument("--weights",  default=None,
                        help="Custom weights path for --eval")
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
