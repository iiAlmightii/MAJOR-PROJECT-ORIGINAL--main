"""
Phase 3 Full Pipeline Runner
──────────────────────────────
Runs the complete action recognition pipeline in order:

  Step 1: Extract poses from Dataset/Action recognition/Spike.mp4 using tagged timestamps
  Step 2: Train Phase-1 LSTM (spike vs background)
  Step 3: Validate — cross-check against all 16 tagged timestamps
  Step 4: Print VERDICT (proceed / adjust)

Usage
─────
  python run_phase3_pipeline.py

  # With GPU (much faster):
  python run_phase3_pipeline.py --device 0

  # Ensure Dataset/Action recognition/Spike.mp4 and annotations.json exist first.
  # Skip extraction if already done:
  python run_phase3_pipeline.py --skip-extract

  # Skip training, just validate (must have weights):
  python run_phase3_pipeline.py --skip-extract --skip-train

Requirements
────────────
  pip install rtmlib torch numpy opencv-python matplotlib scikit-learn
  (rtmlib handles RTMPose ONNX weights auto-download)
"""

import subprocess
import sys
import os
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent
HERE    = Path(__file__).parent
VENV    = ROOT / "backend" / ".venv"

def run(cmd, desc):
    print(f"\n{'─'*60}")
    print(f"  ▶  {desc}")
    print(f"{'─'*60}")
    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\n  ✗ Step failed (exit code {result.returncode})")
        print("  Fix the error above and re-run, or skip with --skip-* flags")
        sys.exit(result.returncode)
    print(f"\n  ✓ Done")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-train",   action="store_true")
    p.add_argument("--epochs",       type=int, default=80)
    p.add_argument("--device",       default="cpu")
    p.add_argument("--plot",         action="store_true")
    args = p.parse_args()

    video_path = str(ROOT / "Dataset" / "Action recognition" / "Spike.mp4")
    ann_path   = str(ROOT / "Dataset" / "Action recognition" / "annotations.json")

    print("=" * 60)
    print("  VolleyVision — Phase 3 Action Recognition Pipeline")
    print("  Spike action (validation run)")
    print("=" * 60)

    # ── Step 1: Extract poses ──────────────────────────────────────────────
    if not args.skip_extract:
        run(
            [str(HERE / "extract_poses.py"),
             "--video",       video_path,
             "--annotations", ann_path,
             "--output",      str(HERE / "pose_data")],
            "STEP 1/3: Extracting RTMPose keypoints from Spike.mp4"
        )
    else:
        print("\n  [STEP 1 SKIPPED] Using existing pose data")

    # ── Step 2: Train LSTM ─────────────────────────────────────────────────
    if not args.skip_train:
        run(
            [str(HERE / "train_lstm.py"),
             "--phase",  "1",
             "--epochs", str(args.epochs),
             "--device", args.device],
            f"STEP 2/3: Training Pose-LSTM (phase 1, {args.epochs} epochs)"
        )
    else:
        print("\n  [STEP 2 SKIPPED] Using existing weights")

    # ── Step 3: Validate ──────────────────────────────────────────────────
    validate_cmd = [
        str(HERE / "validate_spike.py"),
        "--video",       video_path,
        "--annotations", ann_path,
    ]
    if args.plot:
        validate_cmd.append("--plot")

    run(validate_cmd, "STEP 3/3: Validating spike detection")

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Report: training/action_recognition/models/validation_report.json")
    if args.plot:
        print(f"  Plot:   training/action_recognition/models/validation_timeline.png")
    print()
    print("  NEXT STEPS:")
    print("  ─ If VERDICT = PROCEED:")
    print("    Download 4 more videos (serve, block, defense, set)")
    print("    Tag timestamps in annotations.json (same format)")
    print("    Run: python run_phase3_pipeline.py --skip-extract  (add to existing data)")
    print("    Then train Phase 2: python train_lstm.py --phase 2")
    print()
    print("  ─ If VERDICT = NEEDS WORK:")
    print("    Check validation_report.json for which timestamps failed")
    print("    Adjust annotations.json start/end times (add ±0.5s)")
    print("    Re-run this script")
    print("=" * 60)


if __name__ == "__main__":
    main()
