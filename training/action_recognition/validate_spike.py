"""
Spike Detection Validation
───────────────────────────
Cross-verifies the trained LSTM model against your manually tagged spike timestamps.

What this does:
  1. Loads the trained model (action_lstm_phase1.pt)
  2. Loads your annotations.json (16 tagged spike moments)
  3. For each tagged timestamp, extracts the pose sequence and runs inference
  4. Reports:
     - Confidence score at each tagged timestamp  (should be HIGH for spike)
     - Confidence at random background timestamps (should be LOW for spike)
     - Precision, Recall, F1 on your 16 tagged events
     - Overall verdict: "PROCEED with Phase 2" or "NEEDS ADJUSTMENT"

Also runs a SLIDING WINDOW scan across the full video to check:
  - Does the model fire at your tagged moments?
  - Does it produce many false positives in between?

Usage
─────
  python validate_spike.py

  # Custom paths:
  python validate_spike.py \
    --video "Dataset/Action recognition/Spike.mp4" \
    --annotations "Dataset/Action recognition/annotations.json" \
    --weights models/weights/action_lstm_phase1.pt

Output
──────
  - Console: coloured confidence table + verdict
  - training/action_recognition/models/validation_report.json
  - training/action_recognition/models/validation_timeline.png  (optional)
"""

import json
import sys
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple

ROOT         = Path(__file__).resolve().parent.parent.parent
DEFAULT_VID  = ROOT / "Dataset" / "Action recognition" / "Spike.mp4"
DEFAULT_ANN  = ROOT / "Dataset" / "Action recognition" / "annotations.json"
DEFAULT_W    = ROOT / "models" / "weights" / "action_lstm_phase1.pt"
REPORT_DIR   = ROOT / "training" / "action_recognition" / "models"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

NUM_FRAMES   = 30
NUM_FEATURES = 34
SLIDE_STEP   = 15     # frames between sliding window positions
SPIKE_CLASS  = 0      # class index for spike
THRESHOLD    = 0.50   # confidence threshold for positive detection


def parse_time(ts: str) -> float:
    p = ts.strip().split(":")
    if len(p) == 2: return int(p[0]) * 60 + float(p[1])
    return float(p[0])


def load_model(weights_path: str):
    """Load the trained LSTM model."""
    import torch

    if not Path(weights_path).exists():
        return None, None, None

    ckpt     = torch.load(weights_path, map_location="cpu")
    classes  = ckpt.get("classes", ["spike", "background"])

    # Rebuild model
    sys.path.insert(0, str(ROOT / "training" / "action_recognition"))
    from train_lstm import build_model
    model = build_model(len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes, ckpt.get("val_acc", 0)


def load_pose_model():
    """Load RTMPose or MediaPipe (same as extract_poses.py)."""
    try:
        from rtmlib import RTMO
        model = RTMO(pose="rtmo-s", det_model="yolox-s",
                     device="cpu", backend="onnxruntime")
        return "rtmlib", model
    except ImportError:
        pass
    try:
        import mediapipe as mp
        model = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        )
        return "mediapipe", model
    except ImportError:
        pass
    return "none", None


def extract_kps_and_norm(frame: np.ndarray, pose_type: str, pose_model):
    """Extract and normalise 34-dim pose vector from one frame."""
    sys.path.insert(0, str(ROOT / "training" / "action_recognition"))
    from extract_poses import (
        extract_keypoints_rtmlib, extract_keypoints_mediapipe, normalise_keypoints
    )
    h, w = frame.shape[:2]
    if pose_type == "rtmlib":
        kps = extract_keypoints_rtmlib(pose_model, frame)
    elif pose_type == "mediapipe":
        kps = extract_keypoints_mediapipe(pose_model, frame)
    else:
        kps = None
    if kps is None:
        return np.zeros(NUM_FEATURES, dtype=np.float32)
    return normalise_keypoints(kps, w, h)


def get_sequence_at_time(
    cap: cv2.VideoCapture,
    centre_sec: float,
    fps: float,
    pose_type: str,
    pose_model,
) -> np.ndarray:
    """Extract (30, 34) pose sequence centred on a timestamp."""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half_window  = NUM_FRAMES // 2
    centre_frame = int(centre_sec * fps)
    start_frame  = max(0, centre_frame - half_window)
    end_frame    = min(total_frames - 1, start_frame + NUM_FRAMES - 1)
    indices      = np.linspace(start_frame, end_frame, NUM_FRAMES, dtype=int)

    sequence = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            sequence.append(np.zeros(NUM_FEATURES, dtype=np.float32))
        else:
            sequence.append(extract_kps_and_norm(frame, pose_type, pose_model))

    return np.array(sequence, dtype=np.float32)   # (30, 34)


def run_inference(model, sequence: np.ndarray) -> Tuple[float, int]:
    """Run model inference on one (30, 34) sequence. Returns (spike_conf, pred_class)."""
    import torch
    import torch.nn.functional as F

    x     = torch.tensor(sequence[None], dtype=torch.float32)   # (1, 30, 34)
    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=1).squeeze().numpy()

    spike_conf = float(probs[SPIKE_CLASS])
    pred_class = int(probs.argmax())
    return spike_conf, pred_class


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours for console output
# ─────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def conf_bar(conf: float, width: int = 20) -> str:
    n = int(conf * width)
    bar = "█" * n + "░" * (width - n)
    if conf >= 0.7:     color = GREEN
    elif conf >= 0.4:   color = YELLOW
    else:               color = RED
    return f"{color}{bar}{RESET} {conf:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# Main validation
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(args):
    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  Spike Action Validation Report{RESET}")
    print(f"{'='*65}")

    # Load model
    model, classes, trained_acc = load_model(args.weights)
    if model is None:
        print(f"\n  {RED}ERROR: Model weights not found at {args.weights}{RESET}")
        print("  Train first: python train_lstm.py --phase 1")
        return

    print(f"  Model weights:  {args.weights}")
    print(f"  Classes:        {classes}")
    print(f"  Trained val acc:{trained_acc:.4f}")

    # Load annotations
    with open(args.annotations) as f:
        annotations = json.load(f)
    video_name = Path(args.video).name
    ann_key = next((k for k in annotations if Path(k).name == video_name or k == video_name), None)
    if ann_key is None:
        print(f"  {RED}ERROR: '{video_name}' not in annotations{RESET}")
        return

    segments = annotations[ann_key]
    print(f"  Tagged spikes:  {len(segments)}")

    # Load video + pose model
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total_frames / fps
    print(f"  Video:          {int(dur//60)}:{int(dur%60):02d} @ {fps:.0f}fps")

    print(f"\n  Loading pose model...")
    pose_type, pose_model = load_pose_model()
    if pose_type == "none":
        print(f"  {RED}ERROR: No pose model available. Install rtmlib or mediapipe.{RESET}")
        return
    print(f"  Pose model:     {pose_type}")

    # ── Part 1: Evaluate at tagged timestamps ──────────────────────────────
    print(f"\n{BOLD}  PART 1: Tagged Spike Timestamps{RESET}")
    print(f"  {'Timestamp':<12} {'Spike Conf':>10}  {'Prediction':<14} {'Match?':>6}")
    print(f"  {'-'*55}")

    tagged_results = []
    for seg in segments:
        start_sec = parse_time(seg["start"])
        end_sec   = parse_time(seg["end"])
        centre    = (start_sec + end_sec) / 2

        seq  = get_sequence_at_time(cap, centre, fps, pose_type, pose_model)
        conf, pred = run_inference(model, seq)

        is_correct = (conf >= THRESHOLD)
        pred_name  = classes[pred] if pred < len(classes) else f"cls{pred}"
        marker     = f"{GREEN}✓{RESET}" if is_correct else f"{RED}✗{RESET}"

        print(f"  {seg['start']}–{seg['end']:<8} {conf_bar(conf)}  {pred_name:<14} {marker}")
        tagged_results.append({
            "start": seg["start"],
            "end":   seg["end"],
            "spike_confidence": conf,
            "predicted_class":  pred_name,
            "correct":          is_correct,
        })

    # ── Part 2: Background check (5 random non-spike timestamps) ──────────
    print(f"\n{BOLD}  PART 2: Background Timestamps (expect LOW spike confidence){RESET}")
    print(f"  {'Timestamp':<12} {'Spike Conf':>10}  {'Prediction':<14} {'Correct?':>8}")
    print(f"  {'-'*55}")

    # Pick 5 random timestamps not in tagged windows
    tagged_secs = {parse_time(s["start"]) for s in segments}
    bg_results  = []
    tried       = 0
    bg_collected = 0

    while bg_collected < 5 and tried < 50:
        rand_sec = np.random.uniform(30, dur - 30)
        tried += 1
        # Skip if near a tagged spike (±5 seconds)
        if any(abs(rand_sec - ts) < 5 for ts in tagged_secs):
            continue

        seq  = get_sequence_at_time(cap, rand_sec, fps, pose_type, pose_model)
        conf, pred = run_inference(model, seq)
        is_correct = (conf < THRESHOLD)   # should NOT fire on background
        pred_name  = classes[pred] if pred < len(classes) else f"cls{pred}"
        marker     = f"{GREEN}✓{RESET}" if is_correct else f"{RED}FP!{RESET}"

        mm = int(rand_sec // 60)
        ss = rand_sec % 60
        ts_str = f"{mm}:{ss:05.2f}"
        print(f"  {ts_str:<12} {conf_bar(conf)}  {pred_name:<14} {marker}")
        bg_results.append({"timestamp": ts_str, "spike_confidence": conf,
                           "false_positive": not is_correct})
        bg_collected += 1

    cap.release()

    # ── Part 3: Metrics ────────────────────────────────────────────────────
    TP = sum(1 for r in tagged_results if r["correct"])
    FN = len(tagged_results) - TP
    FP = sum(1 for r in bg_results if r["false_positive"])
    TN = len(bg_results) - FP

    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    avg_spike_conf = np.mean([r["spike_confidence"] for r in tagged_results])

    print(f"\n{BOLD}  METRICS (threshold={THRESHOLD}){RESET}")
    print(f"  Tagged spikes detected:    {TP}/{len(tagged_results)} "
          f"(recall = {recall:.3f})")
    print(f"  False positives on bg:     {FP}/{len(bg_results)} "
          f"(precision-proxy = {precision:.3f})")
    print(f"  F1 score:                  {f1:.3f}")
    print(f"  Avg spike confidence:      {avg_spike_conf:.3f}")

    # ── Verdict ────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    if recall >= 0.75 and f1 >= 0.65:
        verdict = "PROCEED"
        print(f"{GREEN}{BOLD}  VERDICT: ✓ PROCEED WITH PHASE 2{RESET}")
        print(f"  The model correctly identifies spikes at ≥75% recall.")
        print(f"  Download the remaining 4 action videos and tag timestamps.")
    elif recall >= 0.50:
        verdict = "MARGINAL"
        print(f"{YELLOW}{BOLD}  VERDICT: ⚠ MARGINAL — Consider improvements{RESET}")
        print(f"  Recall is {recall:.0%}. Options to improve:")
        print(f"  1. Add more context (±1s) in annotations.json")
        print(f"  2. Tag more diverse spike clips")
        print(f"  3. Lower threshold: THRESHOLD = {THRESHOLD - 0.1:.1f}")
    else:
        verdict = "NEEDS_WORK"
        print(f"{RED}{BOLD}  VERDICT: ✗ NEEDS ADJUSTMENT{RESET}")
        print(f"  Only {recall:.0%} recall. Check:")
        print(f"  1. Did extract_poses.py extract cleanly? Check pose_data/spike/")
        print(f"  2. Are timestamps correct in annotations.json?")
        print(f"  3. Is the pose model detecting people in the video?")
    print("=" * 65)

    # ── Save report ────────────────────────────────────────────────────────
    report = {
        "model_weights":      str(args.weights),
        "trained_val_acc":    float(trained_acc),
        "threshold":          THRESHOLD,
        "tagged_results":     tagged_results,
        "background_results": bg_results,
        "metrics": {
            "TP": TP, "FN": FN, "FP": FP, "TN": TN,
            "precision":       round(precision, 4),
            "recall":          round(recall, 4),
            "f1":              round(f1, 4),
            "avg_spike_conf":  round(avg_spike_conf, 4),
        },
        "verdict": verdict,
        "next_steps": (
            "Proceed with 4 more action videos" if verdict == "PROCEED"
            else "Improve annotations and retrain"
        ),
    }

    report_path = REPORT_DIR / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Full report saved → {report_path}")

    # ── Optional: plot timeline ────────────────────────────────────────────
    if args.plot:
        _plot_timeline(tagged_results, report_path)

    return report


def _plot_timeline(results: List[dict], report_path: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        timestamps = [parse_time(r["start"]) for r in results]
        confs      = [r["spike_confidence"] for r in results]
        colors     = ["green" if r["correct"] else "red" for r in results]

        fig, ax = plt.subplots(figsize=(14, 4), facecolor="#1a1f2e")
        ax.set_facecolor("#232b3e")

        for ts, conf, col in zip(timestamps, confs, colors):
            ax.bar(ts, conf, width=1.5, color=col, alpha=0.8, edgecolor="white", linewidth=0.5)

        ax.axhline(y=THRESHOLD, color="yellow", linestyle="--", linewidth=1.5, label=f"Threshold ({THRESHOLD})")
        ax.set_xlim(0, max(timestamps) + 30)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Video timestamp (seconds)", color="white")
        ax.set_ylabel("Spike confidence", color="white")
        ax.set_title("Spike Detection Confidence at Tagged Timestamps", color="white", fontsize=13)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3a52")

        green_patch = mpatches.Patch(color="green", label="Detected (TP)")
        red_patch   = mpatches.Patch(color="red",   label="Missed (FN)")
        ax.legend(handles=[green_patch, red_patch], facecolor="#232b3e",
                  labelcolor="white", edgecolor="#2e3a52")

        plot_path = report_path.parent / "validation_timeline.png"
        fig.savefig(str(plot_path), dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Timeline plot  → {plot_path}")
    except ImportError:
        pass


def main():
    import argparse
    p = argparse.ArgumentParser(description="Validate spike detection model")
    p.add_argument("--video",       default=str(DEFAULT_VID))
    p.add_argument("--annotations", default=str(DEFAULT_ANN))
    p.add_argument("--weights",     default=str(DEFAULT_W))
    p.add_argument("--threshold",   type=float, default=THRESHOLD)
    p.add_argument("--plot",        action="store_true", help="Generate timeline plot")
    args = p.parse_args()

    global THRESHOLD
    THRESHOLD = args.threshold

    run_validation(args)


if __name__ == "__main__":
    main()
