"""
Pose Extraction Pipeline
─────────────────────────
Extracts RTMPose keypoints from tagged video clips.

Pipeline per annotated segment:
  1. Parse timestamp JSON  →  (start_sec, end_sec, label)
  2. Extract frames from the video at those timestamps
  3. Detect person in each frame using YOLOv8 (or COCO pretrain)
  4. Run RTMPose (via rtmlib) to get 17 COCO keypoints per person
  5. Normalise keypoints (relative to bounding box centre)
  6. Build a fixed-length (30-frame) sequence per clip
  7. Save to  pose_data/{label}/clip_{n}.npy  (shape: 30 × 34)

The 30×34 arrays are the training input for the LSTM.
(17 keypoints × 2 coordinates = 34 features per frame)

Usage
─────
  # Extract poses from Spike.mp4
  python extract_poses.py \
    --video "Dataset/Action recognition/Spike.mp4" \
    --annotations "Dataset/Action recognition/annotations.json" \
    --output training/action_recognition/pose_data

  # Dry-run: show what would be extracted
  python extract_poses.py --dry-run

Requirements
────────────
  pip install rtmlib ultralytics opencv-python numpy
  (rtmlib wraps RTMPose — no mmpose/mmcv needed)
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional

ROOT         = Path(__file__).resolve().parent.parent.parent
DEFAULT_VID  = ROOT / "Dataset" / "Action recognition" / "Spike.mp4"
DEFAULT_ANN  = ROOT / "Dataset" / "Action recognition" / "annotations.json"
DEFAULT_OUT  = ROOT / "training" / "action_recognition" / "pose_data"

NUM_FRAMES   = 30        # fixed temporal window per clip
NUM_KPS      = 17        # COCO keypoints
CONTEXT_PAD  = 0.5       # seconds of context added around each annotation
MIN_CONF     = 0.3       # minimum keypoint confidence to accept


def parse_timestamp(ts: str) -> float:
    """'1:37' → 97.0 seconds"""
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(parts[0])


def load_annotations(ann_path: str) -> Dict[str, List[dict]]:
    with open(ann_path) as f:
        return json.load(f)


def load_pose_model():
    """
    Load RTMPose via rtmlib (lightweight, no mmpose dependency).
    Falls back to MediaPipe if rtmlib is not installed.
    """
    try:
        from rtmlib import RTMO, Wholebody, PoseTracker
        print("  [Pose] Using RTMPose (rtmlib)")
        # RTMO — single-stage, faster; auto-downloads ONNX weights
        pose_model = RTMO(
            pose="rtmo-s",         # small model, fast
            det_model="yolox-s",   # built-in person detector
            device="cpu",          # change to "cuda" if available
            backend="onnxruntime",
        )
        return "rtmlib", pose_model
    except ImportError:
        pass

    try:
        import mediapipe as mp
        print("  [Pose] Using MediaPipe (rtmlib not installed)")
        pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        )
        return "mediapipe", pose
    except ImportError:
        pass

    print("  [Pose] WARNING: No pose estimator found.")
    print("  Install: pip install rtmlib   (recommended)")
    print("  Or:      pip install mediapipe")
    return "none", None


def extract_keypoints_rtmlib(model, frame: np.ndarray) -> Optional[np.ndarray]:
    """Returns shape (17, 3): x, y, confidence — or None if no person."""
    try:
        keypoints, scores = model(frame)   # rtmlib returns (N_persons, 17, 2), (N, 17)
        if keypoints is None or len(keypoints) == 0:
            return None
        # Pick person with highest mean score
        best = int(scores.mean(axis=1).argmax())
        kps   = keypoints[best]       # (17, 2)
        confs = scores[best]          # (17,)
        return np.concatenate([kps, confs[:, None]], axis=1)   # (17, 3)
    except Exception as e:
        return None


def extract_keypoints_mediapipe(model, frame: np.ndarray) -> Optional[np.ndarray]:
    """Returns shape (17, 3): x, y, confidence."""
    import mediapipe as mp
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = model.process(rgb)
    if not res.pose_landmarks:
        return None
    h, w = frame.shape[:2]
    # MediaPipe 33 landmarks → map to 17 COCO keypoints
    MP_TO_COCO = [0,12,11,14,13,16,15,24,23,26,25,28,27,2,1,4,3]
    lm = res.pose_landmarks.landmark
    kps = np.zeros((17, 3), dtype=np.float32)
    for i, idx in enumerate(MP_TO_COCO):
        l = lm[idx]
        kps[i] = [l.x * w, l.y * h, l.visibility]
    return kps


def normalise_keypoints(kps: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Normalise (17, 3) keypoints to be scale/position invariant.
    Strategy: subtract torso midpoint, divide by torso height.
    Falls back to image-size normalisation if torso not found.
    """
    valid = kps[:, 2] > MIN_CONF
    if valid.sum() < 4:
        # Minimal fallback
        out = kps[:, :2].copy()
        out[:, 0] /= img_w
        out[:, 1] /= img_h
        return out.flatten()   # shape (34,)

    # COCO: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
    torso_indices = [5, 6, 11, 12]
    torso_pts = kps[torso_indices, :2]
    centre    = torso_pts.mean(axis=0)

    hip_mid   = kps[[11, 12], :2].mean(axis=0)
    shldr_mid = kps[[5, 6],   :2].mean(axis=0)
    scale     = np.linalg.norm(shldr_mid - hip_mid) + 1e-6

    normed = (kps[:, :2] - centre) / scale
    # Zero out low-confidence keypoints
    normed[~valid] = 0.0
    return normed.flatten()   # shape (34,)


def extract_clip_sequence(
    cap: cv2.VideoCapture,
    start_sec: float,
    end_sec:   float,
    fps:       float,
    pose_type: str,
    pose_model,
    target_frames: int = NUM_FRAMES,
) -> Optional[np.ndarray]:
    """
    Extract a (target_frames × 34) numpy array for one clip.
    Returns None if fewer than 5 valid pose frames found.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_f = int(start_sec * fps)
    end_f   = int(end_sec   * fps)
    clip_len = max(end_f - start_f, 1)

    # Sample target_frames evenly from the clip
    sample_indices = np.linspace(start_f, end_f - 1, target_frames, dtype=int)
    sample_indices = np.clip(sample_indices, 0, total_frames - 1)

    sequence = []
    for fi in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            sequence.append(np.zeros(NUM_KPS * 2, dtype=np.float32))
            continue

        if pose_type == "rtmlib":
            kps = extract_keypoints_rtmlib(pose_model, frame)
        elif pose_type == "mediapipe":
            kps = extract_keypoints_mediapipe(pose_model, frame)
        else:
            kps = None

        if kps is not None:
            normed = normalise_keypoints(kps, img_w, img_h)
        else:
            normed = np.zeros(NUM_KPS * 2, dtype=np.float32)
        sequence.append(normed)

    seq = np.array(sequence, dtype=np.float32)   # (30, 34)

    # Reject clip if more than 50% of frames have zero pose
    zero_rows = (seq.sum(axis=1) == 0).sum()
    if zero_rows > target_frames * 0.5:
        return None

    return seq


def run_extraction(args):
    video_path = Path(args.video)
    ann_path   = Path(args.annotations)
    out_dir    = Path(args.output)

    if not video_path.exists():
        print(f"  ERROR: Video not found: {video_path}")
        sys.exit(1)
    if not ann_path.exists():
        print(f"  ERROR: Annotations not found: {ann_path}")
        sys.exit(1)

    annotations = load_annotations(str(ann_path))
    video_name  = video_path.name

    # Find the entry for this video
    ann_key = None
    for key in annotations:
        if Path(key).name == video_name or key == video_name:
            ann_key = key
            break

    if ann_key is None:
        print(f"  ERROR: No annotations found for '{video_name}' in {ann_path}")
        print(f"  Available keys: {list(annotations.keys())}")
        sys.exit(1)

    segments = annotations[ann_key]
    print(f"\n  Video: {video_path}")
    print(f"  Annotations: {len(segments)} segments for '{ann_key}'")

    # Load video
    cap  = cv2.VideoCapture(str(video_path))
    fps  = cap.get(cv2.CAP_PROP_FPS)
    dur  = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    print(f"  Video: {int(dur//60)}:{int(dur%60):02d} @ {fps:.1f}fps")

    # Load pose model
    print("\n  Loading pose estimator...")
    pose_type, pose_model = load_pose_model()

    if pose_type == "none" and not args.dry_run:
        print("  Cannot extract poses without a pose model. Install rtmlib or mediapipe.")
        cap.release()
        sys.exit(1)

    # Process each segment
    results = {"extracted": [], "skipped": [], "errors": []}
    t0 = time.time()

    for i, seg in enumerate(segments):
        label     = seg["label"]
        start_sec = parse_timestamp(seg["start"]) - CONTEXT_PAD
        end_sec   = parse_timestamp(seg["end"])   + CONTEXT_PAD
        start_sec = max(0, start_sec)
        end_sec   = min(dur, end_sec)

        out_class_dir = out_dir / label
        out_class_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_class_dir / f"clip_{i:04d}.npy"

        print(f"\n  [{i+1}/{len(segments)}] {label}  "
              f"{seg['start']}→{seg['end']}  ({start_sec:.1f}s–{end_sec:.1f}s)")

        if args.dry_run:
            print(f"    [DRY-RUN] Would save to {out_path}")
            continue

        if out_path.exists() and not args.force:
            print(f"    Already exists, skipping (--force to re-extract)")
            results["extracted"].append(str(out_path))
            continue

        try:
            seq = extract_clip_sequence(
                cap, start_sec, end_sec, fps, pose_type, pose_model,
                target_frames=NUM_FRAMES,
            )
            if seq is None:
                print(f"    SKIPPED — too few valid pose frames")
                results["skipped"].append(f"{label}_{i}")
            else:
                np.save(str(out_path), seq)
                print(f"    ✓ Saved {seq.shape} → {out_path.name}")
                results["extracted"].append(str(out_path))
        except Exception as e:
            print(f"    ERROR: {e}")
            results["errors"].append(f"{label}_{i}: {e}")

    cap.release()

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  Extraction complete in {elapsed:.1f}s")
    print(f"  Extracted: {len(results['extracted'])}")
    print(f"  Skipped:   {len(results['skipped'])}")
    print(f"  Errors:    {len(results['errors'])}")
    print(f"  Output:    {out_dir}")

    # Save extraction report
    report_path = out_dir / "extraction_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Report:    {report_path}")

    return results


def main():
    p = argparse.ArgumentParser(description="Extract RTMPose keypoints from annotated clips")
    p.add_argument("--video",       default=str(DEFAULT_VID), help="Path to video file")
    p.add_argument("--annotations", default=str(DEFAULT_ANN), help="Path to annotations JSON")
    p.add_argument("--output",      default=str(DEFAULT_OUT), help="Output directory for .npy files")
    p.add_argument("--dry-run",     action="store_true",      help="Preview without extracting")
    p.add_argument("--force",       action="store_true",      help="Re-extract existing clips")
    args = p.parse_args()

    print("=" * 55)
    print("  Pose Extraction Pipeline  (RTMPose / MediaPipe)")
    print("=" * 55)
    run_extraction(args)


if __name__ == "__main__":
    main()
