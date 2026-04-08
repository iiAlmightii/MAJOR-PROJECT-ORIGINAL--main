"""
Action Recognition Service
───────────────────────────
Integrates the trained Pose+LSTM model into the CV pipeline.

Given a frame + bounding box of a player, this service:
  1. Crops the player region
  2. Runs RTMPose to get 17 keypoints
  3. Queues the frame in a sliding temporal window (30 frames)
  4. Runs the LSTM when the window is full
  5. Returns the detected action + confidence

Used by cv_pipeline.py during match analysis.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from collections import deque

logger = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parent.parent.parent.parent
WEIGHTS_PATH = ROOT / "models" / "weights" / "action_lstm_phase1.pt"  # grows to phase2 later
CLASS_MAP_PATH = ROOT / "training" / "action_recognition" / "models" / "class_map.json"

NUM_FRAMES   = 30
NUM_FEATURES = 34
ACTION_THRESHOLD = 0.55    # minimum confidence to report an action


class ActionService:
    """
    Per-match action recogniser.
    Maintains a separate temporal frame buffer per tracked player.
    """

    def __init__(self):
        self._model      = None
        self._classes: List[str] = []
        self._loaded     = False
        self._pose_type  = "none"
        self._pose_model = None
        self._torch_device = "cpu"
        # Per-player sliding window: track_id → deque of (34,) vectors
        self._buffers: Dict[int, deque] = {}

    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load LSTM model and pose estimator."""
        # Prefer CUDA when available for LSTM inference.
        try:
            import torch
            self._torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self._torch_device = "cpu"

        # ── Pose model ──
        pose_loaded = False
        try:
            from rtmlib import RTMO
            preferred_pose_device = "cuda" if self._torch_device == "cuda" else "cpu"

            try:
                self._pose_model = RTMO(
                    pose="rtmo-s", det_model="yolox-s",
                    device=preferred_pose_device, backend="onnxruntime",
                )
                self._pose_type = "rtmlib"
                pose_loaded = True
                logger.info(f"ActionService: RTMPose loaded on {preferred_pose_device}")
            except Exception as exc:
                logger.warning(f"ActionService: RTMPose init failed on {preferred_pose_device}: {exc}")
                if preferred_pose_device == "cuda":
                    try:
                        self._pose_model = RTMO(
                            pose="rtmo-s", det_model="yolox-s",
                            device="cpu", backend="onnxruntime",
                        )
                        self._pose_type = "rtmlib"
                        pose_loaded = True
                        logger.info("ActionService: RTMPose fallback loaded on cpu")
                    except Exception as cpu_exc:
                        logger.warning(f"ActionService: RTMPose cpu fallback failed: {cpu_exc}")
        except ImportError:
            pass

        if not pose_loaded:
            try:
                import mediapipe as mp
                self._pose_model = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._pose_type = "mediapipe"
                logger.info("ActionService: MediaPipe loaded")
            except Exception as exc:
                logger.warning(f"ActionService: No usable pose model — action recognition disabled ({exc})")
                return False

        # ── LSTM model ──
        if not WEIGHTS_PATH.exists():
            logger.warning(f"ActionService: LSTM weights not found at {WEIGHTS_PATH}")
            logger.info("  Run training/action_recognition/run_phase3_pipeline.py first")
            return False

        try:
            import torch
            ckpt = torch.load(str(WEIGHTS_PATH), map_location=self._torch_device)
            self._classes = ckpt.get("classes", ["spike", "background"])

            # Import and rebuild model
            import sys
            sys.path.insert(0, str(ROOT / "training" / "action_recognition"))
            from train_lstm import build_model
            self._model = build_model(len(self._classes))
            self._model.load_state_dict(ckpt["model_state"])
            self._model.to(self._torch_device)
            self._model.eval()
            self._loaded = True

            logger.info(f"ActionService: LSTM loaded on {self._torch_device}. Classes: {self._classes}")
            return True
        except Exception as e:
            logger.error(f"ActionService: Failed to load LSTM: {e}")
            return False

    def reset(self):
        """Clear per-player buffers (call between matches)."""
        self._buffers.clear()

    # ──────────────────────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        player_detections: List[Dict[str, Any]],
        frame_idx: int,
        fps: float,
    ) -> List[Dict[str, Any]]:
        """
        Process one video frame for all detected players.

        Returns list of action detections:
            [{track_id, action, confidence, timestamp, frame_number}]
        Only returns entries when confidence > ACTION_THRESHOLD.
        """
        if not self._loaded:
            return []

        results = []
        timestamp = frame_idx / fps

        for player in player_detections:
            tid  = player["track_id"]
            bbox = (player["bbox_x"], player["bbox_y"],
                    player["bbox_w"], player["bbox_h"])

            # Extract pose for this player
            pose_vec = self._extract_player_pose(frame, bbox)

            # Update buffer
            if tid not in self._buffers:
                self._buffers[tid] = deque(maxlen=NUM_FRAMES)
            self._buffers[tid].append(pose_vec)

            # Only infer when buffer is full
            if len(self._buffers[tid]) < NUM_FRAMES:
                continue

            action, confidence = self._infer(self._buffers[tid])

            if action and confidence >= ACTION_THRESHOLD:
                results.append({
                    "track_id":     tid,
                    "action":       action,
                    "confidence":   round(confidence, 4),
                    "timestamp":    round(timestamp, 4),
                    "frame_number": frame_idx,
                })

        return results

    # ──────────────────────────────────────────────────────────────────────────

    def _extract_player_pose(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Extract normalised 34-dim pose vector for one player."""
        try:
            x, y, w, h = [int(v) for v in bbox]
            # Expand crop slightly for context
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.10)
            fh, fw = frame.shape[:2]
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(fw, x + w + pad_x)
            y2 = min(fh, y + h + pad_y)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                return np.zeros(NUM_FEATURES, dtype=np.float32)

            import sys
            sys.path.insert(0, str(ROOT / "training" / "action_recognition"))
            from extract_poses import (
                extract_keypoints_rtmlib, extract_keypoints_mediapipe,
                normalise_keypoints,
            )

            ch, cw = crop.shape[:2]
            if self._pose_type == "rtmlib":
                kps = extract_keypoints_rtmlib(self._pose_model, crop)
            else:
                kps = extract_keypoints_mediapipe(self._pose_model, crop)

            if kps is None:
                return np.zeros(NUM_FEATURES, dtype=np.float32)
            return normalise_keypoints(kps, cw, ch)

        except Exception as e:
            logger.debug(f"ActionService._extract_player_pose error: {e}")
            return np.zeros(NUM_FEATURES, dtype=np.float32)

    def _infer(self, buffer: deque) -> Tuple[Optional[str], float]:
        """Run LSTM on a full (30, 34) buffer. Returns (action_name, confidence)."""
        try:
            import torch
            import torch.nn.functional as F

            seq = np.array(list(buffer), dtype=np.float32)   # (30, 34)
            x   = torch.tensor(seq[None], device=self._torch_device)  # (1, 30, 34)

            with torch.no_grad():
                logits = self._model(x)
                probs  = F.softmax(logits, dim=1).squeeze().detach().cpu().numpy()

            best_idx  = int(probs.argmax())
            best_conf = float(probs[best_idx])
            action    = self._classes[best_idx]

            # Don't report background as an action
            if action == "background":
                return None, best_conf

            return action, best_conf
        except Exception as e:
            logger.debug(f"ActionService._infer error: {e}")
            return None, 0.0

    def is_ready(self) -> bool:
        return self._loaded
