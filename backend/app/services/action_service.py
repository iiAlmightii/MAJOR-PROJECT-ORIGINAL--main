"""
Action Recognition Service
───────────────────────────
Two-mode action recogniser:

  Mode A — Pose+LSTM (preferred, high quality):
    Requires: models/weights/action_lstm_phase1.pt
    Training: training/action_recognition/run_phase3_pipeline.py

  Mode B — YOLO detector fallback (frame-level, no LSTM needed):
    Requires: models/weights/action_detection.pt
    Training: training/action_recognition/train_action_v2.py   ← USE THIS
              (trains on Dataset/Action recognition/ — 11k images, 5 classes)
    Classes: block, defense, serve, set, spike  (clean, no noise)

If neither weights file exists the service gracefully disables itself.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from collections import deque

logger = logging.getLogger(__name__)

ROOT              = Path(__file__).resolve().parent.parent.parent.parent
# Prefer phase 2 (all 5 classes) over phase 1 (spike only)
_W = ROOT / "models" / "weights"
WEIGHTS_PATH      = _W / "action_lstm_phase2.pt" if (_W / "action_lstm_phase2.pt").exists() else _W / "action_lstm_phase1.pt"
YOLO_ACTION_WEIGHTS = ROOT / "models" / "weights" / "action_detection.pt"
CLASS_MAP_PATH    = ROOT / "training" / "action_recognition" / "models" / "class_map.json"

NUM_FRAMES   = 30
NUM_FEATURES = 34
ACTION_THRESHOLD = 0.45    # minimum confidence to report an action

# YOLO action class names → normalised action type names used in the DB
YOLO_CLASS_MAP = {
    "Defense-Move": "dig",
    "attack":       "attack",   # maps to ActionType.attack (DB enum)
    "block":        "block",
    "reception":    "reception",
    "service":      "serve",
    "setting":      "set",
    "stand":        None,        # not a reportable action
    "spike":        "attack",   # treat spike as attack for DB enum compat
    "serve":        "serve",
    "defense":      "dig",
    "set":          "set",
}


class ActionService:
    """
    Per-match action recogniser.
    Supports two modes:
      'lstm'  – Pose extraction + BiLSTM (best accuracy, needs training)
      'yolo'  – YOLO detector on full frame (frame-level, works after training on Volleyball Activity Dataset)
    """

    def __init__(self):
        self._model        = None
        self._yolo_model   = None
        self._mode         = "none"           # "lstm" | "yolo" | "none"
        self._classes: List[str] = []
        self._loaded       = False
        self._pose_type    = "none"
        self._pose_model   = None
        self._torch_device = "cpu"
        # Per-player sliding window: track_id → deque of (34,) vectors (LSTM only)
        self._buffers: Dict[int, deque] = {}

    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load best available model: LSTM preferred, YOLO fallback."""
        try:
            import torch
            self._torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self._torch_device = "cpu"

        # ── Try LSTM first ───────────────────────────────────────────────────
        if WEIGHTS_PATH.exists():
            if self._load_lstm():
                return True

        # ── Try YOLO action detector ─────────────────────────────────────────
        if YOLO_ACTION_WEIGHTS.exists():
            if self._load_yolo():
                return True

        logger.warning(
            "ActionService: no weights found. "
            "Train action_detection.pt via: python training/action_recognition/train_action_v2.py "
            "or action_lstm_phase1.pt via: python training/action_recognition/run_phase3_pipeline.py"
        )
        return False

    def _load_lstm(self) -> bool:
        """Load BiLSTM model + pose estimator."""
        # Pose model
        pose_loaded = False
        try:
            from rtmlib import RTMO
            dev = "cuda" if self._torch_device == "cuda" else "cpu"
            for d in ([dev, "cpu"] if dev == "cuda" else [dev]):
                try:
                    self._pose_model = RTMO(
                        pose="rtmo-s", det_model="yolox-s",
                        device=d, backend="onnxruntime",
                    )
                    self._pose_type = "rtmlib"
                    pose_loaded = True
                    logger.info(f"ActionService: RTMPose loaded on {d}")
                    break
                except Exception as exc:
                    logger.warning(f"ActionService: RTMPose failed on {d}: {exc}")
        except ImportError:
            pass

        if not pose_loaded:
            try:
                import mediapipe as mp
                self._pose_model = mp.solutions.pose.Pose(
                    static_image_mode=False, model_complexity=1,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5,
                )
                self._pose_type = "mediapipe"
                logger.info("ActionService: MediaPipe loaded")
                pose_loaded = True
            except Exception as exc:
                logger.warning(f"ActionService: no pose model available: {exc}")
                return False

        try:
            import torch, sys
            ckpt = torch.load(str(WEIGHTS_PATH), map_location=self._torch_device,
                              weights_only=False)
            self._classes = ckpt.get("classes", ["spike", "background"])
            sys.path.insert(0, str(ROOT / "training" / "action_recognition"))
            from train_lstm import build_model
            self._model = build_model(len(self._classes))
            self._model.load_state_dict(ckpt["model_state"])
            self._model.to(self._torch_device)
            self._model.eval()
            self._mode   = "lstm"
            self._loaded = True
            logger.info(f"ActionService: LSTM loaded [{', '.join(self._classes)}]")
            return True
        except Exception as e:
            logger.error(f"ActionService: LSTM load failed: {e}")
            return False

    def _load_yolo(self) -> bool:
        """Load YOLO action detection model."""
        try:
            from ultralytics import YOLO
            self._yolo_model = YOLO(str(YOLO_ACTION_WEIGHTS))
            self._mode   = "yolo"
            self._loaded = True
            self._classes = list(self._yolo_model.names.values())
            logger.info(f"ActionService: YOLO action detector loaded [{', '.join(self._classes)}]")
            return True
        except Exception as e:
            logger.error(f"ActionService: YOLO action load failed: {e}")
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
        Dispatches to LSTM mode or YOLO mode depending on what was loaded.

        Returns list of action detections:
            [{track_id, action, confidence, timestamp, frame_number}]
        Only returns entries when confidence > ACTION_THRESHOLD.
        """
        if not self._loaded:
            return []

        if self._mode == "yolo":
            return self._process_frame_yolo(frame, player_detections, frame_idx, fps)
        return self._process_frame_lstm(frame, player_detections, frame_idx, fps)

    def _process_frame_lstm(
        self, frame, player_detections, frame_idx, fps,
    ) -> List[Dict[str, Any]]:
        results = []
        timestamp = frame_idx / fps

        for player in player_detections:
            tid  = player["track_id"]
            bbox = (player["bbox_x"], player["bbox_y"],
                    player["bbox_w"], player["bbox_h"])

            pose_vec = self._extract_player_pose(frame, bbox)
            if tid not in self._buffers:
                self._buffers[tid] = deque(maxlen=NUM_FRAMES)
            self._buffers[tid].append(pose_vec)

            if len(self._buffers[tid]) < NUM_FRAMES:
                continue

            action, confidence = self._infer_lstm(self._buffers[tid])
            if action and confidence >= ACTION_THRESHOLD:
                results.append({
                    "track_id":     tid,
                    "action":       action,
                    "confidence":   round(confidence, 4),
                    "timestamp":    round(timestamp, 4),
                    "frame_number": frame_idx,
                })
        return results

    def _process_frame_yolo(
        self, frame, player_detections, frame_idx, fps,
    ) -> List[Dict[str, Any]]:
        """YOLO mode: detect action bounding boxes on the full frame,
        then assign each detection to the nearest tracked player."""
        try:
            results_yolo = self._yolo_model.predict(
                frame, conf=ACTION_THRESHOLD, verbose=False,
                device=self._torch_device,
            )[0]
        except Exception as e:
            logger.debug(f"ActionService YOLO predict error: {e}")
            return []

        timestamp = frame_idx / fps
        out: List[Dict] = []
        if len(results_yolo.boxes) == 0:
            return out

        for i in range(len(results_yolo.boxes)):
            x1, y1, x2, y2 = results_yolo.boxes.xyxy[i].cpu().numpy()
            conf = float(results_yolo.boxes.conf[i].cpu())
            cls  = int(results_yolo.boxes.cls[i].cpu())
            raw_name = self._yolo_model.names.get(cls, "")
            action = YOLO_CLASS_MAP.get(raw_name)
            if action is None:
                continue          # "stand" or unknown → skip

            # Find the tracked player whose bbox overlaps most with this detection
            det_cx = (x1 + x2) / 2
            det_cy = (y1 + y2) / 2
            best_tid, best_dist = None, float("inf")
            for p in player_detections:
                pcx = p["bbox_x"] + p["bbox_w"] / 2
                pcy = p["bbox_y"] + p["bbox_h"] / 2
                dist = (pcx - det_cx) ** 2 + (pcy - det_cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_tid  = p["track_id"]

            if best_tid is None:
                continue

            out.append({
                "track_id":     best_tid,
                "action":       action,
                "confidence":   round(conf, 4),
                "timestamp":    round(timestamp, 4),
                "frame_number": frame_idx,
            })
        return out

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

    def _infer_lstm(self, buffer: deque) -> Tuple[Optional[str], float]:
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
