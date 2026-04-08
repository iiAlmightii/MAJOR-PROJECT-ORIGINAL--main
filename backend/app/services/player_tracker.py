"""
Player Tracker
──────────────
Uses YOLOv8 for person detection + supervision ByteTrack for consistent IDs.

Model priority:
  1. Custom fine-tuned weights at  models/weights/player_detection.pt
  2. Pre-trained YOLOv8n (COCO person class) as fallback

ByteTrack assigns a stable track_id across frames so each player keeps
the same integer ID throughout the entire match.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Try importing heavy CV deps gracefully so the API still loads without GPU
try:
    from ultralytics import YOLO
    import supervision as sv
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    logger.warning("ultralytics / supervision not installed – player tracking disabled")

WEIGHTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "models", "weights"
)
CUSTOM_WEIGHTS = os.path.join(WEIGHTS_DIR, "player_detection.pt")
FALLBACK_WEIGHTS = "yolov8n.pt"         # auto-downloaded by ultralytics

PERSON_CLASS_ID = 0                     # COCO class 0 = person
CONF_THRESHOLD  = 0.45
IOU_THRESHOLD   = 0.45


class PlayerTracker:
    """
    Stateful per-match player tracker.

    Usage
    -----
    tracker = PlayerTracker()
    tracker.load()
    for frame in video_frames:
        detections = tracker.process_frame(frame, frame_idx, fps)
        # detections: List[Dict] with keys bbox, track_id, conf, court_x/y
    """

    def __init__(self):
        self._model  = None
        self._tracker: Optional[Any] = None
        self._loaded  = False
        self._device  = "cpu"

    def load(self) -> bool:
        if not CV_AVAILABLE:
            return False
        try:
            try:
                import torch
                self._device = 0 if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device = "cpu"

            weights = CUSTOM_WEIGHTS if os.path.exists(CUSTOM_WEIGHTS) else FALLBACK_WEIGHTS
            logger.info(f"Loading player detection model: {weights}")
            logger.info(f"PlayerTracker: inference device {'cuda:0' if self._device == 0 else 'cpu'}")
            self._model   = YOLO(weights)
            self._tracker = sv.ByteTrack(
                track_activation_threshold=CONF_THRESHOLD,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=25,
            )
            self._loaded = True
            return True
        except Exception as exc:
            logger.error(f"PlayerTracker load failed: {exc}")
            return False

    def reset(self):
        """Reset tracker state between matches / rallies."""
        if CV_AVAILABLE and self._tracker:
            self._tracker = sv.ByteTrack(
                track_activation_threshold=CONF_THRESHOLD,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=25,
            )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        homography=None,
    ) -> List[Dict[str, Any]]:
        """
        Detect and track players in a single frame.

        Returns
        -------
        List of dicts:
            track_id, bbox_x, bbox_y, bbox_w, bbox_h,
            confidence, court_x, court_y, timestamp
        """
        if not self._loaded:
            return []

        try:
            results = self._model.predict(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=self._device,
                verbose=False,
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            if len(detections) == 0:
                return []

            detections = self._tracker.update_with_detections(detections)
            timestamp  = frame_idx / fps if fps > 0 else 0.0

            output = []
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                bx, by = float(x1), float(y1)
                bw = float(x2 - x1)
                bh = float(y2 - y1)
                tid  = int(detections.tracker_id[i]) if detections.tracker_id is not None else i
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0

                cx, cy = -1.0, -1.0
                if homography and homography.is_calibrated():
                    cx, cy = homography.transform_bbox_center(bx, by, bw, bh)

                output.append({
                    "track_id":    tid,
                    "bbox_x":      bx,
                    "bbox_y":      by,
                    "bbox_w":      bw,
                    "bbox_h":      bh,
                    "confidence":  conf,
                    "court_x":     cx,
                    "court_y":     cy,
                    "frame_number":frame_idx,
                    "timestamp":   round(timestamp, 4),
                })
            return output

        except Exception as exc:
            logger.error(f"PlayerTracker.process_frame error at frame {frame_idx}: {exc}")
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # Annotation helper (writes boxes + IDs onto the frame in-place)
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def annotate_frame(
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        team_map: Optional[Dict[int, str]] = None,
    ) -> np.ndarray:
        """Draw bounding boxes + track IDs. Returns the annotated frame."""
        out = frame.copy()
        TEAM_COLORS = {
            "A": (219, 130, 50),   # blue (BGR)
            "B": (50, 80, 219),    # red
        }
        DEFAULT_COLOR = (100, 200, 100)

        for d in detections:
            tid   = d["track_id"]
            x, y  = int(d["bbox_x"]), int(d["bbox_y"])
            w, h  = int(d["bbox_w"]), int(d["bbox_h"])
            team  = (team_map or {}).get(tid, None)
            color = TEAM_COLORS.get(team, DEFAULT_COLOR)

            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

            label = f"#{tid}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x, y - lh - 6), (x + lw + 4, y), color, -1)
            cv2.putText(out, label, (x + 2, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return out
