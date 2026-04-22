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
CUSTOM_WEIGHTS    = os.path.join(WEIGHTS_DIR, "player_detection.pt")
REFEREE_WEIGHTS   = os.path.join(WEIGHTS_DIR, "referee_detection.pt")
FALLBACK_WEIGHTS  = "yolov8n.pt"        # auto-downloaded by ultralytics

PERSON_CLASS_ID   = 0                   # COCO class 0 = person
CONF_THRESHOLD    = 0.45
IOU_THRESHOLD     = 0.45
REFEREE_IOU_THRESH = 0.30               # overlap with referee box → suppress player

# Court boundary margins: allow detections slightly outside the court
# (homography calibration is imperfect) but reject clear audience members.
# Values are in normalised court coords (0=near side, 1=far side).
COURT_X_MIN = -0.25
COURT_X_MAX =  1.25
COURT_Y_MIN = -0.30   # allow some area above the court (players jump)
COURT_Y_MAX =  1.30

# Court boundary margins: allow detections slightly outside the court
# (homography calibration is imperfect) but reject clear audience members.
# Values are in normalised court coords (0=near side, 1=far side).
COURT_X_MIN = -0.25
COURT_X_MAX =  1.25
COURT_Y_MIN = -0.30   # allow some area above the court (players jump)
COURT_Y_MAX =  1.30


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
        self._model          = None
        self._referee_model  = None   # optional: suppress referee detections
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
            logger.info(f"PlayerTracker: loading {weights}")
            logger.info(f"PlayerTracker: device {'cuda:0' if self._device == 0 else 'cpu'}")
            self._model   = YOLO(weights)
            self._tracker = sv.ByteTrack(
                track_activation_threshold=CONF_THRESHOLD,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=25,
            )

            # Load referee suppressor if weights available
            if os.path.exists(REFEREE_WEIGHTS):
                try:
                    self._referee_model = YOLO(REFEREE_WEIGHTS)
                    logger.info("PlayerTracker: referee suppressor loaded")
                except Exception as e:
                    logger.warning(f"PlayerTracker: referee model failed to load: {e}")

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

            # Build referee bounding boxes for suppression (if model loaded)
            referee_boxes: List[List[float]] = []
            if self._referee_model is not None:
                try:
                    ref_res = self._referee_model.predict(
                        frame, conf=0.40, verbose=False, device=self._device,
                    )[0]
                    for box in ref_res.boxes.xyxy.cpu().numpy():
                        referee_boxes.append(box.tolist())
                except Exception:
                    pass

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
                    # Skip detections clearly outside the court (audience members,
                    # referees near the sideline, scoreboard operators, etc.)
                    if not (COURT_X_MIN <= cx <= COURT_X_MAX and
                            COURT_Y_MIN <= cy <= COURT_Y_MAX):
                        continue

                # Skip very small boxes — likely crowd noise, not players
                frame_h, frame_w = frame.shape[:2]
                min_height_px = frame_h * 0.06   # player must be ≥6% of frame height
                if bh < min_height_px:
                    continue

                # Suppress if this box overlaps a detected referee
                if referee_boxes and self._iou_any([bx, by, bx+bw, by+bh], referee_boxes) > REFEREE_IOU_THRESH:
                    continue

                output.append({
                    "track_id":    tid,
                    "bbox_x":      bx,
                    "bbox_y":      by,
                    "bbox_w":      bw,
                    "bbox_h":      bh,
                    "confidence":  conf,
                    "court_x":     cx if cx >= 0 else None,
                    "court_y":     cy if cy >= 0 else None,
                    "frame_number":frame_idx,
                    "timestamp":   round(timestamp, 4),
                })
            return output

        except Exception as exc:
            logger.error(f"PlayerTracker.process_frame error at frame {frame_idx}: {exc}")
            return []

    @staticmethod
    def _iou_any(box: List[float], ref_boxes: List[List[float]]) -> float:
        """Return the maximum IoU between box and any reference box."""
        x1, y1, x2, y2 = box
        best = 0.0
        for rb in ref_boxes:
            rx1, ry1, rx2, ry2 = rb
            ix1, iy1 = max(x1, rx1), max(y1, ry1)
            ix2, iy2 = min(x2, rx2), min(y2, ry2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            union = (x2-x1)*(y2-y1) + (rx2-rx1)*(ry2-ry1) - inter
            iou = inter / union if union > 0 else 0.0
            best = max(best, iou)
        return best

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
