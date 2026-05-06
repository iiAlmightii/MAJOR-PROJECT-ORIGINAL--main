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
from app.services.jersey_ocr import read_jersey_number

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
CONF_THRESHOLD    = 0.50
IOU_THRESHOLD     = 0.45
REFEREE_IOU_THRESH = 0.30               # overlap with referee box → suppress player
MAX_TRACKS        = 16     # 12 players + up to 4 sideline staff hard ceiling
OCR_INTERVAL      = 15    # run jersey OCR every N frames to limit GPU cost

# Court boundary margins in normalised court coords (used when homography IS calibrated).
# Tighter than before so sideline coaches/refs are excluded.
COURT_X_MIN = -0.10
COURT_X_MAX =  1.10
COURT_Y_MIN = -0.15   # allow a little above the court (jumping)
COURT_Y_MAX =  1.15

# Pixel-space heuristics applied regardless of homography calibration.
# These catch audience members, camera operators, and scoreboard staff
# without needing a calibrated court map.
MIN_HEIGHT_FRACTION  = 0.08   # bbox_h must be ≥ 8% of frame height
FEET_Y_MIN_FRACTION  = 0.30   # feet (bbox_y+bbox_h) must be below top 30% of frame
EDGE_X_MARGIN        = 0.09   # reject if bbox center_x < 9% or > 91% of frame width
MAX_WIDTH_HEIGHT_RATIO = 1.5  # reject if bbox is wider than 1.5× its height (not a standing person)


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
                lost_track_buffer=120,
                minimum_matching_threshold=0.50,
                frame_rate=8,
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
                lost_track_buffer=120,
                minimum_matching_threshold=0.50,
                frame_rate=8,
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

                frame_h, frame_w = frame.shape[:2]

                # ── Pixel-space heuristics (always applied, no homography needed) ──
                # Reject boxes that are too small
                if bh < frame_h * MIN_HEIGHT_FRACTION:
                    continue
                # Reject boxes wider than tall (not a standing person)
                if bw > bh * MAX_WIDTH_HEIGHT_RATIO:
                    continue
                # Reject if feet (bottom of box) are in the top 30% — audience/scaffolding
                feet_y = by + bh
                if feet_y < frame_h * FEET_Y_MIN_FRACTION:
                    continue
                # Reject detections at extreme left/right edges (sideline equipment/staff)
                center_x = bx + bw / 2
                if center_x < frame_w * EDGE_X_MARGIN or center_x > frame_w * (1 - EDGE_X_MARGIN):
                    continue
                # Reject small elevated boxes — scoreboard staff / elevated refs
                # (head above top 22% AND box height < 22% = not a court player)
                if by < frame_h * 0.22 and bh < frame_h * 0.22:
                    continue

                # ── Homography-based court boundary filter ───────────────────────
                cx, cy = -1.0, -1.0
                if homography and homography.is_calibrated():
                    cx, cy = homography.transform_bbox_center(bx, by, bw, bh)
                    if not (COURT_X_MIN <= cx <= COURT_X_MAX and
                            COURT_Y_MIN <= cy <= COURT_Y_MAX):
                        continue

                # ── Referee suppression ──────────────────────────────────────────
                if referee_boxes and self._iou_any([bx, by, bx+bw, by+bh], referee_boxes) > REFEREE_IOU_THRESH:
                    continue

                # ── Jersey hue (for team clustering downstream) ──────────────────
                jersey_hue = self._sample_jersey_hue(frame, bx, by, bw, bh)

                # Jersey OCR — only every OCR_INTERVAL frames
                jersey_number = None
                if frame_idx % OCR_INTERVAL == 0:
                    x1i = max(0, int(bx))
                    y1i = max(0, int(by))
                    x2i = min(frame.shape[1], int(bx + bw))
                    y2i = min(frame.shape[0], int(by + bh))
                    crop = frame[y1i:y2i, x1i:x2i]
                    jersey_number = read_jersey_number(crop)

                output.append({
                    "track_id":      tid,
                    "bbox_x":        bx,
                    "bbox_y":        by,
                    "bbox_w":        bw,
                    "bbox_h":        bh,
                    "confidence":    conf,
                    "court_x":       cx if cx >= 0 else None,
                    "court_y":       cy if cy >= 0 else None,
                    "frame_number":  frame_idx,
                    "timestamp":     round(timestamp, 4),
                    "jersey_hue":    jersey_hue,
                    "jersey_number": jersey_number,
                })

            # Hard cap: keep only the MAX_TRACKS detections with highest confidence
            if len(output) > MAX_TRACKS:
                output.sort(key=lambda d: d["confidence"], reverse=True)
                output = output[:MAX_TRACKS]

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

    @staticmethod
    def _sample_jersey_hue(
        frame: np.ndarray, bx: float, by: float, bw: float, bh: float
    ) -> float:
        """
        Extract the dominant jersey hue (0-180) from the torso region of a bbox.
        Uses circular mean over saturated pixels only.
        Returns -1.0 if the jersey has no distinctive colour (white/black/gray).
        """
        fh, fw = frame.shape[:2]
        x1 = max(0, int(bx + bw * 0.15))
        x2 = min(fw, int(bx + bw * 0.85))
        y1 = max(0, int(by + bh * 0.20))  # skip head
        y2 = min(fh, int(by + bh * 0.65))  # skip legs
        if x2 <= x1 or y2 <= y1:
            return -1.0
        torso = frame[y1:y2, x1:x2]
        if torso.size < 100:
            return -1.0
        try:
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            sat  = hsv[:, :, 1]
            hue  = hsv[:, :, 0].astype(np.float32)
            mask = sat > 50
            if mask.sum() < 20:
                return -1.0  # mostly unsaturated = white/gray/black jersey
            h_rad = hue[mask] * (np.pi / 90.0)
            circ_hue = np.arctan2(np.sin(h_rad).mean(), np.cos(h_rad).mean())
            return float(circ_hue * 90.0 / np.pi % 180.0)
        except Exception:
            return -1.0

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
