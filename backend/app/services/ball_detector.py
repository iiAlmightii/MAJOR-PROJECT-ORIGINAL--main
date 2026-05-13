"""
Ball Detector
─────────────
Detects the volleyball in each frame using:
  1. Custom YOLOv8 model (models/weights/ball_detection.pt) — preferred
  2. COCO "sports ball" class (class 32) — fallback
  3. Motion + shape heuristics (Hough circles) — last resort

Trajectory smoothing is applied with a Kalman-like exponential smoother.
"""

import os
import math
import logging
from collections import deque
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2

COURT_WIDTH_M  = 9.0    # volleyball court width in metres
COURT_HEIGHT_M = 18.0   # volleyball court length in metres

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    import supervision as sv
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

WEIGHTS_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "weights")
CUSTOM_WEIGHTS  = os.path.join(WEIGHTS_DIR, "ball_detection.pt")
FALLBACK_WEIGHTS = "yolov8n.pt"

BALL_COCO_CLASS  = 32       # COCO sports ball
CONF_THRESHOLD   = 0.50     # raised — below this catches logos/rings
SMOOTH_ALPHA     = 0.6      # EMA weight
TRAJECTORY_LEN   = 30       # frames to keep in trail


class BallDetector:
    """Per-match ball detector with EMA smoothing and trajectory history."""

    def __init__(self):
        self._model  = None
        self._loaded = False
        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None
        self._trajectory: deque = deque(maxlen=TRAJECTORY_LEN)
        self._use_custom = False
        self._device = "cpu"
        self._prev_court_x: Optional[float] = None
        self._prev_court_y: Optional[float] = None
        self._prev_timestamp: Optional[float] = None

    def load(self) -> bool:
        if not CV_AVAILABLE:
            return False
        try:
            try:
                import torch
                self._device = 0 if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device = "cpu"

            if os.path.exists(CUSTOM_WEIGHTS):
                self._model     = YOLO(CUSTOM_WEIGHTS)
                self._use_custom = True
                logger.info("BallDetector: loaded custom weights")
            else:
                self._model     = YOLO(FALLBACK_WEIGHTS)
                self._use_custom = False
                logger.info("BallDetector: using COCO fallback (class 32)")
            logger.info(f"BallDetector: inference device {'cuda:0' if self._device == 0 else 'cpu'}")
            self._loaded = True
            return True
        except Exception as exc:
            logger.error(f"BallDetector.load failed: {exc}")
            return False

    def reset(self):
        self._prev_x = None
        self._prev_y = None
        self._trajectory.clear()
        self._prev_court_x = None
        self._prev_court_y = None
        self._prev_timestamp = None

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        homography=None,
    ) -> Optional[Dict[str, Any]]:
        """
        Detect ball in a single frame.

        Returns None if not found, otherwise a dict:
            x, y, radius, confidence, court_x, court_y,
            frame_number, timestamp, trajectory
        """
        result = None

        if self._loaded:
            result = self._detect_yolo(frame)

        # Hough circle fallback is disabled — it detects logos/rings as balls.
        # YOLO-only is more accurate even when it misses some frames.

        if result is None:
            return None

        # EMA smoothing
        rx, ry = result["x"], result["y"]
        if self._prev_x is not None:
            rx = SMOOTH_ALPHA * rx + (1 - SMOOTH_ALPHA) * self._prev_x
            ry = SMOOTH_ALPHA * ry + (1 - SMOOTH_ALPHA) * self._prev_y
        self._prev_x, self._prev_y = rx, ry

        timestamp = frame_idx / fps if fps > 0 else 0.0
        self._trajectory.append((rx, ry, timestamp))

        cx, cy = -1.0, -1.0
        if homography and homography.is_calibrated():
            cx, cy = homography.frame_to_court(rx, ry)

        # Compute instantaneous speed from consecutive court positions
        speed_kmh, vx_norm, vy_norm = None, None, None
        if (cx >= 0 and cy >= 0
                and self._prev_court_x is not None
                and self._prev_timestamp is not None):
            dt = timestamp - self._prev_timestamp
            if dt > 0:
                dcx = cx - self._prev_court_x
                dcy = cy - self._prev_court_y
                dx_m = dcx * COURT_WIDTH_M
                dy_m = dcy * COURT_HEIGHT_M
                speed_ms = math.sqrt(dx_m ** 2 + dy_m ** 2) / dt
                speed_kmh = round(speed_ms * 3.6, 1)
                vx_norm = round(dcx / dt, 4)
                vy_norm = round(dcy / dt, 4)

        if cx >= 0:
            self._prev_court_x = cx
            self._prev_court_y = cy
            self._prev_timestamp = timestamp

        return {
            "x":           round(rx, 2),
            "y":           round(ry, 2),
            "radius":      result.get("radius", 8),
            "confidence":  result.get("confidence", 0.0),
            "court_x":     round(cx, 4) if cx >= 0 else None,
            "court_y":     round(cy, 4) if cy >= 0 else None,
            "speed_kmh":   speed_kmh,
            "vx":          vx_norm,
            "vy":          vy_norm,
            "frame_number":frame_idx,
            "timestamp":   round(timestamp, 4),
            "trajectory":  list(self._trajectory),
        }

    # ──────────────────────────────────────────────────────────────────────────

    def _detect_yolo(self, frame: np.ndarray) -> Optional[Dict]:
        classes = None if self._use_custom else [BALL_COCO_CLASS]
        try:
            results = self._model.predict(
                frame,
                classes=classes,
                conf=CONF_THRESHOLD,
                device=self._device,
                verbose=False,
            )[0]
            if len(results.boxes) == 0:
                return None

            # Pick the detection with highest confidence
            confs = results.boxes.conf.cpu().numpy()
            best  = int(confs.argmax())
            x1, y1, x2, y2 = results.boxes.xyxy[best].cpu().numpy()
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            r  = float(min(x2 - x1, y2 - y1) / 2)
            return {"x": cx, "y": cy, "radius": r, "confidence": float(confs[best])}
        except Exception:
            return None

    def _detect_hough(self, frame: np.ndarray) -> Optional[Dict]:
        """Hough circle detection as last-resort fallback."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=1, minDist=30,
                param1=50, param2=28,
                minRadius=6, maxRadius=30,
            )
            if circles is not None:
                c = circles[0, 0]
                return {"x": float(c[0]), "y": float(c[1]),
                        "radius": float(c[2]), "confidence": 0.3}
        except Exception:
            pass
        return None

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def annotate_frame(frame: np.ndarray, ball: Optional[Dict]) -> np.ndarray:
        """Draw ball circle + trajectory on frame."""
        out = frame.copy()
        if not ball:
            return out

        bx, by = int(ball["x"]), int(ball["y"])
        br     = max(8, int(ball.get("radius", 8)))

        # Glow effect
        cv2.circle(out, (bx, by), br + 4, (0, 200, 255), 1)
        cv2.circle(out, (bx, by), br,     (0, 230, 255), 2)
        cv2.circle(out, (bx, by), 3,      (255, 255, 0), -1)

        # Trajectory trail (fade out)
        traj = ball.get("trajectory", [])
        for i in range(1, len(traj)):
            alpha = i / len(traj)
            color = (int(0 * alpha), int(200 * alpha), int(255 * alpha))
            p1 = (int(traj[i - 1][0]), int(traj[i - 1][1]))
            p2 = (int(traj[i][0]),     int(traj[i][1]))
            cv2.line(out, p1, p2, color, max(1, int(3 * alpha)))

        return out
