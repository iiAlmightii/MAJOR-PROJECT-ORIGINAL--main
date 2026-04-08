"""
Homography Service
──────────────────
Converts camera-view pixel coordinates → normalised top-down court coordinates.

Volleyball court standard: 18 m × 9 m
Origin = top-left of the court (team A back line, left sideline).

The four src_points are the court corners visible in the video frame,
ordered: [top-left, top-right, bottom-right, bottom-left].

Normalised output: x ∈ [0, 1]  (left → right), y ∈ [0, 1]  (top → bottom).
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional

# Standard volleyball court in pixels (scaled for computation)
COURT_W_PX = 900   # 18 m
COURT_H_PX = 450   #  9 m

# Default court corners in the output (top-down) image
DST_POINTS = np.array([
    [0,          0         ],   # top-left
    [COURT_W_PX, 0         ],   # top-right
    [COURT_W_PX, COURT_H_PX],   # bottom-right
    [0,          COURT_H_PX],   # bottom-left
], dtype=np.float32)


class HomographyService:
    """Compute and apply the perspective transformation."""

    def __init__(self):
        self._H: Optional[np.ndarray] = None          # 3×3 homography matrix
        self._H_inv: Optional[np.ndarray] = None      # inverse (top-down → frame)
        self._src_points: Optional[np.ndarray] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Calibration
    # ──────────────────────────────────────────────────────────────────────────

    def calibrate(self, src_points: List[List[float]]) -> bool:
        """
        Set the four court corners from the video frame.

        Parameters
        ----------
        src_points : [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
            Pixel coordinates of the four court corners in the video,
            ordered top-left, top-right, bottom-right, bottom-left.
        """
        pts = np.array(src_points, dtype=np.float32)
        if pts.shape != (4, 2):
            return False
        self._src_points = pts
        self._H, _ = cv2.findHomography(pts, DST_POINTS)
        if self._H is not None:
            self._H_inv = np.linalg.inv(self._H)
        return self._H is not None

    def auto_calibrate_from_lines(self, frame: np.ndarray) -> bool:
        """
        Attempt automatic calibration using Hough line detection.
        Falls back to a fixed default if the court lines are not detected.
        """
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=w // 5, maxLineGap=20)

        if lines is not None and len(lines) >= 4:
            # Simplified: use extreme points of detected lines as corners
            pts = lines[:, 0, :]          # shape (N, 4)
            xs = np.concatenate([pts[:, 0], pts[:, 2]])
            ys = np.concatenate([pts[:, 1], pts[:, 3]])
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            src = [[x_min, y_min], [x_max, y_min],
                   [x_max, y_max], [x_min, y_max]]
            return self.calibrate(src)

        # Default fallback: assume full frame = court
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        src = [
            [margin_x,     margin_y    ],
            [w - margin_x, margin_y    ],
            [w - margin_x, h - margin_y],
            [margin_x,     h - margin_y],
        ]
        return self.calibrate(src)

    # ──────────────────────────────────────────────────────────────────────────
    # Transformation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def is_calibrated(self) -> bool:
        return self._H is not None

    def frame_to_court(self, x: float, y: float) -> Tuple[float, float]:
        """
        Map a pixel (x, y) in the video frame → normalised court coords (cx, cy).
        Returns (-1, -1) if not calibrated or outside court.
        """
        if self._H is None:
            return -1.0, -1.0
        pt = np.array([[[x, y]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H)
        cx = float(result[0, 0, 0]) / COURT_W_PX
        cy = float(result[0, 0, 1]) / COURT_H_PX
        # Clamp to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        return cx, cy

    def court_to_frame(self, cx: float, cy: float) -> Tuple[float, float]:
        """Inverse: normalised court → pixel frame coords."""
        if self._H_inv is None:
            return -1.0, -1.0
        px = cx * COURT_W_PX
        py = cy * COURT_H_PX
        pt = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H_inv)
        return float(result[0, 0, 0]), float(result[0, 0, 1])

    def transform_bbox_center(
        self, bbox_x: float, bbox_y: float,
        bbox_w: float, bbox_h: float
    ) -> Tuple[float, float]:
        """Use the foot point (bottom-center of bounding box) for court mapping."""
        foot_x = bbox_x + bbox_w / 2.0
        foot_y = bbox_y + bbox_h          # bottom edge = foot
        return self.frame_to_court(foot_x, foot_y)

    def get_court_image(self, w: int = COURT_W_PX, h: int = COURT_H_PX) -> np.ndarray:
        """
        Return a blank top-down court image with standard markings.
        Useful for rendering the mini-map in the backend.
        """
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (20, 28, 40)  # dark background

        line_color = (50, 80, 120)
        thick = 2

        # Outer boundary
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), line_color, thick)
        # Net (centre line)
        cv2.line(img, (0, h // 2), (w, h // 2), (80, 120, 180), thick + 1)
        # 3-metre attack lines
        atk = h // 6
        cv2.line(img, (0, atk),     (w, atk),     line_color, 1)
        cv2.line(img, (0, h - atk), (w, h - atk), line_color, 1)
        # Centre vertical
        cv2.line(img, (w // 2, 0), (w // 2, h), line_color, 1)
        return img
