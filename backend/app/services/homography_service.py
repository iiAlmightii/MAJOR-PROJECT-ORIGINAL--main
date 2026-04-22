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
        Automatic calibration using Hough line detection.

        Strategy
        --------
        1. Work only in the lower 70% of the frame (exclude audience stands).
        2. Detect lines with multiple Hough passes at different thresholds.
        3. Separate lines into horizontal vs vertical clusters by angle.
        4. Use robust percentile clipping (5th/95th) instead of absolute min/max
           to ignore outlier noise lines that would skew the corner estimate.
        5. Fall back to a court-region default (lower 65% of frame, full width)
           if fewer than 4 lines are found.
        """
        h, w = frame.shape[:2]

        # Only examine the lower portion — avoid crowd / scoreboard areas
        roi_top = int(h * 0.30)
        roi = frame[roi_top:, :]

        gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges   = cv2.Canny(blurred, 30, 120, apertureSize=3)

        # Try progressively lower thresholds to find lines
        lines = None
        for threshold in (80, 60, 40):
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=threshold,
                minLineLength=w // 8, maxLineGap=30,
            )
            if lines is not None and len(lines) >= 4:
                break

        if lines is not None and len(lines) >= 4:
            pts = lines[:, 0, :]                          # (N, 4)
            # Re-offset y coordinates back to full-frame space
            pts[:, 1] += roi_top
            pts[:, 3] += roi_top

            # Separate horizontal vs vertical lines
            dx = (pts[:, 2] - pts[:, 0]).astype(float)
            dy = (pts[:, 3] - pts[:, 1]).astype(float)
            angles = np.abs(np.degrees(np.arctan2(dy, dx)))
            horiz_mask = angles < 30                      # near-horizontal
            vert_mask  = angles > 60                      # near-vertical

            def pct_range(vals, lo=5, hi=95):
                return (int(np.percentile(vals, lo)),
                        int(np.percentile(vals, hi)))

            # Horizontal lines → define y_min, y_max
            if horiz_mask.sum() >= 2:
                hy = np.concatenate([pts[horiz_mask, 1], pts[horiz_mask, 3]])
                y_min, y_max = pct_range(hy)
            else:
                all_ys = np.concatenate([pts[:, 1], pts[:, 3]])
                y_min, y_max = pct_range(all_ys)

            # Vertical lines → define x_min, x_max
            if vert_mask.sum() >= 2:
                vx = np.concatenate([pts[vert_mask, 0], pts[vert_mask, 2]])
                x_min, x_max = pct_range(vx)
            else:
                all_xs = np.concatenate([pts[:, 0], pts[:, 2]])
                x_min, x_max = pct_range(all_xs)

            # Sanity check: result must be a reasonable rectangle
            if (x_max - x_min) > w * 0.3 and (y_max - y_min) > h * 0.15:
                src = [
                    [x_min, y_min], [x_max, y_min],
                    [x_max, y_max], [x_min, y_max],
                ]
                return self.calibrate(src)

        # Fallback: assume the court occupies the lower 70% of the frame,
        # full width with a small margin.  This works for standard broadcast angles
        # where the camera is mounted above and slightly behind the end line.
        margin_x = int(w * 0.03)
        y_top    = int(h * 0.30)          # court starts at ~30% down
        y_bot    = int(h * 0.97)          # court ends near the bottom
        src = [
            [margin_x,     y_top],
            [w - margin_x, y_top],
            [w - margin_x, y_bot],
            [margin_x,     y_bot],
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
        # Do NOT clamp — callers use the raw value to filter out-of-court detections
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
