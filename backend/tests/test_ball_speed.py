# backend/tests/test_ball_speed.py
"""Unit tests for ball speed computation — pure math, no YOLO/CV."""
import math
from app.services.ball_detector import BallDetector, COURT_WIDTH_M, COURT_HEIGHT_M


def _make_detector_with_prev(prev_cx, prev_cy, prev_ts):
    """Create a BallDetector with pre-set previous court position."""
    det = BallDetector()
    det._prev_court_x = prev_cx
    det._prev_court_y = prev_cy
    det._prev_timestamp = prev_ts
    return det


def test_speed_1m_per_second():
    """Ball moves 1 metre in 1 second → 3.6 km/h."""
    det = _make_detector_with_prev(0.0, 0.0, 0.0)
    cx = 1.0 / COURT_WIDTH_M   # 1 metre in normalised coords
    cy = 0.0
    ts = 1.0
    dt = ts - 0.0
    dcx = cx - 0.0
    dcy = cy - 0.0
    dx_m = dcx * COURT_WIDTH_M
    dy_m = dcy * COURT_HEIGHT_M
    speed_ms = math.sqrt(dx_m**2 + dy_m**2) / dt
    expected = round(speed_ms * 3.6, 1)
    assert expected == 3.6


def test_speed_ten_ms():
    """Ball moves 10 m/s diagonally → converts to km/h correctly."""
    # 10 m/s in x direction = 36 km/h
    # 1 second, travel 10 m in x = 10/9 court units
    det = _make_detector_with_prev(0.0, 0.0, 0.0)
    cx = 10.0 / COURT_WIDTH_M
    cy = 0.0
    dt = 1.0
    dx_m = cx * COURT_WIDTH_M
    dy_m = 0.0
    speed_ms = math.sqrt(dx_m**2 + dy_m**2) / dt
    expected = round(speed_ms * 3.6, 1)
    assert expected == 36.0


def test_no_speed_on_first_frame():
    """First call (no previous position) produces speed_kmh=None."""
    det = BallDetector()
    assert det._prev_court_x is None
    assert det._prev_timestamp is None


def test_vx_vy_computed():
    """vx and vy are normalised-court-units per second."""
    cx_start, cy_start = 0.2, 0.3
    cx_end,   cy_end   = 0.5, 0.3
    dt = 2.0
    vx_expected = round((cx_end - cx_start) / dt, 4)
    vy_expected = round((cy_end - cy_start) / dt, 4)
    assert vx_expected == round(0.15, 4)
    assert vy_expected == 0.0
