# backend/tests/test_reception_quality.py
"""Unit tests for reception quality 0-3 rating."""
from app.services.scoring_engine import ScoringEngine


def test_error_is_always_zero():
    assert ScoringEngine.compute_reception_quality("error", None)   == 0
    assert ScoringEngine.compute_reception_quality("error", 80.0)   == 0
    assert ScoringEngine.compute_reception_quality("error", 10.0)   == 0


def test_success_slow_ball_is_perfect():
    """Slow incoming ball (<35 km/h) + success → Rtg. 3 (perfect)."""
    assert ScoringEngine.compute_reception_quality("success", 20.0)  == 3
    assert ScoringEngine.compute_reception_quality("success", 34.9)  == 3


def test_success_medium_ball_is_good():
    """Medium ball (35-60 km/h) + success → Rtg. 2 (good)."""
    assert ScoringEngine.compute_reception_quality("success", 35.0)  == 2
    assert ScoringEngine.compute_reception_quality("success", 59.9)  == 2


def test_success_fast_ball_is_poor():
    """Fast ball (≥60 km/h) + success → Rtg. 1 (poor but controlled)."""
    assert ScoringEngine.compute_reception_quality("success", 60.0)  == 1
    assert ScoringEngine.compute_reception_quality("success", 90.0)  == 1


def test_neutral_is_poor():
    """Neutral result → Rtg. 1."""
    assert ScoringEngine.compute_reception_quality("neutral", None)  == 1
    assert ScoringEngine.compute_reception_quality("neutral", 50.0)  == 1


def test_no_speed_data():
    """Missing speed treated as 0 km/h (slow) → 3 if success."""
    assert ScoringEngine.compute_reception_quality("success", None)  == 3
