# backend/tests/test_migrations.py
"""Verify ORM models declare all new columns."""
from app.models.tracking import BallTracking
from app.models.actions import Action
from app.models.analytics import Analytics


def test_ball_tracking_has_speed_columns():
    cols = {c.key for c in BallTracking.__table__.columns}
    assert "speed_kmh" in cols
    assert "vx" in cols
    assert "vy" in cols


def test_action_has_new_columns():
    cols = {c.key for c in Action.__table__.columns}
    assert "landing_x" in cols
    assert "landing_y" in cols
    assert "ball_speed_kmh" in cols
    assert "reception_quality" in cols


def test_analytics_has_movement_columns():
    cols = {c.key for c in Analytics.__table__.columns}
    assert "distance_covered_m" in cols
    assert "avg_speed_kmh" in cols
    assert "max_speed_kmh" in cols
    assert "reception_quality_avg" in cols
    assert "avg_serve_speed_kmh" in cols
    assert "avg_attack_speed_kmh" in cols
