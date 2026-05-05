# backend/tests/test_landing_zones.py
"""Unit tests for landing zone assignment — uses plain objects, no DB."""
import types


def make_ball(timestamp, x, y, court_x=0.5, court_y=0.5, speed_kmh=None):
    b = types.SimpleNamespace()
    b.timestamp = timestamp
    b.x = x
    b.y = y
    b.court_x = court_x
    b.court_y = court_y
    b.speed_kmh = speed_kmh
    return b


def make_action(action_type, timestamp, result="neutral"):
    a = types.SimpleNamespace()
    a.action_type = action_type
    a.timestamp = timestamp
    a.result = result
    a.landing_x = None
    a.landing_y = None
    a.ball_speed_kmh = None
    a.reception_quality = None
    return a


import asyncio
from app.services.cv_pipeline import _assign_landing_zones_pure


def test_serve_landing_uses_max_pixel_y():
    """Landing zone is the ball row with maximum pixel y in post-action window."""
    balls = [
        make_ball(1.0, x=100, y=200, court_x=0.1, court_y=0.1),
        make_ball(1.5, x=300, y=350, court_x=0.5, court_y=0.5),
        make_ball(2.5, x=500, y=580, court_x=0.9, court_y=0.9),  # landing (max y)
        make_ball(3.5, x=600, y=400, court_x=0.9, court_y=0.8),  # after bounce
    ]
    action = make_action("serve", timestamp=0.8)
    asyncio.run(_assign_landing_zones_pure([action], balls))
    assert action.landing_x == 0.9
    assert action.landing_y == 0.9


def test_non_serve_attack_not_assigned_landing():
    """Set and dig actions do not get landing zones."""
    balls = [make_ball(1.0, x=100, y=500, court_x=0.5, court_y=0.5)]
    action = make_action("set", timestamp=0.8)
    asyncio.run(_assign_landing_zones_pure([action], balls))
    assert action.landing_x is None


def test_ball_speed_assigned_to_serve():
    """ball_speed_kmh from closest ball row ±0.5s is assigned to action."""
    balls = [
        make_ball(0.85, x=100, y=200, court_x=0.1, court_y=0.1, speed_kmh=72.5),
    ]
    action = make_action("serve", timestamp=0.8)
    asyncio.run(_assign_landing_zones_pure([action], balls))
    assert action.ball_speed_kmh == 72.5


def test_reception_quality_set_for_reception():
    """Reception actions get reception_quality from compute_reception_quality."""
    balls = [
        make_ball(5.1, x=100, y=200, court_x=0.5, court_y=0.5, speed_kmh=25.0),
    ]
    action = make_action("reception", timestamp=5.0, result="success")
    asyncio.run(_assign_landing_zones_pure([action], balls))
    # speed=25 + success → Rtg. 3
    assert action.reception_quality == 3
