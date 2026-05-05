# backend/tests/test_player_movement.py
"""Unit tests for player movement computation — pure math, no DB."""
from app.services.cv_pipeline import compute_player_movement


def make_row(court_x, court_y, timestamp):
    import types
    r = types.SimpleNamespace()
    r.court_x = court_x
    r.court_y = court_y
    r.timestamp = timestamp
    return r


def test_stationary_player():
    rows = [make_row(0.5, 0.5, 0.0), make_row(0.5, 0.5, 1.0)]
    dist, avg, mx = compute_player_movement(rows)
    assert dist == 0.0
    assert avg == 0.0
    assert mx == 0.0


def test_one_metre_in_x():
    """Move 1 metre in x over 1 second → dist=1.0, avg=3.6 km/h."""
    rows = [
        make_row(0.0, 0.0, 0.0),
        make_row(1.0 / 9.0, 0.0, 1.0),   # 1 metre in x
    ]
    dist, avg, mx = compute_player_movement(rows)
    assert dist == 1.0
    assert avg == 3.6
    assert mx == 3.6


def test_gap_greater_than_2s_skipped():
    """Gaps > 2 seconds (player lost) are excluded from distance."""
    rows = [
        make_row(0.0, 0.0, 0.0),
        make_row(1.0, 1.0, 10.0),   # 10s gap → skip
    ]
    dist, avg, mx = compute_player_movement(rows)
    assert dist == 0.0


def test_speed_over_40kmh_excluded():
    """Speeds > 40 km/h (sensor noise) are excluded from avg/max."""
    # 100 m/s = impossible for a human
    rows = [
        make_row(0.0, 0.0, 0.0),
        make_row(100.0 / 9.0, 0.0, 1.0),
    ]
    dist, avg, mx = compute_player_movement(rows)
    # Distance still accumulated, but speed not counted in avg/max
    assert avg == 0.0
    assert mx == 0.0


def test_none_court_coords_skipped():
    rows = [
        make_row(None, 0.0, 0.0),
        make_row(0.5,  0.5, 1.0),
    ]
    dist, avg, mx = compute_player_movement(rows)
    assert dist == 0.0
