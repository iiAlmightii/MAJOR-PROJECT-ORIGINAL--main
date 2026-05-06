import pytest
from app.services.cv_pipeline import _build_team_map


def test_left_side_is_team_a():
    rows = [{"track_id": 1, "court_x": 0.2}, {"track_id": 1, "court_x": 0.3}]
    result = _build_team_map(rows)
    assert result[1] == "A"


def test_right_side_is_team_b():
    rows = [{"track_id": 2, "court_x": 0.7}, {"track_id": 2, "court_x": 0.8}]
    result = _build_team_map(rows)
    assert result[2] == "B"


def test_player_crossing_net_uses_majority():
    # 3 frames on left, 1 frame on right → median is 0.35 → Team A
    rows = [
        {"track_id": 3, "court_x": 0.2},
        {"track_id": 3, "court_x": 0.3},
        {"track_id": 3, "court_x": 0.4},
        {"track_id": 3, "court_x": 0.6},
    ]
    result = _build_team_map(rows)
    assert result[3] == "A"


def test_no_court_position_returns_none():
    rows = [{"track_id": 4, "court_x": None}]
    result = _build_team_map(rows)
    assert result[4] is None


def test_twelve_players_six_per_side():
    rows = []
    for tid in range(1, 7):
        # Multiple rows per track to exercise accumulation
        rows.append({"track_id": tid, "court_x": 0.1 + tid * 0.05})
        rows.append({"track_id": tid, "court_x": 0.15 + tid * 0.05})
    for tid in range(7, 13):
        rows.append({"track_id": tid, "court_x": 0.55 + (tid - 7) * 0.05})
        rows.append({"track_id": tid, "court_x": 0.60 + (tid - 7) * 0.05})
    result = _build_team_map(rows)
    for tid in range(1, 7):
        assert result[tid] == "A", f"track {tid} should be A"
    for tid in range(7, 13):
        assert result[tid] == "B", f"track {tid} should be B"
