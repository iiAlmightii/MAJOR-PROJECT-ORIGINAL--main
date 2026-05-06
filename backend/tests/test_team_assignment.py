import pytest

def _assign_teams_by_court_side(player_rows):
    """Pure extraction of the new logic — mirrors what cv_pipeline will do."""
    from collections import defaultdict
    track_xs = defaultdict(list)
    for r in player_rows:
        if r.get("court_x") is not None:
            track_xs[r["track_id"]].append(r["court_x"])

    team_map = {}
    for tid, xs in track_xs.items():
        median_x = sorted(xs)[len(xs) // 2]
        team_map[tid] = "A" if median_x < 0.5 else "B"

    # Tracks with no court position → None
    all_tids = {r["track_id"] for r in player_rows}
    for tid in all_tids - set(team_map.keys()):
        team_map[tid] = None

    return team_map


def test_left_side_is_team_a():
    rows = [{"track_id": 1, "court_x": 0.2}, {"track_id": 1, "court_x": 0.3}]
    result = _assign_teams_by_court_side(rows)
    assert result[1] == "A"


def test_right_side_is_team_b():
    rows = [{"track_id": 2, "court_x": 0.7}, {"track_id": 2, "court_x": 0.8}]
    result = _assign_teams_by_court_side(rows)
    assert result[2] == "B"


def test_player_crossing_net_uses_majority():
    # 3 frames on left, 1 frame on right → still Team A
    rows = [
        {"track_id": 3, "court_x": 0.2},
        {"track_id": 3, "court_x": 0.3},
        {"track_id": 3, "court_x": 0.4},
        {"track_id": 3, "court_x": 0.6},  # one frame crossing net
    ]
    result = _assign_teams_by_court_side(rows)
    assert result[3] == "A"


def test_no_court_position_returns_none():
    rows = [{"track_id": 4, "court_x": None}]
    result = _assign_teams_by_court_side(rows)
    assert result[4] is None


def test_twelve_players_six_per_side():
    rows = []
    # Tracks 1-6 on left, 7-12 on right
    for tid in range(1, 7):
        rows.append({"track_id": tid, "court_x": 0.1 + tid * 0.05})
    for tid in range(7, 13):
        rows.append({"track_id": tid, "court_x": 0.55 + (tid - 7) * 0.05})
    result = _assign_teams_by_court_side(rows)
    for tid in range(1, 7):
        assert result[tid] == "A", f"track {tid} should be A"
    for tid in range(7, 13):
        assert result[tid] == "B", f"track {tid} should be B"
