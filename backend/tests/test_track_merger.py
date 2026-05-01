"""Unit tests for track merger logic — pure Python, no DB."""
import pytest
from app.services.track_merger import (
    _find_merge_pairs,
    _assign_display_numbers,
)


def make_track(player_id, team, t_start, t_end, last_cx, last_cy, first_cx, first_cy, frame_count):
    return {
        "player_id": player_id,
        "team": team,
        "t_start": t_start,
        "t_end": t_end,
        "last_cx": last_cx,
        "last_cy": last_cy,
        "first_cx": first_cx,
        "first_cy": first_cy,
        "frame_count": frame_count,
    }


def test_find_merge_pairs_same_team_non_overlapping_close():
    tracks = [
        make_track("p1", "A", 0.0, 10.0, 0.3, 0.5, None, None, 500),
        make_track("p2", "A", 12.0, 30.0, None, None, 0.32, 0.51, 200),
    ]
    pairs = _find_merge_pairs(tracks)
    assert ("p1", "p2") in pairs or ("p2", "p1") in pairs


def test_find_merge_pairs_different_team_not_merged():
    tracks = [
        make_track("p1", "A", 0.0, 10.0, 0.3, 0.5, None, None, 500),
        make_track("p2", "B", 12.0, 30.0, None, None, 0.32, 0.51, 200),
    ]
    pairs = _find_merge_pairs(tracks)
    assert len(pairs) == 0


def test_find_merge_pairs_overlapping_times_not_merged():
    tracks = [
        make_track("p1", "A", 0.0, 20.0, 0.3, 0.5, None, None, 500),
        make_track("p2", "A", 15.0, 30.0, None, None, 0.32, 0.51, 200),
    ]
    pairs = _find_merge_pairs(tracks)
    assert len(pairs) == 0


def test_find_merge_pairs_too_far_spatially_not_merged():
    tracks = [
        make_track("p1", "A", 0.0, 10.0, 0.1, 0.1, None, None, 500),
        make_track("p2", "A", 12.0, 30.0, None, None, 0.9, 0.9, 200),
    ]
    pairs = _find_merge_pairs(tracks)
    assert len(pairs) == 0


def test_assign_display_numbers_team_a_first():
    players = [
        {"player_id": "a1", "team": "A", "frame_count": 100, "t_start": 5.0},
        {"player_id": "a2", "team": "A", "frame_count": 200, "t_start": 2.0},
        {"player_id": "b1", "team": "B", "frame_count": 150, "t_start": 3.0},
    ]
    result = _assign_display_numbers(players)
    nums = {p["player_id"]: p["display_number"] for p in result}
    # Team A: sorted by t_start → a2 (2.0)=#1, a1 (5.0)=#2
    assert nums["a2"] == 1
    assert nums["a1"] == 2
    # Team B: starts at #7
    assert nums["b1"] == 7


def test_assign_display_numbers_none_team_gets_high_number():
    players = [
        {"player_id": "x1", "team": None, "frame_count": 50, "t_start": 1.0},
    ]
    result = _assign_display_numbers(players)
    assert result[0]["display_number"] >= 13
