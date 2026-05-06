from app.services.track_merger import _find_merge_pairs, _assign_display_numbers


def test_find_merge_pairs_same_team_no_overlap():
    tracks = [
        {"player_id": "a", "team": "A", "t_start": 0.0, "t_end": 5.0,
         "last_cx": 0.2, "last_cy": 0.5, "first_cx": 0.2, "first_cy": 0.5},
        {"player_id": "b", "team": "A", "t_start": 5.5, "t_end": 10.0,
         "last_cx": 0.2, "last_cy": 0.5, "first_cx": 0.22, "first_cy": 0.5},
    ]
    pairs = _find_merge_pairs(tracks)
    assert len(pairs) == 1
    assert pairs[0] == ("a", "b")


def test_find_merge_pairs_different_team_not_merged():
    tracks = [
        {"player_id": "a", "team": "A", "t_start": 0.0, "t_end": 5.0,
         "last_cx": 0.2, "last_cy": 0.5, "first_cx": 0.2, "first_cy": 0.5},
        {"player_id": "b", "team": "B", "t_start": 5.5, "t_end": 10.0,
         "last_cx": 0.2, "last_cy": 0.5, "first_cx": 0.22, "first_cy": 0.5},
    ]
    pairs = _find_merge_pairs(tracks)
    assert len(pairs) == 0


def test_find_merge_pairs_overlap_not_merged():
    tracks = [
        {"player_id": "a", "team": "A", "t_start": 0.0, "t_end": 8.0,
         "last_cx": 0.2, "last_cy": 0.5, "first_cx": 0.2, "first_cy": 0.5},
        {"player_id": "b", "team": "A", "t_start": 5.0, "t_end": 12.0,
         "last_cx": 0.22, "last_cy": 0.5, "first_cx": 0.22, "first_cy": 0.5},
    ]
    pairs = _find_merge_pairs(tracks)
    assert len(pairs) == 0


def test_assign_display_numbers_uses_jersey_when_available():
    players = [
        {"player_id": "a", "team": "A", "t_start": 0.0, "jersey_number": 7},
        {"player_id": "b", "team": "B", "t_start": 0.0, "jersey_number": 11},
        {"player_id": "c", "team": "A", "t_start": 1.0, "jersey_number": None},
    ]
    result = _assign_display_numbers(players)
    p = {pl["player_id"]: pl for pl in result}
    assert p["a"]["display_number"] == 7
    assert p["b"]["display_number"] == 11
    assert p["c"]["display_number"] is not None
    assert isinstance(p["c"]["display_number"], int)


def test_assign_display_numbers_no_jersey_uses_sequential():
    players = [
        {"player_id": "a", "team": "A", "t_start": 0.0, "jersey_number": None},
        {"player_id": "b", "team": "B", "t_start": 0.0, "jersey_number": None},
    ]
    result = _assign_display_numbers(players)
    p = {pl["player_id"]: pl for pl in result}
    assert p["a"]["display_number"] in range(1, 20)
    assert p["b"]["display_number"] in range(1, 20)
    assert p["a"]["display_number"] != p["b"]["display_number"]
