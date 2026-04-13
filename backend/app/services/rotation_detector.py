"""
Rotation Detector
─────────────────
Assigns players to one of 6 volleyball court slots based on their normalized
court coordinates (court_x, court_y ∈ [0, 1]) at a given point in time.

Court slot layout (viewed from above, team side occupying y ∈ [0, 0.5]):

  Front row (y close to net, y < 0.25):
    Slot 4 (front-left)  | Slot 5 (front-center) | Slot 6 (front-right)
  ─────────────────────────────────────────────────────────────────────
  Back row (y farther from net, y ≥ 0.25):
    Slot 1 (back-left)   | Slot 2 (back-center)  | Slot 3 (back-right)

If the team occupies y ∈ [0.5, 1.0] (other half), the same logic mirrors
along the y-axis first.

Usage::

    from app.services.rotation_detector import detect_rotation
    rotation = detect_rotation(match_id, rally_id, timestamp, frame_number,
                               player_positions, team_side="home")
"""

import logging
import uuid
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Slot grid definition:
#   slot index (1-based) → (row, col) where row 0 = back, row 1 = front
#                                            col 0 = left, col 1 = center, col 2 = right
_SLOT_GRID = {
    1: (0, 0),  # back-left
    2: (0, 1),  # back-center
    3: (0, 2),  # back-right
    4: (1, 0),  # front-left
    5: (1, 1),  # front-center
    6: (1, 2),  # front-right
}

# Reverse lookup: (row, col) → slot number
_GRID_TO_SLOT = {v: k for k, v in _SLOT_GRID.items()}


def _assign_slot(court_x: float, court_y: float, team_side: str = "home") -> int:
    """
    Map a single player's court position to a slot number (1-6).

    Parameters
    ----------
    court_x : float
        Normalized x coordinate (0 = left, 1 = right).
    court_y : float
        Normalized y coordinate (0 = top/net side for home, 1 = bottom/baseline).
    team_side : str
        "home" occupies y ∈ [0, 0.5], "away" occupies y ∈ [0.5, 1.0].

    Returns
    -------
    int
        Slot number 1-6.
    """
    # Normalize y to [0, 1] within the team's half
    if team_side == "away":
        y_norm = (court_y - 0.5) * 2.0  # map [0.5, 1.0] → [0, 1]
    else:
        y_norm = court_y * 2.0           # map [0, 0.5] → [0, 1]

    y_norm = max(0.0, min(1.0, y_norm))
    x_norm = max(0.0, min(1.0, court_x))

    # Row: 0 = back (y_norm ≥ 0.5), 1 = front (y_norm < 0.5)
    row = 0 if y_norm >= 0.5 else 1

    # Column: 0 = left, 1 = center, 2 = right
    if x_norm < 0.33:
        col = 0
    elif x_norm < 0.67:
        col = 1
    else:
        col = 2

    return _GRID_TO_SLOT[(row, col)]


def detect_rotation(
    match_id: uuid.UUID,
    rally_id: Optional[uuid.UUID],
    timestamp: float,
    frame_number: Optional[int],
    player_positions: List[Dict[str, Any]],
    team_side: str = "unknown",
) -> Dict[str, Any]:
    """
    Assign each player in *player_positions* to a court slot.

    Parameters
    ----------
    match_id : uuid.UUID
    rally_id : uuid.UUID or None
    timestamp : float
        Seconds into the video.
    frame_number : int or None
    player_positions : list of dict
        Each dict must have keys: ``player_id`` (str), ``court_x`` (float),
        ``court_y`` (float).  Players outside [0,1]×[0,1] are skipped.
    team_side : str
        "home", "away", or "unknown".  When "unknown" the detector uses the
        median court_y to infer which half the team occupies.

    Returns
    -------
    dict
        Ready for construction of a ``Rotation`` model instance:
        {
            "match_id": ...,
            "rally_id": ...,
            "timestamp": ...,
            "frame_number": ...,
            "team": ...,
            "slot_1" … "slot_6": player_id str or None,
            "player_positions": [...],
        }
    """
    if not player_positions:
        logger.debug("detect_rotation: no player positions supplied")
        return _empty_rotation(match_id, rally_id, timestamp, frame_number, team_side)

    # Filter positions that have valid court coords
    valid: List[Dict[str, Any]] = [
        p for p in player_positions
        if p.get("court_x") is not None and p.get("court_y") is not None
        and 0.0 <= p["court_x"] <= 1.0
        and 0.0 <= p["court_y"] <= 1.0
    ]

    if not valid:
        logger.debug("detect_rotation: no players with valid court coords")
        return _empty_rotation(match_id, rally_id, timestamp, frame_number, team_side)

    # Infer team side from median y if unknown
    inferred_side = team_side
    if team_side == "unknown":
        median_y = sorted(p["court_y"] for p in valid)[len(valid) // 2]
        inferred_side = "home" if median_y <= 0.5 else "away"
        logger.debug("detect_rotation: inferred team_side=%s from median_y=%.2f", inferred_side, median_y)

    # Assign slots — if two players compete for the same slot, keep closest to ideal center
    slot_map: Dict[int, Dict[str, Any]] = {}  # slot → player dict

    for player in valid:
        slot = _assign_slot(player["court_x"], player["court_y"], inferred_side)
        if slot not in slot_map:
            slot_map[slot] = player
        else:
            # Keep the player whose position is closer to the slot's ideal center
            existing = slot_map[slot]
            if _slot_distance(player, slot, inferred_side) < _slot_distance(existing, slot, inferred_side):
                slot_map[slot] = player

    rotation = {
        "match_id": match_id,
        "rally_id": rally_id,
        "timestamp": timestamp,
        "frame_number": frame_number,
        "team": inferred_side,
        "slot_1": str(slot_map[1]["player_id"]) if 1 in slot_map else None,
        "slot_2": str(slot_map[2]["player_id"]) if 2 in slot_map else None,
        "slot_3": str(slot_map[3]["player_id"]) if 3 in slot_map else None,
        "slot_4": str(slot_map[4]["player_id"]) if 4 in slot_map else None,
        "slot_5": str(slot_map[5]["player_id"]) if 5 in slot_map else None,
        "slot_6": str(slot_map[6]["player_id"]) if 6 in slot_map else None,
        "player_positions": [
            {"player_id": str(p["player_id"]), "court_x": p["court_x"], "court_y": p["court_y"]}
            for p in valid
        ],
    }
    return rotation


def _slot_distance(player: Dict[str, Any], slot: int, team_side: str) -> float:
    """Euclidean distance from a player's court position to the ideal center of a slot."""
    row, col = _SLOT_GRID[slot]
    # Ideal center in team-local coords
    ideal_local_y = 0.25 if row == 1 else 0.75  # front → 0.25, back → 0.75
    ideal_local_x = col / 2.0 + 1 / 6.0         # 0.17, 0.50, 0.83

    # Convert ideal local coords back to absolute court coords
    if team_side == "away":
        ideal_y = ideal_local_y / 2.0 + 0.5
    else:
        ideal_y = ideal_local_y / 2.0

    dx = player["court_x"] - ideal_local_x
    dy = player["court_y"] - ideal_y
    return dx * dx + dy * dy


def _empty_rotation(match_id, rally_id, timestamp, frame_number, team) -> Dict[str, Any]:
    return {
        "match_id": match_id,
        "rally_id": rally_id,
        "timestamp": timestamp,
        "frame_number": frame_number,
        "team": team,
        "slot_1": None,
        "slot_2": None,
        "slot_3": None,
        "slot_4": None,
        "slot_5": None,
        "slot_6": None,
        "player_positions": [],
    }
