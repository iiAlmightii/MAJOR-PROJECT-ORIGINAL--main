"""
Scoring Engine
──────────────
Processes all Action records for a match and:

  1. Determines point winner per rally using volleyball rules
  2. Computes per-player and per-team statistics
  3. Generates match summary with key moments

Volleyball point rules implemented
───────────────────────────────────
  POINT WON by team when:
    • Spike/attack lands in opponent court (attack kill)
    • Block point (ball stuffed back into attacker's court)
    • Ace serve (opponent fails reception)
    • Opponent commits error (net, out, service fault)

  POINT LOST when:
    • Service error (fault / out)
    • Attack error (out / net)
    • Reception error (ball hits floor on own side)
    • Net violation

  RALLY WINNER heuristic (from our RallyDetector):
    • Floor hit in top half  (court_y > 0.5) → Team A scored
    • Floor hit in bottom half               → Team B scored
    • Ball lost (blocked, out, etc.)         → inferred from last action team
"""

import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import uuid

logger = logging.getLogger(__name__)

# Action → typical outcome for the acting team
ACTION_OUTCOMES = {
    "spike":             {"kill": +1, "error": -1, "neutral": 0},
    "serve":             {"ace":  +1, "error": -1, "neutral": 0},
    "block":             {"point":+1, "error":  0, "neutral": 0},
    "reception":         {"success": 0, "error": -1},
    "set":               {"success": 0, "error": -1},
    "dig":               {"success": 0, "error": -1},
    "free_ball_sent":    {"success": 0},
    "free_ball_received":{"success": 0, "error": -1},
}


class ScoringEngine:
    """
    Stateless scoring engine — call compute() with rally/action data.
    """

    def infer_action_results(
        self,
        actions: List[Dict[str, Any]],
        rallies: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rule-based result inference when CV pipeline has no result context.

        Rules:
          1. If action result is already set (not 'neutral'), keep it.
          2. Last action in a rally + rally has a winner:
             - If action team == rally.winner_team  → success (for spike/serve/block)
             - If action team != rally.winner_team  → error   (for spike/serve/block)
          3. Serves and attacks that are not the last action → neutral (not resolved yet).
          4. Receptions, digs, sets → always neutral unless speech-tagged.
        """
        if not actions:
            return actions

        # Build rally boundaries
        rally_intervals = []
        for r in rallies:
            rally_intervals.append({
                "start":  r.get("start_time", 0),
                "end":    r.get("end_time",   0),
                "winner": r.get("winner_team"),
            })
        rally_intervals.sort(key=lambda x: x["start"])

        def get_rally(ts: float):
            for ri in rally_intervals:
                if ri["start"] <= ts <= ri["end"] + 0.5:
                    return ri
            return None

        # Sort by timestamp to find "last action in rally"
        sorted_actions = sorted(actions, key=lambda a: a.get("timestamp", 0))

        # For each rally, mark the last action in it
        rally_last: Dict[str, int] = {}  # interval idx → last action index in sorted_actions
        for i, act in enumerate(sorted_actions):
            ri = get_rally(act.get("timestamp", 0))
            if ri:
                key = f"{ri['start']:.2f}"
                rally_last[key] = i

        last_action_indices = set(rally_last.values())

        updated = []
        for i, act in enumerate(sorted_actions):
            act = dict(act)
            if act.get("result", "neutral") != "neutral":
                updated.append(act)
                continue

            atype = act.get("action_type", "").lower()
            ri    = get_rally(act.get("timestamp", 0))

            if ri and ri.get("winner") and i in last_action_indices:
                winner = ri["winner"]
                team   = act.get("team")

                if atype in ("spike", "attack", "serve", "block"):
                    if team == winner:
                        act["result"] = "success"
                    elif team is not None:
                        act["result"] = "error"
                    else:
                        # No team info → assume success if last action winner
                        act["result"] = "success"

            updated.append(act)

        logger.info(
            f"ScoringEngine.infer_action_results: "
            f"{sum(1 for a in updated if a.get('result') != 'neutral')} / {len(updated)} "
            f"actions have resolved results"
        )
        return updated

    def compute(
        self,
        rallies: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        players: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        rallies : list of rally dicts from DB (rally_number, start_time, end_time,
                  winner_team, point_reason, events)
        actions : list of action dicts (action_type, result, timestamp, player_id, team)
        players : list of player dicts (id, team, player_track_id)

        Returns
        -------
        summary dict with score, player_stats, team_stats, key_moments
        """
        player_map   = {str(p["id"]): p for p in players}
        team_scores  = {"A": 0, "B": 0}
        player_stats = defaultdict(self._empty_player_stats)
        team_stats   = {"A": self._empty_team_stats(), "B": self._empty_team_stats()}
        key_moments  = []

        # ── Score from rally winners ──────────────────────────────────────────
        for r in rallies:
            winner = r.get("winner_team")
            if winner in ("A", "B"):
                team_scores[winner] += 1

        # ── Action statistics ─────────────────────────────────────────────────
        for action in actions:
            pid    = str(action.get("player_id", ""))
            team   = action.get("team") or (player_map.get(pid, {}).get("team", "?"))
            atype  = action.get("action_type", "unknown")
            result = action.get("result", "neutral")

            ps = player_stats[pid]
            ts = team_stats.get(team, self._empty_team_stats())

            self._update_player_stats(ps, atype, result)
            self._update_team_stats(ts, atype, result)

            # Key moment detection
            if atype == "spike" and result == "success":
                key_moments.append({
                    "timestamp": action.get("timestamp", 0),
                    "type":      "attack_kill",
                    "team":      team,
                    "player_id": pid,
                    "label":     f"Team {team} — Kill spike",
                })
            elif atype == "serve" and result == "success":
                key_moments.append({
                    "timestamp": action.get("timestamp", 0),
                    "type":      "ace",
                    "team":      team,
                    "player_id": pid,
                    "label":     f"Team {team} — Ace serve",
                })
            elif atype == "block" and result == "success":
                key_moments.append({
                    "timestamp": action.get("timestamp", 0),
                    "type":      "block_point",
                    "team":      team,
                    "player_id": pid,
                    "label":     f"Team {team} — Block point",
                })

        # Compute efficiencies
        for pid, ps in player_stats.items():
            ps["serve_efficiency"]    = self._efficiency(ps["aces"], ps["serve_errors"],   ps["total_serves"])
            ps["attack_efficiency"]   = self._efficiency(ps["kills"],ps["attack_errors"],  ps["total_attacks"])
            ps["reception_efficiency"]= self._efficiency(
                ps["total_receptions"] - ps["reception_errors"],
                ps["reception_errors"], ps["total_receptions"]
            )
            ps["player_id"] = pid

        key_moments.sort(key=lambda m: m["timestamp"])

        return {
            "team_a_score": team_scores["A"],
            "team_b_score": team_scores["B"],
            "total_rallies": len(rallies),
            "player_stats":  dict(player_stats),
            "team_stats":    team_stats,
            "key_moments":   key_moments[:20],   # top 20
        }

    # ──────────────────────────────────────────────────────────────────────────

    def _update_player_stats(self, ps: dict, atype: str, result: str):
        if atype == "serve":
            ps["total_serves"] += 1
            if result == "success":  ps["aces"] += 1
            elif result == "error":  ps["serve_errors"] += 1

        elif atype in ("spike", "attack"):
            ps["total_attacks"] += 1
            if result == "success":  ps["kills"] += 1
            elif result == "error":  ps["attack_errors"] += 1

        elif atype == "block":
            ps["total_blocks"] += 1
            if result == "success":  ps["block_points"] += 1
            elif result == "error":  ps["block_errors"] += 1

        elif atype == "reception":
            ps["total_receptions"] += 1
            if result == "error":    ps["reception_errors"] += 1

        elif atype == "dig":
            ps["total_digs"] += 1
            if result == "error":    ps["dig_errors"] += 1

        elif atype == "set":
            ps["total_sets"] += 1

    def _update_team_stats(self, ts: dict, atype: str, result: str):
        ts["total_actions"] += 1
        if result == "error":
            ts["total_errors"] += 1
        if atype in ("spike", "attack") and result == "success":
            ts["kills"] += 1
        if atype == "serve" and result == "success":
            ts["aces"] += 1
        if atype == "block" and result == "success":
            ts["block_points"] += 1

    @staticmethod
    def _efficiency(positive: int, errors: int, total: int) -> float:
        if total == 0:
            return 0.0
        return round((positive - errors) / total, 4)

    @staticmethod
    def _empty_player_stats() -> dict:
        return {
            "total_serves": 0, "aces": 0, "serve_errors": 0, "serve_efficiency": 0.0,
            "total_attacks": 0, "kills": 0, "attack_errors": 0, "attack_efficiency": 0.0,
            "total_blocks": 0, "block_points": 0, "block_errors": 0,
            "total_receptions": 0, "reception_errors": 0, "reception_efficiency": 0.0,
            "total_digs": 0, "dig_errors": 0,
            "total_sets": 0,
        }

    @staticmethod
    def _empty_team_stats() -> dict:
        return {
            "total_actions": 0, "total_errors": 0,
            "kills": 0, "aces": 0, "block_points": 0,
        }
