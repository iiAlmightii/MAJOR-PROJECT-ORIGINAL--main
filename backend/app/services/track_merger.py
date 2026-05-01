"""
Track Merger
────────────
Post-processing pass that runs after the full CV pipeline completes.

Steps:
  1. Load all Player records for the match.
  2. Find pairs that are likely the same physical person (same team,
     non-overlapping time, spatially close last/first position).
  3. Merge ghost tracks into their parent: re-assign player_tracking,
     actions rows; delete the ghost Player record.
  4. If still >16 players, prune the lowest frame-count ones.
  5. Assign stable display_number: Team A → #1-6, Team B → #7-12, rest → #13+.
"""

import logging
import uuid
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

SPATIAL_MERGE_THRESHOLD = 0.25   # normalized court units
MAX_PLAYERS = 16


# ─────────────────────────────────────────────────────────────────────────────
# Pure-logic helpers (tested without DB)
# ─────────────────────────────────────────────────────────────────────────────

def _find_merge_pairs(tracks: List[Dict]) -> List[Tuple[str, str]]:
    """
    Return list of (parent_id, ghost_id) pairs to merge.

    A pair is mergeable when ALL three hold:
      • same team
      • time ranges do not overlap (one ends before the other begins)
      • spatial gap between last position of earlier and first position
        of later is ≤ SPATIAL_MERGE_THRESHOLD
    """
    pairs = []
    n = len(tracks)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tracks[i], tracks[j]
            if a["team"] != b["team"] or a["team"] is None:
                continue
            # Ensure a ends before b starts (or swap)
            if a["t_end"] > b["t_start"] and b["t_end"] > a["t_start"]:
                continue  # overlap — cannot merge
            earlier, later = (a, b) if a["t_end"] <= b["t_start"] else (b, a)
            lx, ly = earlier.get("last_cx"), earlier.get("last_cy")
            fx, fy = later.get("first_cx"), later.get("first_cy")
            if lx is None or fx is None:
                continue
            dist = ((lx - fx) ** 2 + (ly - fy) ** 2) ** 0.5
            if dist <= SPATIAL_MERGE_THRESHOLD:
                pairs.append((earlier["player_id"], later["player_id"]))
    return pairs


def _assign_display_numbers(players: List[Dict]) -> List[Dict]:
    """
    Assign display_number to each player dict in place.
    Team A: #1–6 sorted by t_start.
    Team B: #7–12 sorted by t_start.
    Others (refs, coaches, unknown): #13+ sorted by t_start.
    Returns the same list with 'display_number' key added.
    """
    team_a = sorted([p for p in players if p.get("team") == "A"], key=lambda p: p["t_start"])
    team_b = sorted([p for p in players if p.get("team") == "B"], key=lambda p: p["t_start"])
    others = sorted([p for p in players if p.get("team") not in ("A", "B")], key=lambda p: p["t_start"])

    counter = 1
    for p in team_a:
        p["display_number"] = counter
        counter += 1

    counter = 7
    for p in team_b:
        p["display_number"] = counter
        counter += 1

    counter = 13
    for p in others:
        p["display_number"] = counter
        counter += 1

    return players


# ─────────────────────────────────────────────────────────────────────────────
# DB operations
# ─────────────────────────────────────────────────────────────────────────────

async def merge_tracks(match_id: str) -> Dict:
    """
    Run the full merge pass for a match. Called from analysis_worker after
    pipeline.run() completes successfully.

    Returns a summary dict: {merged, pruned, final_count}
    """
    from app.database import AsyncSessionLocal
    from app.models.player import Player
    from app.models.tracking import PlayerTracking
    from app.models.actions import Action
    from sqlalchemy import select, func, delete as sa_delete, update as sa_update

    logger.info(f"TrackMerger: starting for match {match_id}")
    mid = uuid.UUID(match_id)

    async with AsyncSessionLocal() as db:

        # ── Load player summaries ────────────────────────────────────────────
        p_result = await db.execute(
            select(Player).where(Player.match_id == mid)
        )
        players = p_result.scalars().all()

        if not players:
            logger.info("TrackMerger: no players found, skipping")
            return {"merged": 0, "pruned": 0, "final_count": 0}

        # For each player: get t_start, t_end, frame_count, last/first court pos
        track_info: List[Dict] = []
        for p in players:
            stats = await db.execute(
                select(
                    func.min(PlayerTracking.timestamp),
                    func.max(PlayerTracking.timestamp),
                    func.count(PlayerTracking.id),
                ).where(PlayerTracking.player_id == p.id)
            )
            row = stats.one()
            t_start, t_end, frame_count = row

            # Last known court position (foot of earlier track)
            last_pos = await db.execute(
                select(PlayerTracking.court_x, PlayerTracking.court_y)
                .where(
                    PlayerTracking.player_id == p.id,
                    PlayerTracking.court_x.isnot(None),
                )
                .order_by(PlayerTracking.timestamp.desc())
                .limit(1)
            )
            last_row = last_pos.one_or_none()

            # First known court position
            first_pos = await db.execute(
                select(PlayerTracking.court_x, PlayerTracking.court_y)
                .where(
                    PlayerTracking.player_id == p.id,
                    PlayerTracking.court_x.isnot(None),
                )
                .order_by(PlayerTracking.timestamp.asc())
                .limit(1)
            )
            first_row = first_pos.one_or_none()

            track_info.append({
                "player_id":   str(p.id),
                "team":        p.team,
                "t_start":     t_start or 0.0,
                "t_end":       t_end or 0.0,
                "frame_count": frame_count or 0,
                "last_cx":     last_row[0] if last_row else None,
                "last_cy":     last_row[1] if last_row else None,
                "first_cx":    first_row[0] if first_row else None,
                "first_cy":    first_row[1] if first_row else None,
            })

        # ── Fix team assignment using median court_x ─────────────────────────
        for info in track_info:
            if info["team"] is None:
                xs_r = await db.execute(
                    select(PlayerTracking.court_x)
                    .where(
                        PlayerTracking.player_id == uuid.UUID(info["player_id"]),
                        PlayerTracking.court_x.isnot(None),
                    )
                )
                xs = [r[0] for r in xs_r.all()]
                if xs:
                    median_x = sorted(xs)[len(xs) // 2]
                    new_team = "A" if median_x < 0.5 else "B"
                    info["team"] = new_team
                    await db.execute(
                        sa_update(Player)
                        .where(Player.id == uuid.UUID(info["player_id"]))
                        .values(team=new_team)
                    )

        # ── Find merge candidates ────────────────────────────────────────────
        pairs = _find_merge_pairs(track_info)
        logger.info(f"TrackMerger: found {len(pairs)} merge pairs")

        # Build parent→ghost mapping (resolve chains)
        parent_map: Dict[str, str] = {}  # ghost_id → parent_id
        for parent_id, ghost_id in pairs:
            # Follow chains: if parent is already a ghost, follow to root
            root = parent_id
            while root in parent_map:
                root = parent_map[root]
            parent_map[ghost_id] = root

        # ── Merge ghost rows into parent ─────────────────────────────────────
        merged_count = 0
        for ghost_id, parent_id in parent_map.items():
            ghost_uuid  = uuid.UUID(ghost_id)
            parent_uuid = uuid.UUID(parent_id)

            await db.execute(
                sa_update(PlayerTracking)
                .where(PlayerTracking.player_id == ghost_uuid)
                .values(player_id=parent_uuid)
            )
            await db.execute(
                sa_update(Action)
                .where(Action.player_id == ghost_uuid)
                .values(player_id=parent_uuid)
            )
            await db.execute(
                sa_delete(Player).where(Player.id == ghost_uuid)
            )
            merged_count += 1

        await db.flush()

        # ── Prune overflow (keep top MAX_PLAYERS by frame count) ─────────────
        remaining = await db.execute(
            select(Player).where(Player.match_id == mid)
        )
        remaining_players = remaining.scalars().all()

        pruned_count = 0
        if len(remaining_players) > MAX_PLAYERS:
            # Count frames per player
            counts = []
            for p in remaining_players:
                cnt = await db.execute(
                    select(func.count(PlayerTracking.id))
                    .where(PlayerTracking.player_id == p.id)
                )
                counts.append((p.id, cnt.scalar() or 0))

            counts.sort(key=lambda x: x[1], reverse=True)
            to_delete = [pid for pid, _ in counts[MAX_PLAYERS:]]
            for pid in to_delete:
                await db.execute(
                    sa_delete(PlayerTracking).where(PlayerTracking.player_id == pid)
                )
                await db.execute(
                    sa_delete(Action).where(Action.player_id == pid)
                )
                await db.execute(
                    sa_delete(Player).where(Player.id == pid)
                )
                pruned_count += 1
            await db.flush()

        # ── Assign display numbers ────────────────────────────────────────────
        final_result = await db.execute(
            select(Player).where(Player.match_id == mid)
        )
        final_players = final_result.scalars().all()

        # Build summary list for display number assignment
        summary = []
        for p in final_players:
            t_start_r = await db.execute(
                select(func.min(PlayerTracking.timestamp))
                .where(PlayerTracking.player_id == p.id)
            )
            t_start_val = t_start_r.scalar() or 0.0
            frame_cnt_r = await db.execute(
                select(func.count(PlayerTracking.id))
                .where(PlayerTracking.player_id == p.id)
            )
            summary.append({
                "player_id":    str(p.id),
                "team":         p.team,
                "t_start":      t_start_val,
                "frame_count":  frame_cnt_r.scalar() or 0,
            })

        numbered = _assign_display_numbers(summary)
        for item in numbered:
            await db.execute(
                sa_update(Player)
                .where(Player.id == uuid.UUID(item["player_id"]))
                .values(
                    display_number=item["display_number"],
                    display_name=f"Player #{item['display_number']} (Team {item['team'] or '?'})",
                )
            )

        await db.commit()

        final_count = len(final_players)
        logger.info(
            f"TrackMerger: done — merged={merged_count}, pruned={pruned_count}, "
            f"final={final_count}"
        )
        return {"merged": merged_count, "pruned": pruned_count, "final_count": final_count}
