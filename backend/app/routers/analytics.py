from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional
import uuid

from app.database import get_db
from app.models.user import User, UserRole
from app.models.match import Match, MatchStatus
from app.models.analytics import Analytics
from app.models.actions import Action, ActionType, Rally
from app.models.logs import UserActivityLog
from app.utils.dependencies import get_current_user, require_admin

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/dashboard/admin")
async def admin_dashboard(
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    from app.models.video import Video

    total_users = await db.execute(select(func.count(User.id)))
    total_matches = await db.execute(select(func.count(Match.id)))
    total_videos = await db.execute(select(func.count(Video.id)))
    completed_matches = await db.execute(
        select(func.count(Match.id)).where(Match.status == MatchStatus.completed)
    )
    processing_matches = await db.execute(
        select(func.count(Match.id)).where(Match.status == MatchStatus.processing)
    )
    recent_logs = await db.execute(
        select(UserActivityLog, User.username)
        .outerjoin(User, UserActivityLog.user_id == User.id)
        .order_by(UserActivityLog.timestamp.desc())
        .limit(15)
    )

    # Match status breakdown over last 10 matches
    recent_matches_q = await db.execute(
        select(Match).order_by(Match.created_at.desc()).limit(10)
    )
    recent_matches = recent_matches_q.scalars().all()

    return {
        "stats": {
            "total_users": total_users.scalar(),
            "total_matches": total_matches.scalar(),
            "total_videos": total_videos.scalar(),
            "completed_analyses": completed_matches.scalar(),
            "processing_matches": processing_matches.scalar(),
        },
        "recent_activity": [
            {
                "id": str(log.id),
                "user_id": str(log.user_id),
                "username": username or "unknown",
                "action": log.action,
                "resource_type": log.resource_type,
                "timestamp": log.timestamp.isoformat(),
            }
            for log, username in recent_logs.all()
        ],
        "recent_matches": [
            {
                "id": str(m.id),
                "title": m.title,
                "team_a": m.team_a,
                "team_b": m.team_b,
                "status": m.status.value,
                "team_a_score": m.team_a_score,
                "team_b_score": m.team_b_score,
                "created_at": m.created_at.isoformat(),
            }
            for m in recent_matches
        ],
    }


@router.get("/dashboard/coach")
async def coach_dashboard(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.role not in (UserRole.admin, UserRole.coach):
        raise HTTPException(status_code=403, detail="Access denied")

    my_matches = await db.execute(
        select(Match).where(Match.uploaded_by == current_user.id)
    )
    matches = my_matches.scalars().all()
    total = len(matches)
    completed = sum(1 for m in matches if m.status == MatchStatus.completed)
    processing = sum(1 for m in matches if m.status == MatchStatus.processing)

    # Per-match performance from real analytics table
    match_perf_q = await db.execute(
        select(
            Match.id,
            Match.title,
            Match.team_a,
            Match.team_b,
            Match.team_a_score,
            Match.team_b_score,
            Match.status,
            Match.created_at,
            func.coalesce(func.sum(Analytics.total_attacks), 0).label("attacks"),
            func.coalesce(func.sum(Analytics.attack_kills), 0).label("kills"),
            func.coalesce(func.sum(Analytics.total_serves), 0).label("serves"),
            func.coalesce(func.sum(Analytics.aces), 0).label("aces"),
            func.coalesce(func.sum(Analytics.total_blocks), 0).label("blocks"),
            func.coalesce(func.sum(Analytics.block_points), 0).label("block_pts"),
            func.coalesce(func.sum(Analytics.total_digs), 0).label("digs"),
            func.coalesce(func.avg(Analytics.attack_efficiency), 0.0).label("attack_eff"),
        )
        .outerjoin(Analytics, Analytics.match_id == Match.id)
        .where(Match.uploaded_by == current_user.id)
        .group_by(Match.id)
        .order_by(Match.created_at.asc())
        .limit(10)
    )
    match_rows = match_perf_q.all()

    match_performance = []
    for i, row in enumerate(match_rows):
        match_performance.append({
            "label": f"M{i+1}",
            "match_id": str(row.id),
            "title": row.title,
            "team_a": row.team_a or "Team A",
            "team_b": row.team_b or "Team B",
            "team_a_score": row.team_a_score,
            "team_b_score": row.team_b_score,
            "status": row.status.value,
            "attacks": int(row.attacks),
            "kills": int(row.kills),
            "serves": int(row.serves),
            "aces": int(row.aces),
            "blocks": int(row.blocks),
            "block_pts": int(row.block_pts),
            "digs": int(row.digs),
            "attack_eff": round(float(row.attack_eff) * 100, 1),
        })

    recent = sorted(matches, key=lambda m: m.created_at, reverse=True)[:5]

    return {
        "stats": {
            "total_matches": total,
            "completed_matches": completed,
            "processing_matches": processing,
            "pending_matches": total - completed - processing,
        },
        "match_performance": match_performance,
        "recent_matches": [
            {
                "id": str(m.id),
                "title": m.title,
                "team_a": m.team_a,
                "team_b": m.team_b,
                "status": m.status.value,
                "team_a_score": m.team_a_score,
                "team_b_score": m.team_b_score,
                "created_at": m.created_at.isoformat(),
            }
            for m in recent
        ],
    }


@router.get("/dashboard/player")
async def player_dashboard(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from app.models.player import Player

    player_records = await db.execute(
        select(Player).where(Player.user_id == current_user.id)
    )
    players = player_records.scalars().all()

    if not players:
        return {
            "message": "No match data found for your player profile yet.",
            "stats": {},
            "match_history": [],
        }

    player_ids = [p.id for p in players]
    analytics_result = await db.execute(
        select(Analytics, Match.title, Match.created_at)
        .join(Match, Analytics.match_id == Match.id)
        .where(Analytics.player_id.in_(player_ids))
        .order_by(Match.created_at.asc())
    )
    rows = analytics_result.all()

    if not rows:
        return {"message": "No analytics data yet", "stats": {}, "match_history": []}

    all_analytics = [r[0] for r in rows]

    aggregated = {
        "total_matches": len(set(str(a.match_id) for a in all_analytics)),
        "total_serves": sum(a.total_serves for a in all_analytics),
        "total_aces": sum(a.aces for a in all_analytics),
        "total_attacks": sum(a.total_attacks for a in all_analytics),
        "total_kills": sum(a.attack_kills for a in all_analytics),
        "total_blocks": sum(a.total_blocks for a in all_analytics),
        "block_points": sum(a.block_points for a in all_analytics),
        "total_digs": sum(a.total_digs for a in all_analytics),
        "total_receptions": sum(a.total_receptions for a in all_analytics),
        "avg_attack_efficiency": round(
            sum(a.attack_efficiency for a in all_analytics) / len(all_analytics), 3
        ),
        "avg_serve_efficiency": round(
            sum(a.serve_efficiency for a in all_analytics) / len(all_analytics), 3
        ),
        "avg_reception_efficiency": round(
            sum(a.reception_efficiency for a in all_analytics) / len(all_analytics), 3
        ),
    }

    # Per-match progression for line chart
    match_history = []
    for i, (a, title, created_at) in enumerate(rows):
        match_history.append({
            "label": f"M{i+1}",
            "title": title,
            "attacks": a.total_attacks,
            "kills": a.attack_kills,
            "serves": a.total_serves,
            "aces": a.aces,
            "blocks": a.total_blocks,
            "digs": a.total_digs,
            "attack_eff": round(a.attack_efficiency * 100, 1),
            "serve_eff": round(a.serve_efficiency * 100, 1),
        })

    return {
        "player": {
            "username": current_user.username,
            "position": current_user.position,
            "team": current_user.team_name,
        },
        "stats": aggregated,
        "match_history": match_history,
        "matches_played": len(players),
    }


@router.get("/leaderboard")
async def get_leaderboard(
    metric: str = Query("attacks", regex="^(attacks|kills|blocks|aces|digs|attack_eff)$"),
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Top players across all completed matches (coach sees own matches; admin sees all)."""
    from app.models.player import Player as PlayerModel

    q = (
        select(
            PlayerModel.display_name,
            PlayerModel.player_track_id,
            PlayerModel.team,
            func.coalesce(func.sum(Analytics.total_attacks), 0).label("attacks"),
            func.coalesce(func.sum(Analytics.attack_kills), 0).label("kills"),
            func.coalesce(func.avg(Analytics.attack_efficiency), 0.0).label("attack_eff"),
            func.coalesce(func.sum(Analytics.total_serves), 0).label("serves"),
            func.coalesce(func.sum(Analytics.aces), 0).label("aces"),
            func.coalesce(func.sum(Analytics.total_blocks), 0).label("blocks"),
            func.coalesce(func.sum(Analytics.block_points), 0).label("block_pts"),
            func.coalesce(func.sum(Analytics.total_digs), 0).label("digs"),
        )
        .join(Analytics, Analytics.player_id == PlayerModel.id)
        .join(Match, Match.id == Analytics.match_id)
        .where(Match.status == MatchStatus.completed)
    )

    if current_user.role != UserRole.admin:
        q = q.where(Match.uploaded_by == current_user.id)

    q = q.group_by(
        PlayerModel.id,
        PlayerModel.display_name,
        PlayerModel.player_track_id,
        PlayerModel.team,
    )

    metric_col = {
        "attacks": func.sum(Analytics.total_attacks),
        "kills":   func.sum(Analytics.attack_kills),
        "blocks":  func.sum(Analytics.total_blocks),
        "aces":    func.sum(Analytics.aces),
        "digs":    func.sum(Analytics.total_digs),
        "attack_eff": func.avg(Analytics.attack_efficiency),
    }[metric]

    q = q.order_by(metric_col.desc()).limit(limit)
    result = await db.execute(q)
    rows = result.all()

    return [
        {
            "name": row.display_name or f"Player #{row.player_track_id}",
            "team": row.team or "?",
            "attacks": int(row.attacks),
            "kills": int(row.kills),
            "attack_eff": round(float(row.attack_eff) * 100, 1),
            "serves": int(row.serves),
            "aces": int(row.aces),
            "blocks": int(row.blocks),
            "block_pts": int(row.block_pts),
            "digs": int(row.digs),
        }
        for row in rows
    ]


@router.get("/players")
async def list_all_players(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all players with analytics data for the compare dropdowns."""
    from app.models.player import Player as PlayerModel

    q = (
        select(PlayerModel.id, PlayerModel.display_name, PlayerModel.player_track_id, PlayerModel.team, Match.title)
        .join(Analytics, Analytics.player_id == PlayerModel.id)
        .join(Match, Match.id == PlayerModel.match_id)
        .where(Match.status == MatchStatus.completed)
        .distinct()
    )
    if current_user.role != UserRole.admin:
        q = q.where(Match.uploaded_by == current_user.id)

    result = await db.execute(q)
    rows = result.all()
    return [
        {
            "id": str(r.id),
            "name": r.display_name or f"Player #{r.player_track_id}",
            "team": r.team or "?",
            "match_title": r.title,
        }
        for r in rows
    ]


@router.get("/players/compare")
async def player_comparison(
    player_a: uuid.UUID,
    player_b: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Head-to-head stat comparison for two players across all their matches."""
    from app.models.player import Player as PlayerModel

    async def fetch_player(pid: uuid.UUID):
        p_result = await db.execute(
            select(PlayerModel).where(PlayerModel.id == pid)
        )
        player = p_result.scalar_one_or_none()
        if not player:
            raise HTTPException(404, f"Player {pid} not found")

        stats_result = await db.execute(
            select(
                func.coalesce(func.sum(Analytics.total_attacks), 0).label("attacks"),
                func.coalesce(func.sum(Analytics.attack_kills), 0).label("kills"),
                func.coalesce(func.avg(Analytics.attack_efficiency), 0.0).label("attack_eff"),
                func.coalesce(func.sum(Analytics.total_serves), 0).label("serves"),
                func.coalesce(func.sum(Analytics.aces), 0).label("aces"),
                func.coalesce(func.avg(Analytics.serve_efficiency), 0.0).label("serve_eff"),
                func.coalesce(func.sum(Analytics.total_blocks), 0).label("blocks"),
                func.coalesce(func.sum(Analytics.block_points), 0).label("block_pts"),
                func.coalesce(func.sum(Analytics.total_digs), 0).label("digs"),
                func.coalesce(func.sum(Analytics.total_receptions), 0).label("receptions"),
                func.coalesce(func.avg(Analytics.reception_efficiency), 0.0).label("reception_eff"),
            ).where(Analytics.player_id == pid)
        )
        row = stats_result.one()
        return {
            "id": str(player.id),
            "name": player.display_name or f"Player #{player.player_track_id}",
            "team": player.team or "?",
            "attacks": int(row.attacks),
            "kills": int(row.kills),
            "attack_eff": round(float(row.attack_eff) * 100, 1),
            "serves": int(row.serves),
            "aces": int(row.aces),
            "serve_eff": round(float(row.serve_eff) * 100, 1),
            "blocks": int(row.blocks),
            "block_pts": int(row.block_pts),
            "digs": int(row.digs),
            "receptions": int(row.receptions),
            "reception_eff": round(float(row.reception_eff) * 100, 1),
        }

    pa, pb = await fetch_player(player_a), await fetch_player(player_b)
    return {"player_a": pa, "player_b": pb}


@router.get("/match/{match_id}/compare")
async def team_comparison(
    match_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Team A vs Team B head-to-head stats for a completed match."""
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    if current_user.role not in (UserRole.admin,) and match.uploaded_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    rows = await db.execute(
        select(
            Analytics.team,
            func.coalesce(func.sum(Analytics.total_attacks), 0).label("attacks"),
            func.coalesce(func.sum(Analytics.attack_kills), 0).label("kills"),
            func.coalesce(func.sum(Analytics.attack_errors), 0).label("attack_errors"),
            func.coalesce(func.avg(Analytics.attack_efficiency), 0.0).label("attack_eff"),
            func.coalesce(func.sum(Analytics.total_serves), 0).label("serves"),
            func.coalesce(func.sum(Analytics.aces), 0).label("aces"),
            func.coalesce(func.sum(Analytics.serve_errors), 0).label("serve_errors"),
            func.coalesce(func.avg(Analytics.serve_efficiency), 0.0).label("serve_eff"),
            func.coalesce(func.sum(Analytics.total_blocks), 0).label("blocks"),
            func.coalesce(func.sum(Analytics.block_points), 0).label("block_pts"),
            func.coalesce(func.sum(Analytics.total_digs), 0).label("digs"),
            func.coalesce(func.sum(Analytics.total_receptions), 0).label("receptions"),
            func.coalesce(func.avg(Analytics.reception_efficiency), 0.0).label("reception_eff"),
        )
        .where(Analytics.match_id == match_id)
        .group_by(Analytics.team)
    )
    team_rows = rows.all()

    teams = {}
    for row in team_rows:
        key = row.team or "unknown"
        teams[key] = {
            "team": key,
            "attacks": int(row.attacks),
            "kills": int(row.kills),
            "attack_errors": int(row.attack_errors),
            "attack_eff": round(float(row.attack_eff) * 100, 1),
            "serves": int(row.serves),
            "aces": int(row.aces),
            "serve_errors": int(row.serve_errors),
            "serve_eff": round(float(row.serve_eff) * 100, 1),
            "blocks": int(row.blocks),
            "block_pts": int(row.block_pts),
            "digs": int(row.digs),
            "receptions": int(row.receptions),
            "reception_eff": round(float(row.reception_eff) * 100, 1),
        }

    return {
        "match_id": str(match_id),
        "title": match.title,
        "team_a": match.team_a or "Team A",
        "team_b": match.team_b or "Team B",
        "score": {"a": match.team_a_score, "b": match.team_b_score},
        "team_stats": teams,
        "comparison": [
            {"stat": "Attacks",     "a": teams.get("A", {}).get("attacks", 0),     "b": teams.get("B", {}).get("attacks", 0)},
            {"stat": "Kills",       "a": teams.get("A", {}).get("kills", 0),       "b": teams.get("B", {}).get("kills", 0)},
            {"stat": "Aces",        "a": teams.get("A", {}).get("aces", 0),        "b": teams.get("B", {}).get("aces", 0)},
            {"stat": "Blocks",      "a": teams.get("A", {}).get("blocks", 0),      "b": teams.get("B", {}).get("blocks", 0)},
            {"stat": "Block Pts",   "a": teams.get("A", {}).get("block_pts", 0),   "b": teams.get("B", {}).get("block_pts", 0)},
            {"stat": "Digs",        "a": teams.get("A", {}).get("digs", 0),        "b": teams.get("B", {}).get("digs", 0)},
            {"stat": "Receptions",  "a": teams.get("A", {}).get("receptions", 0),  "b": teams.get("B", {}).get("receptions", 0)},
            {"stat": "Atk Eff %",   "a": teams.get("A", {}).get("attack_eff", 0),  "b": teams.get("B", {}).get("attack_eff", 0)},
            {"stat": "Serve Eff %", "a": teams.get("A", {}).get("serve_eff", 0),   "b": teams.get("B", {}).get("serve_eff", 0)},
        ],
    }


@router.get("/logs")
async def get_system_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    action_filter: Optional[str] = None,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    query = select(UserActivityLog, User.username).outerjoin(
        User, UserActivityLog.user_id == User.id
    )
    if action_filter:
        query = query.where(UserActivityLog.action.ilike(f"%{action_filter}%"))

    query = query.offset(skip).limit(limit).order_by(UserActivityLog.timestamp.desc())
    result = await db.execute(query)

    return [
        {
            "id": str(log.id),
            "user_id": str(log.user_id),
            "username": username or "unknown",
            "action": log.action,
            "resource_type": log.resource_type,
            "resource_id": log.resource_id,
            "ip_address": log.ip_address,
            "timestamp": log.timestamp.isoformat(),
        }
        for log, username in result.all()
    ]
