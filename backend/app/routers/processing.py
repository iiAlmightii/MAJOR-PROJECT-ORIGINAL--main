"""
Processing Router
─────────────────
Endpoints for:
  • POST /matches/{id}/analyze  — start CV pipeline
  • WS  /ws/match/{id}/progress — real-time progress stream
  • GET /matches/{id}/tracking  — per-frame tracking data for canvas overlay
  • POST /matches/{id}/homography — set/update court corners
"""

import uuid
import asyncio
from typing import Optional, List
from fastapi import (
    APIRouter, Depends, HTTPException, BackgroundTasks,
    WebSocket, WebSocketDisconnect, Query, Request
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.database import get_db
from app.models.match import Match, MatchStatus
from app.models.video import Video
from app.models.tracking import PlayerTracking, BallTracking
from app.models.actions import Action, ActionType
from app.models.player import Player
from app.models.rotations import Rotation
from app.models.user import User, UserRole
from app.utils.dependencies import get_current_user, require_coach
from app.workers.analysis_worker import run_analysis, register_ws, unregister_ws

router = APIRouter(prefix="/matches", tags=["Processing"])


# ─────────────────────────────────────────────────────────────────────────────
# Start Analysis
# ─────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    court_corners: Optional[List[List[float]]] = None  # 4 × [x, y] in pixels


@router.post("/{match_id}/analyze")
async def start_analysis(
    match_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    request: Request,
    body: AnalyzeRequest = AnalyzeRequest(),
    current_user: User = Depends(require_coach),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Match).where(Match.id == match_id))
    match  = result.scalar_one_or_none()
    if not match:
        raise HTTPException(404, "Match not found")
    if current_user.role != UserRole.admin and match.uploaded_by != current_user.id:
        raise HTTPException(403, "Access denied")
    if match.status == MatchStatus.processing:
        raise HTTPException(400, "Analysis already running")

    # Get video path
    vid_result = await db.execute(select(Video).where(Video.id == match.video_id))
    video = vid_result.scalar_one_or_none()
    if not video:
        raise HTTPException(404, "Video file not found")

    match.status = MatchStatus.processing
    match.processing_progress = 0
    await db.flush()

    background_tasks.add_task(
        run_analysis,
        str(match_id),
        video.file_path,
        body.court_corners,
    )
    return {"message": "Analysis started", "match_id": str(match_id)}


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket – real-time progress
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/{match_id}/ws/progress")
async def ws_progress(match_id: uuid.UUID, websocket: WebSocket):
    await websocket.accept()
    mid = str(match_id)

    async def send_fn(msg: str):
        await websocket.send_text(msg)

    register_ws(mid, send_fn)
    try:
        # Keep alive – echo any client pings
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_text('{"type":"pong"}')
            except asyncio.TimeoutError:
                await websocket.send_text('{"type":"heartbeat"}')
    except WebSocketDisconnect:
        pass
    finally:
        unregister_ws(mid, send_fn)


# ─────────────────────────────────────────────────────────────────────────────
# Tracking data API  (used by frontend canvas overlay)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{match_id}/tracking")
async def get_tracking_data(
    match_id: uuid.UUID,
    timestamp: float = Query(..., description="Video timestamp in seconds"),
    window: float    = Query(0.1, description="±window seconds around timestamp"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """
    Returns player + ball tracking data for a given video timestamp.
    The frontend calls this periodically as the video plays to update
    the canvas overlay.
    """
    t_min = timestamp - window
    t_max = timestamp + window

    # Players
    p_result = await db.execute(
        select(PlayerTracking)
        .where(
            PlayerTracking.match_id == match_id,
            PlayerTracking.timestamp >= t_min,
            PlayerTracking.timestamp <= t_max,
        )
        .order_by(PlayerTracking.timestamp)
    )
    players = p_result.scalars().all()

    # Ball
    b_result = await db.execute(
        select(BallTracking)
        .where(
            BallTracking.match_id == match_id,
            BallTracking.timestamp >= t_min,
            BallTracking.timestamp <= t_max,
        )
        .order_by(BallTracking.timestamp)
    )
    balls = b_result.scalars().all()

    # Deduplicate: take closest frame per track_id
    seen_tracks: dict = {}
    for pt in players:
        tid = str(pt.player_id)
        if tid not in seen_tracks or abs(pt.timestamp - timestamp) < abs(seen_tracks[tid]["timestamp"] - timestamp):
            seen_tracks[tid] = {
                "player_id":   tid,
                "track_id":    pt.player_id,
                "bbox_x":      pt.bbox_x,
                "bbox_y":      pt.bbox_y,
                "bbox_w":      pt.bbox_w,
                "bbox_h":      pt.bbox_h,
                "court_x":     pt.court_x,
                "court_y":     pt.court_y,
                "timestamp":   pt.timestamp,
            }

    ball_data = None
    if balls:
        closest = min(balls, key=lambda b: abs(b.timestamp - timestamp))
        ball_data = {
            "x":        closest.x,
            "y":        closest.y,
            "court_x":  closest.court_x,
            "court_y":  closest.court_y,
            "timestamp":closest.timestamp,
        }

    return {
        "timestamp": timestamp,
        "players":   list(seen_tracks.values()),
        "ball":      ball_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Set homography corners
# ─────────────────────────────────────────────────────────────────────────────

class HomographyRequest(BaseModel):
    court_corners: List[List[float]]  # [[x,y]×4] top-left, top-right, bottom-right, bottom-left


# ─────────────────────────────────────────────────────────────────────────────
# Actions API  (detected actions from CV pipeline)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{match_id}/actions")
async def get_actions(
    match_id: uuid.UUID,
    action_type: Optional[str]  = Query(None, description="Filter by action type (spike, serve, block, …)"),
    player_id:   Optional[str]  = Query(None, description="Filter by player UUID"),
    t_start:     Optional[float] = Query(None, description="Min timestamp (seconds)"),
    t_end:       Optional[float] = Query(None, description="Max timestamp (seconds)"),
    min_confidence: float        = Query(0.0,  description="Minimum confidence threshold"),
    limit:       int             = Query(200,  ge=1, le=1000),
    offset:      int             = Query(0,    ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """
    List detected actions for a match.
    Supports filtering by action type, player, time range, and confidence.
    """
    q = select(Action, Player).outerjoin(
        Player, Action.player_id == Player.id
    ).where(Action.match_id == match_id)

    if action_type:
        q = q.where(Action.action_type == action_type)
    if player_id:
        try:
            pid = uuid.UUID(player_id)
            q = q.where(Action.player_id == pid)
        except ValueError:
            pass
    if t_start is not None:
        q = q.where(Action.timestamp >= t_start)
    if t_end is not None:
        q = q.where(Action.timestamp <= t_end)
    if min_confidence > 0:
        q = q.where(Action.confidence >= min_confidence)

    q = q.order_by(Action.timestamp).offset(offset).limit(limit)
    rows = (await db.execute(q)).all()

    # Count total for pagination
    count_q = select(Action).where(Action.match_id == match_id)
    if action_type:
        count_q = count_q.where(Action.action_type == action_type)
    if player_id:
        try:
            count_q = count_q.where(Action.player_id == uuid.UUID(player_id))
        except ValueError:
            pass
    if t_start is not None:
        count_q = count_q.where(Action.timestamp >= t_start)
    if t_end is not None:
        count_q = count_q.where(Action.timestamp <= t_end)
    if min_confidence > 0:
        count_q = count_q.where(Action.confidence >= min_confidence)

    from sqlalchemy import func
    total = (await db.execute(select(func.count()).select_from(count_q.subquery()))).scalar()

    items = []
    for action, player in rows:
        items.append({
            "id":           str(action.id),
            "action_type":  action.action_type,
            "result":       action.result,
            "timestamp":    action.timestamp,
            "frame_number": action.frame_number,
            "confidence":   action.confidence,
            "zone":         action.zone,
            "player_id":    str(action.player_id) if action.player_id else None,
            "player_track_id": player.player_track_id if player else None,
            "team":         player.team if player else None,
            "rally_id":     str(action.rally_id) if action.rally_id else None,
        })

    return {
        "total":  total,
        "offset": offset,
        "limit":  limit,
        "items":  items,
    }


@router.get("/{match_id}/rotations")
async def get_rotations(
    match_id: uuid.UUID,
    rally_id: Optional[str]  = Query(None, description="Filter by rally UUID"),
    team:     Optional[str]  = Query(None, description="Filter by team side (home/away/unknown)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """
    Return rotation snapshots for a match.
    Each entry represents the 6-slot assignment at the end of one rally.
    """
    q = select(Rotation).where(Rotation.match_id == match_id)
    if rally_id:
        try:
            q = q.where(Rotation.rally_id == uuid.UUID(rally_id))
        except ValueError:
            pass
    if team:
        q = q.where(Rotation.team == team)
    q = q.order_by(Rotation.timestamp)

    rows = (await db.execute(q)).scalars().all()
    return {"rotations": [r.to_dict() for r in rows]}


@router.post("/{match_id}/homography")
async def set_homography(
    match_id: uuid.UUID,
    body: HomographyRequest,
    current_user: User = Depends(require_coach),
    db: AsyncSession   = Depends(get_db),
):
    """Store court corners in the match summary for use during/after analysis."""
    result = await db.execute(select(Match).where(Match.id == match_id))
    match  = result.scalar_one_or_none()
    if not match:
        raise HTTPException(404, "Match not found")

    summary = match.summary or {}
    summary["court_corners"] = body.court_corners
    match.summary = summary
    await db.flush()
    return {"message": "Court corners saved", "corners": body.court_corners}
