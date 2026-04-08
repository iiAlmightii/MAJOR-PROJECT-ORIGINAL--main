"""
Analysis Worker
───────────────
Background task that drives the CVPipeline for a match.
Progress is broadcast to all connected WebSocket clients for that match.

Usage (called from the /matches/{id}/analyze endpoint):
    background_tasks.add_task(run_analysis, match_id, video_path, court_corners)
"""

import asyncio
import logging
import uuid
from typing import Optional, List

logger = logging.getLogger(__name__)

# In-memory registry: match_id → set of WebSocket send callables
_ws_registry: dict[str, set] = {}


def register_ws(match_id: str, send_fn) -> None:
    _ws_registry.setdefault(match_id, set()).add(send_fn)


def unregister_ws(match_id: str, send_fn) -> None:
    if match_id in _ws_registry:
        _ws_registry[match_id].discard(send_fn)


async def _broadcast(match_id: str, pct: int, msg: str, failed: bool = False):
    """Send progress to every WS client watching this match."""
    import json
    payload = json.dumps({"progress": pct, "message": msg, "failed": failed})
    dead = set()
    for send_fn in list(_ws_registry.get(match_id, [])):
        try:
            await send_fn(payload)
        except Exception:
            dead.add(send_fn)
    for fn in dead:
        _ws_registry.get(match_id, set()).discard(fn)


async def run_analysis(
    match_id: str,
    video_path: str,
    court_corners: Optional[List] = None,
):
    """
    Entry-point background task.
    Updates match.status and match.processing_progress throughout.
    """
    from app.database import AsyncSessionLocal
    from app.models.match import Match, MatchStatus
    from sqlalchemy import select
    from app.services.cv_pipeline import CVPipeline

    logger.info(f"Starting analysis for match {match_id}")

    # Mark as processing
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Match).where(Match.id == uuid.UUID(match_id)))
        match  = result.scalar_one_or_none()
        if not match:
            logger.error(f"Match {match_id} not found")
            return
        match.status = MatchStatus.processing
        match.processing_progress = 0
        await db.commit()

    async def progress_cb(pct: int, msg: str):
        # Update DB progress
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(Match).where(Match.id == uuid.UUID(match_id))
                )
                m = result.scalar_one_or_none()
                if m:
                    m.processing_progress = pct
                    await db.commit()
        except Exception as e:
            logger.warning(f"Progress DB update failed: {e}")

        # Broadcast to WebSocket clients
        await _broadcast(match_id, pct, msg)

    pipeline = CVPipeline(
        match_id=match_id,
        video_path=video_path,
        progress_cb=progress_cb,
        court_corners=court_corners,
    )

    try:
        summary = await pipeline.run()
        logger.info(f"Analysis done for match {match_id}: {summary}")
    except Exception as exc:
        logger.error(f"Analysis failed for match {match_id}: {exc}", exc_info=True)
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Match).where(Match.id == uuid.UUID(match_id)))
            m = result.scalar_one_or_none()
            if m:
                from app.models.match import MatchStatus
                m.status = MatchStatus.failed
                await db.commit()
        await _broadcast(match_id, -1, f"Analysis failed: {exc}", failed=True)
