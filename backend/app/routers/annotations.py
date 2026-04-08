# backend/app/routers/annotations.py
import uuid
import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.database import get_db
from app.models.annotations import VideoAnnotation
from app.models.match import Match
from app.models.video import Video
from app.models.user import User, UserRole
from app.utils.dependencies import get_current_user, require_coach

router = APIRouter(tags=["Annotations"])

VALID_ACTION_TYPES = {"spike", "serve", "block", "dig", "set", "reception"}


class AnnotationCreate(BaseModel):
    timestamp: float
    action_type: str


@router.get("/matches/{match_id}/annotations")
async def list_annotations(
    match_id: uuid.UUID,
    current_user: User = Depends(require_coach),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(VideoAnnotation)
        .where(VideoAnnotation.match_id == match_id)
        .order_by(VideoAnnotation.timestamp)
    )
    annotations = result.scalars().all()
    return [
        {
            "id": str(a.id),
            "timestamp": a.timestamp,
            "action_type": a.action_type,
            "tagged_by": str(a.tagged_by) if a.tagged_by else None,
            "created_at": a.created_at.isoformat(),
        }
        for a in annotations
    ]


@router.post("/matches/{match_id}/annotations", status_code=201)
async def create_annotation(
    match_id: uuid.UUID,
    body: AnnotationCreate,
    current_user: User = Depends(require_coach),
    db: AsyncSession = Depends(get_db),
):
    if body.action_type not in VALID_ACTION_TYPES:
        raise HTTPException(400, f"Invalid action_type. Must be one of: {VALID_ACTION_TYPES}")

    match_result = await db.execute(select(Match).where(Match.id == match_id))
    match = match_result.scalar_one_or_none()
    if not match:
        raise HTTPException(404, "Match not found")

    vid_result = await db.execute(select(Video).where(Video.id == match.video_id))
    video = vid_result.scalar_one_or_none()
    video_path = video.file_path if video else ""

    annotation = VideoAnnotation(
        match_id=match_id,
        video_path=video_path,
        timestamp=body.timestamp,
        action_type=body.action_type,
        tagged_by=current_user.id,
    )
    db.add(annotation)
    await db.commit()
    await db.refresh(annotation)
    return {"id": str(annotation.id), "timestamp": annotation.timestamp, "action_type": annotation.action_type}


@router.delete("/annotations/{annotation_id}", status_code=204)
async def delete_annotation(
    annotation_id: uuid.UUID,
    current_user: User = Depends(require_coach),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(VideoAnnotation).where(VideoAnnotation.id == annotation_id)
    )
    annotation = result.scalar_one_or_none()
    if not annotation:
        raise HTTPException(404, "Annotation not found")
    await db.delete(annotation)
    await db.commit()


@router.get("/annotations/export")
async def export_annotations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export all annotations in training JSON format (admin only)."""
    if current_user.role != UserRole.admin:
        raise HTTPException(403, "Admin only")

    result = await db.execute(
        select(VideoAnnotation).order_by(VideoAnnotation.video_path, VideoAnnotation.timestamp)
    )
    annotations = result.scalars().all()

    from collections import defaultdict
    grouped = defaultdict(list)
    for a in annotations:
        grouped[a.video_path].append({"timestamp": a.timestamp, "action": a.action_type})

    export = [{"video_path": vp, "actions": acts} for vp, acts in grouped.items()]

    return JSONResponse(
        content=export,
        headers={"Content-Disposition": 'attachment; filename="annotations.json"'},
    )
