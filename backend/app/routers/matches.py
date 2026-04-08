from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
import uuid

from app.database import get_db
from app.models.match import Match, MatchStatus
from app.models.video import Video
from app.models.user import User, UserRole
from app.models.actions import Rally
from app.schemas.match import MatchCreate, MatchUpdate, MatchResponse, MatchListResponse, RallyResponse
from app.utils.dependencies import get_current_user, require_coach, log_activity

router = APIRouter(prefix="/matches", tags=["Matches"])


@router.post("/", response_model=MatchResponse, status_code=status.HTTP_201_CREATED)
async def create_match(
    data: MatchCreate,
    request: Request,
    current_user: User = Depends(require_coach),
    db: AsyncSession = Depends(get_db),
):
    # Verify video exists and belongs to user
    result = await db.execute(select(Video).where(Video.id == data.video_id))
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if current_user.role != UserRole.admin and video.uploaded_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied to this video")

    match = Match(
        title=data.title,
        description=data.description,
        team_a=data.team_a,
        team_b=data.team_b,
        match_date=data.match_date,
        venue=data.venue,
        video_id=data.video_id,
        uploaded_by=current_user.id,
    )
    db.add(match)
    await db.flush()
    await db.refresh(match)

    await log_activity(db, current_user.id, "create_match", "match", str(match.id), request=request)
    return MatchResponse.model_validate(match)


@router.get("/", response_model=MatchListResponse)
async def list_matches(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=50),
    status_filter: Optional[MatchStatus] = None,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    query = select(Match)

    # Players only see matches for their team
    if current_user.role == UserRole.player:
        query = query.where(Match.uploaded_by == current_user.id)
    elif current_user.role == UserRole.coach:
        query = query.where(Match.uploaded_by == current_user.id)

    if status_filter:
        query = query.where(Match.status == status_filter)
    if search:
        query = query.where(Match.title.ilike(f"%{search}%"))

    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    offset = (page - 1) * per_page
    query = query.offset(offset).limit(per_page).order_by(Match.created_at.desc())
    result = await db.execute(query)
    matches = result.scalars().all()

    return MatchListResponse(
        matches=[MatchResponse.model_validate(m) for m in matches],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/{match_id}", response_model=MatchResponse)
async def get_match(
    match_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    if current_user.role == UserRole.player and match.uploaded_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    return MatchResponse.model_validate(match)


@router.patch("/{match_id}", response_model=MatchResponse)
async def update_match(
    match_id: uuid.UUID,
    data: MatchUpdate,
    request: Request,
    current_user: User = Depends(require_coach),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    if current_user.role != UserRole.admin and match.uploaded_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    for field, value in data.model_dump(exclude_none=True).items():
        setattr(match, field, value)
    await db.flush()

    await log_activity(db, current_user.id, "update_match", "match", str(match_id), request=request)
    return MatchResponse.model_validate(match)


@router.delete("/{match_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_match(
    match_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    if current_user.role not in (UserRole.admin,) and match.uploaded_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    await db.delete(match)



@router.get("/{match_id}/rallies", response_model=List[RallyResponse])
async def get_rallies(
    match_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Rally)
        .where(Rally.match_id == match_id)
        .order_by(Rally.rally_number)
    )
    rallies = result.scalars().all()
    return [RallyResponse.model_validate(r) for r in rallies]


@router.get("/{match_id}/analytics")
async def get_match_analytics(
    match_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from app.models.analytics import Analytics
    from app.models.actions import Action

    result = await db.execute(
        select(Analytics).where(Analytics.match_id == match_id)
    )
    analytics = result.scalars().all()

    if not analytics:
        return {"message": "No analytics available yet. Run analysis first.", "data": []}

    return {
        "match_id": str(match_id),
        "players": [
            {
                "player_id": str(a.player_id),
                "team": a.team,
                "serves": a.total_serves,
                "aces": a.aces,
                "serve_errors": a.serve_errors,
                "serve_efficiency": a.serve_efficiency,
                "attacks": a.total_attacks,
                "kills": a.attack_kills,
                "attack_errors": a.attack_errors,
                "attack_efficiency": a.attack_efficiency,
                "blocks": a.total_blocks,
                "block_points": a.block_points,
                "receptions": a.total_receptions,
                "reception_errors": a.reception_errors,
                "reception_efficiency": a.reception_efficiency,
                "digs": a.total_digs,
            }
            for a in analytics
        ],
    }
