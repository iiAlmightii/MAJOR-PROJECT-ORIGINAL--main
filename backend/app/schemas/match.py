from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
from app.models.match import MatchStatus


class MatchCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    description: Optional[str] = None
    team_a: Optional[str] = None
    team_b: Optional[str] = None
    match_date: Optional[datetime] = None
    venue: Optional[str] = None
    video_id: uuid.UUID


class MatchUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    team_a: Optional[str] = None
    team_b: Optional[str] = None
    match_date: Optional[datetime] = None
    venue: Optional[str] = None


class MatchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    description: Optional[str] = None
    team_a: Optional[str] = None
    team_b: Optional[str] = None
    match_date: Optional[datetime] = None
    venue: Optional[str] = None
    video_id: uuid.UUID
    uploaded_by: uuid.UUID
    status: MatchStatus
    processing_progress: int
    total_rallies: int
    team_a_score: int
    team_b_score: int
    summary: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


class MatchListResponse(BaseModel):
    matches: List[MatchResponse]
    total: int
    page: int
    per_page: int


class RallyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    match_id: uuid.UUID
    rally_number: int
    start_time: float
    end_time: float
    video_clip_path: Optional[str] = None
    winner_team: Optional[str] = None
    point_reason: Optional[str] = None
    events: Optional[List[Dict[str, Any]]] = None
