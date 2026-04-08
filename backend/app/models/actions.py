import uuid
from datetime import datetime
from sqlalchemy import String, Float, DateTime, Enum as SAEnum, ForeignKey, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.database import Base
import enum


class ActionType(str, enum.Enum):
    serve = "serve"
    reception = "reception"
    set = "set"
    attack = "attack"
    block = "block"
    dig = "dig"
    free_ball_sent = "free_ball_sent"
    free_ball_received = "free_ball_received"
    unknown = "unknown"


class ActionResult(str, enum.Enum):
    success = "success"
    error = "error"
    neutral = "neutral"


class Action(Base):
    __tablename__ = "actions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id"), nullable=False
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("players.id"), nullable=True
    )
    rally_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rallies.id"), nullable=True
    )
    action_type: Mapped[ActionType] = mapped_column(SAEnum(ActionType), nullable=False)
    result: Mapped[ActionResult] = mapped_column(
        SAEnum(ActionResult), default=ActionResult.neutral, nullable=False
    )
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)    # seconds in video
    frame_number: Mapped[int] = mapped_column(Integer, nullable=True)
    zone: Mapped[int] = mapped_column(Integer, nullable=True)          # court zone 1-6
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    match = relationship("Match", back_populates="actions")
    player = relationship("Player", back_populates="actions")
    rally = relationship("Rally", back_populates="actions")

    def __repr__(self):
        return f"<Action {self.action_type} ({self.result}) at {self.timestamp}s>"


class Rally(Base):
    __tablename__ = "rallies"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id"), nullable=False
    )
    rally_number: Mapped[int] = mapped_column(Integer, nullable=False)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)   # seconds
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    start_frame: Mapped[int] = mapped_column(Integer, nullable=True)
    end_frame: Mapped[int] = mapped_column(Integer, nullable=True)
    video_clip_path: Mapped[str] = mapped_column(String(1000), nullable=True)
    winner_team: Mapped[str] = mapped_column(String(10), nullable=True)
    point_reason: Mapped[str] = mapped_column(String(200), nullable=True)
    error_by_player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("players.id"), nullable=True
    )
    events: Mapped[dict] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    match = relationship("Match", back_populates="rallies")
    actions = relationship("Action", back_populates="rally")

    def __repr__(self):
        return f"<Rally #{self.rally_number} Match:{self.match_id}>"


class Event(Base):
    __tablename__ = "events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id"), nullable=False
    )
    rally_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rallies.id"), nullable=True
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("players.id"), nullable=True
    )
    team: Mapped[str] = mapped_column(String(10), nullable=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    result: Mapped[str] = mapped_column(String(50), nullable=True)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    metadata_json: Mapped[dict] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
