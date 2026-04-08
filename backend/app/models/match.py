import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Enum as SAEnum, ForeignKey, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.database import Base
import enum


class MatchStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    analyzing = "analyzing"
    completed = "completed"
    failed = "failed"


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    team_a: Mapped[str] = mapped_column(String(200), nullable=True)
    team_b: Mapped[str] = mapped_column(String(200), nullable=True)
    match_date: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    venue: Mapped[str] = mapped_column(String(300), nullable=True)

    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id"), nullable=False
    )
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    status: Mapped[MatchStatus] = mapped_column(
        SAEnum(MatchStatus), default=MatchStatus.pending, nullable=False
    )
    processing_progress: Mapped[int] = mapped_column(Integer, default=0)  # 0–100
    total_rallies: Mapped[int] = mapped_column(Integer, default=0)
    team_a_score: Mapped[int] = mapped_column(Integer, default=0)
    team_b_score: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[dict] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    video = relationship("Video", back_populates="matches")
    uploader = relationship("User", back_populates="uploaded_matches", foreign_keys=[uploaded_by])
    players = relationship("Player", back_populates="match")
    rallies = relationship("Rally", back_populates="match")
    actions = relationship("Action", back_populates="match")
    analytics = relationship("Analytics", back_populates="match")

    def __repr__(self):
        return f"<Match {self.title} ({self.status})>"
