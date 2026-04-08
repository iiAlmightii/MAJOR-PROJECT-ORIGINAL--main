import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base


class Player(Base):
    __tablename__ = "players"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    player_track_id: Mapped[int] = mapped_column(Integer, nullable=False)  # AI assigned ID
    team: Mapped[str] = mapped_column(String(10), nullable=True)  # "A" or "B"
    position: Mapped[str] = mapped_column(String(50), nullable=True)
    jersey_number: Mapped[str] = mapped_column(String(10), nullable=True)
    display_name: Mapped[str] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    match = relationship("Match", back_populates="players")
    user = relationship("User", foreign_keys=[user_id])
    tracking = relationship("PlayerTracking", back_populates="player")
    actions = relationship("Action", back_populates="player")

    def __repr__(self):
        return f"<Player {self.display_name or self.player_track_id} (Match: {self.match_id})>"
