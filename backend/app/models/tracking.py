import uuid
from datetime import datetime
from sqlalchemy import Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base


class PlayerTracking(Base):
    __tablename__ = "player_tracking"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("players.id"), nullable=False
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id"), nullable=False
    )
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)  # seconds
    # Bounding box (pixel coords)
    bbox_x: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_w: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_h: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    # Homography court coords (normalized 0-1)
    court_x: Mapped[float] = mapped_column(Float, nullable=True)
    court_y: Mapped[float] = mapped_column(Float, nullable=True)

    # Relationships
    player = relationship("Player", back_populates="tracking")

    def __repr__(self):
        return f"<PlayerTracking Player:{self.player_id} Frame:{self.frame_number}>"


class BallTracking(Base):
    __tablename__ = "ball_tracking"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id"), nullable=False
    )
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    court_x: Mapped[float] = mapped_column(Float, nullable=True)
    court_y: Mapped[float] = mapped_column(Float, nullable=True)

    def __repr__(self):
        return f"<BallTracking Match:{self.match_id} Frame:{self.frame_number}>"
