import uuid
from datetime import datetime
from sqlalchemy import Integer, Float, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.database import Base


class Rotation(Base):
    """
    Stores a detected team rotation snapshot at a point in a rally.

    The court is divided into a 2-row × 3-column grid (6 slots total):
      Row 0 (back row): slots 1, 2, 3  (left → right from team's perspective)
      Row 1 (front row): slots 4, 5, 6

    Each slot contains the player_id occupying that position (or None).
    """
    __tablename__ = "rotations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id"), nullable=False
    )
    rally_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("rallies.id"), nullable=True
    )
    # Seconds into the video when this rotation was captured
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=True)

    # Which team: "home" or "away" (or "unknown" if undifferentiated)
    team: Mapped[str] = mapped_column(String(20), default="unknown", nullable=False)

    # Slot assignments: slot_1 … slot_6 hold player UUID strings (nullable = empty slot)
    slot_1: Mapped[str] = mapped_column(String(36), nullable=True)  # back-left
    slot_2: Mapped[str] = mapped_column(String(36), nullable=True)  # back-center
    slot_3: Mapped[str] = mapped_column(String(36), nullable=True)  # back-right
    slot_4: Mapped[str] = mapped_column(String(36), nullable=True)  # front-left
    slot_5: Mapped[str] = mapped_column(String(36), nullable=True)  # front-center
    slot_6: Mapped[str] = mapped_column(String(36), nullable=True)  # front-right

    # Raw positions at snapshot time: list of {player_id, court_x, court_y}
    player_positions: Mapped[dict] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    match = relationship("Match", back_populates="rotations")

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "match_id": str(self.match_id),
            "rally_id": str(self.rally_id) if self.rally_id else None,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "team": self.team,
            "slots": {
                "1": self.slot_1,
                "2": self.slot_2,
                "3": self.slot_3,
                "4": self.slot_4,
                "5": self.slot_5,
                "6": self.slot_6,
            },
            "player_positions": self.player_positions or [],
        }

    def __repr__(self):
        return f"<Rotation Match:{self.match_id} t={self.timestamp:.1f}s>"
