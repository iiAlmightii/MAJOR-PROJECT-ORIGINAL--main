import uuid
from datetime import datetime
from sqlalchemy import String, Float, DateTime, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.database import Base


class Analytics(Base):
    __tablename__ = "analytics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id"), nullable=False
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("players.id"), nullable=True
    )
    team: Mapped[str] = mapped_column(String(10), nullable=True)

    # Serve stats
    total_serves: Mapped[int] = mapped_column(Integer, default=0)
    serve_errors: Mapped[int] = mapped_column(Integer, default=0)
    aces: Mapped[int] = mapped_column(Integer, default=0)
    serve_efficiency: Mapped[float] = mapped_column(Float, default=0.0)

    # Attack stats
    total_attacks: Mapped[int] = mapped_column(Integer, default=0)
    attack_errors: Mapped[int] = mapped_column(Integer, default=0)
    attack_kills: Mapped[int] = mapped_column(Integer, default=0)
    attack_efficiency: Mapped[float] = mapped_column(Float, default=0.0)

    # Block stats
    total_blocks: Mapped[int] = mapped_column(Integer, default=0)
    block_errors: Mapped[int] = mapped_column(Integer, default=0)
    block_points: Mapped[int] = mapped_column(Integer, default=0)

    # Reception stats
    total_receptions: Mapped[int] = mapped_column(Integer, default=0)
    reception_errors: Mapped[int] = mapped_column(Integer, default=0)
    reception_efficiency: Mapped[float] = mapped_column(Float, default=0.0)

    # Dig stats
    total_digs: Mapped[int] = mapped_column(Integer, default=0)
    dig_errors: Mapped[int] = mapped_column(Integer, default=0)

    # Set stats
    total_sets: Mapped[int] = mapped_column(Integer, default=0)

    # Extra data
    extra_data: Mapped[dict] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    match = relationship("Match", back_populates="analytics")
    player = relationship("Player", foreign_keys=[player_id])

    def __repr__(self):
        return f"<Analytics Match:{self.match_id} Player:{self.player_id}>"
