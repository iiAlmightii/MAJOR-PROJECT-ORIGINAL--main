import uuid
from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, Enum as SAEnum, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base
import enum


class VideoStatus(str, enum.Enum):
    uploading = "uploading"
    uploaded = "uploaded"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)  # bytes
    duration: Mapped[float] = mapped_column(Float, nullable=True)    # seconds
    width: Mapped[int] = mapped_column(Integer, nullable=True)
    height: Mapped[int] = mapped_column(Integer, nullable=True)
    fps: Mapped[float] = mapped_column(Float, nullable=True)
    format: Mapped[str] = mapped_column(String(20), nullable=True)
    thumbnail_path: Mapped[str] = mapped_column(String(1000), nullable=True)
    status: Mapped[VideoStatus] = mapped_column(
        SAEnum(VideoStatus), default=VideoStatus.uploading, nullable=False
    )
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    uploader = relationship("User", foreign_keys=[uploaded_by])
    matches = relationship("Match", back_populates="video")

    def __repr__(self):
        return f"<Video {self.original_filename} ({self.status})>"
