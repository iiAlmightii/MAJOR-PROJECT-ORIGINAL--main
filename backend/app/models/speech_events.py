"""
Speech Event Model
──────────────────
Stores Whisper transcription segments + NLP-extracted volleyball events.
One row per extracted event (not per transcription segment).
"""

import uuid
from datetime import datetime
from sqlalchemy import String, Float, DateTime, ForeignKey, Text, Integer, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.database import Base


class SpeechTranscription(Base):
    """
    One per audio file uploaded for a match.
    Stores the full Whisper output and audio file reference.
    """
    __tablename__ = "speech_transcriptions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id"), nullable=False
    )
    audio_file_path: Mapped[str] = mapped_column(String(1000), nullable=True)
    audio_source: Mapped[str] = mapped_column(
        String(50), nullable=True
    )  # 'upload' | 'video_audio'
    whisper_model: Mapped[str] = mapped_column(String(50), nullable=True)
    full_text: Mapped[str] = mapped_column(Text, nullable=True)     # complete transcription
    segments_json: Mapped[dict] = mapped_column(JSONB, nullable=True)   # raw Whisper segments
    language: Mapped[str] = mapped_column(String(10), nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending | processing | completed | failed
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    match = relationship("Match", foreign_keys=[match_id])
    speech_events = relationship("SpeechEvent", back_populates="transcription")

    def __repr__(self):
        return f"<SpeechTranscription match={self.match_id} status={self.status}>"


class SpeechEvent(Base):
    """
    One row per volleyball event extracted from speech commentary.
    Multiple events can come from one transcription.
    """
    __tablename__ = "speech_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id"), nullable=False
    )
    transcription_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("speech_transcriptions.id"), nullable=True
    )

    # Source text
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)  # in video (seconds)
    end_time: Mapped[float] = mapped_column(Float, nullable=True)

    # Extracted event fields
    event_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # serve|spike|block|receive|dig|set|unknown
    player_number: Mapped[int] = mapped_column(Integer, nullable=True)  # jersey #
    team: Mapped[str] = mapped_column(String(10), nullable=True)       # A | B
    result: Mapped[str] = mapped_column(
        String(20), default="neutral"
    )  # success | error | neutral
    extraction_confidence: Mapped[float] = mapped_column(Float, default=0.5)

    # Fusion info
    fused_action_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("actions.id"), nullable=True
    )
    fusion_status: Mapped[str] = mapped_column(
        String(20), default="standalone"
    )  # standalone | fused | conflict

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    match = relationship("Match", foreign_keys=[match_id])
    transcription = relationship("SpeechTranscription", back_populates="speech_events")

    def to_dict(self) -> dict:
        return {
            "id":                  str(self.id),
            "match_id":            str(self.match_id),
            "transcription_id":    str(self.transcription_id) if self.transcription_id else None,
            "raw_text":            self.raw_text,
            "start_time":          self.start_time,
            "end_time":            self.end_time,
            "event_type":          self.event_type,
            "player_number":       self.player_number,
            "team":                self.team,
            "result":              self.result,
            "extraction_confidence": self.extraction_confidence,
            "fused_action_id":     str(self.fused_action_id) if self.fused_action_id else None,
            "fusion_status":       self.fusion_status,
        }

    def __repr__(self):
        return f"<SpeechEvent {self.event_type} t={self.start_time:.1f}s>"
