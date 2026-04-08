from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime
import uuid
from app.models.video import VideoStatus


class VideoResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    original_filename: str
    file_size: int
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    format: Optional[str] = None
    thumbnail_path: Optional[str] = None
    status: VideoStatus
    error_message: Optional[str] = None
    uploaded_by: uuid.UUID
    created_at: datetime


class VideoUploadResponse(BaseModel):
    video_id: uuid.UUID
    filename: str
    file_size: int
    status: VideoStatus
    message: str
