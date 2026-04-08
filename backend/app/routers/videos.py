import os
import uuid
import aiofiles
from typing import Optional
from fastapi import (
    APIRouter, Depends, HTTPException, UploadFile, File,
    Form, status, Request, BackgroundTasks
)
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.video import Video, VideoStatus
from app.models.user import User, UserRole
from app.schemas.video import VideoResponse, VideoUploadResponse
from app.utils.dependencies import get_current_user, require_coach, log_activity
from app.utils.jwt_handler import verify_access_token
from app.config import settings
from app.services.video_service import extract_video_metadata, generate_thumbnail

router = APIRouter(prefix="/videos", tags=["Videos"])

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@router.post("/upload", response_model=VideoUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(require_coach),
    db: AsyncSession = Depends(get_db),
):
    # Validate extension
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read file in chunks and validate size
    safe_name = f"{uuid.uuid4()}{ext.lower()}"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_name)

    total_size = 0
    chunk_size = 1024 * 1024  # 1 MB

    async with aiofiles.open(file_path, "wb") as out_file:
        while chunk := await file.read(chunk_size):
            total_size += len(chunk)
            if total_size > settings.max_upload_bytes:
                await out_file.close()
                os.remove(file_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB",
                )
            await out_file.write(chunk)

    # Create video record
    video = Video(
        filename=safe_name,
        original_filename=file.filename or safe_name,
        file_path=file_path,
        file_size=total_size,
        format=ext.lower().lstrip("."),
        status=VideoStatus.uploaded,
        uploaded_by=current_user.id,
    )
    db.add(video)
    await db.flush()
    await db.refresh(video)

    # Extract metadata in background
    background_tasks.add_task(
        process_video_metadata, str(video.id), file_path
    )

    await log_activity(
        db, current_user.id, "upload_video", "video", str(video.id),
        {"filename": file.filename, "size": total_size}, request
    )

    return VideoUploadResponse(
        video_id=video.id,
        filename=video.original_filename,
        file_size=total_size,
        status=video.status,
        message="Video uploaded successfully. Processing metadata...",
    )


async def process_video_metadata(video_id: str, file_path: str):
    from app.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Video).where(Video.id == uuid.UUID(video_id)))
        video = result.scalar_one_or_none()
        if not video:
            return
        try:
            metadata = await extract_video_metadata(file_path)
            video.duration = metadata.get("duration")
            video.width = metadata.get("width")
            video.height = metadata.get("height")
            video.fps = metadata.get("fps")

            # Generate thumbnail
            thumb_path = await generate_thumbnail(file_path, video_id)
            video.thumbnail_path = thumb_path
            video.status = VideoStatus.uploaded
        except Exception as e:
            video.status = VideoStatus.failed
            video.error_message = str(e)
        await db.commit()


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if current_user.role not in (UserRole.admin, UserRole.coach):
        if video.uploaded_by != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

    return VideoResponse.model_validate(video)


@router.get("/{video_id}/stream")
async def stream_video(
    video_id: uuid.UUID,
    request: Request,
    token: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    access_token = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        access_token = auth_header.split(" ", 1)[1].strip()
    elif token:
        access_token = token

    if not access_token:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    payload = verify_access_token(access_token)
    if not payload or not payload.get("sub"):
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    try:
        user_id = uuid.UUID(payload["sub"])
    except ValueError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    user_result = await db.execute(select(User).where(User.id == user_id))
    current_user = user_result.scalar_one_or_none()
    if not current_user or not current_user.is_active:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if current_user.role not in (UserRole.admin, UserRole.coach):
        if video.uploaded_by != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(video.file_path):
        raise HTTPException(status_code=404, detail="Video file not found on server")

    file_size = os.path.getsize(video.file_path)
    range_header = request.headers.get("Range")

    if range_header:
        # Parse range header for partial content
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0])
        end = int(range_match[1]) if range_match[1] else file_size - 1
        chunk_size = end - start + 1

        async def iter_file():
            async with aiofiles.open(video.file_path, "rb") as f:
                await f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    data = await f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_file(),
            status_code=206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
                "Content-Type": f"video/{video.format or 'mp4'}",
            },
        )

    return FileResponse(
        video.file_path,
        media_type=f"video/{video.format or 'mp4'}",
        filename=video.original_filename,
    )


@router.get("/{video_id}/thumbnail")
async def get_thumbnail(
    video_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    if not video or not video.thumbnail_path or not os.path.exists(video.thumbnail_path):
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return FileResponse(video.thumbnail_path, media_type="image/jpeg")


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if current_user.role not in (UserRole.admin,) and video.uploaded_by != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Delete file
    if os.path.exists(video.file_path):
        os.remove(video.file_path)
    if video.thumbnail_path and os.path.exists(video.thumbnail_path):
        os.remove(video.thumbnail_path)

    await db.delete(video)
