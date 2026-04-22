"""
Speech-to-Knowledge Router
───────────────────────────
Endpoints:
  POST /matches/{id}/speech/transcribe   — upload audio commentary file → transcribe + extract events
  POST /matches/{id}/speech/transcribe-video — transcribe audio track of the match video itself
  GET  /matches/{id}/speech/events       — list extracted speech events
  GET  /matches/{id}/speech/transcriptions — list transcription runs
  POST /matches/{id}/speech/fuse         — run event fusion against existing CV actions
"""

import os
import uuid
import shutil
import tempfile
import asyncio
import logging
from typing import Optional, List
from fastapi import (
    APIRouter, Depends, HTTPException, UploadFile, File,
    Query, BackgroundTasks
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete as sa_delete

from app.database import get_db
from app.models.match import Match, MatchStatus
from app.models.video import Video
from app.models.speech_events import SpeechTranscription, SpeechEvent
from app.models.actions import Action, ActionType, ActionResult
from app.models.player import Player
from app.utils.dependencies import get_current_user, require_coach
from app.models.user import User
from app.config import settings

router = APIRouter(prefix="/matches", tags=["Speech-to-Knowledge"])
logger = logging.getLogger(__name__)

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".mp4", ".webm"}
MAX_AUDIO_SIZE_MB = 500


# ─────────────────────────────────────────────────────────────────────────────
# Upload & transcribe commentary audio
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/{match_id}/speech/transcribe")
async def transcribe_commentary(
    match_id:         uuid.UUID,
    background_tasks: BackgroundTasks,
    audio_file:       UploadFile = File(..., description="Commentary audio/video file"),
    language:         str        = Query("en", description="ISO 639-1 language code"),
    current_user: User           = Depends(require_coach),
    db: AsyncSession             = Depends(get_db),
):
    """
    Upload a commentary audio file and start Whisper transcription + NLP extraction.
    Processing happens in the background.  Poll /speech/transcriptions for status.
    """
    result = await db.execute(select(Match).where(Match.id == match_id))
    match  = result.scalar_one_or_none()
    if not match:
        raise HTTPException(404, "Match not found")

    # Validate file extension
    ext = os.path.splitext(audio_file.filename or "")[1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )

    # Save uploaded file
    upload_dir = os.path.join(settings.UPLOAD_DIR, str(match_id), "speech")
    os.makedirs(upload_dir, exist_ok=True)
    dest_path = os.path.join(upload_dir, f"commentary_{uuid.uuid4().hex}{ext}")

    with open(dest_path, "wb") as out:
        shutil.copyfileobj(audio_file.file, out)

    # Create pending transcription record
    transcription = SpeechTranscription(
        match_id        = match_id,
        audio_file_path = dest_path,
        audio_source    = "upload",
        status          = "pending",
        language        = language,
    )
    db.add(transcription)
    await db.commit()
    await db.refresh(transcription)

    background_tasks.add_task(
        _run_transcription_and_extraction,
        str(transcription.id),
        dest_path,
        str(match_id),
        language,
    )

    return {
        "message":          "Transcription started",
        "transcription_id": str(transcription.id),
        "match_id":         str(match_id),
        "status":           "pending",
    }


@router.post("/{match_id}/speech/transcribe-video")
async def transcribe_video_audio(
    match_id:         uuid.UUID,
    background_tasks: BackgroundTasks,
    language:         str = Query("en", description="ISO 639-1 language code"),
    current_user: User    = Depends(require_coach),
    db: AsyncSession      = Depends(get_db),
):
    """
    Transcribe the audio track of the match video itself.
    Useful when the commentator's voice is recorded in the match video.
    """
    result = await db.execute(select(Match).where(Match.id == match_id))
    match  = result.scalar_one_or_none()
    if not match:
        raise HTTPException(404, "Match not found")

    vid_result = await db.execute(select(Video).where(Video.id == match.video_id))
    video = vid_result.scalar_one_or_none()
    if not video or not video.file_path:
        raise HTTPException(404, "Video file not found")

    if not os.path.exists(video.file_path):
        raise HTTPException(404, f"Video file missing on disk: {video.file_path}")

    # Create pending transcription record
    transcription = SpeechTranscription(
        match_id        = match_id,
        audio_file_path = video.file_path,
        audio_source    = "video_audio",
        status          = "pending",
        language        = language,
    )
    db.add(transcription)
    await db.commit()
    await db.refresh(transcription)

    background_tasks.add_task(
        _run_transcription_and_extraction,
        str(transcription.id),
        video.file_path,
        str(match_id),
        language,
    )

    return {
        "message":          "Video audio transcription started",
        "transcription_id": str(transcription.id),
        "match_id":         str(match_id),
        "status":           "pending",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Get speech events
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{match_id}/speech/events")
async def get_speech_events(
    match_id:    uuid.UUID,
    event_type:  Optional[str]  = Query(None),
    fusion_status: Optional[str] = Query(None, description="standalone|fused|conflict"),
    t_start:     Optional[float] = Query(None),
    t_end:       Optional[float] = Query(None),
    limit:       int             = Query(200, ge=1, le=1000),
    offset:      int             = Query(0, ge=0),
    current_user: User           = Depends(get_current_user),
    db: AsyncSession             = Depends(get_db),
):
    """List speech-extracted events for a match."""
    q = select(SpeechEvent).where(SpeechEvent.match_id == match_id)

    if event_type:
        q = q.where(SpeechEvent.event_type == event_type)
    if fusion_status:
        q = q.where(SpeechEvent.fusion_status == fusion_status)
    if t_start is not None:
        q = q.where(SpeechEvent.start_time >= t_start)
    if t_end is not None:
        q = q.where(SpeechEvent.start_time <= t_end)

    q = q.order_by(SpeechEvent.start_time).offset(offset).limit(limit)
    rows = (await db.execute(q)).scalars().all()

    return {
        "total":  len(rows),
        "offset": offset,
        "limit":  limit,
        "items":  [r.to_dict() for r in rows],
    }


@router.get("/{match_id}/speech/transcriptions")
async def get_transcriptions(
    match_id: uuid.UUID,
    current_user: User   = Depends(get_current_user),
    db: AsyncSession     = Depends(get_db),
):
    """List all transcription runs for a match."""
    q = (
        select(SpeechTranscription)
        .where(SpeechTranscription.match_id == match_id)
        .order_by(SpeechTranscription.created_at.desc())
    )
    rows = (await db.execute(q)).scalars().all()

    return {
        "transcriptions": [
            {
                "id":               str(t.id),
                "audio_source":     t.audio_source,
                "whisper_model":    t.whisper_model,
                "status":           t.status,
                "language":         t.language,
                "duration_seconds": t.duration_seconds,
                "error_message":    t.error_message,
                "created_at":       t.created_at.isoformat() if t.created_at else None,
            }
            for t in rows
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Manual fusion trigger
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/{match_id}/speech/fuse")
async def run_event_fusion(
    match_id: uuid.UUID,
    current_user: User   = Depends(require_coach),
    db: AsyncSession     = Depends(get_db),
):
    """
    Run event fusion between existing CV-detected actions and speech events.
    Useful to re-run after CV analysis completes if speech was processed first.
    """
    from app.services.event_fusion import EventFusionEngine

    # Load CV actions
    cv_result = await db.execute(
        select(Action).where(Action.match_id == match_id)
    )
    cv_actions = cv_result.scalars().all()

    # Load speech events
    se_result = await db.execute(
        select(SpeechEvent).where(SpeechEvent.match_id == match_id)
    )
    speech_events_db = se_result.scalars().all()

    if not speech_events_db:
        return {"message": "No speech events to fuse", "fused_count": 0}

    # Convert to dicts for fusion engine
    cv_dicts = [
        {
            "action_type": str(a.action_type),
            "confidence":  a.confidence or 0.5,
            "timestamp":   a.timestamp,
            "result":      str(a.result),
            "id":          str(a.id),
        }
        for a in cv_actions
    ]
    se_dicts = [
        {
            "event_type": e.event_type,
            "start_time": e.start_time,
            "result":     e.result,
            "team":       e.team,
            "confidence": e.extraction_confidence,
        }
        for e in speech_events_db
    ]

    engine = EventFusionEngine()
    _, updated_se = engine.fuse(cv_dicts, se_dicts)

    # Update fusion status on speech events
    for i, se_db in enumerate(speech_events_db):
        if i < len(updated_se):
            se_db.fusion_status = updated_se[i].get("fusion_status", "standalone")

    await db.commit()

    fused_count = sum(1 for se in updated_se if se.get("fusion_status") == "fused")
    return {
        "message":     f"Fusion complete: {fused_count} speech events matched to CV actions",
        "fused_count": fused_count,
        "total_speech": len(speech_events_db),
        "total_cv":     len(cv_actions),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Background task: transcribe + extract + persist
# ─────────────────────────────────────────────────────────────────────────────

async def _run_transcription_and_extraction(
    transcription_id: str,
    audio_path:       str,
    match_id:         str,
    language:         str = "en",
):
    """Background task: Whisper → NLP → DB persist."""
    from app.database import AsyncSessionLocal
    from app.services.speech_service import SpeechService
    from app.services.nlp_extractor  import NLPExtractor
    from app.services.event_fusion   import EventFusionEngine
    from app.models.actions import Action

    logger.info(f"SpeechPipeline: starting for transcription {transcription_id}")

    async with AsyncSessionLocal() as db:
        # Mark as processing
        t_res = await db.execute(
            select(SpeechTranscription).where(
                SpeechTranscription.id == uuid.UUID(transcription_id)
            )
        )
        transcription = t_res.scalar_one_or_none()
        if not transcription:
            return
        transcription.status = "processing"
        await db.commit()

    try:
        loop = asyncio.get_event_loop()

        # ── Step 1: Whisper transcription ─────────────────────────────────────
        svc = SpeechService()
        if not svc.load():
            raise RuntimeError("Whisper model failed to load")

        segments = await loop.run_in_executor(
            None, lambda: svc.transcribe(audio_path, language)
        )

        full_text = " ".join(s["text"] for s in segments)

        # ── Step 2: NLP extraction ────────────────────────────────────────────
        extractor     = NLPExtractor()
        speech_events = extractor.extract_events(segments)

        logger.info(
            f"SpeechPipeline: {len(segments)} segments → {len(speech_events)} events"
        )

        # ── Step 3: Event fusion with existing CV actions ─────────────────────
        async with AsyncSessionLocal() as db:
            cv_result = await db.execute(
                select(Action).where(Action.match_id == uuid.UUID(match_id))
            )
            cv_actions = cv_result.scalars().all()

        cv_dicts = [
            {
                "action_type": str(a.action_type),
                "confidence":  a.confidence or 0.5,
                "timestamp":   a.timestamp,
                "result":      str(a.result),
                "id":          str(a.id),
            }
            for a in cv_actions
        ]

        engine     = EventFusionEngine()
        _, fused_se = engine.fuse(cv_dicts, speech_events)

        # ── Step 4: Persist to DB ─────────────────────────────────────────────
        async with AsyncSessionLocal() as db:
            t_res = await db.execute(
                select(SpeechTranscription).where(
                    SpeechTranscription.id == uuid.UUID(transcription_id)
                )
            )
            transcription = t_res.scalar_one_or_none()
            if transcription:
                transcription.status        = "completed"
                transcription.full_text     = full_text
                transcription.segments_json = segments
                transcription.whisper_model = svc.model_size
                # Estimate audio duration from last segment end time
                if segments:
                    transcription.duration_seconds = segments[-1].get("end", 0.0)

            # Insert speech events
            # Normalise event_type for DB storage
            _NORMALISE = {"spike": "attack", "receive": "reception"}
            for se in fused_se:
                event_type = _NORMALISE.get(se.get("event_type", "unknown"), se.get("event_type", "unknown"))
                db.add(SpeechEvent(
                    match_id             = uuid.UUID(match_id),
                    transcription_id     = uuid.UUID(transcription_id),
                    raw_text             = se.get("raw_text", ""),
                    start_time           = float(se.get("start_time", 0.0)),
                    end_time             = float(se.get("end_time") or se.get("start_time", 0.0)),
                    event_type           = event_type,
                    player_number        = se.get("player_number"),
                    team                 = se.get("team"),
                    result               = se.get("result", "neutral"),
                    extraction_confidence= float(se.get("confidence", 0.5)),
                    fusion_status        = se.get("fusion_status", "standalone"),
                ))

            await db.commit()
            logger.info(
                f"SpeechPipeline: saved {len(fused_se)} speech events "
                f"for match {match_id}"
            )

    except Exception as e:
        logger.error(f"SpeechPipeline failed: {e}", exc_info=True)
        async with AsyncSessionLocal() as db:
            t_res = await db.execute(
                select(SpeechTranscription).where(
                    SpeechTranscription.id == uuid.UUID(transcription_id)
                )
            )
            transcription = t_res.scalar_one_or_none()
            if transcription:
                transcription.status        = "failed"
                transcription.error_message = str(e)
                await db.commit()
