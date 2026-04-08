# backend/app/routers/training.py
"""
Training trigger router.
Admin-only: runs run_phase3_pipeline.py as a subprocess,
streams stdout/stderr via WebSocket.
"""
import asyncio
import os
import json
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.models.user import User, UserRole
from app.utils.dependencies import get_current_user

router = APIRouter(prefix="/training", tags=["Training"])
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
PIPELINE_SCRIPT = ROOT / "training" / "action_recognition" / "run_phase3_pipeline.py"
PYTHON_BIN = ROOT / "backend" / ".venv" / "bin" / "python"

# Simple in-memory status (one training job at a time)
_training_status = {"running": False, "last_log": "", "last_exit_code": None}


class TrainRequest(BaseModel):
    phase: int = 2  # 1 or 2


@router.get("/status")
async def training_status(current_user: User = Depends(get_current_user)):
    if current_user.role != UserRole.admin:
        raise HTTPException(403, "Admin only")
    return _training_status


@router.post("/run")
async def trigger_training(
    body: TrainRequest,
    current_user: User = Depends(get_current_user),
):
    if current_user.role != UserRole.admin:
        raise HTTPException(403, "Admin only")
    if _training_status["running"]:
        raise HTTPException(409, "Training already running")
    if not PIPELINE_SCRIPT.exists():
        raise HTTPException(503, f"Training script not found at {PIPELINE_SCRIPT}")

    python = str(PYTHON_BIN) if PYTHON_BIN.exists() else "python"
    asyncio.create_task(_run_training(python, body.phase))
    return {"message": f"Training phase {body.phase} started"}


async def _run_training(python: str, phase: int):
    _training_status["running"] = True
    _training_status["last_log"] = "Starting..."
    _training_status["last_exit_code"] = None

    cmd = [python, str(PIPELINE_SCRIPT), "--phase", str(phase)]
    logger.info(f"Running training: {' '.join(cmd)}")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(ROOT),
        )
        async for line in proc.stdout:
            decoded = line.decode().rstrip()
            _training_status["last_log"] = decoded
            logger.info(f"[training] {decoded}")

        await proc.wait()
        _training_status["last_exit_code"] = proc.returncode
        logger.info(f"Training finished with exit code {proc.returncode}")

        if proc.returncode == 0:
            logger.info("Training complete — ActionService will reload weights on next analysis")

    except Exception as e:
        logger.error(f"Training subprocess error: {e}")
        _training_status["last_log"] = f"Error: {e}"
        _training_status["last_exit_code"] = -1
    finally:
        _training_status["running"] = False


@router.websocket("/ws")
async def training_ws(websocket: WebSocket):
    """Stream training log lines to connected clients."""
    await websocket.accept()
    last_log = ""
    try:
        while True:
            current_log = _training_status["last_log"]
            running = _training_status["running"]
            exit_code = _training_status["last_exit_code"]

            if current_log != last_log or not running:
                await websocket.send_text(json.dumps({
                    "running": running,
                    "log": current_log,
                    "exit_code": exit_code,
                }))
                last_log = current_log

            if not running:
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
