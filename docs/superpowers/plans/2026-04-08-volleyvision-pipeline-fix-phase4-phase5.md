# VolleyVision Pipeline Fix + Phase 4 + Phase 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the broken end-to-end CV analysis pipeline, then build Phase 4 (multi-action LSTM annotation + training trigger) and Phase 5 (player comparison, rotation detection, match summary + heat maps).

**Architecture:** Fix the 5 identified bugs in the pipeline first (WebSocket proxy, failure signal handling, early-return in scoring, dep-check on startup, failed UI state), then add new DB models/routers for annotation and training, then add Phase 5 analytics features on top of the working pipeline.

**Tech Stack:** FastAPI + SQLAlchemy async (backend), React 18 + Vite + Recharts + Tailwind (frontend), PostgreSQL, YOLOv8/ByteTrack, BiLSTM, MediaPipe/rtmlib

---

## File Map

### New files
| Path | Purpose |
|------|---------|
| `backend/app/models/annotations.py` | `VideoAnnotation` ORM model |
| `backend/app/models/rotations.py` | `Rotation` ORM model |
| `backend/app/routers/annotations.py` | Annotation CRUD + export endpoint |
| `backend/app/routers/training.py` | Training trigger + WS status endpoint |
| `backend/app/services/rotation_detector.py` | Rotation slot assignment logic |
| `frontend/src/pages/AnnotatePage.jsx` | In-app timestamp-tagging UI |
| `frontend/src/pages/PlayerComparePage.jsx` | Player comparison radar + diff table |
| `frontend/src/components/Video/MatchSummaryTab.jsx` | Heat map + zone chart + key moments |
| `frontend/src/components/Video/RotationPanel.jsx` | Court rotation diagram per rally |

### Modified files
| Path | Changes |
|------|---------|
| `frontend/vite.config.js` | Add `ws: true` to `/api` proxy |
| `frontend/src/hooks/useAnalysisProgress.js` | Expose `failed` state from `progress === -1` |
| `frontend/src/pages/MatchDetailPage.jsx` | Handle `failed` status; add Summary tab; add RotationPanel |
| `frontend/src/App.jsx` | Add `/matches/:id/annotate` and `/analytics/compare` routes |
| `frontend/src/services/api.js` | Add annotation, training, compare, heatmap API calls |
| `backend/app/services/cv_pipeline.py` | Remove early return in `_run_scoring`; integrate rotation detection |
| `backend/app/workers/analysis_worker.py` | Emit `type:"failed"` field in failure broadcast |
| `backend/app/main.py` | Add CV dep-check log in `lifespan()`; register new routers |
| `backend/app/models/__init__.py` | Import new models so `create_all` picks them up |
| `backend/requirements.txt` | Add `torch` CPU line for development |
| `backend/app/routers/processing.py` | Add ball heatmap endpoint |
| `backend/app/routers/analytics.py` | Add player-to-player compare endpoint |

---

## Section 1 — Pipeline Fixes

---

### Task 1: Fix WebSocket proxy in Vite

**Files:**
- Modify: `frontend/vite.config.js`

- [ ] **Step 1: Add `ws: true` to the `/api` proxy block**

```js
// frontend/vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        ws: true,
      },
      '/uploads': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/rallies': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
})
```

- [ ] **Step 2: Restart the Vite dev server and verify `/api/health` responds**

```bash
cd frontend
npm run dev
# In a new tab:
curl http://localhost:5173/api/health
# Expected: {"status":"healthy","app":"Volleyball Analytics Platform","version":"1.0.0"}
```

- [ ] **Step 3: Commit**

```bash
cd frontend
git add vite.config.js
git commit -m "fix: enable WebSocket proxying in Vite dev server"
```

---

### Task 2: Fix failure signal — backend broadcast

**Files:**
- Modify: `backend/app/workers/analysis_worker.py:107`

The failure broadcast currently sends `progress: -1`. Add a `type` field so the frontend can reliably distinguish failure from in-progress.

- [ ] **Step 1: Update the failure broadcast call in `run_analysis`**

Open `backend/app/workers/analysis_worker.py`. Find line 107:
```python
        await _broadcast(match_id, -1, f"Analysis failed: {exc}")
```
Replace with:
```python
        await _broadcast(match_id, -1, f"Analysis failed: {exc}", failed=True)
```

- [ ] **Step 2: Update `_broadcast` to accept and include the `failed` flag**

Find the `_broadcast` function (lines 31-41) and replace it:
```python
async def _broadcast(match_id: str, pct: int, msg: str, failed: bool = False):
    """Send progress to every WS client watching this match."""
    import json
    payload = json.dumps({"progress": pct, "message": msg, "failed": failed})
    dead = set()
    for send_fn in list(_ws_registry.get(match_id, [])):
        try:
            await send_fn(payload)
        except Exception:
            dead.add(send_fn)
    for fn in dead:
        _ws_registry.get(match_id, set()).discard(fn)
```

- [ ] **Step 3: Commit**

```bash
cd backend
git add app/workers/analysis_worker.py
git commit -m "fix: broadcast type:failed on analysis exception via WebSocket"
```

---

### Task 3: Fix failure signal — frontend hook and MatchDetailPage

**Files:**
- Modify: `frontend/src/hooks/useAnalysisProgress.js`
- Modify: `frontend/src/pages/MatchDetailPage.jsx`

- [ ] **Step 1: Expose `failed` state in `useAnalysisProgress`**

Replace the entire content of `frontend/src/hooks/useAnalysisProgress.js`:

```js
import { useState, useEffect, useRef } from 'react'

export function useAnalysisProgress(matchId, enabled = false) {
  const [progress,  setProgress]  = useState(0)
  const [message,   setMessage]   = useState('')
  const [connected, setConnected] = useState(false)
  const [failed,    setFailed]    = useState(false)
  const wsRef  = useRef(null)
  const pingRef = useRef(null)

  useEffect(() => {
    if (!matchId || !enabled) return

    setFailed(false)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host     = window.location.host
    const url      = `${protocol}//${host}/api/matches/${matchId}/ws/progress`

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      pingRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send('ping')
      }, 20_000)
    }

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data)
        if (data.failed) {
          setFailed(true)
          setMessage(data.message || 'Analysis failed')
          setProgress(0)
        } else if (data.progress !== undefined) {
          setProgress(Math.max(0, data.progress))
          setMessage(data.message || '')
        }
      } catch { /* ignore malformed */ }
    }

    ws.onclose  = () => { setConnected(false); clearInterval(pingRef.current) }
    ws.onerror  = () => setConnected(false)

    return () => {
      clearInterval(pingRef.current)
      ws.close()
    }
  }, [matchId, enabled])

  return { progress, message, connected, failed }
}
```

- [ ] **Step 2: Handle `failed` state in MatchDetailPage**

In `frontend/src/pages/MatchDetailPage.jsx`, find where `useAnalysisProgress` is destructured (search for `useAnalysisProgress`). It will look like:
```js
const { progress, message, connected } = useAnalysisProgress(...)
```
Change to:
```js
const { progress, message, connected, failed: analysisFailed } = useAnalysisProgress(...)
```

Find the section in MatchDetailPage that renders the analysis progress bar / processing state. It will have a condition like `match.status === 'processing'`. Add a failed state display directly after:

```jsx
{analysisFailed && (
  <div className="flex items-center gap-3 p-4 bg-red-900/30 border border-red-700 rounded-lg">
    <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
    <div>
      <div className="text-red-300 font-medium text-sm">Analysis Failed</div>
      <div className="text-red-400/80 text-xs mt-0.5">{message}</div>
    </div>
  </div>
)}
```

Also find the section that checks `match.status === 'failed'` (or add it if missing) to show a persistent failed banner when the user navigates back to the page:

```jsx
{match.status === 'failed' && !analysisFailed && (
  <div className="flex items-center gap-3 p-4 bg-red-900/30 border border-red-700 rounded-lg">
    <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
    <div className="text-red-300 text-sm">Analysis failed. Check server logs and re-run analysis.</div>
  </div>
)}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/useAnalysisProgress.js frontend/src/pages/MatchDetailPage.jsx
git commit -m "fix: show analysis failure state in UI via WebSocket failed signal"
```

---

### Task 4: Fix scoring engine early return

**Files:**
- Modify: `backend/app/services/cv_pipeline.py:406`

Currently `_run_scoring` returns early when both `action_rows` and `rallies` are empty. This means zero Analytics rows are written when action recognition is disabled (no LSTM weights). We need to write zero-value rows for each detected player so the Analytics tab is never blank for a completed match.

- [ ] **Step 1: Remove the early return guard and always run scoring**

In `backend/app/services/cv_pipeline.py`, find the `_run_scoring` method. Find this block (around line 406):
```python
        if not action_rows and not rallies:
            return
```
Remove those two lines entirely.

- [ ] **Step 2: Ensure Analytics rows are written for players with zero actions**

In the same `_run_scoring` method, after the block that persists `Analytics` per player from `summary["player_stats"]`, add a zero-row fallback for any detected player that didn't appear in `player_stats`:

Find the section (near the end of `_run_scoring`) that writes Analytics rows. It will look like:
```python
            for pid_str, stats in summary.get("player_stats", {}).items():
```

After the closing block of that loop (after the `await db.commit()`), add:

```python
            # Write zero-stat Analytics rows for any player not in player_stats
            written_pids = set(summary.get("player_stats", {}).keys())
            for p in db_players:
                if str(p.id) not in written_pids:
                    db.add(Analytics(
                        match_id=uuid.UUID(self.match_id),
                        player_id=p.id,
                        team=p.team,
                    ))
            await db.commit()
```

Make sure `Analytics` is imported inside `_run_scoring` — it already is via `from app.models.analytics import Analytics`.

- [ ] **Step 3: Commit**

```bash
cd backend
git add app/services/cv_pipeline.py
git commit -m "fix: always write Analytics rows per player even when action recognition is disabled"
```

---

### Task 5: Add CV dependency check on startup

**Files:**
- Modify: `backend/app/main.py`

- [ ] **Step 1: Add `_check_cv_deps()` and call it from `lifespan`**

In `backend/app/main.py`, after the imports block, add this function:

```python
def _check_cv_deps():
    """Log which CV dependencies are available at startup."""
    deps = {
        "torch": False,
        "ultralytics": False,
        "supervision": False,
        "mediapipe": False,
        "rtmlib": False,
        "cv2": False,
    }
    for name in deps:
        try:
            __import__(name)
            deps[name] = True
        except ImportError:
            pass

    available = [k for k, v in deps.items() if v]
    missing   = [k for k, v in deps.items() if not v]

    if available:
        logger.info(f"CV deps available: {', '.join(available)}")
    if missing:
        logger.warning(
            f"CV deps MISSING (analysis will be limited): {', '.join(missing)}"
        )
```

Then in `lifespan`, add a call to it:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Volleyball Analytics API...")
    await create_tables()
    await seed_admin()
    _check_cv_deps()
    logger.info("Database tables ready.")
    yield
    logger.info("Shutting down...")
```

- [ ] **Step 2: Restart backend and verify the log output**

```bash
cd backend
source .venv/bin/activate
python run.py
# Expected in logs: lines like "CV deps available: torch, ultralytics, cv2, ..." 
# and "CV deps MISSING: ..." for anything not installed
```

- [ ] **Step 3: Commit**

```bash
git add app/main.py
git commit -m "feat: log CV dependency availability at startup"
```

---

### Task 6: Ensure torch is in requirements.txt

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Add CPU torch line**

Open `backend/requirements.txt`. Find the comment block:
```
# torch + torchvision: install separately for your CUDA version
# CPU fallback (included via ultralytics auto-install):
# torch>=2.0.0  torchvision>=0.15.0
```

Replace those three commented lines with:
```
# torch CPU build — for GPU install: pip install torch --index-url https://download.pytorch.org/whl/cu121
torch>=2.0.0
torchvision>=0.15.0
```

- [ ] **Step 2: Install in the venv**

```bash
cd backend
source .venv/bin/activate
pip install torch>=2.0.0 torchvision>=0.15.0
```

Expected: installs successfully (CPU build).

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "fix: add torch and torchvision to requirements.txt"
```

---

## Section 2 — Phase 4: Annotation + Training

---

### Task 7: Create VideoAnnotation model

**Files:**
- Create: `backend/app/models/annotations.py`
- Modify: `backend/app/models/__init__.py`

- [ ] **Step 1: Create the model file**

```python
# backend/app/models/annotations.py
import uuid
from datetime import datetime
from sqlalchemy import String, Float, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base


class VideoAnnotation(Base):
    __tablename__ = "video_annotations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id"), nullable=False
    )
    video_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    tagged_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    match = relationship("Match")
    tagger = relationship("User", foreign_keys=[tagged_by])

    def __repr__(self):
        return f"<VideoAnnotation {self.action_type} @ {self.timestamp}s>"
```

- [ ] **Step 2: Import the model in `__init__.py` so `create_all` picks it up**

Open `backend/app/models/__init__.py`. Add:
```python
from app.models.annotations import VideoAnnotation  # noqa: F401
```

- [ ] **Step 3: Restart backend and confirm table is created**

```bash
cd backend && source .venv/bin/activate && python run.py
# Check logs for: no errors about "video_annotations" table
# Or: psql -U postgres volleyball_analytics -c "\dt" | grep video_annotations
```

- [ ] **Step 4: Commit**

```bash
git add app/models/annotations.py app/models/__init__.py
git commit -m "feat: add VideoAnnotation model for action timestamp tagging"
```

---

### Task 8: Create annotation CRUD router

**Files:**
- Create: `backend/app/routers/annotations.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Create the router**

```python
# backend/app/routers/annotations.py
import uuid
import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.database import get_db
from app.models.annotations import VideoAnnotation
from app.models.match import Match
from app.models.video import Video
from app.models.user import User, UserRole
from app.utils.dependencies import get_current_user, require_coach

router = APIRouter(tags=["Annotations"])

VALID_ACTION_TYPES = {"spike", "serve", "block", "dig", "set", "reception"}


class AnnotationCreate(BaseModel):
    timestamp: float
    action_type: str


@router.get("/matches/{match_id}/annotations")
async def list_annotations(
    match_id: uuid.UUID,
    current_user: User = Depends(require_coach),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(VideoAnnotation)
        .where(VideoAnnotation.match_id == match_id)
        .order_by(VideoAnnotation.timestamp)
    )
    annotations = result.scalars().all()
    return [
        {
            "id": str(a.id),
            "timestamp": a.timestamp,
            "action_type": a.action_type,
            "tagged_by": str(a.tagged_by) if a.tagged_by else None,
            "created_at": a.created_at.isoformat(),
        }
        for a in annotations
    ]


@router.post("/matches/{match_id}/annotations", status_code=201)
async def create_annotation(
    match_id: uuid.UUID,
    body: AnnotationCreate,
    current_user: User = Depends(require_coach),
    db: AsyncSession = Depends(get_db),
):
    if body.action_type not in VALID_ACTION_TYPES:
        raise HTTPException(400, f"Invalid action_type. Must be one of: {VALID_ACTION_TYPES}")

    match_result = await db.execute(select(Match).where(Match.id == match_id))
    match = match_result.scalar_one_or_none()
    if not match:
        raise HTTPException(404, "Match not found")

    vid_result = await db.execute(select(Video).where(Video.id == match.video_id))
    video = vid_result.scalar_one_or_none()
    video_path = video.file_path if video else ""

    annotation = VideoAnnotation(
        match_id=match_id,
        video_path=video_path,
        timestamp=body.timestamp,
        action_type=body.action_type,
        tagged_by=current_user.id,
    )
    db.add(annotation)
    await db.commit()
    await db.refresh(annotation)
    return {"id": str(annotation.id), "timestamp": annotation.timestamp, "action_type": annotation.action_type}


@router.delete("/annotations/{annotation_id}", status_code=204)
async def delete_annotation(
    annotation_id: uuid.UUID,
    current_user: User = Depends(require_coach),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(VideoAnnotation).where(VideoAnnotation.id == annotation_id)
    )
    annotation = result.scalar_one_or_none()
    if not annotation:
        raise HTTPException(404, "Annotation not found")
    await db.delete(annotation)
    await db.commit()


@router.get("/annotations/export")
async def export_annotations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export all annotations in training JSON format (admin only)."""
    if current_user.role != UserRole.admin:
        raise HTTPException(403, "Admin only")

    result = await db.execute(
        select(VideoAnnotation).order_by(VideoAnnotation.video_path, VideoAnnotation.timestamp)
    )
    annotations = result.scalars().all()

    # Group by video_path, matching the existing annotations.json format:
    # { "video_path": "...", "actions": [{"timestamp": 1.2, "action": "spike"}, ...] }
    from collections import defaultdict
    grouped = defaultdict(list)
    for a in annotations:
        grouped[a.video_path].append({"timestamp": a.timestamp, "action": a.action_type})

    export = [{"video_path": vp, "actions": acts} for vp, acts in grouped.items()]

    return JSONResponse(
        content=export,
        headers={"Content-Disposition": 'attachment; filename="annotations.json"'},
    )
```

- [ ] **Step 2: Register the router in `main.py`**

In `backend/app/main.py`, add to the imports:
```python
from app.routers import annotations as annotations_router
```

And after the existing `app.include_router(processing.router, prefix="/api")` line add:
```python
app.include_router(annotations_router.router, prefix="/api")
```

- [ ] **Step 3: Test the endpoints**

```bash
# Restart backend first
cd backend && source .venv/bin/activate && python run.py

# In another terminal (get a coach/admin token first):
TOKEN="<paste JWT here>"

# List (should be empty)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8001/api/matches/<match_id>/annotations
# Expected: []

# Create
curl -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"timestamp": 12.5, "action_type": "spike"}' \
  http://localhost:8001/api/matches/<match_id>/annotations
# Expected: {"id":"...","timestamp":12.5,"action_type":"spike"}
```

- [ ] **Step 4: Commit**

```bash
git add app/routers/annotations.py app/main.py
git commit -m "feat: add annotation CRUD endpoints for action timestamp tagging"
```

---

### Task 9: Build AnnotatePage frontend

**Files:**
- Create: `frontend/src/pages/AnnotatePage.jsx`
- Modify: `frontend/src/services/api.js`
- Modify: `frontend/src/App.jsx`

- [ ] **Step 1: Add annotation API calls to `api.js`**

In `frontend/src/services/api.js`, add after the `matchesAPI` block:

```js
// ─── Annotations ─────────────────────────────────────────────
export const annotationsAPI = {
  list:   (matchId)         => api.get(`/matches/${matchId}/annotations`),
  create: (matchId, data)   => api.post(`/matches/${matchId}/annotations`, data),
  delete: (id)              => api.delete(`/annotations/${id}`),
  export: ()                => api.get('/annotations/export', { responseType: 'blob' }),
}
```

- [ ] **Step 2: Create AnnotatePage**

```jsx
// frontend/src/pages/AnnotatePage.jsx
import React, { useState, useRef } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { ChevronLeft, Plus, Trash2, Download, Tag } from 'lucide-react'
import toast from 'react-hot-toast'
import { matchesAPI, annotationsAPI } from '../services/api'
import useAuthStore from '../store/authStore'

const ACTION_TYPES = ['spike', 'serve', 'block', 'dig', 'set', 'reception']
const ACTION_COLORS = {
  spike: 'bg-orange-500/20 text-orange-400 border-orange-600',
  serve: 'bg-blue-500/20 text-blue-400 border-blue-600',
  block: 'bg-green-500/20 text-green-400 border-green-600',
  dig: 'bg-cyan-500/20 text-cyan-400 border-cyan-600',
  set: 'bg-yellow-500/20 text-yellow-400 border-yellow-600',
  reception: 'bg-purple-500/20 text-purple-400 border-purple-600',
}

function fmtTime(secs) {
  const m = Math.floor(secs / 60)
  const s = secs.toFixed(2).padStart(5, '0')
  return `${m}:${s}`
}

export default function AnnotatePage() {
  const { id: matchId } = useParams()
  const { isAdmin }     = useAuthStore()
  const videoRef        = useRef(null)
  const qc              = useQueryClient()
  const [selectedType, setSelectedType] = useState('spike')

  const { data: match } = useQuery({
    queryKey: ['match', matchId],
    queryFn: () => matchesAPI.get(matchId).then(r => r.data),
  })

  const { data: annotations = [] } = useQuery({
    queryKey: ['annotations', matchId],
    queryFn: () => annotationsAPI.list(matchId).then(r => r.data),
  })

  const token = localStorage.getItem('access_token')
  const streamUrl = `/api/videos/${match?.video_id}/stream${token ? `?token=${encodeURIComponent(token)}` : ''}`

  const createMutation = useMutation({
    mutationFn: (data) => annotationsAPI.create(matchId, data),
    onSuccess: () => {
      qc.invalidateQueries(['annotations', matchId])
      toast.success('Tag added')
    },
    onError: () => toast.error('Failed to add tag'),
  })

  const deleteMutation = useMutation({
    mutationFn: (id) => annotationsAPI.delete(id),
    onSuccess: () => {
      qc.invalidateQueries(['annotations', matchId])
      toast.success('Tag removed')
    },
  })

  function handleTagCurrent() {
    if (!videoRef.current) return
    const ts = videoRef.current.currentTime
    createMutation.mutate({ timestamp: ts, action_type: selectedType })
  }

  async function handleExport() {
    const res = await annotationsAPI.export()
    const url = URL.createObjectURL(res.data)
    const a = document.createElement('a')
    a.href = url
    a.download = 'annotations.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Link to={`/matches/${matchId}`} className="text-slate-400 hover:text-white">
          <ChevronLeft className="w-5 h-5" />
        </Link>
        <div>
          <h2 className="text-xl font-bold text-white">Annotate Match</h2>
          <p className="text-slate-400 text-sm">{match?.title}</p>
        </div>
        {isAdmin() && (
          <button
            onClick={handleExport}
            className="ml-auto flex items-center gap-2 px-3 py-1.5 text-sm bg-[#2e3a52] hover:bg-[#3e4a62] text-slate-300 rounded-lg"
          >
            <Download className="w-4 h-4" />
            Export JSON
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Video + tagging */}
        <div className="lg:col-span-2 space-y-3">
          <div className="bg-black rounded-xl overflow-hidden aspect-video">
            {match?.video_id && (
              <video
                ref={videoRef}
                src={streamUrl}
                controls
                className="w-full h-full"
              />
            )}
          </div>

          {/* Tag controls */}
          <div className="card flex items-center gap-3 flex-wrap">
            <Tag className="w-4 h-4 text-slate-400 flex-shrink-0" />
            <span className="text-sm text-slate-400">Tag current position as:</span>
            <div className="flex gap-2 flex-wrap">
              {ACTION_TYPES.map(t => (
                <button
                  key={t}
                  onClick={() => setSelectedType(t)}
                  className={`px-3 py-1 text-xs rounded-full border capitalize font-medium transition-all ${
                    selectedType === t
                      ? ACTION_COLORS[t]
                      : 'bg-transparent text-slate-500 border-slate-700 hover:border-slate-500'
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
            <button
              onClick={handleTagCurrent}
              disabled={createMutation.isPending}
              className="ml-auto flex items-center gap-1.5 px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg disabled:opacity-50"
            >
              <Plus className="w-4 h-4" />
              Add Tag
            </button>
          </div>
        </div>

        {/* Tag list */}
        <div className="card overflow-hidden">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-white">Tags ({annotations.length})</h3>
          </div>
          <div className="space-y-1 max-h-[500px] overflow-y-auto">
            {annotations.length === 0 && (
              <p className="text-slate-500 text-sm text-center py-8">No tags yet. Play the video and tag actions.</p>
            )}
            {annotations.map(a => (
              <div key={a.id} className="flex items-center gap-2 py-1.5 px-2 rounded hover:bg-[#1a1f2e] group">
                <button
                  onClick={() => {
                    if (videoRef.current) videoRef.current.currentTime = a.timestamp
                  }}
                  className="flex-1 flex items-center gap-2 text-left"
                >
                  <span className={`px-2 py-0.5 rounded text-xs border capitalize font-medium ${ACTION_COLORS[a.action_type] || 'bg-slate-700 text-slate-300 border-slate-600'}`}>
                    {a.action_type}
                  </span>
                  <span className="text-xs text-slate-400 font-mono">{fmtTime(a.timestamp)}</span>
                </button>
                <button
                  onClick={() => deleteMutation.mutate(a.id)}
                  className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 transition-opacity"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Add route in `App.jsx`**

In `frontend/src/App.jsx`, add the import:
```js
import AnnotatePage from './pages/AnnotatePage'
```

Inside the protected routes (inside `<Route path="/" ...>`), add:
```jsx
<Route path="matches/:id/annotate" element={<RequireAuth roles={['admin','coach']}><AnnotatePage /></RequireAuth>} />
```

- [ ] **Step 4: Add "Annotate" button on MatchDetailPage**

In `frontend/src/pages/MatchDetailPage.jsx`, find the header area where match title is displayed (near the back chevron/link). Import `Link` if not already imported. Add a button linking to the annotate page:

```jsx
<Link
  to={`/matches/${match.id}/annotate`}
  className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-[#2e3a52] hover:bg-[#3e4a62] text-slate-300 rounded-lg"
>
  <Tag className="w-4 h-4" />
  Annotate
</Link>
```

Import `Tag` from `lucide-react` at the top of MatchDetailPage.

- [ ] **Step 5: Verify in browser**

```
1. Open http://localhost:5173
2. Log in as admin/coach
3. Navigate to any match
4. Click "Annotate" button → should open /matches/:id/annotate
5. Play video, click action type buttons, click "Add Tag"
6. Tag should appear in the list on the right
7. Click a tag timestamp → video should seek to that position
8. Click delete → tag removed from list
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/pages/AnnotatePage.jsx frontend/src/services/api.js frontend/src/App.jsx frontend/src/pages/MatchDetailPage.jsx
git commit -m "feat: add in-app annotation page for tagging action timestamps"
```

---

### Task 10: Training trigger router

**Files:**
- Create: `backend/app/routers/training.py`
- Modify: `backend/app/main.py`
- Modify: `frontend/src/services/api.js`

- [ ] **Step 1: Create training router**

```python
# backend/app/routers/training.py
"""
Training trigger router.
Admin-only: runs run_phase3_pipeline.py as a subprocess,
streams stdout/stderr via WebSocket.
"""
import asyncio
import os
import subprocess
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

        # Reload ActionService weights if training succeeded
        if proc.returncode == 0:
            try:
                from app.services.action_service import ActionService
                # Force reload by creating a fresh instance check
                logger.info("Training complete — ActionService will reload weights on next analysis")
            except Exception as e:
                logger.warning(f"Post-training reload notice failed: {e}")

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
                import json
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
```

- [ ] **Step 2: Register training router in `main.py`**

In `backend/app/main.py`, add to imports:
```python
from app.routers import training as training_router
```

After the existing `app.include_router` lines:
```python
app.include_router(training_router.router, prefix="/api")
```

- [ ] **Step 3: Add training API in `api.js`**

In `frontend/src/services/api.js`, add:
```js
// ─── Training ─────────────────────────────────────────────────
export const trainingAPI = {
  status: () => api.get('/training/status'),
  run:    (phase = 2) => api.post('/training/run', { phase }),
}
```

- [ ] **Step 4: Test trigger endpoint**

```bash
TOKEN="<admin JWT>"
curl -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"phase": 2}' http://localhost:8001/api/training/run
# Expected: {"message":"Training phase 2 started"} (or 503 if script not found)

curl -H "Authorization: Bearer $TOKEN" http://localhost:8001/api/training/status
# Expected: {"running": true/false, "last_log": "...", "last_exit_code": null/0/-1}
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/routers/training.py backend/app/main.py frontend/src/services/api.js
git commit -m "feat: add training trigger API and WebSocket status stream"
```

---

## Section 3 — Phase 5: Advanced Analytics

---

### Task 11: Player comparison — backend endpoint

**Files:**
- Modify: `backend/app/routers/analytics.py`

- [ ] **Step 1: Add `/analytics/players/compare` endpoint**

In `backend/app/routers/analytics.py`, add after the existing `team_comparison` endpoint:

```python
@router.get("/players/compare")
async def player_comparison(
    player_a: uuid.UUID,
    player_b: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Head-to-head stat comparison for two players across all their matches."""
    from app.models.player import Player as PlayerModel

    async def fetch_player(pid: uuid.UUID):
        p_result = await db.execute(
            select(PlayerModel).where(PlayerModel.id == pid)
        )
        player = p_result.scalar_one_or_none()
        if not player:
            raise HTTPException(404, f"Player {pid} not found")

        stats_result = await db.execute(
            select(
                func.coalesce(func.sum(Analytics.total_attacks), 0).label("attacks"),
                func.coalesce(func.sum(Analytics.attack_kills), 0).label("kills"),
                func.coalesce(func.avg(Analytics.attack_efficiency), 0.0).label("attack_eff"),
                func.coalesce(func.sum(Analytics.total_serves), 0).label("serves"),
                func.coalesce(func.sum(Analytics.aces), 0).label("aces"),
                func.coalesce(func.avg(Analytics.serve_efficiency), 0.0).label("serve_eff"),
                func.coalesce(func.sum(Analytics.total_blocks), 0).label("blocks"),
                func.coalesce(func.sum(Analytics.block_points), 0).label("block_pts"),
                func.coalesce(func.sum(Analytics.total_digs), 0).label("digs"),
                func.coalesce(func.sum(Analytics.total_receptions), 0).label("receptions"),
                func.coalesce(func.avg(Analytics.reception_efficiency), 0.0).label("reception_eff"),
            ).where(Analytics.player_id == pid)
        )
        row = stats_result.one()
        return {
            "id": str(player.id),
            "name": player.display_name or f"Player #{player.player_track_id}",
            "team": player.team or "?",
            "attacks": int(row.attacks),
            "kills": int(row.kills),
            "attack_eff": round(float(row.attack_eff) * 100, 1),
            "serves": int(row.serves),
            "aces": int(row.aces),
            "serve_eff": round(float(row.serve_eff) * 100, 1),
            "blocks": int(row.blocks),
            "block_pts": int(row.block_pts),
            "digs": int(row.digs),
            "receptions": int(row.receptions),
            "reception_eff": round(float(row.reception_eff) * 100, 1),
        }

    pa, pb = await fetch_player(player_a), await fetch_player(player_b)
    return {"player_a": pa, "player_b": pb}
```

Also add the new API call to `api.js`:
```js
// In analyticsAPI object, add:
playerCompare: (a, b) => api.get('/analytics/players/compare', { params: { player_a: a, player_b: b } }),
playersList: () => api.get('/analytics/players'),
```

And add a players list endpoint in `analytics.py`:
```python
@router.get("/players")
async def list_all_players(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all players with analytics data for the compare dropdowns."""
    from app.models.player import Player as PlayerModel

    q = (
        select(PlayerModel.id, PlayerModel.display_name, PlayerModel.player_track_id, PlayerModel.team, Match.title)
        .join(Analytics, Analytics.player_id == PlayerModel.id)
        .join(Match, Match.id == PlayerModel.match_id)
        .where(Match.status == MatchStatus.completed)
        .distinct()
    )
    if current_user.role != UserRole.admin:
        q = q.where(Match.uploaded_by == current_user.id)

    result = await db.execute(q)
    rows = result.all()
    return [
        {
            "id": str(r.id),
            "name": r.display_name or f"Player #{r.player_track_id}",
            "team": r.team or "?",
            "match_title": r.title,
        }
        for r in rows
    ]
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/routers/analytics.py frontend/src/services/api.js
git commit -m "feat: add player-to-player comparison and players list endpoints"
```

---

### Task 12: PlayerComparePage frontend

**Files:**
- Create: `frontend/src/pages/PlayerComparePage.jsx`
- Modify: `frontend/src/App.jsx`

- [ ] **Step 1: Create PlayerComparePage**

```jsx
// frontend/src/pages/PlayerComparePage.jsx
import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip, Legend,
} from 'recharts'
import { Users, ChevronUp, ChevronDown, Minus, Loader2 } from 'lucide-react'
import { analyticsAPI } from '../services/api'

const CHART_STYLE = {
  contentStyle: { background: '#232b3e', border: '1px solid #2e3a52', borderRadius: 8, color: '#f1f5f9' },
}

const STAT_ROWS = [
  { key: 'attacks',      label: 'Attacks'          },
  { key: 'kills',        label: 'Kills'             },
  { key: 'attack_eff',   label: 'Attack Eff %'      },
  { key: 'serves',       label: 'Serves'            },
  { key: 'aces',         label: 'Aces'              },
  { key: 'serve_eff',    label: 'Serve Eff %'       },
  { key: 'blocks',       label: 'Blocks'            },
  { key: 'block_pts',    label: 'Block Points'      },
  { key: 'digs',         label: 'Digs'              },
  { key: 'receptions',   label: 'Receptions'        },
  { key: 'reception_eff',label: 'Reception Eff %'   },
]

function DeltaBadge({ a, b }) {
  if (a === b) return <Minus className="w-3 h-3 text-slate-500 mx-auto" />
  if (a > b) return <ChevronUp className="w-3 h-3 text-green-400 mx-auto" />
  return <ChevronDown className="w-3 h-3 text-red-400 mx-auto" />
}

export default function PlayerComparePage() {
  const [playerAId, setPlayerAId] = useState('')
  const [playerBId, setPlayerBId] = useState('')

  const { data: players = [], isLoading: loadingPlayers } = useQuery({
    queryKey: ['players-list'],
    queryFn: () => analyticsAPI.playersList().then(r => r.data),
  })

  const enabled = !!(playerAId && playerBId && playerAId !== playerBId)

  const { data: comparison, isLoading: loadingCompare } = useQuery({
    queryKey: ['player-compare', playerAId, playerBId],
    queryFn: () => analyticsAPI.playerCompare(playerAId, playerBId).then(r => r.data),
    enabled,
  })

  const pa = comparison?.player_a
  const pb = comparison?.player_b

  const radarData = pa && pb ? [
    { stat: 'Attacking',  a: Math.min(Math.abs(pa.attack_eff), 100),     b: Math.min(Math.abs(pb.attack_eff), 100) },
    { stat: 'Serving',    a: Math.min(Math.abs(pa.serve_eff), 100),      b: Math.min(Math.abs(pb.serve_eff), 100) },
    { stat: 'Blocking',   a: Math.min((pa.blocks / Math.max(pa.blocks + pb.blocks, 1)) * 100, 100),
                          b: Math.min((pb.blocks / Math.max(pa.blocks + pb.blocks, 1)) * 100, 100) },
    { stat: 'Digging',    a: Math.min((pa.digs / Math.max(pa.digs + pb.digs, 1)) * 100, 100),
                          b: Math.min((pb.digs / Math.max(pa.digs + pb.digs, 1)) * 100, 100) },
    { stat: 'Reception',  a: Math.min(Math.abs(pa.reception_eff), 100),  b: Math.min(Math.abs(pb.reception_eff), 100) },
  ] : []

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h2 className="text-2xl font-bold text-white">Player Comparison</h2>
        <p className="text-slate-400 text-sm mt-1">Select two players to compare their stats</p>
      </div>

      {/* Player selectors */}
      <div className="card grid grid-cols-1 md:grid-cols-2 gap-4">
        {[
          { label: 'Player A', value: playerAId, set: setPlayerAId, color: 'text-blue-400' },
          { label: 'Player B', value: playerBId, set: setPlayerBId, color: 'text-orange-400' },
        ].map(({ label, value, set, color }) => (
          <div key={label}>
            <label className={`text-xs font-semibold ${color} mb-1 block`}>{label}</label>
            <select
              value={value}
              onChange={e => set(e.target.value)}
              className="w-full bg-[#1a1f2e] text-slate-300 border border-[#2e3a52] rounded-lg px-3 py-2 text-sm outline-none focus:border-blue-500"
            >
              <option value="">— Select player —</option>
              {players.map(p => (
                <option key={p.id} value={p.id}>
                  {p.name} ({p.team}) — {p.match_title}
                </option>
              ))}
            </select>
          </div>
        ))}
      </div>

      {loadingCompare && (
        <div className="flex items-center justify-center h-32">
          <Loader2 className="w-6 h-6 animate-spin text-blue-500" />
        </div>
      )}

      {pa && pb && !loadingCompare && (
        <>
          {/* Player name headers */}
          <div className="grid grid-cols-2 gap-4">
            <div className="card text-center">
              <Users className="w-8 h-8 text-blue-400 mx-auto mb-1" />
              <div className="text-white font-bold">{pa.name}</div>
              <div className="text-xs text-slate-400">Team {pa.team}</div>
            </div>
            <div className="card text-center">
              <Users className="w-8 h-8 text-orange-400 mx-auto mb-1" />
              <div className="text-white font-bold">{pb.name}</div>
              <div className="text-xs text-slate-400">Team {pb.team}</div>
            </div>
          </div>

          {/* Radar chart */}
          <div className="card">
            <h3 className="text-sm font-semibold text-white mb-3">Skill Comparison</h3>
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#2e3a52" />
                <PolarAngleAxis dataKey="stat" tick={{ fill: '#64748b', fontSize: 11 }} />
                <PolarRadiusAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 9 }} />
                <Radar name={pa.name} dataKey="a" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} />
                <Radar name={pb.name} dataKey="b" stroke="#f97316" fill="#f97316" fillOpacity={0.25} />
                <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
                <Tooltip contentStyle={CHART_STYLE.contentStyle} formatter={v => `${v.toFixed(1)}`} />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Stat diff table */}
          <div className="card overflow-x-auto">
            <h3 className="text-sm font-semibold text-white mb-3">Stat Breakdown</h3>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-slate-500 border-b border-[#2e3a52] text-xs">
                  <th className="text-right py-2 pr-4 text-blue-400">{pa.name}</th>
                  <th className="text-center py-2 px-3">Stat</th>
                  <th className="text-center py-2 px-2">Δ</th>
                  <th className="text-left py-2 pl-4 text-orange-400">{pb.name}</th>
                </tr>
              </thead>
              <tbody>
                {STAT_ROWS.map(({ key, label }) => (
                  <tr key={key} className="border-b border-[#2e3a52]/50 last:border-0">
                    <td className="py-2 pr-4 text-right text-blue-300 font-mono">{pa[key]}</td>
                    <td className="py-2 px-3 text-center text-slate-400 text-xs">{label}</td>
                    <td className="py-2 px-2 text-center">
                      <DeltaBadge a={pa[key]} b={pb[key]} />
                    </td>
                    <td className="py-2 pl-4 text-left text-orange-300 font-mono">{pb[key]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {!enabled && !loadingCompare && (
        <div className="card text-center py-12 text-slate-500">
          Select two different players above to compare their stats.
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Add route in `App.jsx`**

Import and register:
```js
import PlayerComparePage from './pages/PlayerComparePage'
```
```jsx
<Route path="analytics/compare" element={<PlayerComparePage />} />
```

- [ ] **Step 3: Add a "Compare Players" link on AnalyticsPage**

In `frontend/src/pages/AnalyticsPage.jsx`, add this button at the top of the page header (next to the "Analytics" h2):

```jsx
import { Link } from 'react-router-dom'
// ...
<div className="flex items-center justify-between">
  <div>
    <h2 className="text-2xl font-bold text-white">Analytics</h2>
    <p className="text-slate-400 text-sm mt-1">Performance insights across all matches</p>
  </div>
  <Link
    to="/analytics/compare"
    className="flex items-center gap-2 px-3 py-1.5 text-sm bg-[#2e3a52] hover:bg-[#3e4a62] text-slate-300 rounded-lg"
  >
    <Users className="w-4 h-4" />
    Compare Players
  </Link>
</div>
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/PlayerComparePage.jsx frontend/src/App.jsx frontend/src/pages/AnalyticsPage.jsx
git commit -m "feat: add player comparison page with radar chart and stat diff table"
```

---

### Task 13: Rotation model + detection service

**Files:**
- Create: `backend/app/models/rotations.py`
- Create: `backend/app/services/rotation_detector.py`
- Modify: `backend/app/models/__init__.py`

- [ ] **Step 1: Create the Rotation model**

```python
# backend/app/models/rotations.py
import uuid
from datetime import datetime
from sqlalchemy import String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.database import Base


class Rotation(Base):
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
    team: Mapped[str] = mapped_column(String(10), nullable=True)
    rotation_number: Mapped[int] = mapped_column(Integer, nullable=True)  # inferred 1-6
    player_positions: Mapped[dict] = mapped_column(JSONB, nullable=True)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

- [ ] **Step 2: Create the rotation detection service**

```python
# backend/app/services/rotation_detector.py
"""
Rotation Detector
─────────────────
Given a list of player detections (court_x, court_y, team) for a single frame,
assigns each player to one of 6 rotation slots based on a 2x3 court grid:

  Court layout (normalised 0-1):
    Top half (court_y < 0.5):  slots 1, 2, 3  (back row)  — left to right
    Bottom half (court_y ≥ 0.5): slots 4, 5, 6 (front row) — left to right

  Slot mapping:
    court_x < 0.33 → column 0 (left)
    court_x < 0.67 → column 1 (centre)
    else           → column 2 (right)

    Back row:  left=1, centre=2, right=3
    Front row: left=4, centre=5, right=6
"""

from typing import List, Dict, Any, Optional


SLOT_MAP = {
    (0, 0): 1, (0, 1): 2, (0, 2): 3,   # back row
    (1, 0): 4, (1, 1): 5, (1, 2): 6,   # front row
}


def assign_slot(court_x: float, court_y: float) -> int:
    """Return rotation slot 1-6 for given normalised court coordinates."""
    row = 0 if court_y < 0.5 else 1
    col = 0 if court_x < 0.33 else (1 if court_x < 0.67 else 2)
    return SLOT_MAP.get((row, col), 0)


def detect_rotation(
    players: List[Dict[str, Any]],
    team: str,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    players : list of player dicts with keys track_id, court_x, court_y, team
    team    : "A" or "B" — which team's rotation to detect

    Returns
    -------
    dict with rotation_number (most common slot of first player sorted by court_y)
    and player_positions mapping track_id → {court_x, court_y, slot}
    """
    team_players = [
        p for p in players
        if p.get("team") == team
        and p.get("court_x") is not None
        and p.get("court_y") is not None
    ]

    if not team_players:
        return {"rotation_number": None, "player_positions": {}}

    positions = {}
    for p in team_players:
        slot = assign_slot(p["court_x"], p["court_y"])
        positions[str(p["track_id"])] = {
            "court_x": p["court_x"],
            "court_y": p["court_y"],
            "slot": slot,
        }

    # Infer rotation number as the slot of the server (player closest to back-right: court_y < 0.2, court_x > 0.67)
    # Fall back to the slot of the player with lowest court_y (deepest back row)
    server = min(team_players, key=lambda p: p["court_y"])
    rotation_number = assign_slot(server["court_x"], server["court_y"])

    return {
        "rotation_number": rotation_number,
        "player_positions": positions,
    }
```

- [ ] **Step 3: Register new model in `__init__.py`**

In `backend/app/models/__init__.py`, add:
```python
from app.models.rotations import Rotation  # noqa: F401
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/models/rotations.py backend/app/services/rotation_detector.py backend/app/models/__init__.py
git commit -m "feat: add Rotation model and rotation slot detection service"
```

---

### Task 14: Integrate rotation detection into CV pipeline

**Files:**
- Modify: `backend/app/services/cv_pipeline.py`

- [ ] **Step 1: Import and call `detect_rotation` in the frame processing loop**

In `backend/app/services/cv_pipeline.py`, add to imports at top of file:
```python
from app.services.rotation_detector import detect_rotation
```

In the `__init__` method, add a new collector:
```python
self._rotation_rows: List[Dict] = []
```

In the frame loop (inside `if frame_idx % PROCESS_EVERY_N == 0:`), after the `completed = self.rally_detector.update(...)` line, add:

```python
                # Rotation snapshot — record once per detected rally start
                if completed and players:
                    for team in ("A", "B"):
                        rot = detect_rotation(players, team)
                        if rot["rotation_number"] is not None:
                            self._rotation_rows.append({
                                "match_id": self.match_id,
                                "rally_number": completed.rally_number,
                                "team": team,
                                "rotation_number": rot["rotation_number"],
                                "player_positions": rot["player_positions"],
                                "frame_number": frame_idx,
                            })
```

- [ ] **Step 2: Persist rotation rows in `_save_to_db`**

In `cv_pipeline._save_to_db`, after the rallies section and before `await db.commit()`, add:

```python
            # ── Rotations ───────────────────────────────────────────────────
            from app.models.rotations import Rotation
            # Build rally_number → rally UUID map
            rally_num_to_id = {}
            rally_q = await db.execute(
                select(Rally).where(Rally.match_id == uuid.UUID(self.match_id))
            )
            for rally_obj in rally_q.scalars().all():
                rally_num_to_id[rally_obj.rally_number] = rally_obj.id

            for row in getattr(self, '_rotation_rows', []):
                db.add(Rotation(
                    match_id=uuid.UUID(self.match_id),
                    rally_id=rally_num_to_id.get(row["rally_number"]),
                    team=row["team"],
                    rotation_number=row["rotation_number"],
                    player_positions=row["player_positions"],
                    frame_number=row["frame_number"],
                ))
```

Also add `from app.models.actions import Rally` to the imports inside `_save_to_db` if `Rally` is not already imported there (it is, check line ~277).

- [ ] **Step 3: Add the rotation API endpoint to `processing.py`**

In `backend/app/routers/processing.py`, add after the existing tracking endpoint:

```python
@router.get("/{match_id}/rotations")
async def get_rotations(
    match_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all rotation snapshots for a match, grouped by rally."""
    from app.models.rotations import Rotation

    result = await db.execute(
        select(Rotation)
        .where(Rotation.match_id == match_id)
        .order_by(Rotation.frame_number)
    )
    rotations = result.scalars().all()
    grouped: dict = {}
    for r in rotations:
        key = str(r.rally_id or "")
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r.team] = {
            "rotation_number": r.rotation_number,
            "player_positions": r.player_positions or {},
        }
    return {"match_id": str(match_id), "rotations_by_rally": grouped}
```

Add this to `api.js` in `matchesAPI`:
```js
rotations: (id) => api.get(`/matches/${id}/rotations`),
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/cv_pipeline.py backend/app/routers/processing.py frontend/src/services/api.js
git commit -m "feat: detect and store rotation slots per rally; add rotations endpoint"
```

---

### Task 15: RotationPanel component + MatchDetailPage integration

**Files:**
- Create: `frontend/src/components/Video/RotationPanel.jsx`
- Modify: `frontend/src/pages/MatchDetailPage.jsx`

- [ ] **Step 1: Create RotationPanel**

```jsx
// frontend/src/components/Video/RotationPanel.jsx
import React from 'react'

const SLOT_POSITIONS = {
  // slot → {x%, y%} center in the court SVG (percentage)
  1: { x: 16, y: 25 }, 2: { x: 50, y: 25 }, 3: { x: 84, y: 25 },
  4: { x: 16, y: 75 }, 5: { x: 50, y: 75 }, 6: { x: 84, y: 75 },
}

const TEAM_COLORS = { A: '#3b82f6', B: '#f97316' }

export default function RotationPanel({ rotationData }) {
  if (!rotationData) return null

  const teamA = rotationData['A'] || {}
  const teamB = rotationData['B'] || {}

  function renderPlayers(teamData, team) {
    if (!teamData?.player_positions) return null
    return Object.entries(teamData.player_positions).map(([trackId, pos]) => {
      const slot = pos.slot
      const coords = SLOT_POSITIONS[slot]
      if (!coords) return null
      return (
        <g key={`${team}-${trackId}`}>
          <circle
            cx={`${coords.x}%`} cy={`${coords.y}%`}
            r="8" fill={TEAM_COLORS[team]} fillOpacity={0.85}
          />
          <text
            x={`${coords.x}%`} y={`${coords.y}%`}
            textAnchor="middle" dominantBaseline="central"
            fontSize="9" fill="white" fontWeight="bold"
          >
            {trackId}
          </text>
        </g>
      )
    })
  }

  return (
    <div className="card">
      <div className="text-xs font-semibold text-slate-400 mb-2">Rotation Snapshot</div>
      <svg viewBox="0 0 200 120" className="w-full rounded overflow-hidden bg-[#1a3a1a]">
        {/* Court outline */}
        <rect x="10" y="10" width="180" height="100" fill="none" stroke="#2d5a2d" strokeWidth="1.5" rx="2" />
        {/* Net */}
        <line x1="100" y1="10" x2="100" y2="110" stroke="#4ade80" strokeWidth="1" strokeDasharray="3,2" />
        {/* Slot dividers */}
        <line x1="10" y1="60" x2="190" y2="60" stroke="#2d5a2d" strokeWidth="0.5" strokeDasharray="2,2" />
        <line x1="70" y1="10" x2="70" y2="110" stroke="#2d5a2d" strokeWidth="0.5" strokeDasharray="2,2" />
        <line x1="130" y1="10" x2="130" y2="110" stroke="#2d5a2d" strokeWidth="0.5" strokeDasharray="2,2" />
        {/* Players */}
        {renderPlayers(teamA, 'A')}
        {renderPlayers(teamB, 'B')}
        {/* Legend */}
        <circle cx="15" cy="118" r="4" fill="#3b82f6" />
        <text x="22" y="118" fontSize="7" fill="#94a3b8" dominantBaseline="central">Team A</text>
        <circle cx="60" cy="118" r="4" fill="#f97316" />
        <text x="67" y="118" fontSize="7" fill="#94a3b8" dominantBaseline="central">Team B</text>
      </svg>
      <div className="flex gap-4 mt-2 text-xs text-slate-400">
        {teamA.rotation_number && <span className="text-blue-400">A: Rotation {teamA.rotation_number}</span>}
        {teamB.rotation_number && <span className="text-orange-400">B: Rotation {teamB.rotation_number}</span>}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Add RotationPanel to MatchDetailPage Rallies tab**

In `frontend/src/pages/MatchDetailPage.jsx`:

1. Import:
```js
import RotationPanel from '../components/Video/RotationPanel'
```

2. Add a query to fetch rotations (near the other `useQuery` calls):
```js
const { data: rotationsData } = useQuery({
  queryKey: ['rotations', match?.id],
  queryFn: () => matchesAPI.rotations(match.id).then(r => r.data),
  enabled: match?.status === 'completed',
})
```

3. In the Rallies tab, find where individual rallies are rendered. For each rally, get its rotation data:
```jsx
// Inside the rally list render, for each rally item:
const rallyRotation = rotationsData?.rotations_by_rally?.[rally.id]

// Add below the rally card:
{rallyRotation && <RotationPanel rotationData={rallyRotation} />}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/Video/RotationPanel.jsx frontend/src/pages/MatchDetailPage.jsx
git commit -m "feat: add rotation panel showing player positions per rally"
```

---

### Task 16: Ball heatmap endpoint

**Files:**
- Modify: `backend/app/routers/processing.py`
- Modify: `frontend/src/services/api.js`

- [ ] **Step 1: Add the heatmap endpoint to `processing.py`**

In `backend/app/routers/processing.py`, add after the rotations endpoint:

```python
@router.get("/{match_id}/tracking/ball-heatmap")
async def ball_heatmap(
    match_id: uuid.UUID,
    max_points: int = Query(2000, ge=100, le=10000),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns downsampled ball court positions for heat map rendering.
    Only includes frames where court_x and court_y are non-null.
    """
    result = await db.execute(
        select(BallTracking.court_x, BallTracking.court_y)
        .where(
            BallTracking.match_id == match_id,
            BallTracking.court_x.isnot(None),
            BallTracking.court_y.isnot(None),
        )
        .order_by(BallTracking.frame_number)
    )
    rows = result.all()

    # Downsample if needed
    if len(rows) > max_points:
        step = len(rows) // max_points
        rows = rows[::step]

    return {
        "match_id": str(match_id),
        "count": len(rows),
        "points": [{"x": r.court_x, "y": r.court_y} for r in rows],
    }
```

- [ ] **Step 2: Add to `api.js`**

In `matchesAPI` in `frontend/src/services/api.js`, add:
```js
ballHeatmap: (id) => api.get(`/matches/${id}/tracking/ball-heatmap`),
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/routers/processing.py frontend/src/services/api.js
git commit -m "feat: add ball position heatmap endpoint"
```

---

### Task 17: MatchSummaryTab component + MatchDetailPage 5th tab

**Files:**
- Create: `frontend/src/components/Video/MatchSummaryTab.jsx`
- Modify: `frontend/src/pages/MatchDetailPage.jsx`

- [ ] **Step 1: Create MatchSummaryTab**

```jsx
// frontend/src/components/Video/MatchSummaryTab.jsx
import React, { useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer,
} from 'recharts'
import { Flame, BarChart3, Star, Loader2 } from 'lucide-react'
import { matchesAPI } from '../../services/api'

const CHART_STYLE = {
  contentStyle: { background: '#232b3e', border: '1px solid #2e3a52', borderRadius: 8, color: '#f1f5f9' },
}

// Draw a 2D density heat map on a canvas using the ball court positions
function HeatmapCanvas({ points }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!canvasRef.current || !points?.length) return
    const canvas = canvasRef.current
    const ctx    = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height

    ctx.clearRect(0, 0, W, H)

    // Draw court background
    ctx.fillStyle = '#1a3a1a'
    ctx.fillRect(0, 0, W, H)

    // Draw net line
    ctx.strokeStyle = '#4ade80'
    ctx.lineWidth = 1.5
    ctx.setLineDash([4, 3])
    ctx.beginPath()
    ctx.moveTo(W / 2, 0)
    ctx.lineTo(W / 2, H)
    ctx.stroke()
    ctx.setLineDash([])

    // Build density grid
    const GRID = 40
    const grid = Array.from({ length: GRID }, () => new Array(GRID).fill(0))
    for (const p of points) {
      const gx = Math.min(Math.floor(p.x * GRID), GRID - 1)
      const gy = Math.min(Math.floor(p.y * GRID), GRID - 1)
      grid[gy][gx]++
    }

    const maxVal = Math.max(...grid.flat(), 1)
    const cellW = W / GRID
    const cellH = H / GRID

    for (let gy = 0; gy < GRID; gy++) {
      for (let gx = 0; gx < GRID; gx++) {
        const val = grid[gy][gx]
        if (val === 0) continue
        const intensity = val / maxVal
        // Blue → yellow → red color scale
        const r = Math.round(intensity > 0.5 ? 255 : intensity * 2 * 255)
        const g = Math.round(intensity < 0.5 ? intensity * 2 * 255 : (1 - intensity) * 2 * 255)
        const b = Math.round(intensity < 0.5 ? 255 - intensity * 2 * 255 : 0)
        ctx.fillStyle = `rgba(${r},${g},${b},${0.3 + intensity * 0.6})`
        ctx.fillRect(gx * cellW, gy * cellH, cellW, cellH)
      }
    }

    // Draw court outline
    ctx.strokeStyle = '#2d5a2d'
    ctx.lineWidth = 2
    ctx.setLineDash([])
    ctx.strokeRect(2, 2, W - 4, H - 4)
  }, [points])

  return <canvas ref={canvasRef} width={400} height={240} className="w-full rounded-lg" />
}

function fmtTime(secs) {
  const m = Math.floor(secs / 60)
  const s = Math.floor(secs % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

export default function MatchSummaryTab({ match, rallies, actionsData }) {
  const { data: heatmapData, isLoading: heatLoading } = useQuery({
    queryKey: ['ball-heatmap', match?.id],
    queryFn: () => matchesAPI.ballHeatmap(match.id).then(r => r.data),
    enabled: match?.status === 'completed',
  })

  const actions = actionsData?.items || []

  // Zone distribution
  const zoneData = [1, 2, 3, 4, 5, 6].map(zone => {
    const aActions = actions.filter(a => a.zone === zone && (!a.team || a.team === 'A'))
    const bActions = actions.filter(a => a.zone === zone && a.team === 'B')
    return { zone: `Zone ${zone}`, teamA: aActions.length, teamB: bActions.length }
  })
  const hasZoneData = zoneData.some(d => d.teamA > 0 || d.teamB > 0)

  // Key moments — top 10 rallies by duration
  const rallyList = rallies || []
  const topRallies = [...rallyList]
    .sort((a, b) => (b.end_time - b.start_time) - (a.end_time - a.start_time))
    .slice(0, 10)

  return (
    <div className="space-y-6">

      {/* Ball Heat Map */}
      <div className="card">
        <div className="flex items-center gap-2 mb-3">
          <Flame className="w-4 h-4 text-orange-400" />
          <h3 className="text-sm font-semibold text-white">Ball Position Heat Map</h3>
          {heatmapData && <span className="text-xs text-slate-500 ml-auto">{heatmapData.count} positions</span>}
        </div>
        {heatLoading && (
          <div className="flex items-center justify-center h-40">
            <Loader2 className="w-6 h-6 animate-spin text-blue-500" />
          </div>
        )}
        {!heatLoading && heatmapData?.points?.length > 0 && (
          <HeatmapCanvas points={heatmapData.points} />
        )}
        {!heatLoading && (!heatmapData?.points?.length) && (
          <div className="text-center py-8 text-slate-500 text-sm">
            No ball tracking data available. Run analysis first.
          </div>
        )}
      </div>

      {/* Attack Zone Distribution */}
      {hasZoneData && (
        <div className="card">
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 className="w-4 h-4 text-blue-400" />
            <h3 className="text-sm font-semibold text-white">Attack Zone Distribution</h3>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={zoneData} barSize={14}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2e3a52" />
              <XAxis dataKey="zone" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={CHART_STYLE.contentStyle} />
              <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
              <Bar dataKey="teamA" name="Team A" fill="#3b82f6" radius={[3, 3, 0, 0]} />
              <Bar dataKey="teamB" name="Team B" fill="#f97316" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Key Moment Timeline */}
      {topRallies.length > 0 && (
        <div className="card">
          <div className="flex items-center gap-2 mb-3">
            <Star className="w-4 h-4 text-yellow-400" />
            <h3 className="text-sm font-semibold text-white">Key Moments (Longest Rallies)</h3>
          </div>
          <div className="space-y-2">
            {topRallies.map((rally) => {
              const duration = (rally.end_time - rally.start_time).toFixed(1)
              const winnerColor = rally.winner_team === 'A' ? 'text-blue-400' : rally.winner_team === 'B' ? 'text-orange-400' : 'text-slate-400'
              return (
                <div
                  key={rally.id}
                  className="flex items-center gap-3 p-3 bg-[#1a1f2e] rounded-lg cursor-pointer hover:bg-[#232b3e] transition-colors"
                  onClick={() => {
                    const video = document.querySelector('video')
                    if (video) video.currentTime = rally.start_time
                  }}
                >
                  <div className="w-7 h-7 rounded-full bg-yellow-500/20 flex items-center justify-center text-xs text-yellow-400 font-bold flex-shrink-0">
                    {rally.rally_number}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-xs text-white">
                      {fmtTime(rally.start_time)} → {fmtTime(rally.end_time)}
                      <span className="text-slate-500 ml-2">{duration}s</span>
                    </div>
                    {rally.point_reason && (
                      <div className="text-xs text-slate-500 truncate">{rally.point_reason}</div>
                    )}
                  </div>
                  {rally.winner_team && (
                    <span className={`text-xs font-bold flex-shrink-0 ${winnerColor}`}>
                      Team {rally.winner_team}
                    </span>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {topRallies.length === 0 && !heatLoading && (
        <div className="card text-center py-12 text-slate-500 text-sm">
          No summary data yet. Complete a match analysis to see heat maps and key moments.
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Add Summary tab to MatchDetailPage**

In `frontend/src/pages/MatchDetailPage.jsx`:

1. Import the component:
```js
import MatchSummaryTab from '../components/Video/MatchSummaryTab'
```

2. Find the tab list (the row with "Overview", "Rallies", "Actions", "Analytics" buttons). Add a 5th tab:
```jsx
<button
  onClick={() => setActiveTab('summary')}
  className={clsx('tab-btn', activeTab === 'summary' && 'active')}
>
  Summary
</button>
```

3. Find the tab content section and add:
```jsx
{activeTab === 'summary' && (
  <MatchSummaryTab
    match={match}
    rallies={ralliesData}
    actionsData={actionsData}
  />
)}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/Video/MatchSummaryTab.jsx frontend/src/pages/MatchDetailPage.jsx
git commit -m "feat: add match summary tab with ball heat map, zone chart, and key moments"
```

---

## Self-Review

### Spec Coverage Check

| Spec requirement | Covered by task |
|------------------|----------------|
| Fix import crashes | Task 4 (startup check), Task 5 (error guard in scoring) |
| WebSocket proxy | Task 1 |
| Failure signal — backend | Task 2 |
| Failure signal — frontend | Task 3 |
| torch in requirements | Task 6 |
| Scoring engine empty actions | Task 5 |
| VideoAnnotation model | Task 7 |
| Annotation CRUD API | Task 8 |
| In-app annotation page | Task 9 |
| Training trigger API + WS | Task 10 |
| Player compare endpoint | Task 11 |
| PlayerComparePage | Task 12 |
| Rotation model | Task 13 |
| Rotation detection in pipeline | Task 14 |
| Rotation panel UI | Task 15 |
| Ball heatmap endpoint | Task 16 |
| Match summary tab | Task 17 |

### Notes
- `AnnotatePage` uses `match.video_id` which is present on the match object returned by `matchesAPI.get()`.
- `MatchSummaryTab` receives `ralliesData` — verify MatchDetailPage already fetches rallies (it does, via `matchesAPI.rallies(id)`).
- All new routers are registered in `main.py` in Tasks 8 and 10.
- All new models are imported in `models/__init__.py` in Tasks 7, 13, and the relevant tasks.
- `detect_rotation` is called only when `completed` (a `RallySegment`) is non-None, which is the correct hook point.
