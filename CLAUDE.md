# CLAUDE.md — VolleyVision Project Context

## Project
**AI-Based Volleyball Match Analytics Platform**
A Balltime/Hudl-like platform for volleyball coaches, players, and admins.

## Tech Stack
| Layer     | Tech                                      |
|-----------|-------------------------------------------|
| Frontend  | React 18 + Vite + Tailwind CSS + Zustand + React Query |
| Backend   | FastAPI (Python 3.11) + SQLAlchemy async  |
| Database  | PostgreSQL 16                             |
| CV / AI   | YOLOv8, supervision (ByteTrack), OpenCV   |
| Video     | HTML5 streaming, FFmpeg for rally clips   |

> **No Docker** — run backend and frontend directly (see Startup below).

## Directory Structure
```
MAJOR-PROJECT-ORIGINAL--main/
├── backend/           FastAPI app
│   ├── app/
│   │   ├── models/    SQLAlchemy ORM models
│   │   ├── routers/   API endpoints (auth, videos, matches, processing …)
│   │   ├── services/  CV pipeline, player tracker, ball detector, homography
│   │   └── workers/   Background analysis worker + WebSocket registry
│   ├── uploads/       Uploaded video files
│   ├── rallies/       Rally clip segments
│   └── run.py         Entry point → uvicorn on :8000
├── frontend/          React SPA
│   └── src/
│       ├── pages/     LoginPage, MatchDetailPage, UploadPage …
│       ├── components/Video/  VideoPlayer.jsx + FilterPanel.jsx
│       └── hooks/     useTrackingData, useAnalysisProgress (WebSocket)
├── Dataset/           Training datasets (all models)
│   ├── Action recognition/         (5 classes: block, defense, serve, set, spike)
│   │   ├── data.yaml
│   │   ├── Spike.mp4               ← place action video here for LSTM training
│   │   └── annotations.json        ← tagged action timestamps (export from AnnotatePage)
│   ├── Action Recognition 2/       (13 action classes)
│   ├── Action recognition 3/       (6 classes: blocking, digging, passing, serving, setting, spiking)
│   ├── Ball detector/              (1 class: Balls)
│   ├── Court detector/             (1 class: court — segmentation)
│   ├── Court detector 2/           (1 class: court_boundary)
│   ├── Net detector/               (1 class: net)
│   ├── Refree detector/            (1 class: referee)
│   ├── Refree detector 2/          (1 class: referee)
│   ├── rally detector/             (2 classes: break, in-play — classification)
│   └── Volleyball Activity Dataset.v1i.yolov8 (1)/  (7 action classes, ~25k images)
├── training/          Training scripts for all models
│   ├── player_detection/  train_player_local.py → Dataset/Refree detector/
│   ├── ball_detection/    train_ball.py → Dataset/Ball detector/
│   └── action_recognition/
│       ├── extract_poses.py       (RTMPose/MediaPipe → .npy clip sequences)
│       ├── train_lstm.py          (BiLSTM 30×34 → phase1/phase2)
│       ├── validate_spike.py      (cross-validate tagged timestamps)
│       └── run_phase3_pipeline.py (one-command: extract→train→validate)
├── models/weights/    Trained .pt weight files go here
└── CLAUDE.md          ← this file
```

## User Roles
- **admin** — full access, user management, system logs
- **coach** — upload videos, start analysis, view analytics
- **player** — view personal stats and match replays

## Startup (No Docker)
```bash
# 1. Start PostgreSQL (must be running on port 5432)
#    DB: volleyball_analytics  user: postgres  password: password

# 2. Backend
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python run.py                   # → http://localhost:8000

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev                     # → http://localhost:5173
```
Default admin: `admin@volleyball.com` / `Admin@123456`

## Development Phases
| Phase | Status   | Description                                        |
|-------|----------|----------------------------------------------------|
| 1     | ✅ Done   | Auth, video upload, basic UI, filter panel         |
| 2     | ✅ Done   | CV pipeline: player tracking, ball detection, homography, rally detection, WebSocket |
| 3     | ✅ Done   | Action recognition: Pose+BiLSTM pipeline, ScoringEngine, actions API + UI tab |
| 4     | 🔜 Next   | Phase 2 LSTM (add serve/block/defense/set videos + tag timestamps) |
| 5     | 🔜 Next   | Advanced analytics dashboards, player comparison, rotation detection |

## Key Architecture Notes

### CV Pipeline (`backend/app/services/cv_pipeline.py`)
- Entry via `POST /api/matches/{id}/analyze`
- Runs in background task via `workers/analysis_worker.py`
- Progress broadcast via WebSocket at `WS /api/matches/{id}/ws/progress`
- Tracking data stored in `player_tracking` + `ball_tracking` tables
- Frontend polls `/api/matches/{id}/tracking?timestamp=X` every 100ms for overlay

### Homography (`backend/app/services/homography_service.py`)
- Auto-calibrates from first keyframe using Hough line detection
- Manual override: `POST /api/matches/{id}/homography` with 4 court corners
- Maps pixel coords → normalised (0–1) court coords for mini-map

### Video Streaming
- Range-request streaming at `GET /api/videos/{id}/stream`
- Frontend VideoPlayer uses HTML5 `<video>` + canvas overlay

### Filter Panel (Balltime-style)
- Slides in from right side of video
- Filters: Players, Actions, Positions, Court Zones (1–6), Action Time, Labels
- Adjusts rally list and (Phase 3) highlights in video

## Models (weights → `models/weights/`)
| File                        | Used by              | Training script                                       |
|-----------------------------|----------------------|-------------------------------------------------------|
| `player_detection.pt`       | PlayerTracker        | `training/player_detection/train_player_local.py`     |
| `ball_detection.pt`         | BallDetector         | `training/ball_detection/stream_coco_ball.py`         |
| `action_lstm_phase1.pt`     | ActionService        | `training/action_recognition/train_lstm.py --phase 1` |
| `action_lstm_phase2.pt`     | ActionService        | `training/action_recognition/train_lstm.py --phase 2` |

Without custom weights, YOLOv8n COCO pretrain (auto-downloaded) is used as fallback.
ActionService gracefully disables itself if no LSTM weights are present.

## Action Recognition Workflow
1. Place source video at `Dataset/Action recognition/Spike.mp4`
2. Tag timestamps via the in-app AnnotatePage, then export:
   `GET /annotations/export` → save as `Dataset/Action recognition/annotations.json`
3. Run `python training/action_recognition/run_phase3_pipeline.py` (extract→train→validate)
4. If VERDICT = PROCEED: add more videos to `Dataset/Action recognition/` + retrain with `--phase 2`
5. Weights auto-loaded by `ActionService` on analysis start

## Database
All tables auto-created on startup via SQLAlchemy `create_all`.
Connection string in `backend/.env` → `DATABASE_URL`.

## Important Files
- `backend/app/main.py` — FastAPI app, CORS, lifespan, admin seeding
- `backend/app/services/cv_pipeline.py` — full analysis orchestrator (Phase 1–3)
- `backend/app/services/action_service.py` — Pose+LSTM inference, per-player sliding window
- `backend/app/services/scoring_engine.py` — volleyball rules, efficiency calculations
- `backend/app/routers/processing.py` — analyze, WebSocket, tracking, homography, actions endpoints
- `frontend/src/components/Video/VideoPlayer.jsx` — video + canvas overlay + mini-map
- `frontend/src/components/Video/FilterPanel.jsx` — Balltime-style sliding filter UI
- `frontend/src/pages/MatchDetailPage.jsx` — main analysis view (4 tabs: overview/rallies/actions/analytics)

## Preferences / Conventions
- No Docker — run directly with venv + npm
- Python 3.11, async SQLAlchemy throughout
- Pydantic v2 for all schemas
- React with Zustand for auth state, React Query for server state
- Tailwind utility classes only (no CSS modules)
- Dark theme: `court-bg (#1a1f2e)`, `court-panel (#232b3e)`, `court-border (#2e3a52)`
