# Volleyball Analytics — Project Context

This document summarizes the entire codebase and project context for the Volleyball Analytics platform (referred to in the repo as VolleyVision / Volleyball Analytics). Use this file as a single, comprehensive source of truth when generating a thesis or report.

## 1. Project Overview

- Purpose: AI-based volleyball match analytics for coaches, players, and admins. Provides automated tracking, rally detection, action recognition, clip generation, and analytics dashboards similar to Balltime/Hudl.
- Primary users: `admin`, `coach`, `player` (roles supported via auth and UI permissions).
- Main capabilities:
  - Video upload + range-request streaming
  - Player detection & tracking (multi-object tracking)
  - Ball detection and ball-tracking
  - Homography mapping from video to normalized court coordinates
  - Rally detection and rally clip extraction
  - Action recognition (pose extraction + BiLSTM pipeline)
  - Analytics dashboards and scoring engine

## 2. Tech Stack

- Frontend: React 18, Vite, Tailwind CSS, Zustand (state), React Query (server state)
- Backend: FastAPI (Python 3.11), async SQLAlchemy, Pydantic v2
- Database: PostgreSQL 16 (expected default) — connection via `backend/.env` `DATABASE_URL`
- CV & ML: YOLOv8 (object detection), ByteTrack/supervision (tracking), OpenCV (image processing), pose pipelines (RTMPose / MediaPipe-like), PyTorch weights in `models/weights`
- Video/Streaming: HTML5 range-request streaming endpoint and FFmpeg for rally clip creation
- Testing: pytest unit tests located in `backend/tests`

## 3. Directory Structure (high-level)

- `backend/` — FastAPI app and services
  - `backend/app/main.py` — FastAPI app initialization (lifespan, CORS, seeding)
  - `backend/app/config.py`, `database.py` — config and DB utilities
  - `backend/app/routers/` — API routers (auth, matches, processing, videos, tracking)
  - `backend/app/services/` — CV pipeline, homography, action_service, scoring engine
  - `backend/app/workers/` — background analysis worker + websocket registry
  - `backend/run.py` — runserver entrypoint
- `frontend/` — React SPA
  - `frontend/src/components/Video/VideoPlayer.jsx` — video player + overlay + canvas
  - `frontend/src/components/Video/FilterPanel.jsx` — sliding filter UI
  - `frontend/src/pages/MatchDetailPage.jsx` — main analysis view
- `Dataset/` — datasets for training detectors and action recognition
- `training/` — training scripts for players, ball, actions (pose extraction + LSTM)
- `models/weights/` — trained model files (.pt)
- `uploads/` & `rallies/` — uploaded videos and rally clips

Note: See `CLAUDE.md` for a compact high-level project map; this repository contains that file.

## 4. Key Components and Flow

1. User uploads a match video via the frontend.
2. Backend stores uploads under `uploads/` and can spawn an analysis job.
3. Analysis orchestration: `cv_pipeline.py` (services) is the orchestrator. It:
   - Runs background tasks through `workers/analysis_worker.py`.
   - Detects court & calibrates homography using `homography_service.py`.
   - Runs detector+tracker to produce `player_tracking` and `ball_tracking` entries in DB.
   - Detects rallies and emits rally segments (clip generation via FFmpeg).
   - Runs action recognition `action_service.py` if LSTM weights are present.
4. Progress and realtime updates: WebSocket `WS /api/matches/{id}/ws/progress` used by frontend to track analysis progress.
5. Frontend overlays tracking and action predictions onto the video using a canvas above the `<video>` element and fetches tracking frames via `GET /api/matches/{id}/tracking?timestamp=X`.

## 5. Important Endpoints and Interfaces

- POST `/api/matches/{id}/analyze` — start analysis (triggers background worker)
- WS `/api/matches/{id}/ws/progress` — analysis progress updates
- GET `/api/videos/{id}/stream` — range-request streaming for frontend player
- POST `/api/matches/{id}/homography` — manual homography override (4 corners)
- GET `/annotations/export` — export annotations for training pipelines (used in dataset creation)

Exact router locations: `backend/app/routers/processing.py` contains processing endpoints, and `backend/app/services` contains the implementation.

## 6. CV / ML Details

- Detection: YOLOv8 models (repo includes `yolov8n.pt`, `yolov8s.pt`, `yolo26n.pt`, and `best.pt`/`last.pt` artifacts). If custom weights are missing, YOLOv8n COCO pretrain is used as fallback.
- Tracking: ByteTrack / supervision integration handles track association across frames.
- Ball detection: separate model under `Dataset/Ball detector/` and training script `training/ball_detection/`.
- Player detection: `training/player_detection/train_player_local.py` trains player detector weights.
- Action recognition pipeline:
  - Pose extraction: `training/action_recognition/extract_poses.py` (extracts pose sequences to .npy clips)
  - LSTM training: `training/action_recognition/train_lstm.py` (BiLSTM model)
  - Phase3 pipeline wrapper: `training/action_recognition/run_phase3_pipeline.py` (extract → train → validate)
  - Weights: `models/weights/action_lstm_phase1.pt`, `action_lstm_phase2.pt` (if present)

Notes: ActionService is designed to gracefully disable if no LSTM weights are present.

## 7. Models & Weights

- Location: `models/weights/`
- Common files present in repo root: `yolov8n.pt`, `yolov8s.pt`, `yolo26n.pt`, `best.pt`, `last.pt`.
- Training outputs are stored under `training_results/` and `runs/` for specific experiments.

## 8. Datasets

- `Dataset/Action recognition/` — contains `data.yaml` and a place to put videos such as `Spike.mp4` and `annotations.json` exports from the AnnotatePage.
- Separate dataset folders for ball detection, court detection, net, refree, rally detector, etc.

## 9. Startup & Development

Recommended local startup (from CLAUDE.md / quickstart):

1. PostgreSQL running on port `5432`, DB name: `volleyball_analytics`, user: `postgres`, password: `password` (adjust `backend/.env`).
2. Backend (Windows example):

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py  # starts uvicorn on :8000
```

3. Frontend (new terminal):

```bash
cd frontend
npm install
npm run dev  # Vite dev server on :5173
```

Default admin credentials (seeded on first run): `admin@volleyball.com` / `Admin@123456`

## 10. Tests and QA

- Unit tests: `backend/tests/` contains tests such as:
  - `test_ball_speed.py`, `test_jersey_ocr.py`, `test_landing_zones.py`, `test_player_movement.py`, `test_track_merger.py`, etc.
- Run tests (from `backend/`):

```bash
pytest -q
```

Note: tests assume a configured test DB and relevant assets; check tests for fixtures and setup.

## 11. Notable Files (where to look for thesis sections)

- `CLAUDE.md` — existing project summary and architecture notes (useful starting point)
- `backend/app/services/cv_pipeline.py` — orchestrator for analysis pipelines
- `backend/app/services/homography_service.py` — court detection & homography logic
- `backend/app/services/action_service.py` — pose extraction + LSTM inference
- `backend/app/services/scoring_engine.py` — volleyball rules + analytics calculations
- `backend/app/routers/processing.py` — API endpoints for analysis operations
- `frontend/src/components/Video/VideoPlayer.jsx` — overlay and visualization implementation
- `training/action_recognition/*` — pose extraction & training pipeline

## 12. Recommended Report / Thesis Structure (mapped to repository artifacts)

Use these sections in your thesis and map each to relevant files, figures, or experiments:

1. Introduction & Motivation — cite project goal and target users (coaches, players).
2. Related Work — mention Balltime/Hudl-style systems and prior volleyball analytics work.
3. System Architecture — use `CLAUDE.md` and `backend/app/main.py` to describe components and interactions; include a diagram mapping frontend, backend, DB, and ML pipeline.
4. Data & Datasets — document `Dataset/` subfolders and annotation/export workflow.
5. Detection & Tracking — describe YOLOv8 detectors, training scripts, and ByteTrack-based tracker; reference `training/player_detection` and `training/ball_detection`.
6. Homography & Court Mapping — discuss `homography_service.py`, auto-calibration, manual override API.
7. Action Recognition — detail pose extraction (`extract_poses.py`), BiLSTM architecture (`train_lstm.py`), and evaluation scripts (`validate_spike.py`). Include training procedure and hyperparameters from `training/` logs.
8. Rally Detection & Clip Generation — explain pipeline steps and FFmpeg usage for clip extraction.
9. Analytics & Scoring — present formulas and logic in `scoring_engine.py`.
10. Frontend Visualization — explain `VideoPlayer.jsx`, overlays, mini-map, and filters.
11. Experiments & Results — link to `training_results/`, `runs/`, and `models/weights` for quantitative evaluation.
12. Limitations & Future Work — discuss datasets, edge cases, real-time constraints, and planned Phase 4/5 items.

## 13. Practical Notes for Generating the Thesis with Claude/Code

- Use this `project_context.md` as the canonical repository summary input.
- For figures, extract sample frames, detection overlays, and sample rally clips from `rallies/` and `uploads/`.
- To reproduce experiments, use commands in `training/` and collect `runs/` artifacts.
- When asking Claude for prose, point it to specific files or sections in this document and include relevant code snippets or training logs.

## 14. Contact / Maintainers

- Repo root contains `CLAUDE.md` and `README.md` with additional context and next-phase goals.

---

Generated from repository scan and `CLAUDE.md`.
