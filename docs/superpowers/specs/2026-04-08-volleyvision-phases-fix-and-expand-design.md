# VolleyVision — Pipeline Fix + Phase 4 + Phase 5 Design

**Date:** 2026-04-08  
**Scope:** Fix broken CV pipeline (Phases 1-3), build LSTM expansion (Phase 4), build advanced analytics (Phase 5)  
**Approach:** Approach A — Fix hard first, then build. No mock/synthetic data at any point.

---

## Section 1: Pipeline Fix (Phases 1–3)

### Problem Summary

The analysis pipeline is broken end-to-end. Root causes identified from code audit:

1. **Hard import crashes at module load** — `cv_pipeline.py` imports `PlayerTracker`, `BallDetector`, `ActionService` directly. If `ultralytics`, `supervision`, or `torch` are missing from the venv, FastAPI either fails to start or crashes on the first `/analyze` call with an unhandled `ImportError`.

2. **Missing model weights** — `player_detection.pt` and `ball_detection.pt` may not exist in `models/weights/`. `PlayerTracker` attempts an auto-download of `yolov8n.pt` as fallback which silently fails in offline environments. `ActionService` disables itself gracefully when no LSTM weights are present — but this means zero actions are ever written to the DB.

3. **Pose backend not guaranteed** — `ActionService` relies on RTMPose or MediaPipe for keypoint extraction. Neither is guaranteed to be installed. Without a working pose extractor, the LSTM receives no input and produces no actions even when weights are present.

4. **WebSocket never signals failure** — `analysis_worker.py` runs the pipeline in a FastAPI background task. If the pipeline throws an exception mid-way, the WebSocket never sends a completion or failure message. The frontend spinner hangs permanently.

5. **Analytics table never written on partial failure** — If the pipeline crashes after frame processing but before `ScoringEngine.compute()` is called, the `Analytics` table stays empty. The Analytics tab shows all zeros.

6. **Frontend `/api` proxy** — If `vite.config.js` does not proxy `/api` → `localhost:8000`, all API calls return 404.

### Fixes

| # | File | Fix |
|---|------|-----|
| 1 | `backend/app/services/player_tracker.py` | Wrap ultralytics/supervision import in try/except; raise `RuntimeError("CV deps missing")` with clear message on `load()` |
| 2 | `backend/app/services/ball_detector.py` | Same pattern as PlayerTracker |
| 3 | `backend/app/services/action_service.py` | Wrap torch/mediapipe/rtmlib imports; on load failure log exactly which dep is missing |
| 4 | `backend/app/services/cv_pipeline.py` | Catch `RuntimeError` from tracker/detector load; propagate as `HTTPException(503)` with human-readable message |
| 5 | `backend/app/main.py` | Add startup dep-check in `lifespan()` — log which CV packages are available/missing at boot |
| 6 | `backend/app/workers/analysis_worker.py` | Wrap entire pipeline run in try/except; always emit `{"type": "failed", "reason": str(e)}` via WebSocket on exception; set match status to `failed` in DB |
| 7 | `backend/app/services/scoring_engine.py` | Call `compute()` even when `actions` list is empty — writes zero-value Analytics rows so the table is never empty after a completed analysis |
| 8 | `backend/requirements.txt` | Audit and add: `ultralytics`, `supervision`, `torch`, `mediapipe` (or `rtmlib`), `ffmpeg-python` |
| 9 | `frontend/vite.config.js` | Confirm `/api` proxy target is `http://localhost:8000`; add WebSocket proxy for `/api/matches/*/ws/*` |

### Success Criteria

- `python run.py` starts without import errors even when GPU/CV deps are missing
- Clicking "Analyze" on a match with CV deps installed runs to completion and emits 0–100% progress via WebSocket
- If analysis fails, match status becomes `failed` and the frontend shows an error state instead of a spinner
- After a successful analysis, the Actions tab and Analytics tab show real data

---

## Section 2: Phase 4 — LSTM Expansion (Multi-Action Training)

### Goal

Expand action recognition from spike-only (Phase 1) to 6 action types: **spike, serve, block, dig, set, reception**.

### Current State

- `training/action_recognition/annotations.json` — 16 spike timestamps from `Spike.mp4`
- `training/action_recognition/train_lstm.py` — supports `--phase 1` (spike) and `--phase 2` (multi-class)
- `training/action_recognition/extract_poses.py` — extracts pose sequences from tagged timestamps
- `training/action_recognition/run_phase3_pipeline.py` — one-command: extract → train → validate

### New: Annotation Table

A new `video_annotations` table stores user-tagged action timestamps:

```
video_annotations
  id            UUID PK
  match_id      UUID FK → matches
  video_path    String        -- absolute path used by training scripts
  timestamp     Float         -- seconds in video
  action_type   String        -- spike | serve | block | dig | set | reception
  tagged_by     UUID FK → users
  created_at    DateTime
```

### New: Annotation API

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/api/matches/{id}/annotations` | coach+ | List all annotations for a match |
| POST | `/api/matches/{id}/annotations` | coach+ | Create one annotation `{timestamp, action_type}` |
| DELETE | `/api/annotations/{id}` | coach+ | Remove an annotation |
| GET | `/api/annotations/export` | admin | Export all annotations as JSON (training format) |

### New: In-App Annotation Page

Route: `/matches/:id/annotate`  
Access: coach and admin only

Layout:
- Video player (same `VideoPlayer` component, read-only controls)
- Timeline bar showing existing tagged timestamps as coloured markers
- "Tag Action" panel: current timestamp display + action type dropdown (spike/serve/block/dig/set/reception) + "Add Tag" button
- Tag list below: scrollable table of all tags with delete buttons
- "Export for Training" button (admin only) — downloads JSON in `annotations.json` format

### New: Training Trigger API

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/training/run` | admin | Trigger `run_phase3_pipeline.py --phase 2` as subprocess |
| GET | `/api/training/status` | admin | Returns current training status and last log line |

Training runs as a background subprocess. Progress streamed via WebSocket at `WS /api/training/ws`.

### Weight Reload

After training completes, `ActionService` calls an internal `reload()` method that re-loads the model from `models/weights/action_lstm_phase2.pt` without restarting the server.

### Success Criteria

- A coach can open any completed match, scrub the video, and tag timestamps for all 6 action types
- Admin can export the full annotation set as JSON matching the existing training format
- Admin can trigger Phase 2 retraining from the UI and watch progress
- After retraining, new analysis runs use the Phase 2 weights automatically

---

## Section 3: Phase 5 — Advanced Analytics

### Feature A: Player Comparison

**Route:** `/analytics/compare` (new page, linked from AnalyticsPage)  
**Access:** coach and admin

UI:
- Two player-select dropdowns (populated from players in completed matches)
- Side-by-side radar chart (same 5-axis radar as PlayerDashboard: Serving, Attacking, Blocking, Digging, Reception)
- Stat diff table: each row shows stat name, Player A value, Player B value, and a delta badge (green if A > B, red if A < B)
- Stats covered: attacks, kills, aces, blocks, block_pts, digs, receptions, attack_eff%, serve_eff%, reception_eff%

**New Backend Endpoint:**

`GET /api/analytics/players/compare?player_a={id}&player_b={id}`  
Aggregates from `Analytics` table, same joins as existing leaderboard. Returns both players' full stat sets.

### Feature B: Rotation Detection

**What it detects:** During analysis, after homography maps player pixel coords → normalised court coords (0–1), players are assigned to one of 6 rotation positions based on a 2×3 court grid:

```
Court top half (court_y < 0.5):  positions 1, 2, 3  (back row)
Court bottom half (court_y >= 0.5): positions 4, 5, 6  (front row)
Within each half: left/centre/right split by court_x thirds
```

**New `rotations` table:**

```
rotations
  id              UUID PK
  match_id        UUID FK → matches
  rally_id        UUID FK → rallies
  team            String(10)
  rotation_number Int         -- 1-6 inferred rotation
  player_positions JSONB      -- {track_id: {court_x, court_y, slot}} per player
  frame_number    Int
  created_at      DateTime
```

Populated by `cv_pipeline.py` once per rally (on the first frame of each rally).

**Frontend:** New "Rotation" sub-panel on MatchDetailPage > Rallies tab. Shows a small court diagram (same mini-map SVG) with 6 numbered slots and player markers placed in their detected positions for the selected rally.

### Feature C: Match Summary + Heat Maps

**New "Summary" tab on MatchDetailPage** (5th tab after Overview/Rallies/Actions/Analytics):

1. **Ball position heat map** — Canvas element rendering a 2D density plot of all `ball_tracking` rows for the match, overlaid on a court outline SVG. Colour scale: blue (cold) → red (hot). Computed client-side from raw court_x/court_y data fetched from `GET /api/matches/{id}/tracking/ball-heatmap`.

2. **Attack zone distribution** — Bar chart showing count of `actions` grouped by `zone` (1–6), split by team A and B. Uses existing `actions` data already in DB.

3. **Key moment timeline** — Horizontal scrollable timeline of the top 10 rallies by duration. Each card shows: rally number, duration, winner team, point reason. Clicking seeks the video to that rally's start_time.

**New Backend Endpoint:**

`GET /api/matches/{id}/tracking/ball-heatmap` — returns aggregated `{court_x, court_y}` pairs from `ball_tracking` table (downsampled to max 2000 points).

### Success Criteria

- Player comparison page renders correct radar + diff table for any two players with analytics data
- Rotation panel shows player slot assignments per rally on MatchDetailPage
- Summary tab shows heat map, zone chart, and key moment timeline for any completed match

---

## Implementation Order

1. **Pipeline fix** — unblock everything; nothing in Phases 4-5 is useful if analysis doesn't run
2. **Annotation page + API** — enables tagging new training data immediately
3. **Training trigger UI** — admin can kick off Phase 2 retraining
4. **Phase 5 Feature A** — Player comparison (pure frontend + one new endpoint, low risk)
5. **Phase 5 Feature B** — Rotation detection (requires pipeline change + new table)
6. **Phase 5 Feature C** — Summary/heat maps (requires new endpoint + canvas rendering)
