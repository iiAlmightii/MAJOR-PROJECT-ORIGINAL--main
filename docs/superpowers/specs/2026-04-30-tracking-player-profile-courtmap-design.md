# Design Spec: Tracking Fix + Player Profile Modal + Court Map Accuracy
**Date:** 2026-04-30  
**Status:** Approved  
**Scope:** Three sequential features — fix tracking first (A), then UI features (B, C)

---

## Background & Problem Statement

The CV pipeline uses ByteTrack to assign player IDs. With the current configuration (`lost_track_buffer: 30` ≈ 1.5s at the effective 20fps processing rate), players who disappear from frame for more than ~1.5 seconds (occlusion, rotation, timeout) are considered "lost." When they reappear, ByteTrack assigns a new `track_id` → the pipeline creates a new `Player` UUID → a ghost player accumulates in the database with near-zero stats.

**Observed result:** ~60 Player records per match instead of the expected ≤16 (12 active players + 2 referees + 2 coaches).

**Downstream effects:**
- Analytics tab shows 60+ players, almost all with zero stats
- Player profile feature (B) would show meaningless data
- Court mini-map (C) has unreliable positions due to both ghost players and poor homography calibration

**Build order:** A → B → C. B and C depend on A being clean.

---

## Feature A — Tracking Fix (Approach B: Tuning + Post-processing Merger)

### A1 — ByteTrack Parameter Changes
File: `backend/app/services/player_tracker.py`

| Parameter | Old | New | Reason |
|---|---|---|---|
| `lost_track_buffer` | 30 frames (~1.5s) | 120 frames (~6s) | Players absent during timeouts/rotations |
| `minimum_matching_threshold` | 0.8 | 0.65 | Match through partial occlusion |
| `track_activation_threshold` | 0.45 | 0.50 | Reduce false positives from audience/scoreboard |
| Max tracks hard cap | None | 16 | 12 players + 2 refs + 2 coaches |

### A2 — Post-processing Track Merger
New file: `backend/app/services/track_merger.py`

Called once after the full CV pipeline completes for a match. Runs four steps:

**Step 1 — Find merge candidates**  
For every pair of Player UUIDs in the match, check all three conditions:
- Same team assignment
- Time ranges do not overlap (one track ends before the other begins)
- Spatial gap ≤ 0.25 normalized court units between the last known foot position of the earlier track and the first known foot position of the later track

**Step 2 — Merge and consolidate**  
Re-assign all rows in `player_tracking`, `actions`, and `analytics` from the ghost UUID to the parent UUID. Delete the ghost `Player` record. Stats from both tracks accumulate on the surviving player.

**Step 3 — Prune overflow**  
If more than 16 Player records still exist after merging, rank by total frames detected. Keep the top 16, delete the rest. Low frame-count records are noise (audience, scoreboard artifacts).

**Step 4 — Assign stable display numbers**  
Assign sequential integer display numbers stored in a new `display_number` column on the `Player` model. Assignment order: Team A players ranked by first-seen frame → #1–#6, Team B players ranked by first-seen frame → #7–#12, remaining detections (referees, coaches, noise not pruned) → #13–#16. If fewer than 6 players are confidently assigned to a team, numbers are assigned up to however many exist.

The video overlay shows `#display_number` instead of the raw UUID prefix.

### A3 — Re-analyze Trigger

**Backend:** `POST /api/matches/{id}/reanalyze`  
- Admin and coach roles only  
- Clears all match-specific data: `player_tracking`, `ball_tracking`, `players`, `actions`, `analytics`, `rallies` rows for this match  
- Resets `match.status` → `"pending"`  
- Fires full CV pipeline in background (same worker as initial analysis)  
- Progress broadcast via existing WebSocket at `WS /api/matches/{id}/ws/progress`

**Frontend:** "Re-analyze" button in match detail header  
- Visible to admin and coach only  
- Confirmation dialog: *"This will delete all existing tracking data and re-run analysis. This cannot be undone. Continue?"*  
- Shows existing WebSocket progress bar during re-run  
- On completion, invalidates React Query cache for match data

### A4 — Team Assignment Fix
Current team assignment is unreliable. Replace with:  
- Cluster players by average `court_x` position across all their tracking frames  
- Players with median `court_x` < 0.5 → Team A; ≥ 0.5 → Team B  
- Applied during the merge pass in `track_merger.py`

---

## Feature B — Player Profile Modal

### B1 — Canvas Click Detection
File: `frontend/src/components/Video/VideoPlayer.jsx`

- Remove `pointer-events: none` from the tracking overlay canvas  
- Add `onClick` handler to the canvas  
- On click: compute click position relative to canvas, scale to video dimensions, iterate current tracking data bounding boxes  
- If click point is inside a bounding box: open player profile modal for that player  
- If no bounding box hit: fall through to toggle video play/pause (existing behavior)  
- Touch events handled the same way for mobile

### B2 — New API Endpoint
`GET /api/matches/{match_id}/players/{player_id}/stats`

Response shape:
```json
{
  "player_id": "uuid",
  "display_number": 3,
  "team": "A",
  "presence": {
    "frames_detected": 8420,
    "total_frames": 11400,
    "involvement_pct": 73.9,
    "time_on_court_seconds": 514
  },
  "actions": {
    "attack":  { "total": 12, "success": 7, "error": 2, "neutral": 3 },
    "block":   { "total": 5,  "success": 2, "error": 1, "neutral": 2 },
    "dig":     { "total": 8,  "success": 6, "error": 0, "neutral": 2 },
    "serve":   { "total": 3,  "success": 2, "error": 1, "neutral": 0 },
    "set":     { "total": 7,  "success": 5, "error": 0, "neutral": 2 }
  },
  "zones": { "1": 4, "2": 11, "3": 18, "4": 9, "5": 3, "6": 2 },
  "efficiency": {
    "attack_eff": 0.41,
    "serve_eff": 0.33
  },
  "recent_actions": [
    { "timestamp": 501.2, "action_type": "attack", "result": "success" },
    { "timestamp": 464.8, "action_type": "block",  "result": "neutral" },
    { "timestamp": 412.1, "action_type": "serve",  "result": "error" }
  ]
}
```
Returns last 10 actions in `recent_actions`, ordered by timestamp descending.

### B3 — Modal Component
New file: `frontend/src/components/Video/PlayerProfileModal.jsx`

**Layout (top to bottom):**
1. **Header band** — team-colored gradient background, circular avatar with display number, player label, team badge, time on court
2. **Involvement bar** — full-width progress bar showing `involvement_pct`, colored green
3. **Stats grid** — 3×2 grid: Attacks, Blocks, Digs, Serves, Sets, Attack Efficiency
4. **Zone activity** — 6 vertical bars (Zones 1–6), height proportional to action count in that zone
5. **Recent actions** — list of last 10 actions with action type, result icon (✓/—/✗), and timestamp. Clicking a row seeks the video to that timestamp and closes the modal.

**Behavior:**
- Opens as a centered modal with backdrop blur
- Closes on backdrop click, Escape key, or ✕ button
- Does not interfere with video playback (video remains playing/paused as it was)
- Data fetched on open via React Query (cached per player per match)
- Independent of the existing Analytics tab — does not share state or components

---

## Feature C — Court Map Accuracy

### C1 — Improved Auto-Calibration
File: `backend/app/services/homography_service.py`

Replace single-frame calibration with multi-frame sampling:
- Sample 5 keyframes evenly distributed across the first 30 seconds of video
- Compute a candidate homography matrix for each frame using existing Hough line detection
- For each candidate, compute reprojection error: project the 4 detected court corners back through the inverse matrix and measure pixel distance from original detected points
- Select the candidate with the lowest reprojection error as the active homography
- Fall back to current heuristic if fewer than 4 valid lines are found in all 5 frames

### C2 — Manual Corner Override UI
New file: `frontend/src/components/Video/CourtCalibrationModal.jsx`

**Trigger:** "Fix Court Map" button in match detail header (admin/coach only, appears after analysis completes)

**Flow:**
1. Modal opens showing the first clean video frame (fetched via existing stream endpoint, seeked to t=2s)
2. Instruction pills show corner order: Top-Left → Top-Right → Bottom-Right → Bottom-Left
3. User clicks on the video frame image; each click places a colored dot and activates the next pill
4. "Reset" clears all dots and restarts
5. "Apply" (enabled only when all 4 corners placed) sends corners to `POST /api/matches/{id}/homography` with the 4 pixel coordinates
6. Modal closes; mini-map updates on next tracking poll (within 100ms)

### C3 — Mini-map Rendering Improvements
File: `frontend/src/components/Video/VideoPlayer.jsx`, `drawMiniMap()` function

- Player dots colored by team: blue for Team A, red for Team B (matching overlay box colors)
- Display number rendered inside each dot (font size 7px)
- Ball rendered as yellow dot (already exists, keep)
- Add ⚠ warning badge in top-right corner of mini-map when >30% of current-frame players have `null` court coordinates — signals to user that calibration may be off and they should use "Fix Court Map"

---

## Data Model Changes

### `players` table — new column
```sql
ALTER TABLE players ADD COLUMN display_number INTEGER;
```
Populated by `track_merger.py` after each pipeline run.

### No other schema changes required
All other data (actions, analytics, player_tracking, ball_tracking) already links via `player_id` UUID — merging UUIDs in the merger pass correctly consolidates everything.

---

## API Changes Summary

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/matches/{id}/reanalyze` | Wipe + re-run CV pipeline |
| `GET` | `/api/matches/{match_id}/players/{player_id}/stats` | Player profile stats |

Existing endpoint used as-is:
- `POST /api/matches/{id}/homography` — manual court calibration (already implemented)

---

## Files Changed / Created

### Backend
| File | Change |
|---|---|
| `app/services/player_tracker.py` | Tune ByteTrack params, add hard cap |
| `app/services/track_merger.py` | **New** — post-processing merger logic |
| `app/services/homography_service.py` | Multi-frame auto-calibration |
| `app/models/player.py` | Add `display_number` column |
| `app/routers/processing.py` | Add `reanalyze` endpoint |
| `app/routers/processing.py` | Add `player stats` endpoint |
| `app/workers/analysis_worker.py` | Call `track_merger` after full pipeline completes (not per-batch) |

### Frontend
| File | Change |
|---|---|
| `src/components/Video/VideoPlayer.jsx` | Canvas click handler, mini-map improvements |
| `src/components/Video/PlayerProfileModal.jsx` | **New** — player profile modal |
| `src/components/Video/CourtCalibrationModal.jsx` | **New** — court corner calibration UI |
| `src/pages/MatchDetailPage.jsx` | Re-analyze button, Fix Court Map button |
| `src/services/api.js` | Add `reanalyze`, `playerStats` API calls |

---

## Constraints & Non-Goals

- Player profile modal is independent of the Analytics tab — no shared state or components
- Re-ID (appearance-based matching) is explicitly out of scope — track merger handles long-occlusion cases
- No changes to the Actions tab, Rallies tab, or Speech tab
- No Docker — runs directly via venv + npm per project convention
- Pydantic v2 for all new schemas
