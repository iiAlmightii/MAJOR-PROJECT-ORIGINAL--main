# Tracking Fix + Player Profile Modal + Court Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix duplicate player tracking (60 ghost players → ≤16), add a player profile modal triggered by clicking bounding boxes, and improve court mini-map accuracy with multi-frame calibration and a manual corner override UI.

**Architecture:** Backend CV pipeline tuning + a new post-processing track merger pass that runs after full analysis. Frontend gets three new components (PlayerProfileModal, CourtCalibrationModal, Re-analyze button) wired to two new API endpoints.

**Tech Stack:** FastAPI + SQLAlchemy async (Python 3.12), React 18 + Vite + Tailwind + React Query, PostgreSQL 16, supervision ByteTrack, OpenCV. No Docker — run via `backend/.venv/bin/python run.py` and `npm run dev`.

---

## File Map

### New files
| Path | Responsibility |
|---|---|
| `backend/app/services/track_merger.py` | Post-pipeline merge: consolidate ghost UUIDs, prune to ≤16, assign display numbers |
| `backend/tests/test_track_merger.py` | Unit tests for merger logic (pure Python, no DB) |
| `frontend/src/components/Video/PlayerProfileModal.jsx` | Player stats modal opened on bounding box click |
| `frontend/src/components/Video/CourtCalibrationModal.jsx` | 4-corner court calibration UI |

### Modified files
| Path | What changes |
|---|---|
| `backend/app/models/player.py` | Add `display_number: Mapped[Optional[int]]` column |
| `backend/app/services/player_tracker.py` | Tune ByteTrack params, add MAX_TRACKS cap |
| `backend/app/services/homography_service.py` | Add `auto_calibrate_best_of(frames)` multi-frame method |
| `backend/app/services/cv_pipeline.py` | Use multi-frame calibration instead of single frame |
| `backend/app/workers/analysis_worker.py` | Call track merger after pipeline.run() |
| `backend/app/routers/processing.py` | Add `/reanalyze` and `/players/{id}/stats` endpoints; include `display_number` in tracking response |
| `frontend/src/services/api.js` | Add `reanalyze()` and `playerStats()` calls |
| `frontend/src/pages/MatchDetailPage.jsx` | Re-analyze button + Fix Court Map button |
| `frontend/src/components/Video/VideoPlayer.jsx` | Canvas click detection, mini-map team colors + warning badge |

---

## Task 1: Add `display_number` to Player model

**Files:**
- Modify: `backend/app/models/player.py`

- [ ] **Step 1: Add the column**

Open `backend/app/models/player.py`. Add the import and column:

```python
from typing import Optional
from sqlalchemy import String, DateTime, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base
import uuid
from datetime import datetime


class Player(Base):
    __tablename__ = "players"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("matches.id"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    player_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    team: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    position: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    jersey_number: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    display_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="players")
    user = relationship("User", foreign_keys=[user_id])
    tracking = relationship("PlayerTracking", back_populates="player")
    actions = relationship("Action", back_populates="player")

    def __repr__(self):
        return f"<Player #{self.display_number or self.player_track_id} (Match: {self.match_id})>"
```

- [ ] **Step 2: Apply migration via SQLAlchemy auto-create**

The project uses `create_all` on startup (not Alembic). Run the migration manually so existing data isn't lost:

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main/backend"
.venv/bin/python -c "
import asyncio
from app.database import engine
from sqlalchemy import text

async def migrate():
    async with engine.begin() as conn:
        await conn.execute(text(
            'ALTER TABLE players ADD COLUMN IF NOT EXISTS display_number INTEGER'
        ))
        print('display_number column added')

asyncio.run(migrate())
"
```

Expected output: `display_number column added`

- [ ] **Step 3: Verify column exists**

```bash
.venv/bin/python -c "
import asyncio
from app.database import engine
from sqlalchemy import text

async def check():
    async with engine.begin() as conn:
        r = await conn.execute(text(
            \"SELECT column_name FROM information_schema.columns \"
            \"WHERE table_name='players' AND column_name='display_number'\"
        ))
        row = r.fetchone()
        print('Column exists:', row is not None)

asyncio.run(check())
"
```

Expected output: `Column exists: True`

- [ ] **Step 4: Commit**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main"
git add backend/app/models/player.py
git commit -m "feat: add display_number column to Player model"
```

---

## Task 2: Tune ByteTrack Parameters

**Files:**
- Modify: `backend/app/services/player_tracker.py`

- [ ] **Step 1: Update constants and ByteTrack init**

In `player_tracker.py`, replace the constants block (lines ~38-57) and ByteTrack constructor calls. Find all occurrences of `sv.ByteTrack(` — there are two (in `load()` and `reset()`). Update both.

Replace the constants at the top:

```python
CONF_THRESHOLD         = 0.50   # raised from 0.45 — reduces audience false positives
IOU_THRESHOLD          = 0.45
REFEREE_IOU_THRESH     = 0.30
MAX_TRACKS             = 16     # 12 players + 2 refs + 2 coaches hard ceiling
```

Replace both `sv.ByteTrack(...)` calls (in `load()` and `reset()`) with:

```python
self._tracker = sv.ByteTrack(
    track_activation_threshold=CONF_THRESHOLD,
    lost_track_buffer=120,
    minimum_matching_threshold=0.65,
    frame_rate=25,
)
```

- [ ] **Step 2: Enforce MAX_TRACKS cap in process_frame**

At the end of `process_frame`, before `return output`, add:

```python
            # Hard cap: keep only the MAX_TRACKS detections with highest confidence
            if len(output) > MAX_TRACKS:
                output.sort(key=lambda d: d["confidence"], reverse=True)
                output = output[:MAX_TRACKS]
```

- [ ] **Step 3: Verify the server still starts**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main/backend"
.venv/bin/python -c "from app.services.player_tracker import PlayerTracker; t = PlayerTracker(); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main"
git add backend/app/services/player_tracker.py
git commit -m "feat: tune ByteTrack params and add MAX_TRACKS=16 hard cap"
```

---

## Task 3: Create Track Merger

**Files:**
- Create: `backend/app/services/track_merger.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_track_merger.py`

- [ ] **Step 1: Write the failing unit tests**

Create `backend/tests/__init__.py` (empty).

Create `backend/tests/test_track_merger.py`:

```python
"""Unit tests for track merger logic — pure Python, no DB."""
import pytest
from app.services.track_merger import (
    _find_merge_pairs,
    _assign_display_numbers,
)


def make_track(player_id, team, t_start, t_end, last_cx, last_cy, first_cx, first_cy, frame_count):
    return {
        "player_id": player_id,
        "team": team,
        "t_start": t_start,
        "t_end": t_end,
        "last_cx": last_cx,
        "last_cy": last_cy,
        "first_cx": first_cx,
        "first_cy": first_cy,
        "frame_count": frame_count,
    }


def test_find_merge_pairs_same_team_non_overlapping_close():
    tracks = [
        make_track("p1", "A", 0.0, 10.0, 0.3, 0.5, None, None, 500),
        make_track("p2", "A", 12.0, 30.0, None, None, 0.32, 0.51, 200),
    ]
    pairs = _find_merge_pairs(tracks)
    assert ("p1", "p2") in pairs or ("p2", "p1") in pairs


def test_find_merge_pairs_different_team_not_merged():
    tracks = [
        make_track("p1", "A", 0.0, 10.0, 0.3, 0.5, None, None, 500),
        make_track("p2", "B", 12.0, 30.0, None, None, 0.32, 0.51, 200),
    ]
    pairs = _find_merge_pairs(tracks)
    assert len(pairs) == 0


def test_find_merge_pairs_overlapping_times_not_merged():
    tracks = [
        make_track("p1", "A", 0.0, 20.0, 0.3, 0.5, None, None, 500),
        make_track("p2", "A", 15.0, 30.0, None, None, 0.32, 0.51, 200),
    ]
    pairs = _find_merge_pairs(tracks)
    assert len(pairs) == 0


def test_find_merge_pairs_too_far_spatially_not_merged():
    tracks = [
        make_track("p1", "A", 0.0, 10.0, 0.1, 0.1, None, None, 500),
        make_track("p2", "A", 12.0, 30.0, None, None, 0.9, 0.9, 200),
    ]
    pairs = _find_merge_pairs(tracks)
    assert len(pairs) == 0


def test_assign_display_numbers_team_a_first():
    players = [
        {"player_id": "a1", "team": "A", "frame_count": 100, "t_start": 5.0},
        {"player_id": "a2", "team": "A", "frame_count": 200, "t_start": 2.0},
        {"player_id": "b1", "team": "B", "frame_count": 150, "t_start": 3.0},
    ]
    result = _assign_display_numbers(players)
    nums = {p["player_id"]: p["display_number"] for p in result}
    # Team A: sorted by t_start → a2 (2.0)=#1, a1 (5.0)=#2
    assert nums["a2"] == 1
    assert nums["a1"] == 2
    # Team B: starts at #7
    assert nums["b1"] == 7


def test_assign_display_numbers_none_team_gets_high_number():
    players = [
        {"player_id": "x1", "team": None, "frame_count": 50, "t_start": 1.0},
    ]
    result = _assign_display_numbers(players)
    assert result[0]["display_number"] >= 13
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main/backend"
.venv/bin/python -m pytest tests/test_track_merger.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'app.services.track_merger'`

- [ ] **Step 3: Create `track_merger.py`**

Create `backend/app/services/track_merger.py`:

```python
"""
Track Merger
────────────
Post-processing pass that runs after the full CV pipeline completes.

Steps:
  1. Load all Player records for the match.
  2. Find pairs that are likely the same physical person (same team,
     non-overlapping time, spatially close last/first position).
  3. Merge ghost tracks into their parent: re-assign player_tracking,
     actions rows; delete the ghost Player record.
  4. If still >16 players, prune the lowest frame-count ones.
  5. Assign stable display_number: Team A → #1-6, Team B → #7-12, rest → #13+.
"""

import logging
import uuid
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

SPATIAL_MERGE_THRESHOLD = 0.25   # normalized court units
MAX_PLAYERS = 16


# ─────────────────────────────────────────────────────────────────────────────
# Pure-logic helpers (tested without DB)
# ─────────────────────────────────────────────────────────────────────────────

def _find_merge_pairs(tracks: List[Dict]) -> List[Tuple[str, str]]:
    """
    Return list of (parent_id, ghost_id) pairs to merge.

    A pair is mergeable when ALL three hold:
      • same team
      • time ranges do not overlap (one ends before the other begins)
      • spatial gap between last position of earlier and first position
        of later is ≤ SPATIAL_MERGE_THRESHOLD
    """
    pairs = []
    n = len(tracks)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tracks[i], tracks[j]
            if a["team"] != b["team"] or a["team"] is None:
                continue
            # Ensure a ends before b starts (or swap)
            if a["t_end"] > b["t_start"] and b["t_end"] > a["t_start"]:
                continue  # overlap — cannot merge
            earlier, later = (a, b) if a["t_end"] <= b["t_start"] else (b, a)
            lx, ly = earlier.get("last_cx"), earlier.get("last_cy")
            fx, fy = later.get("first_cx"), later.get("first_cy")
            if lx is None or fx is None:
                continue
            dist = ((lx - fx) ** 2 + (ly - fy) ** 2) ** 0.5
            if dist <= SPATIAL_MERGE_THRESHOLD:
                pairs.append((earlier["player_id"], later["player_id"]))
    return pairs


def _assign_display_numbers(players: List[Dict]) -> List[Dict]:
    """
    Assign display_number to each player dict in place.
    Team A: #1–6 sorted by t_start.
    Team B: #7–12 sorted by t_start.
    Others (refs, coaches, unknown): #13+ sorted by t_start.
    Returns the same list with 'display_number' key added.
    """
    team_a = sorted([p for p in players if p.get("team") == "A"], key=lambda p: p["t_start"])
    team_b = sorted([p for p in players if p.get("team") == "B"], key=lambda p: p["t_start"])
    others = sorted([p for p in players if p.get("team") not in ("A", "B")], key=lambda p: p["t_start"])

    counter = 1
    for p in team_a:
        p["display_number"] = counter
        counter += 1

    counter = 7
    for p in team_b:
        p["display_number"] = counter
        counter += 1

    counter = 13
    for p in others:
        p["display_number"] = counter
        counter += 1

    return players


# ─────────────────────────────────────────────────────────────────────────────
# DB operations
# ─────────────────────────────────────────────────────────────────────────────

async def merge_tracks(match_id: str) -> Dict:
    """
    Run the full merge pass for a match. Called from analysis_worker after
    pipeline.run() completes successfully.

    Returns a summary dict: {merged, pruned, final_count}
    """
    from app.database import AsyncSessionLocal
    from app.models.player import Player
    from app.models.tracking import PlayerTracking
    from app.models.actions import Action
    from sqlalchemy import select, func, delete as sa_delete, update as sa_update

    logger.info(f"TrackMerger: starting for match {match_id}")
    mid = uuid.UUID(match_id)

    async with AsyncSessionLocal() as db:

        # ── Load player summaries ────────────────────────────────────────────
        p_result = await db.execute(
            select(Player).where(Player.match_id == mid)
        )
        players = p_result.scalars().all()

        if not players:
            logger.info("TrackMerger: no players found, skipping")
            return {"merged": 0, "pruned": 0, "final_count": 0}

        # For each player: get t_start, t_end, frame_count, last/first court pos
        track_info: List[Dict] = []
        for p in players:
            stats = await db.execute(
                select(
                    func.min(PlayerTracking.timestamp),
                    func.max(PlayerTracking.timestamp),
                    func.count(PlayerTracking.id),
                ).where(PlayerTracking.player_id == p.id)
            )
            row = stats.one()
            t_start, t_end, frame_count = row

            # Last known court position (foot of earlier track)
            last_pos = await db.execute(
                select(PlayerTracking.court_x, PlayerTracking.court_y)
                .where(
                    PlayerTracking.player_id == p.id,
                    PlayerTracking.court_x.isnot(None),
                )
                .order_by(PlayerTracking.timestamp.desc())
                .limit(1)
            )
            last_row = last_pos.one_or_none()

            # First known court position
            first_pos = await db.execute(
                select(PlayerTracking.court_x, PlayerTracking.court_y)
                .where(
                    PlayerTracking.player_id == p.id,
                    PlayerTracking.court_x.isnot(None),
                )
                .order_by(PlayerTracking.timestamp.asc())
                .limit(1)
            )
            first_row = first_pos.one_or_none()

            track_info.append({
                "player_id":   str(p.id),
                "team":        p.team,
                "t_start":     t_start or 0.0,
                "t_end":       t_end or 0.0,
                "frame_count": frame_count or 0,
                "last_cx":     last_row[0] if last_row else None,
                "last_cy":     last_row[1] if last_row else None,
                "first_cx":    first_row[0] if first_row else None,
                "first_cy":    first_row[1] if first_row else None,
            })

        # ── Fix team assignment using median court_x ─────────────────────────
        for info in track_info:
            if info["team"] is None:
                xs_r = await db.execute(
                    select(PlayerTracking.court_x)
                    .where(
                        PlayerTracking.player_id == uuid.UUID(info["player_id"]),
                        PlayerTracking.court_x.isnot(None),
                    )
                )
                xs = [r[0] for r in xs_r.all()]
                if xs:
                    median_x = sorted(xs)[len(xs) // 2]
                    new_team = "A" if median_x < 0.5 else "B"
                    info["team"] = new_team
                    await db.execute(
                        sa_update(Player)
                        .where(Player.id == uuid.UUID(info["player_id"]))
                        .values(team=new_team)
                    )

        # ── Find merge candidates ────────────────────────────────────────────
        pairs = _find_merge_pairs(track_info)
        logger.info(f"TrackMerger: found {len(pairs)} merge pairs")

        # Build parent→ghost mapping (resolve chains)
        parent_map: Dict[str, str] = {}  # ghost_id → parent_id
        for parent_id, ghost_id in pairs:
            # Follow chains: if parent is already a ghost, follow to root
            root = parent_id
            while root in parent_map:
                root = parent_map[root]
            parent_map[ghost_id] = root

        # ── Merge ghost rows into parent ─────────────────────────────────────
        merged_count = 0
        for ghost_id, parent_id in parent_map.items():
            ghost_uuid  = uuid.UUID(ghost_id)
            parent_uuid = uuid.UUID(parent_id)

            await db.execute(
                sa_update(PlayerTracking)
                .where(PlayerTracking.player_id == ghost_uuid)
                .values(player_id=parent_uuid)
            )
            await db.execute(
                sa_update(Action)
                .where(Action.player_id == ghost_uuid)
                .values(player_id=parent_uuid)
            )
            await db.execute(
                sa_delete(Player).where(Player.id == ghost_uuid)
            )
            merged_count += 1

        await db.flush()

        # ── Prune overflow (keep top MAX_PLAYERS by frame count) ─────────────
        remaining = await db.execute(
            select(Player).where(Player.match_id == mid)
        )
        remaining_players = remaining.scalars().all()

        pruned_count = 0
        if len(remaining_players) > MAX_PLAYERS:
            # Count frames per player
            counts = []
            for p in remaining_players:
                cnt = await db.execute(
                    select(func.count(PlayerTracking.id))
                    .where(PlayerTracking.player_id == p.id)
                )
                counts.append((p.id, cnt.scalar() or 0))

            counts.sort(key=lambda x: x[1], reverse=True)
            to_delete = [pid for pid, _ in counts[MAX_PLAYERS:]]
            for pid in to_delete:
                await db.execute(
                    sa_delete(PlayerTracking).where(PlayerTracking.player_id == pid)
                )
                await db.execute(
                    sa_delete(Action).where(Action.player_id == pid)
                )
                await db.execute(
                    sa_delete(Player).where(Player.id == pid)
                )
                pruned_count += 1
            await db.flush()

        # ── Assign display numbers ────────────────────────────────────────────
        final_result = await db.execute(
            select(Player).where(Player.match_id == mid)
        )
        final_players = final_result.scalars().all()

        # Build summary list for display number assignment
        summary = []
        for p in final_players:
            t_start_r = await db.execute(
                select(func.min(PlayerTracking.timestamp))
                .where(PlayerTracking.player_id == p.id)
            )
            t_start_val = t_start_r.scalar() or 0.0
            frame_cnt_r = await db.execute(
                select(func.count(PlayerTracking.id))
                .where(PlayerTracking.player_id == p.id)
            )
            summary.append({
                "player_id":    str(p.id),
                "team":         p.team,
                "t_start":      t_start_val,
                "frame_count":  frame_cnt_r.scalar() or 0,
            })

        numbered = _assign_display_numbers(summary)
        for item in numbered:
            await db.execute(
                sa_update(Player)
                .where(Player.id == uuid.UUID(item["player_id"]))
                .values(
                    display_number=item["display_number"],
                    display_name=f"Player #{item['display_number']} (Team {item['team'] or '?'})",
                )
            )

        await db.commit()

        final_count = len(final_players)
        logger.info(
            f"TrackMerger: done — merged={merged_count}, pruned={pruned_count}, "
            f"final={final_count}"
        )
        return {"merged": merged_count, "pruned": pruned_count, "final_count": final_count}
```

- [ ] **Step 4: Run tests — they should pass**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main/backend"
.venv/bin/python -m pytest tests/test_track_merger.py -v
```

Expected output: all 6 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main"
git add backend/app/services/track_merger.py backend/tests/
git commit -m "feat: add track_merger post-processing pass with unit tests"
```

---

## Task 4: Wire Track Merger into Analysis Worker

**Files:**
- Modify: `backend/app/workers/analysis_worker.py`

- [ ] **Step 1: Call merge_tracks after pipeline.run()**

In `analysis_worker.py`, find the `try` block that calls `pipeline.run()`. Replace it:

```python
    try:
        summary = await pipeline.run()
        logger.info(f"Analysis done for match {match_id}: {summary}")

        # Post-processing: merge ghost tracks, prune to ≤16, assign display numbers
        await progress_cb(95, "Merging and deduplicating player tracks...")
        from app.services.track_merger import merge_tracks
        merge_summary = await merge_tracks(match_id)
        logger.info(f"TrackMerger result: {merge_summary}")

        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Match).where(Match.id == uuid.UUID(match_id)))
            m = result.scalar_one_or_none()
            if m:
                m.status = MatchStatus.completed
                await db.commit()

        await _broadcast(match_id, 100, "Analysis complete!")

    except Exception as exc:
        logger.error(f"Analysis failed for match {match_id}: {exc}", exc_info=True)
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Match).where(Match.id == uuid.UUID(match_id)))
            m = result.scalar_one_or_none()
            if m:
                m.status = MatchStatus.failed
                await db.commit()
        await _broadcast(match_id, -1, f"Analysis failed: {exc}", failed=True)
```

Note: remove any existing `m.status = MatchStatus.completed` lines after `pipeline.run()` since the block above handles that.

- [ ] **Step 2: Verify import works**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main/backend"
.venv/bin/python -c "from app.workers.analysis_worker import run_analysis; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main"
git add backend/app/workers/analysis_worker.py
git commit -m "feat: call track_merger after pipeline completes"
```

---

## Task 5: Add Re-analyze Endpoint

**Files:**
- Modify: `backend/app/routers/processing.py`

- [ ] **Step 1: Add the endpoint**

In `processing.py`, after the existing `start_analysis` endpoint (around line 90), add:

```python
@router.post("/{match_id}/reanalyze")
async def reanalyze_match(
    match_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    request: Request,
    body: AnalyzeRequest = AnalyzeRequest(),
    current_user: User = Depends(require_coach),
    db: AsyncSession = Depends(get_db),
):
    """Wipe existing tracking data and re-run the CV pipeline."""
    result = await db.execute(select(Match).where(Match.id == match_id))
    match = result.scalar_one_or_none()
    if not match:
        raise HTTPException(404, "Match not found")
    if current_user.role != UserRole.admin and match.uploaded_by != current_user.id:
        raise HTTPException(403, "Access denied")
    if match.status == MatchStatus.processing:
        raise HTTPException(400, "Analysis already running")

    vid_result = await db.execute(select(Video).where(Video.id == match.video_id))
    video = vid_result.scalar_one_or_none()
    if not video:
        raise HTTPException(404, "Video file not found")

    from app.models.player import Player
    from app.models.tracking import PlayerTracking, BallTracking
    from app.models.actions import Action, Rally
    from app.models.rotations import Rotation
    from app.models.analytics import Analytics
    from sqlalchemy import delete as sa_delete

    for model in (Rotation, Action, BallTracking, PlayerTracking, Rally, Analytics, Player):
        await db.execute(sa_delete(model).where(model.match_id == match_id))

    match.status = MatchStatus.processing
    match.processing_progress = 0
    match.total_rallies = 0
    await db.flush()

    background_tasks.add_task(
        run_analysis,
        str(match_id),
        video.file_path,
        body.court_corners,
    )
    return {"message": "Re-analysis started", "match_id": str(match_id)}
```

- [ ] **Step 2: Test endpoint exists**

```bash
curl -s http://localhost:8001/api/openapi.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
paths = [p for p in d['paths'] if 'reanalyze' in p]
print('Found:', paths)
"
```

Expected: `Found: ['/api/matches/{match_id}/reanalyze']`

- [ ] **Step 3: Commit**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main"
git add backend/app/routers/processing.py
git commit -m "feat: add POST /matches/{id}/reanalyze endpoint"
```

---

## Task 6: Add Player Stats Endpoint + Update Tracking Response

**Files:**
- Modify: `backend/app/routers/processing.py`

- [ ] **Step 1: Add player stats endpoint**

Add this endpoint after the ball-heatmap endpoint in `processing.py`:

```python
@router.get("/{match_id}/players/{player_id}/stats")
async def get_player_stats(
    match_id: uuid.UUID,
    player_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return aggregated stats for a single player in a match."""
    from app.models.analytics import Analytics
    from app.models.actions import Action, ActionType, ActionResult
    from sqlalchemy import func

    # Verify player belongs to this match
    p_result = await db.execute(
        select(Player).where(Player.id == player_id, Player.match_id == match_id)
    )
    player = p_result.scalar_one_or_none()
    if not player:
        raise HTTPException(404, "Player not found in this match")

    # Total frames in match
    match_result = await db.execute(select(Match).where(Match.id == match_id))
    match = match_result.scalar_one_or_none()

    # Presence
    presence_result = await db.execute(
        select(
            func.count(PlayerTracking.id),
            func.min(PlayerTracking.timestamp),
            func.max(PlayerTracking.timestamp),
        ).where(PlayerTracking.player_id == player_id)
    )
    frames_detected, t_min, t_max = presence_result.one()
    frames_detected = frames_detected or 0

    # Total frames from ball tracking as proxy for total match frames
    total_frames_result = await db.execute(
        select(func.count(BallTracking.id)).where(BallTracking.match_id == match_id)
    )
    total_frames = total_frames_result.scalar() or 1
    involvement_pct = round((frames_detected / total_frames) * 100, 1) if total_frames else 0.0
    time_on_court = round((t_max - t_min) if (t_max and t_min) else 0.0, 1)

    # Actions grouped by type + result
    actions_result = await db.execute(
        select(Action.action_type, Action.result, func.count(Action.id))
        .where(Action.player_id == player_id, Action.match_id == match_id)
        .group_by(Action.action_type, Action.result)
    )
    actions_raw = actions_result.all()

    actions: Dict = {}
    for action_type, result, count in actions_raw:
        at = action_type.value if hasattr(action_type, "value") else str(action_type)
        r  = result.value if hasattr(result, "value") else str(result)
        if at not in actions:
            actions[at] = {"total": 0, "success": 0, "error": 0, "neutral": 0}
        actions[at]["total"] += count
        if r in actions[at]:
            actions[at][r] += count

    # Zone counts
    zones_result = await db.execute(
        select(Action.zone, func.count(Action.id))
        .where(Action.player_id == player_id, Action.match_id == match_id, Action.zone.isnot(None))
        .group_by(Action.zone)
    )
    zones = {str(z): c for z, c in zones_result.all()}

    # Efficiency from analytics table
    analytics_result = await db.execute(
        select(Analytics).where(
            Analytics.player_id == player_id,
            Analytics.match_id == match_id
        )
    )
    analytics = analytics_result.scalar_one_or_none()
    efficiency = {
        "attack_eff": float(analytics.attack_efficiency) if analytics else 0.0,
        "serve_eff":  float(analytics.serve_efficiency) if analytics else 0.0,
    }

    # Recent actions (last 10)
    recent_result = await db.execute(
        select(Action.timestamp, Action.action_type, Action.result)
        .where(Action.player_id == player_id, Action.match_id == match_id)
        .order_by(Action.timestamp.desc())
        .limit(10)
    )
    recent_actions = [
        {
            "timestamp":   float(ts),
            "action_type": at.value if hasattr(at, "value") else str(at),
            "result":      r.value if hasattr(r, "value") else str(r),
        }
        for ts, at, r in recent_result.all()
    ]

    return {
        "player_id":      str(player_id),
        "display_number": player.display_number,
        "team":           player.team,
        "presence": {
            "frames_detected":       frames_detected,
            "total_frames":          total_frames,
            "involvement_pct":       involvement_pct,
            "time_on_court_seconds": time_on_court,
        },
        "actions":        actions,
        "zones":          zones,
        "efficiency":     efficiency,
        "recent_actions": recent_actions,
    }
```

You need to add `Dict` to the imports at the top of `processing.py`:
```python
from typing import Optional, List, Dict
```

- [ ] **Step 2: Update tracking endpoint to include display_number**

In the `get_tracking_data` endpoint, find the `seen_tracks[tid] = {...}` dict (around line 177) and add `display_number`:

```python
            seen_tracks[tid] = {
                "player_id":       tid,
                "player_track_id": player.player_track_id,
                "display_number":  player.display_number,
                "team":            player.team,
                "display_name":    player.display_name,
                "bbox_x":          pt.bbox_x,
                "bbox_y":          pt.bbox_y,
                "bbox_w":          pt.bbox_w,
                "bbox_h":          pt.bbox_h,
                "court_x":         pt.court_x,
                "court_y":         pt.court_y,
                "timestamp":       pt.timestamp,
            }
```

- [ ] **Step 3: Verify the new endpoints appear in OpenAPI**

```bash
curl -s http://localhost:8001/api/openapi.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
paths = [p for p in d['paths'] if 'players' in p or 'stats' in p]
print('Found:', paths)
"
```

Expected: `Found: ['/api/matches/{match_id}/players/{player_id}/stats']`

- [ ] **Step 4: Commit**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main"
git add backend/app/routers/processing.py
git commit -m "feat: add player stats endpoint and display_number in tracking response"
```

---

## Task 7: Frontend API Functions

**Files:**
- Modify: `frontend/src/services/api.js`

- [ ] **Step 1: Add reanalyze and playerStats to api.js**

In `api.js`, find the `matchesAPI` or relevant export section. Add these two functions alongside existing API calls:

```js
// In the matchesAPI object or as standalone exports — match the existing pattern in the file.
// Check how other endpoints are called (e.g., videosAPI.streamUrl) and mirror that style.

reanalyze: (matchId) =>
  apiClient.post(`/matches/${matchId}/reanalyze`),

playerStats: (matchId, playerId) =>
  apiClient.get(`/matches/${matchId}/players/${playerId}/stats`),
```

Check `api.js` first to see whether it exports a single object or named functions, and add these following the same pattern.

- [ ] **Step 2: Verify no import errors in frontend**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main/frontend"
npm run build 2>&1 | tail -10
```

Expected: build succeeds with no errors (or only pre-existing warnings).

- [ ] **Step 3: Commit**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main"
git add frontend/src/services/api.js
git commit -m "feat: add reanalyze and playerStats API calls"
```

---

## Task 8: Re-analyze Button in MatchDetailPage

**Files:**
- Modify: `frontend/src/pages/MatchDetailPage.jsx`

- [ ] **Step 1: Add Re-analyze button with confirmation**

In `MatchDetailPage.jsx`, find the match detail header section (look for where match title/status is rendered). Add a Re-analyze button visible only to admin/coach:

```jsx
// At the top of the component, add state:
const [reanalyzing, setReanalyzing] = useState(false)

// Add this handler:
const handleReanalyze = async () => {
  if (!window.confirm(
    'This will delete all existing tracking data and re-run analysis. This cannot be undone. Continue?'
  )) return
  try {
    setReanalyzing(true)
    await matchesAPI.reanalyze(matchId)   // adjust import to match existing API pattern
    // React Query cache invalidation — match the pattern used elsewhere in the file
    queryClient.invalidateQueries(['match', matchId])
    queryClient.invalidateQueries(['tracking', matchId])
  } catch (err) {
    console.error('Re-analyze failed:', err)
    alert('Re-analyze failed: ' + (err?.response?.data?.detail || err.message))
    setReanalyzing(false)
  }
}

// In the JSX header section, add (next to existing action buttons):
{(user?.role === 'admin' || user?.role === 'coach') && (
  <button
    onClick={handleReanalyze}
    disabled={reanalyzing || match?.status === 'processing'}
    className="px-3 py-1.5 text-sm bg-orange-600/20 hover:bg-orange-600/30 
               text-orange-400 border border-orange-600/30 rounded-lg 
               disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
  >
    {reanalyzing ? 'Starting...' : 'Re-analyze'}
  </button>
)}
```

- [ ] **Step 2: Add Fix Court Map button**

In the same header section, add:

```jsx
const [showCalibration, setShowCalibration] = useState(false)

// In JSX (next to Re-analyze button):
{(user?.role === 'admin' || user?.role === 'coach') && match?.status === 'completed' && (
  <button
    onClick={() => setShowCalibration(true)}
    className="px-3 py-1.5 text-sm bg-blue-600/20 hover:bg-blue-600/30 
               text-blue-400 border border-blue-600/30 rounded-lg transition-colors"
  >
    Fix Court Map
  </button>
)}

{showCalibration && (
  <CourtCalibrationModal
    matchId={matchId}
    videoId={match?.video_id}
    onClose={() => setShowCalibration(false)}
  />
)}
```

Import `CourtCalibrationModal` at the top of the file:
```jsx
import CourtCalibrationModal from '../components/Video/CourtCalibrationModal'
```

- [ ] **Step 3: Check page renders without error**

Open http://localhost:5173 in browser, navigate to a match. Confirm:
- Re-analyze button appears in header for admin/coach
- Fix Court Map button appears when match is completed
- No console errors

- [ ] **Step 4: Commit**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main"
git add frontend/src/pages/MatchDetailPage.jsx
git commit -m "feat: add Re-analyze and Fix Court Map buttons to match detail"
```

---

## Task 9: Canvas Click Detection in VideoPlayer

**Files:**
- Modify: `frontend/src/components/Video/VideoPlayer.jsx`

- [ ] **Step 1: Add selectedPlayer state and click handler**

In `VideoPlayer.jsx`, add at the top of the component:

```jsx
const [selectedPlayerId, setSelectedPlayerId] = useState(null)
```

Add the canvas click handler (add after existing `changeRate` function):

```jsx
const handleCanvasClick = (e) => {
  const canvas = canvasRef.current
  const video  = videoRef.current
  if (!canvas || !video || !trackingData?.players?.length) {
    // No bounding boxes — pass click to video
    togglePlay()
    return
  }
  const rect   = canvas.getBoundingClientRect()
  const clickX = (e.clientX - rect.left) * (video.videoWidth  / canvas.clientWidth)
  const clickY = (e.clientY - rect.top)  * (video.videoHeight / canvas.clientHeight)

  for (const p of trackingData.players) {
    const { bbox_x: x, bbox_y: y, bbox_w: w, bbox_h: h } = p
    if (clickX >= x && clickX <= x + w && clickY >= y && clickY <= y + h) {
      setSelectedPlayerId(p.player_id)
      return
    }
  }
  // Clicked canvas but missed all boxes — toggle play
  togglePlay()
}
```

- [ ] **Step 2: Update canvas element**

Find the canvas element (around line 266) and update it:

```jsx
{trackingData && (
  <canvas
    ref={canvasRef}
    className="absolute inset-0 w-full h-full cursor-pointer"
    onClick={handleCanvasClick}
  />
)}
```

Remove the `pointer-events-none` class and add `cursor-pointer`. Also remove `onClick={togglePlay}` from the `<video>` element since the canvas now handles clicks.

- [ ] **Step 3: Wire modal open/close**

Add PlayerProfileModal to VideoPlayer JSX (at the bottom, before closing `</div>`):

```jsx
{selectedPlayerId && matchId && (
  <PlayerProfileModal
    matchId={matchId}
    playerId={selectedPlayerId}
    onClose={() => setSelectedPlayerId(null)}
    onSeek={(ts) => {
      const v = videoRef.current
      if (v) v.currentTime = ts
      setSelectedPlayerId(null)
    }}
  />
)}
```

Import at top of file:
```jsx
import PlayerProfileModal from './PlayerProfileModal'
```

Ensure `matchId` is passed as a prop to `VideoPlayer`. Check `MatchDetailPage.jsx` to confirm `matchId` is already passed; if not, add it to the component signature:
```jsx
export default function VideoPlayer({ videoId, matchId, trackingData, onTimeUpdate, showOverlay }) {
```

- [ ] **Step 4: Verify clicks work in browser**

Open a match with completed analysis. Play the video. Click on a player bounding box. Confirm:
- Clicking a box does NOT toggle play/pause
- Clicking empty canvas does toggle play/pause
- selectedPlayerId is set (check React DevTools or add a temporary console.log)

- [ ] **Step 5: Commit**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main"
git add frontend/src/components/Video/VideoPlayer.jsx
git commit -m "feat: add canvas click detection for player profile modal"
```

---

## Task 10: PlayerProfileModal Component

**Files:**
- Create: `frontend/src/components/Video/PlayerProfileModal.jsx`

- [ ] **Step 1: Create the component**

Create `frontend/src/components/Video/PlayerProfileModal.jsx`:

```jsx
import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { matchesAPI } from '../../services/api'

const TEAM_COLORS = {
  A: { bg: 'from-blue-900 to-blue-800', badge: 'bg-blue-600', dot: '#3b82f6' },
  B: { bg: 'from-red-900 to-red-800',  badge: 'bg-red-600',  dot: '#ef4444' },
}

const RESULT_ICON = { success: '✓', error: '✗', neutral: '—' }
const RESULT_COLOR = { success: 'text-green-400', error: 'text-red-400', neutral: 'text-yellow-400' }

function fmtTime(secs) {
  const m = Math.floor(secs / 60)
  const s = Math.floor(secs % 60).toString().padStart(2, '0')
  return `${m}:${s}`
}

export default function PlayerProfileModal({ matchId, playerId, onClose, onSeek }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ['playerStats', matchId, playerId],
    queryFn:  () => matchesAPI.playerStats(matchId, playerId).then(r => r.data),
    staleTime: 60_000,
  })

  // Close on Escape
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  const colors = TEAM_COLORS[data?.team] || TEAM_COLORS.A

  const ACTION_LABELS = {
    attack: 'Attacks', block: 'Blocks', dig: 'Digs',
    serve: 'Serves', set: 'Sets', reception: 'Receptions',
  }

  const maxZoneCount = data
    ? Math.max(1, ...Object.values(data.zones).map(Number))
    : 1

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="bg-[#0d1117] rounded-xl overflow-hidden w-full max-w-lg shadow-2xl border border-white/10">

        {/* Header */}
        <div className={`bg-gradient-to-r ${colors.bg} p-5 flex items-center gap-4 relative`}>
          <div className={`w-14 h-14 rounded-full ${colors.badge} flex items-center justify-center text-2xl font-bold text-white`}>
            {isLoading ? '?' : `#${data?.display_number ?? '?'}`}
          </div>
          <div>
            <div className="text-lg font-bold text-white">
              {isLoading ? 'Loading...' : `Player #${data?.display_number ?? '?'}`}
            </div>
            {data && (
              <div className="flex items-center gap-2 mt-1">
                <span className={`${colors.badge} text-white text-xs font-semibold px-2 py-0.5 rounded-full`}>
                  Team {data.team}
                </span>
                <span className="text-white/50 text-sm">
                  {fmtTime(data.presence.time_on_court_seconds)} on court
                </span>
              </div>
            )}
          </div>
          <button
            onClick={onClose}
            className="absolute right-4 top-4 text-white/40 hover:text-white/80 text-xl"
          >✕</button>
        </div>

        {isLoading && (
          <div className="p-8 text-center text-white/40">Loading stats...</div>
        )}

        {isError && (
          <div className="p-8 text-center text-red-400">Failed to load player stats.</div>
        )}

        {data && (
          <>
            {/* Involvement bar */}
            <div className="px-5 pt-4 pb-3 border-b border-white/5">
              <div className="flex justify-between mb-1.5">
                <span className="text-xs text-white/40 uppercase tracking-wider">Match Involvement</span>
                <span className="text-sm font-semibold text-green-400">{data.presence.involvement_pct}%</span>
              </div>
              <div className="h-1.5 bg-white/10 rounded-full">
                <div
                  className="h-full bg-gradient-to-r from-green-500 to-green-400 rounded-full"
                  style={{ width: `${Math.min(data.presence.involvement_pct, 100)}%` }}
                />
              </div>
            </div>

            {/* Stats grid */}
            <div className="px-5 py-3 grid grid-cols-3 gap-2 border-b border-white/5">
              {Object.entries(ACTION_LABELS).map(([key, label]) => (
                <div key={key} className="bg-white/4 rounded-lg p-2.5 text-center">
                  <div className="text-xl font-bold text-blue-400">
                    {data.actions[key]?.total ?? 0}
                  </div>
                  <div className="text-xs text-white/40 mt-0.5">{label}</div>
                </div>
              ))}
              <div className="bg-white/4 rounded-lg p-2.5 text-center">
                <div className="text-xl font-bold text-orange-400">
                  {data.efficiency.attack_eff.toFixed(2)}
                </div>
                <div className="text-xs text-white/40 mt-0.5">Atk Eff</div>
              </div>
            </div>

            {/* Zone activity */}
            {Object.keys(data.zones).length > 0 && (
              <div className="px-5 py-3 border-b border-white/5">
                <p className="text-xs text-white/40 uppercase tracking-wider mb-2">Court Zone Activity</p>
                <div className="flex items-end gap-1 h-16">
                  {[1, 2, 3, 4, 5, 6].map((z) => {
                    const count = Number(data.zones[String(z)] ?? 0)
                    const height = Math.max(4, (count / maxZoneCount) * 56)
                    return (
                      <div key={z} className="flex-1 flex flex-col items-center gap-1">
                        <div
                          className="w-full bg-blue-500 rounded-t"
                          style={{ height: `${height}px`, opacity: 0.3 + (count / maxZoneCount) * 0.7 }}
                        />
                        <span className="text-xs text-white/30">Z{z}</span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Recent actions */}
            {data.recent_actions.length > 0 && (
              <div className="px-5 py-3">
                <p className="text-xs text-white/40 uppercase tracking-wider mb-2">Recent Actions</p>
                <div className="space-y-1.5 max-h-36 overflow-y-auto">
                  {data.recent_actions.map((a, i) => (
                    <button
                      key={i}
                      onClick={() => onSeek(a.timestamp)}
                      className="w-full flex justify-between items-center text-sm
                                 hover:bg-white/5 rounded px-2 py-1 transition-colors text-left"
                    >
                      <span className="text-white/70 capitalize">{a.action_type}</span>
                      <span className={RESULT_COLOR[a.result]}>
                        {RESULT_ICON[a.result]} {a.result}
                      </span>
                      <span className="text-white/30">{fmtTime(a.timestamp)}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify the modal renders**

Open the app in browser, navigate to a match with completed analysis, click a bounding box. Confirm:
- Modal opens with player number and team color
- Stats load (or "Loading..." shows while fetching)
- Clicking a recent action seeks the video and closes the modal
- Escape key and backdrop click close the modal
- No console errors

- [ ] **Step 3: Commit**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main"
git add frontend/src/components/Video/PlayerProfileModal.jsx
git commit -m "feat: add PlayerProfileModal component with stats, zones, and recent actions"
```

---

## Task 11: Multi-frame Homography Calibration

**Files:**
- Modify: `backend/app/services/homography_service.py`
- Modify: `backend/app/services/cv_pipeline.py`

- [ ] **Step 1: Add `auto_calibrate_best_of` to HomographyService**

In `homography_service.py`, add this method to the `HomographyService` class after `auto_calibrate_from_lines`:

```python
def auto_calibrate_best_of(self, frames: list) -> bool:
    """
    Try calibrating from multiple frames, pick the one with lowest
    reprojection error. Falls back to the first successful calibration
    if reprojection error cannot be computed.

    Parameters
    ----------
    frames : list of np.ndarray
        Candidate video frames to try (e.g. 5 frames from first 30s).
    """
    import copy

    best_H     = None
    best_H_inv = None
    best_src   = None
    best_err   = float("inf")

    for frame in frames:
        probe = HomographyService()
        success = probe.auto_calibrate_from_lines(frame)
        if not success or probe._H is None or probe._src_points is None:
            continue

        # Reprojection error: project src corners through H, compare to DST_POINTS
        try:
            src_h = np.hstack([probe._src_points, np.ones((4, 1), dtype=np.float32)])
            projected = (probe._H @ src_h.T).T
            projected /= projected[:, 2:3]
            err = float(np.mean(np.linalg.norm(projected[:, :2] - DST_POINTS, axis=1)))
        except Exception:
            err = float("inf")

        if err < best_err:
            best_err   = err
            best_H     = probe._H.copy()
            best_H_inv = probe._H_inv.copy()
            best_src   = probe._src_points.copy()

    if best_H is not None:
        self._H          = best_H
        self._H_inv      = best_H_inv
        self._src_points = best_src
        return True

    return False
```

- [ ] **Step 2: Update cv_pipeline to use multi-frame calibration**

In `cv_pipeline.py`, find the homography calibration block (around lines 122–132). Replace it:

```python
        # ── Homography calibration ──────────────────────────────────────────
        await self._emit(6, "Calibrating court homography...")
        if self.court_corners:
            self.homography.calibrate(self.court_corners)
            logger.info("Homography: using provided court corners")
        else:
            # Sample 5 frames across the first 30 seconds
            sample_frames = []
            sample_duration = min(30.0, total_frames / fps)
            for i in range(5):
                pos = int((i / 4) * sample_duration * fps) if i < 4 else int(sample_duration * fps) - 1
                pos = min(max(pos, 0), total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, f = cap.read()
                if ret:
                    sample_frames.append(f)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if sample_frames:
                self.homography.auto_calibrate_best_of(sample_frames)
                logger.info(f"Homography: multi-frame calibration from {len(sample_frames)} frames")
            else:
                logger.warning("Homography: no frames sampled, skipping calibration")
```

- [ ] **Step 3: Verify import works**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main/backend"
.venv/bin/python -c "
from app.services.homography_service import HomographyService
h = HomographyService()
print(hasattr(h, 'auto_calibrate_best_of'))
"
```

Expected: `True`

- [ ] **Step 4: Commit**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main"
git add backend/app/services/homography_service.py backend/app/services/cv_pipeline.py
git commit -m "feat: multi-frame homography calibration selects best reprojection error"
```

---

## Task 12: CourtCalibrationModal Component

**Files:**
- Create: `frontend/src/components/Video/CourtCalibrationModal.jsx`

- [ ] **Step 1: Create the component**

Create `frontend/src/components/Video/CourtCalibrationModal.jsx`:

```jsx
import { useState, useRef, useEffect } from 'react'
import { matchesAPI } from '../../services/api'

const CORNER_LABELS = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
const CORNER_COLORS = ['#22c55e', '#3b82f6', '#f59e0b', '#ec4899']

export default function CourtCalibrationModal({ matchId, videoId, onClose }) {
  const [corners, setCorners]   = useState([])   // [{x, y}, ...]
  const [applying, setApplying] = useState(false)
  const [error, setError]       = useState(null)
  const imgRef  = useRef(null)
  const streamUrl = videoId
    ? `/api/videos/${videoId}/stream?token=${localStorage.getItem('access_token')}&t=2`
    : null

  // Close on Escape
  useEffect(() => {
    const h = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [onClose])

  const handleImageClick = (e) => {
    if (corners.length >= 4) return
    const rect   = imgRef.current.getBoundingClientRect()
    const scaleX = imgRef.current.naturalWidth  / rect.width
    const scaleY = imgRef.current.naturalHeight / rect.height
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top)  * scaleY
    setCorners(prev => [...prev, { x, y }])
  }

  const handleApply = async () => {
    if (corners.length < 4) return
    setApplying(true)
    setError(null)
    try {
      const court_corners = corners.map(c => [c.x, c.y])
      await matchesAPI.setHomography(matchId, court_corners)
      onClose()
    } catch (err) {
      setError(err?.response?.data?.detail || 'Failed to apply calibration')
      setApplying(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div className="bg-[#0d1117] rounded-xl overflow-hidden w-full max-w-2xl border border-white/10 shadow-2xl">

        {/* Header */}
        <div className="flex justify-between items-start p-4 border-b border-white/10">
          <div>
            <h3 className="text-white font-semibold">Fix Court Map</h3>
            <p className="text-white/40 text-sm mt-0.5">
              Click the 4 court corners in order
            </p>
          </div>
          <button onClick={onClose} className="text-white/40 hover:text-white/80 text-xl">✕</button>
        </div>

        {/* Corner pills */}
        <div className="flex gap-2 px-4 pt-3 flex-wrap">
          {CORNER_LABELS.map((label, i) => {
            const placed  = i < corners.length
            const current = i === corners.length
            return (
              <span
                key={i}
                className="px-3 py-1 rounded-full text-xs font-semibold text-white transition-all"
                style={{
                  background: placed
                    ? CORNER_COLORS[i]
                    : current
                    ? `${CORNER_COLORS[i]}55`
                    : 'rgba(255,255,255,0.08)',
                  border: current ? `1px solid ${CORNER_COLORS[i]}` : '1px solid transparent',
                }}
              >
                {i + 1} {label} {placed ? '✓' : current ? '←' : ''}
              </span>
            )
          })}
        </div>

        {/* Video frame with click dots */}
        <div className="p-4">
          <div className="relative inline-block w-full" style={{ cursor: corners.length < 4 ? 'crosshair' : 'default' }}>
            {streamUrl ? (
              <video
                ref={imgRef}
                src={streamUrl}
                className="w-full rounded-lg border border-white/10"
                style={{ pointerEvents: 'none' }}
                muted
                preload="metadata"
                onLoadedMetadata={(e) => { e.target.currentTime = 2 }}
              />
            ) : (
              <div className="w-full aspect-video bg-white/5 rounded-lg flex items-center justify-center text-white/30">
                No video
              </div>
            )}

            {/* Invisible click layer */}
            <div
              className="absolute inset-0 rounded-lg"
              onClick={handleImageClick}
            />

            {/* Corner dots */}
            {corners.map((c, i) => {
              const el = imgRef.current
              if (!el) return null
              const rect   = el.getBoundingClientRect()
              const scaleX = rect.width  / (el.videoWidth  || rect.width)
              const scaleY = rect.height / (el.videoHeight || rect.height)
              const px = c.x * scaleX
              const py = c.y * scaleY
              return (
                <div
                  key={i}
                  className="absolute w-3 h-3 rounded-full pointer-events-none"
                  style={{
                    left:      px - 6,
                    top:       py - 6,
                    background: CORNER_COLORS[i],
                    boxShadow: `0 0 8px ${CORNER_COLORS[i]}`,
                  }}
                />
              )
            })}
          </div>
        </div>

        {error && (
          <p className="px-4 pb-2 text-red-400 text-sm">{error}</p>
        )}

        {/* Actions */}
        <div className="flex justify-between p-4 border-t border-white/10">
          <button
            onClick={() => setCorners([])}
            className="px-4 py-2 text-sm text-white/50 hover:text-white/80 
                       bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
          >
            Reset
          </button>
          <button
            onClick={handleApply}
            disabled={corners.length < 4 || applying}
            className="px-4 py-2 text-sm text-blue-300 bg-blue-600/20 hover:bg-blue-600/30
                       border border-blue-600/30 rounded-lg disabled:opacity-40 
                       disabled:cursor-not-allowed transition-colors"
          >
            {applying
              ? 'Applying...'
              : corners.length < 4
              ? `Apply (${4 - corners.length} more needed)`
              : 'Apply'}
          </button>
        </div>
      </div>
    </div>
  )
}
```

Add `setHomography` to `api.js` if not already present:
```js
setHomography: (matchId, courtCorners) =>
  apiClient.post(`/matches/${matchId}/homography`, { court_corners: courtCorners }),
```

- [ ] **Step 2: Verify the modal works in browser**

Click "Fix Court Map" in the match header. Confirm:
- Modal opens showing the video frame seeked to ~2 seconds
- Clicking places colored dots in sequence
- Pill labels update as corners are placed
- Reset clears dots
- Apply button enables after 4 corners and sends the request

- [ ] **Step 3: Commit**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main"
git add frontend/src/components/Video/CourtCalibrationModal.jsx frontend/src/services/api.js
git commit -m "feat: add CourtCalibrationModal for manual 4-corner court setup"
```

---

## Task 13: Mini-map Rendering Improvements

**Files:**
- Modify: `frontend/src/components/Video/VideoPlayer.jsx`

- [ ] **Step 1: Update drawMiniMap to use team colors, display numbers, and warning badge**

In `VideoPlayer.jsx`, find the `drawMiniMap` function (around line 101). Replace the player drawing loop and add the warning badge. The existing function structure is:

```js
// Find the section that draws player dots — it will look something like:
// ctx.fillStyle = ...
// ctx.fillRect(...) or ctx.arc(...)
```

Replace the player drawing block inside `drawMiniMap` with:

```js
    // Draw players with team colors and display numbers
    const nullCourtCount = players.filter(p => p.court_x == null).length
    const totalPlayers   = players.length

    players.forEach(p => {
      if (p.court_x == null || p.court_y == null) return
      const px = mapX + p.court_x * mapW
      const py = mapY + p.court_y * mapH
      const dotR = 5

      // Team color
      ctx.fillStyle = p.team === 'A' ? '#3b82f6' : p.team === 'B' ? '#ef4444' : '#9ca3af'
      ctx.beginPath()
      ctx.arc(px, py, dotR, 0, Math.PI * 2)
      ctx.fill()

      // Display number inside dot
      const label = p.display_number != null ? String(p.display_number) : '?'
      ctx.fillStyle = '#fff'
      ctx.font = 'bold 6px sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(label, px, py)
    })

    // Warning badge if >30% of players have null court coords
    if (totalPlayers > 0 && nullCourtCount / totalPlayers > 0.3) {
      ctx.fillStyle = 'rgba(251,191,36,0.9)'
      ctx.font = 'bold 9px sans-serif'
      ctx.textAlign = 'right'
      ctx.textBaseline = 'top'
      ctx.fillText('⚠', mapX + mapW - 2, mapY + 2)
    }
```

- [ ] **Step 2: Verify mini-map in browser**

Open a match with completed analysis. Confirm:
- Team A players are blue dots, Team B are red
- Each dot shows the player's display number
- If calibration is poor, the ⚠ badge appears

- [ ] **Step 3: Commit**

```bash
cd "/media/1RV24MC025_CHANDAN_K/New Volume/MAJOR-PROJECT-ORIGINAL--main"
git add frontend/src/components/Video/VideoPlayer.jsx
git commit -m "feat: mini-map team colors, display numbers, and calibration warning badge"
```

---

## Self-Review Checklist

After completing all tasks:

- [ ] Run all unit tests: `.venv/bin/python -m pytest backend/tests/ -v`
- [ ] Restart backend and verify no import errors: `.venv/bin/python run.py`
- [ ] Trigger re-analyze on a match and watch WebSocket progress reach 100%
- [ ] Confirm Analytics tab shows ≤16 players after re-analysis
- [ ] Click a bounding box — player profile modal opens with real stats
- [ ] Click a recent action in the modal — video seeks to correct timestamp
- [ ] Open "Fix Court Map" — place 4 corners — mini-map updates within 2 seconds
- [ ] Confirm mini-map shows team-colored dots with numbers
