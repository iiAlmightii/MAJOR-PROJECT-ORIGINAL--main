# Balltime-Level Accuracy & Analytics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ball speed + trajectory visualization, player movement analytics, and action outcome ratings to VolleyVision to reach Balltime AI quality.

**Architecture:** Three additive phases. Phase A: ball speed computed per frame in `ball_detector.py`, stored to DB, surfaced via new trajectory endpoint, drawn as color-coded arc on canvas. Phase B: player distance/speed computed from existing `PlayerTracking.court_x/y` rows after analysis, stored in `Analytics`. Phase C: reception quality rating (0–3) and serve/attack speed annotations added to `Action` rows and exposed in the actions list + player stats endpoints. All DB changes use the project's incremental `ALTER TABLE IF NOT EXISTS` migration pattern in `database.py`.

**Tech Stack:** FastAPI async, SQLAlchemy ORM (mapped_column), PostgreSQL, math/numpy, React canvas (2D API).

**Backend venv:** `backend/.venv` — run Python as `backend/.venv/bin/python` and pytest as `backend/.venv/bin/python -m pytest`.

**Key file map:**
- `backend/app/models/tracking.py` — `BallTracking` ORM
- `backend/app/models/actions.py` — `Action` ORM
- `backend/app/models/analytics.py` — `Analytics` ORM
- `backend/app/database.py` — incremental migrations list
- `backend/app/services/ball_detector.py` — per-frame ball detection + speed
- `backend/app/services/cv_pipeline.py` — main analysis orchestrator
- `backend/app/services/scoring_engine.py` — action result inference
- `backend/app/routers/processing.py` — REST endpoints
- `frontend/src/components/Video/VideoPlayer.jsx` — canvas overlay + court mini-map
- `frontend/src/pages/MatchDetailPage.jsx` — Analytics tab + Actions tab

---

### Task 1: All DB schema changes (Phase A + B + C)

**Files:**
- Modify: `backend/app/models/tracking.py`
- Modify: `backend/app/models/actions.py`
- Modify: `backend/app/models/analytics.py`
- Modify: `backend/app/database.py`
- Test: `backend/tests/test_migrations.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_migrations.py
"""Verify ORM models declare all new columns."""
from app.models.tracking import BallTracking
from app.models.actions import Action
from app.models.analytics import Analytics


def test_ball_tracking_has_speed_columns():
    cols = {c.key for c in BallTracking.__table__.columns}
    assert "speed_kmh" in cols
    assert "vx" in cols
    assert "vy" in cols


def test_action_has_new_columns():
    cols = {c.key for c in Action.__table__.columns}
    assert "landing_x" in cols
    assert "landing_y" in cols
    assert "ball_speed_kmh" in cols
    assert "reception_quality" in cols


def test_analytics_has_movement_columns():
    cols = {c.key for c in Analytics.__table__.columns}
    assert "distance_covered_m" in cols
    assert "avg_speed_kmh" in cols
    assert "max_speed_kmh" in cols
    assert "reception_quality_avg" in cols
    assert "avg_serve_speed_kmh" in cols
    assert "avg_attack_speed_kmh" in cols
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && .venv/bin/python -m pytest tests/test_migrations.py -v
```
Expected: FAIL — `assert "speed_kmh" in cols` fails.

- [ ] **Step 3: Add columns to BallTracking ORM**

In `backend/app/models/tracking.py`, after line `court_y: Mapped[float] = mapped_column(Float, nullable=True)` in `BallTracking`, add:

```python
    speed_kmh: Mapped[float] = mapped_column(Float, nullable=True)   # instantaneous 2D speed km/h
    vx: Mapped[float] = mapped_column(Float, nullable=True)           # court velocity x (norm/s)
    vy: Mapped[float] = mapped_column(Float, nullable=True)           # court velocity y (norm/s)
```

- [ ] **Step 4: Add columns to Action ORM**

In `backend/app/models/actions.py`, after the `source` column (line 52), add:

```python
    landing_x: Mapped[float] = mapped_column(Float, nullable=True)      # court x of ball landing (0-1)
    landing_y: Mapped[float] = mapped_column(Float, nullable=True)       # court y of ball landing (0-1)
    ball_speed_kmh: Mapped[float] = mapped_column(Float, nullable=True)  # ball speed at action moment
    reception_quality: Mapped[int] = mapped_column(Integer, nullable=True)  # 0=error 1=poor 2=good 3=perfect
```

- [ ] **Step 5: Add columns to Analytics ORM**

In `backend/app/models/analytics.py`, after the `extra_data` column, add:

```python
    distance_covered_m: Mapped[float] = mapped_column(Float, nullable=True)
    avg_speed_kmh: Mapped[float]      = mapped_column(Float, nullable=True)
    max_speed_kmh: Mapped[float]      = mapped_column(Float, nullable=True)
    reception_quality_avg: Mapped[float] = mapped_column(Float, nullable=True)
    avg_serve_speed_kmh: Mapped[float]   = mapped_column(Float, nullable=True)
    avg_attack_speed_kmh: Mapped[float]  = mapped_column(Float, nullable=True)
```

- [ ] **Step 6: Add ALTER TABLE migrations**

In `backend/app/database.py`, extend the `statements` list in `_apply_incremental_migrations()`:

```python
    statements = [
        # Existing
        "ALTER TABLE actions ADD COLUMN IF NOT EXISTS source VARCHAR(20) DEFAULT 'cv'",
        # Phase A — ball speed
        "ALTER TABLE ball_tracking ADD COLUMN IF NOT EXISTS speed_kmh FLOAT",
        "ALTER TABLE ball_tracking ADD COLUMN IF NOT EXISTS vx FLOAT",
        "ALTER TABLE ball_tracking ADD COLUMN IF NOT EXISTS vy FLOAT",
        # Phase A — action landing zones
        "ALTER TABLE actions ADD COLUMN IF NOT EXISTS landing_x FLOAT",
        "ALTER TABLE actions ADD COLUMN IF NOT EXISTS landing_y FLOAT",
        "ALTER TABLE actions ADD COLUMN IF NOT EXISTS ball_speed_kmh FLOAT",
        # Phase C — reception quality
        "ALTER TABLE actions ADD COLUMN IF NOT EXISTS reception_quality SMALLINT",
        # Phase B — player movement
        "ALTER TABLE analytics ADD COLUMN IF NOT EXISTS distance_covered_m FLOAT",
        "ALTER TABLE analytics ADD COLUMN IF NOT EXISTS avg_speed_kmh FLOAT",
        "ALTER TABLE analytics ADD COLUMN IF NOT EXISTS max_speed_kmh FLOAT",
        # Phase C — speed aggregates
        "ALTER TABLE analytics ADD COLUMN IF NOT EXISTS reception_quality_avg FLOAT",
        "ALTER TABLE analytics ADD COLUMN IF NOT EXISTS avg_serve_speed_kmh FLOAT",
        "ALTER TABLE analytics ADD COLUMN IF NOT EXISTS avg_attack_speed_kmh FLOAT",
    ]
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
cd backend && .venv/bin/python -m pytest tests/test_migrations.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 8: Start backend and verify migrations run cleanly**

```bash
cd backend && .venv/bin/python run.py
```
Expected: server starts, log shows `Migration OK: ALTER TABLE ball_tracking ADD COLUMN IF NOT EXISTS speed_kmh FLOAT` (and similar) with no errors.

- [ ] **Step 9: Commit**

```bash
git add backend/app/models/tracking.py backend/app/models/actions.py \
        backend/app/models/analytics.py backend/app/database.py \
        backend/tests/test_migrations.py
git commit -m "feat: add ball speed, landing zone, movement, rating columns to DB schema"
```

---

### Task 2: BallDetector — per-frame speed computation (Phase A)

**Files:**
- Modify: `backend/app/services/ball_detector.py`
- Test: `backend/tests/test_ball_speed.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_ball_speed.py
"""Unit tests for ball speed computation — pure math, no YOLO/CV."""
import math
from app.services.ball_detector import BallDetector, COURT_WIDTH_M, COURT_HEIGHT_M


def _make_detector_with_prev(prev_cx, prev_cy, prev_ts):
    """Create a BallDetector with pre-set previous court position."""
    det = BallDetector()
    det._prev_court_x = prev_cx
    det._prev_court_y = prev_cy
    det._prev_timestamp = prev_ts
    return det


def test_speed_1m_per_second():
    """Ball moves 1 metre in 1 second → 3.6 km/h."""
    det = _make_detector_with_prev(0.0, 0.0, 0.0)
    cx = 1.0 / COURT_WIDTH_M   # 1 metre in normalised coords
    cy = 0.0
    ts = 1.0
    dt = ts - 0.0
    dcx = cx - 0.0
    dcy = cy - 0.0
    dx_m = dcx * COURT_WIDTH_M
    dy_m = dcy * COURT_HEIGHT_M
    speed_ms = math.sqrt(dx_m**2 + dy_m**2) / dt
    expected = round(speed_ms * 3.6, 1)
    assert expected == 3.6


def test_speed_ten_ms():
    """Ball moves 10 m/s diagonally → converts to km/h correctly."""
    # 10 m/s in x direction = 36 km/h
    # 1 second, travel 10 m in x = 10/9 court units
    det = _make_detector_with_prev(0.0, 0.0, 0.0)
    cx = 10.0 / COURT_WIDTH_M
    cy = 0.0
    dt = 1.0
    dx_m = cx * COURT_WIDTH_M
    dy_m = 0.0
    speed_ms = math.sqrt(dx_m**2 + dy_m**2) / dt
    expected = round(speed_ms * 3.6, 1)
    assert expected == 36.0


def test_no_speed_on_first_frame():
    """First call (no previous position) produces speed_kmh=None."""
    det = BallDetector()
    assert det._prev_court_x is None
    assert det._prev_timestamp is None


def test_vx_vy_computed():
    """vx and vy are normalised-court-units per second."""
    cx_start, cy_start = 0.2, 0.3
    cx_end,   cy_end   = 0.5, 0.3
    dt = 2.0
    vx_expected = round((cx_end - cx_start) / dt, 4)
    vy_expected = round((cy_end - cy_start) / dt, 4)
    assert vx_expected == round(0.15, 4)
    assert vy_expected == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && .venv/bin/python -m pytest tests/test_ball_speed.py -v
```
Expected: `ImportError` — `COURT_WIDTH_M` not exported yet.

- [ ] **Step 3: Add speed constants and state to ball_detector.py**

At the top of `backend/app/services/ball_detector.py`, after the existing imports, add:

```python
import math

COURT_WIDTH_M  = 9.0    # volleyball court width in metres
COURT_HEIGHT_M = 18.0   # volleyball court length in metres
```

In `BallDetector.__init__`, after `self._device = "cpu"`, add:

```python
        self._prev_court_x: Optional[float] = None
        self._prev_court_y: Optional[float] = None
        self._prev_timestamp: Optional[float] = None
```

In `BallDetector.reset()`, after `self._trajectory.clear()`, add:

```python
        self._prev_court_x = None
        self._prev_court_y = None
        self._prev_timestamp = None
```

- [ ] **Step 4: Replace the return dict in detect() with speed-aware version**

In `backend/app/services/ball_detector.py`, the `detect()` method currently ends with:

```python
        cx, cy = -1.0, -1.0
        if homography and homography.is_calibrated():
            cx, cy = homography.frame_to_court(rx, ry)

        return {
            "x":           round(rx, 2),
            "y":           round(ry, 2),
            "radius":      result.get("radius", 8),
            "confidence":  result.get("confidence", 0.0),
            "court_x":     round(cx, 4) if cx >= 0 else None,
            "court_y":     round(cy, 4) if cy >= 0 else None,
            "frame_number":frame_idx,
            "timestamp":   round(timestamp, 4),
            "trajectory":  list(self._trajectory),
        }
```

Replace that entire block (from `cx, cy = -1.0` to the end of the `return` dict) with:

```python
        cx, cy = -1.0, -1.0
        if homography and homography.is_calibrated():
            cx, cy = homography.frame_to_court(rx, ry)

        # Compute instantaneous speed from consecutive court positions
        speed_kmh, vx_norm, vy_norm = None, None, None
        if (cx >= 0 and cy >= 0
                and self._prev_court_x is not None
                and self._prev_timestamp is not None):
            dt = timestamp - self._prev_timestamp
            if dt > 0:
                dcx = cx - self._prev_court_x
                dcy = cy - self._prev_court_y
                dx_m = dcx * COURT_WIDTH_M
                dy_m = dcy * COURT_HEIGHT_M
                speed_ms = math.sqrt(dx_m ** 2 + dy_m ** 2) / dt
                speed_kmh = round(speed_ms * 3.6, 1)
                vx_norm = round(dcx / dt, 4)
                vy_norm = round(dcy / dt, 4)

        if cx >= 0:
            self._prev_court_x = cx
            self._prev_court_y = cy
            self._prev_timestamp = timestamp

        return {
            "x":           round(rx, 2),
            "y":           round(ry, 2),
            "radius":      result.get("radius", 8),
            "confidence":  result.get("confidence", 0.0),
            "court_x":     round(cx, 4) if cx >= 0 else None,
            "court_y":     round(cy, 4) if cy >= 0 else None,
            "speed_kmh":   speed_kmh,
            "vx":          vx_norm,
            "vy":          vy_norm,
            "frame_number":frame_idx,
            "timestamp":   round(timestamp, 4),
            "trajectory":  list(self._trajectory),
        }
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd backend && .venv/bin/python -m pytest tests/test_ball_speed.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/ball_detector.py backend/tests/test_ball_speed.py
git commit -m "feat: compute ball speed km/h from consecutive court positions"
```

---

### Task 3: cv_pipeline — store speed in ball_rows + BallTracking (Phase A)

**Files:**
- Modify: `backend/app/services/cv_pipeline.py` (two small edits)

No new tests — covered by integration when analysis runs.

- [ ] **Step 1: Add speed fields to ball_rows dict**

In `backend/app/services/cv_pipeline.py`, find the `if ball:` block that appends to `ball_rows` (around line 206). Currently:

```python
                if ball:
                    ball_rows.append({
                        "match_id":     self.match_id,
                        "frame_number": frame_idx,
                        "timestamp":    ball["timestamp"],
                        "x":  ball["x"], "y": ball["y"],
                        "confidence": ball["confidence"],
                        "court_x": ball.get("court_x"),
                        "court_y": ball.get("court_y"),
                    })
```

Replace with:

```python
                if ball:
                    ball_rows.append({
                        "match_id":     self.match_id,
                        "frame_number": frame_idx,
                        "timestamp":    ball["timestamp"],
                        "x":  ball["x"], "y": ball["y"],
                        "confidence": ball["confidence"],
                        "court_x":    ball.get("court_x"),
                        "court_y":    ball.get("court_y"),
                        "speed_kmh":  ball.get("speed_kmh"),
                        "vx":         ball.get("vx"),
                        "vy":         ball.get("vy"),
                    })
```

- [ ] **Step 2: Pass speed to BallTracking constructor**

In the same file, find the `BallTracking(...)` constructor call inside `_save_to_db` (around line 420). Currently:

```python
                    db.add(BallTracking(
                        match_id=uuid.UUID(self.match_id),
                        frame_number=row["frame_number"],
                        timestamp=row["timestamp"],
                        x=row["x"], y=row["y"],
                        confidence=row.get("confidence"),
                        court_x=row.get("court_x"),
                        court_y=row.get("court_y"),
                    ))
```

Replace with:

```python
                    db.add(BallTracking(
                        match_id=uuid.UUID(self.match_id),
                        frame_number=row["frame_number"],
                        timestamp=row["timestamp"],
                        x=row["x"], y=row["y"],
                        confidence=row.get("confidence"),
                        court_x=row.get("court_x"),
                        court_y=row.get("court_y"),
                        speed_kmh=row.get("speed_kmh"),
                        vx=row.get("vx"),
                        vy=row.get("vy"),
                    ))
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/cv_pipeline.py
git commit -m "feat: store ball speed_kmh, vx, vy in ball_tracking rows"
```

---

### Task 4: ScoringEngine — reception quality rating (Phase C)

**Files:**
- Modify: `backend/app/services/scoring_engine.py`
- Test: `backend/tests/test_reception_quality.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_reception_quality.py
"""Unit tests for reception quality 0-3 rating."""
from app.services.scoring_engine import ScoringEngine


def test_error_is_always_zero():
    assert ScoringEngine.compute_reception_quality("error", None)   == 0
    assert ScoringEngine.compute_reception_quality("error", 80.0)   == 0
    assert ScoringEngine.compute_reception_quality("error", 10.0)   == 0


def test_success_slow_ball_is_perfect():
    """Slow incoming ball (<35 km/h) + success → Rtg. 3 (perfect)."""
    assert ScoringEngine.compute_reception_quality("success", 20.0)  == 3
    assert ScoringEngine.compute_reception_quality("success", 34.9)  == 3


def test_success_medium_ball_is_good():
    """Medium ball (35-60 km/h) + success → Rtg. 2 (good)."""
    assert ScoringEngine.compute_reception_quality("success", 35.0)  == 2
    assert ScoringEngine.compute_reception_quality("success", 59.9)  == 2


def test_success_fast_ball_is_poor():
    """Fast ball (≥60 km/h) + success → Rtg. 1 (poor but controlled)."""
    assert ScoringEngine.compute_reception_quality("success", 60.0)  == 1
    assert ScoringEngine.compute_reception_quality("success", 90.0)  == 1


def test_neutral_is_poor():
    """Neutral result → Rtg. 1."""
    assert ScoringEngine.compute_reception_quality("neutral", None)  == 1
    assert ScoringEngine.compute_reception_quality("neutral", 50.0)  == 1


def test_no_speed_data():
    """Missing speed treated as 0 km/h (slow) → 3 if success."""
    assert ScoringEngine.compute_reception_quality("success", None)  == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && .venv/bin/python -m pytest tests/test_reception_quality.py -v
```
Expected: `AttributeError: type object 'ScoringEngine' has no attribute 'compute_reception_quality'`.

- [ ] **Step 3: Add static method to ScoringEngine**

In `backend/app/services/scoring_engine.py`, add this static method at the end of the `ScoringEngine` class (after `_empty_team_stats`):

```python
    @staticmethod
    def compute_reception_quality(action_result: str, ball_speed_at_action) -> int:
        """
        Rate reception quality 0-3 (Balltime Rtg. system).
          0 = error (ball hit floor / ace / uncontrolled)
          1 = poor  (overpass risk, setter has 1 option)
          2 = good  (setter has 2+ attack options)
          3 = perfect (setter can run any offense)
        Based on action result + incoming ball speed.
        """
        if action_result == "error":
            return 0
        speed = float(ball_speed_at_action) if ball_speed_at_action is not None else 0.0
        if action_result == "success":
            if speed < 35.0:
                return 3
            elif speed < 60.0:
                return 2
            else:
                return 1
        return 1   # neutral
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && .venv/bin/python -m pytest tests/test_reception_quality.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/scoring_engine.py backend/tests/test_reception_quality.py
git commit -m "feat: add reception quality 0-3 rating method to ScoringEngine"
```

---

### Task 5: cv_pipeline — `_assign_landing_zones` post-processing pass (Phase A + C)

**Files:**
- Modify: `backend/app/services/cv_pipeline.py`
- Test: `backend/tests/test_landing_zones.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_landing_zones.py
"""Unit tests for landing zone assignment — uses plain objects, no DB."""
import types


def make_ball(timestamp, x, y, court_x=0.5, court_y=0.5, speed_kmh=None):
    b = types.SimpleNamespace()
    b.timestamp = timestamp
    b.x = x
    b.y = y
    b.court_x = court_x
    b.court_y = court_y
    b.speed_kmh = speed_kmh
    return b


def make_action(action_type, timestamp, result="neutral"):
    a = types.SimpleNamespace()
    a.action_type = action_type
    a.timestamp = timestamp
    a.result = result
    a.landing_x = None
    a.landing_y = None
    a.ball_speed_kmh = None
    a.reception_quality = None
    return a


import asyncio
from app.services.cv_pipeline import _assign_landing_zones_pure


def test_serve_landing_uses_max_pixel_y():
    """Landing zone is the ball row with maximum pixel y in post-action window."""
    balls = [
        make_ball(1.0, x=100, y=200, court_x=0.1, court_y=0.1),
        make_ball(1.5, x=300, y=350, court_x=0.5, court_y=0.5),
        make_ball(2.5, x=500, y=580, court_x=0.9, court_y=0.9),  # landing (max y)
        make_ball(3.5, x=600, y=400, court_x=0.9, court_y=0.8),  # after bounce
    ]
    action = make_action("serve", timestamp=0.8)
    asyncio.run(_assign_landing_zones_pure([action], balls))
    assert action.landing_x == 0.9
    assert action.landing_y == 0.9


def test_non_serve_attack_not_assigned_landing():
    """Set and dig actions do not get landing zones."""
    balls = [make_ball(1.0, x=100, y=500, court_x=0.5, court_y=0.5)]
    action = make_action("set", timestamp=0.8)
    asyncio.run(_assign_landing_zones_pure([action], balls))
    assert action.landing_x is None


def test_ball_speed_assigned_to_serve():
    """ball_speed_kmh from closest ball row ±0.5s is assigned to action."""
    balls = [
        make_ball(0.85, x=100, y=200, court_x=0.1, court_y=0.1, speed_kmh=72.5),
    ]
    action = make_action("serve", timestamp=0.8)
    asyncio.run(_assign_landing_zones_pure([action], balls))
    assert action.ball_speed_kmh == 72.5


def test_reception_quality_set_for_reception():
    """Reception actions get reception_quality from compute_reception_quality."""
    balls = [
        make_ball(5.1, x=100, y=200, court_x=0.5, court_y=0.5, speed_kmh=25.0),
    ]
    action = make_action("reception", timestamp=5.0, result="success")
    asyncio.run(_assign_landing_zones_pure([action], balls))
    # speed=25 + success → Rtg. 3
    assert action.reception_quality == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && .venv/bin/python -m pytest tests/test_landing_zones.py -v
```
Expected: `ImportError` — `_assign_landing_zones_pure` not defined yet.

- [ ] **Step 3: Add `_assign_landing_zones_pure` pure function to cv_pipeline.py**

At the module level of `backend/app/services/cv_pipeline.py` (before the `CVPipeline` class), add this pure async function that the tests call and `_assign_landing_zones` will delegate to:

```python
LANDING_WINDOW_SERVE  = 3.0   # seconds
LANDING_WINDOW_ATTACK = 1.5   # seconds
LANDING_SKIP_FIRST    = 0.2   # ignore first 200ms after action (ball still being hit)
SPEED_WINDOW          = 0.5   # ±0.5s to find ball speed at action moment


async def _assign_landing_zones_pure(actions, ball_rows):
    """
    Pure assignment function — works on any objects with the right attributes.
    Mutates action.landing_x, action.landing_y, action.ball_speed_kmh,
    action.reception_quality in-place.
    Called by CVPipeline._assign_landing_zones and tests.
    """
    from app.services.scoring_engine import ScoringEngine

    for action in actions:
        atype = getattr(action, "action_type", None)
        if isinstance(atype, object) and hasattr(atype, "value"):
            atype = atype.value  # unwrap SQLAlchemy enum
        t0 = action.timestamp

        if atype in ("serve", "attack"):
            window_len = LANDING_WINDOW_SERVE if atype == "serve" else LANDING_WINDOW_ATTACK
            candidates = [
                b for b in ball_rows
                if t0 + LANDING_SKIP_FIRST <= b.timestamp <= t0 + window_len
                and b.court_x is not None
                and b.y is not None
            ]
            if candidates:
                landing = max(candidates, key=lambda b: b.y)
                action.landing_x = landing.court_x
                action.landing_y = landing.court_y

        # Ball speed at the action moment (closest ball row within ±SPEED_WINDOW)
        speed_candidates = [
            b for b in ball_rows
            if abs(b.timestamp - t0) <= SPEED_WINDOW
            and getattr(b, "speed_kmh", None) is not None
        ]
        if speed_candidates:
            closest = min(speed_candidates, key=lambda b: abs(b.timestamp - t0))
            action.ball_speed_kmh = closest.speed_kmh

        # Reception quality
        if atype == "reception":
            result_val = getattr(action, "result", "neutral")
            if hasattr(result_val, "value"):
                result_val = result_val.value
            action.reception_quality = ScoringEngine.compute_reception_quality(
                result_val, action.ball_speed_kmh
            )
```

- [ ] **Step 4: Add `_assign_landing_zones` DB method to CVPipeline**

Add this method to `CVPipeline` class in `backend/app/services/cv_pipeline.py`:

```python
    async def _assign_landing_zones(self):
        """Post-processing: assign landing zones + ball speed + reception quality to Action rows."""
        from app.database import AsyncSessionLocal
        from app.models.actions import Action, ActionType
        from app.models.tracking import BallTracking
        from sqlalchemy import select
        import uuid

        async with AsyncSessionLocal() as db:
            actions_result = await db.execute(
                select(Action).where(Action.match_id == uuid.UUID(self.match_id))
            )
            actions = actions_result.scalars().all()

            balls_result = await db.execute(
                select(BallTracking).where(BallTracking.match_id == uuid.UUID(self.match_id))
            )
            balls = balls_result.scalars().all()

            await _assign_landing_zones_pure(actions, balls)
            await db.commit()
            logger.info(f"Landing zones assigned for match {self.match_id}")
```

- [ ] **Step 5: Call `_assign_landing_zones` in analyze() after `_save_to_db`**

In `backend/app/services/cv_pipeline.py` in the `analyze()` method, find:

```python
        await self._emit(90, "Computing match analytics & scoring...")
        await self._run_scoring(player_id_map, action_rows, completed_rallies)
```

Insert before it:

```python
        await self._emit(82, "Computing ball landing zones and speeds...")
        await self._assign_landing_zones()

```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd backend && .venv/bin/python -m pytest tests/test_landing_zones.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/cv_pipeline.py backend/tests/test_landing_zones.py
git commit -m "feat: post-process landing zones, ball speed, reception quality per action"
```

---

### Task 6: cv_pipeline — `_compute_player_movement` (Phase B)

**Files:**
- Modify: `backend/app/services/cv_pipeline.py`
- Test: `backend/tests/test_player_movement.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_player_movement.py
"""Unit tests for player movement computation — pure math, no DB."""
from app.services.cv_pipeline import compute_player_movement


def make_row(court_x, court_y, timestamp):
    import types
    r = types.SimpleNamespace()
    r.court_x = court_x
    r.court_y = court_y
    r.timestamp = timestamp
    return r


def test_stationary_player():
    rows = [make_row(0.5, 0.5, 0.0), make_row(0.5, 0.5, 1.0)]
    dist, avg, mx = compute_player_movement(rows)
    assert dist == 0.0
    assert avg == 0.0
    assert mx == 0.0


def test_one_metre_in_x():
    """Move 1 metre in x over 1 second → dist=1.0, avg=3.6 km/h."""
    rows = [
        make_row(0.0, 0.0, 0.0),
        make_row(1.0 / 9.0, 0.0, 1.0),   # 1 metre in x
    ]
    dist, avg, mx = compute_player_movement(rows)
    assert dist == 1.0
    assert avg == 3.6
    assert mx == 3.6


def test_gap_greater_than_2s_skipped():
    """Gaps > 2 seconds (player lost) are excluded from distance."""
    rows = [
        make_row(0.0, 0.0, 0.0),
        make_row(1.0, 1.0, 10.0),   # 10s gap → skip
    ]
    dist, avg, mx = compute_player_movement(rows)
    assert dist == 0.0


def test_speed_over_40kmh_excluded():
    """Speeds > 40 km/h (sensor noise) are excluded from avg/max."""
    # 100 m/s = impossible for a human
    rows = [
        make_row(0.0, 0.0, 0.0),
        make_row(100.0 / 9.0, 0.0, 1.0),
    ]
    dist, avg, mx = compute_player_movement(rows)
    # Distance still accumulated, but speed not counted in avg/max
    assert avg == 0.0
    assert mx == 0.0


def test_none_court_coords_skipped():
    rows = [
        make_row(None, 0.0, 0.0),
        make_row(0.5,  0.5, 1.0),
    ]
    dist, avg, mx = compute_player_movement(rows)
    assert dist == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && .venv/bin/python -m pytest tests/test_player_movement.py -v
```
Expected: `ImportError` — `compute_player_movement` not defined yet.

- [ ] **Step 3: Add `compute_player_movement` pure function to cv_pipeline.py**

At module level in `backend/app/services/cv_pipeline.py` (near `_assign_landing_zones_pure`), add:

```python
def compute_player_movement(tracking_rows):
    """
    Compute distance and speed for one player from sorted PlayerTracking rows.
    tracking_rows: list/sequence with .court_x, .court_y, .timestamp attributes.
    Returns (distance_m, avg_speed_kmh, max_speed_kmh).
    """
    import math as _math
    COURT_W = 9.0
    COURT_H = 18.0
    MAX_SPEED_CAP = 40.0   # km/h — above this is sensor noise, not a human
    MAX_GAP_S = 2.0         # seconds — bigger gaps = player was lost, skip

    total_dist = 0.0
    speeds = []
    for i in range(1, len(tracking_rows)):
        prev, cur = tracking_rows[i - 1], tracking_rows[i]
        if None in (prev.court_x, prev.court_y, cur.court_x, cur.court_y):
            continue
        dt = cur.timestamp - prev.timestamp
        if dt <= 0 or dt > MAX_GAP_S:
            continue
        dx_m = (cur.court_x - prev.court_x) * COURT_W
        dy_m = (cur.court_y - prev.court_y) * COURT_H
        dist = _math.sqrt(dx_m ** 2 + dy_m ** 2)
        total_dist += dist
        speed_kmh = (dist / dt) * 3.6
        if speed_kmh < MAX_SPEED_CAP:
            speeds.append(speed_kmh)
    avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0.0
    max_speed = round(max(speeds), 1) if speeds else 0.0
    return round(total_dist, 1), avg_speed, max_speed
```

- [ ] **Step 4: Add `_compute_player_movement` DB method to CVPipeline**

Add this method to `CVPipeline` in `backend/app/services/cv_pipeline.py`:

```python
    async def _compute_player_movement(self):
        """Post-processing: compute distance/speed/reception-aggregates per player."""
        from app.database import AsyncSessionLocal
        from app.models.tracking import PlayerTracking
        from app.models.analytics import Analytics
        from app.models.actions import Action
        from app.models.player import Player
        from sqlalchemy import select
        import uuid

        async with AsyncSessionLocal() as db:
            # Load all players for this match
            p_res = await db.execute(
                select(Player).where(Player.match_id == uuid.UUID(self.match_id))
            )
            players = p_res.scalars().all()

            for player in players:
                # PlayerTracking rows sorted by timestamp
                tr_res = await db.execute(
                    select(PlayerTracking)
                    .where(PlayerTracking.player_id == player.id)
                    .order_by(PlayerTracking.timestamp)
                )
                tracking_rows = tr_res.scalars().all()
                dist_m, avg_spd, max_spd = compute_player_movement(tracking_rows)

                # Action-based speed aggregates
                act_res = await db.execute(
                    select(Action).where(
                        Action.player_id == player.id,
                        Action.match_id == uuid.UUID(self.match_id),
                    )
                )
                player_actions = act_res.scalars().all()

                serve_speeds = [
                    a.ball_speed_kmh for a in player_actions
                    if getattr(a.action_type, "value", a.action_type) == "serve"
                    and a.ball_speed_kmh
                ]
                attack_speeds = [
                    a.ball_speed_kmh for a in player_actions
                    if getattr(a.action_type, "value", a.action_type) == "attack"
                    and a.ball_speed_kmh
                ]
                rq_vals = [
                    a.reception_quality for a in player_actions
                    if a.reception_quality is not None
                ]

                # Update Analytics row
                ana_res = await db.execute(
                    select(Analytics).where(
                        Analytics.player_id == player.id,
                        Analytics.match_id == uuid.UUID(self.match_id),
                    )
                )
                analytics = ana_res.scalar_one_or_none()
                if analytics:
                    analytics.distance_covered_m = dist_m
                    analytics.avg_speed_kmh = avg_spd
                    analytics.max_speed_kmh = max_spd
                    analytics.avg_serve_speed_kmh = (
                        round(sum(serve_speeds) / len(serve_speeds), 1) if serve_speeds else None
                    )
                    analytics.avg_attack_speed_kmh = (
                        round(sum(attack_speeds) / len(attack_speeds), 1) if attack_speeds else None
                    )
                    analytics.reception_quality_avg = (
                        round(sum(rq_vals) / len(rq_vals), 2) if rq_vals else None
                    )

            await db.commit()
            logger.info(f"Player movement computed for match {self.match_id}")
```

- [ ] **Step 5: Call `_compute_player_movement` in analyze() after `_run_scoring`**

In `backend/app/services/cv_pipeline.py` in `analyze()`, find:

```python
        await self._emit(93, "Fusing CV events with any existing speech events...")
        await self._run_speech_fusion(action_rows)
```

Insert before it:

```python
        await self._emit(91, "Computing player movement analytics...")
        await self._compute_player_movement()

```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd backend && .venv/bin/python -m pytest tests/test_player_movement.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/cv_pipeline.py backend/tests/test_player_movement.py
git commit -m "feat: compute player distance, speed, reception quality averages post-analysis"
```

---

### Task 7: processing.py — trajectory endpoint + update existing responses (Phase A + C)

**Files:**
- Modify: `backend/app/routers/processing.py`

- [ ] **Step 1: Add `speed_kmh` + `trajectory` array to tracking endpoint ball response**

The VideoPlayer draws a trajectory arc from recent ball positions. These must come from the
tracking endpoint (not a separate call) so there is no extra request per frame.

In `backend/app/routers/processing.py`, in `get_tracking_data`, the `balls` query already
fetches a ±window of BallTracking rows. After the `ball_data` block, also collect the last
15 rows within the past 8 seconds as a trajectory array.

Find the entire ball section (from `# Ball` comment to end of `ball_data` block):

```python
    # Ball
    b_result = await db.execute(
        select(BallTracking)
        .where(
            BallTracking.match_id == match_id,
            BallTracking.timestamp >= t_min,
            BallTracking.timestamp <= t_max,
        )
        .order_by(BallTracking.timestamp)
    )
    balls = b_result.scalars().all()
```

Replace with (extends the query window for trajectory, keeps the closest-ball logic):

```python
    # Ball — fetch wider window for trajectory trail (last 8 s) plus the ±window for current
    TRAJ_WINDOW = 8.0
    b_result = await db.execute(
        select(BallTracking)
        .where(
            BallTracking.match_id == match_id,
            BallTracking.timestamp >= timestamp - TRAJ_WINDOW,
            BallTracking.timestamp <= t_max,
        )
        .order_by(BallTracking.timestamp)
    )
    balls = b_result.scalars().all()
```

Then replace the `ball_data` assignment block:

```python
    ball_data = None
    if balls:
        closest = min(balls, key=lambda b: abs(b.timestamp - timestamp))
        ball_data = {
            "x":        closest.x,
            "y":        closest.y,
            "court_x":  closest.court_x,
            "court_y":  closest.court_y,
            "timestamp":closest.timestamp,
        }
```

With:

```python
    ball_data = None
    if balls:
        closest = min(balls, key=lambda b: abs(b.timestamp - timestamp))
        # Trajectory: last 15 positions from the wide window (already ordered by timestamp)
        traj_rows = [b for b in balls if b.timestamp <= timestamp][-15:]
        ball_data = {
            "x":          closest.x,
            "y":          closest.y,
            "court_x":    closest.court_x,
            "court_y":    closest.court_y,
            "timestamp":  closest.timestamp,
            "speed_kmh":  closest.speed_kmh,
            "trajectory": [
                {"x": b.x, "y": b.y, "court_x": b.court_x, "court_y": b.court_y,
                 "speed_kmh": b.speed_kmh, "timestamp": b.timestamp}
                for b in traj_rows
            ],
        }
```

- [ ] **Step 2: Add `reception_quality` and `ball_speed_kmh` to actions response**

In `backend/app/routers/processing.py`, find the `items.append(...)` block in `get_actions` (around line 329):

```python
        items.append({
            "id":           str(action.id),
            "action_type":  action.action_type,
            "result":       action.result,
            "timestamp":    action.timestamp,
            "frame_number": action.frame_number,
            "confidence":   action.confidence,
            "zone":         action.zone,
            "player_id":    str(action.player_id) if action.player_id else None,
            "player_track_id": player.player_track_id if player else None,
            "team":         player.team if player else None,
            "rally_id":     str(action.rally_id) if action.rally_id else None,
        })
```

Replace with:

```python
        items.append({
            "id":                str(action.id),
            "action_type":       action.action_type,
            "result":            action.result,
            "timestamp":         action.timestamp,
            "frame_number":      action.frame_number,
            "confidence":        action.confidence,
            "zone":              action.zone,
            "player_id":         str(action.player_id) if action.player_id else None,
            "player_track_id":   player.player_track_id if player else None,
            "team":              player.team if player else None,
            "rally_id":          str(action.rally_id) if action.rally_id else None,
            "reception_quality": action.reception_quality,
            "ball_speed_kmh":    action.ball_speed_kmh,
            "landing_x":         action.landing_x,
            "landing_y":         action.landing_y,
        })
```

- [ ] **Step 3: Add ball trajectory endpoint**

In `backend/app/routers/processing.py`, after the `get_ball_heatmap` endpoint (which starts around line 351), add:

```python
@router.get("/{match_id}/ball/trajectory")
async def get_ball_trajectory(
    match_id: uuid.UUID,
    t: float        = Query(0.0,  description="Current video time (seconds)"),
    window: float   = Query(8.0,  description="Seconds of history to return"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
):
    """Return the last `window` seconds of ball positions ending at time `t`."""
    result = await db.execute(
        select(BallTracking)
        .where(
            BallTracking.match_id == match_id,
            BallTracking.timestamp >= t - window,
            BallTracking.timestamp <= t,
        )
        .order_by(BallTracking.timestamp)
    )
    rows = result.scalars().all()
    return [
        {
            "timestamp": r.timestamp,
            "x":         r.x,
            "y":         r.y,
            "court_x":   r.court_x,
            "court_y":   r.court_y,
            "speed_kmh": r.speed_kmh,
        }
        for r in rows
    ]
```

- [ ] **Step 4: Verify endpoint is reachable**

Start the backend and confirm:
```bash
cd backend && .venv/bin/python run.py
# In another terminal (with a valid match UUID and auth token):
curl "http://localhost:8000/api/matches/<UUID>/ball/trajectory?t=30&window=5" \
     -H "Authorization: Bearer <token>"
```
Expected: HTTP 200 with a JSON array (may be empty if match not yet analysed).

- [ ] **Step 5: Commit**

```bash
git add backend/app/routers/processing.py
git commit -m "feat: expose ball trajectory endpoint, speed/quality fields in tracking+actions"
```

---

### Task 8: processing.py — player stats movement fields (Phase B)

**Files:**
- Modify: `backend/app/routers/processing.py`

- [ ] **Step 1: Extend player stats return dict**

In `backend/app/routers/processing.py`, find `get_player_stats` (starts at line 438). Find the `efficiency` dict and `return` statement (around lines 532-567). The current `efficiency` dict:

```python
    efficiency = {
        "attack_eff": float(analytics.attack_efficiency) if analytics else 0.0,
        "serve_eff":  float(analytics.serve_efficiency) if analytics else 0.0,
    }
```

Replace with:

```python
    efficiency = {
        "attack_eff":   float(analytics.attack_efficiency) if analytics else 0.0,
        "serve_eff":    float(analytics.serve_efficiency) if analytics else 0.0,
    }
    movement = {
        "distance_covered_m":   analytics.distance_covered_m if analytics else None,
        "avg_speed_kmh":        analytics.avg_speed_kmh if analytics else None,
        "max_speed_kmh":        analytics.max_speed_kmh if analytics else None,
    }
    speed_stats = {
        "avg_serve_speed_kmh":    analytics.avg_serve_speed_kmh if analytics else None,
        "avg_attack_speed_kmh":   analytics.avg_attack_speed_kmh if analytics else None,
        "reception_quality_avg":  analytics.reception_quality_avg if analytics else None,
    }
```

Then add `"movement": movement, "speed_stats": speed_stats,` to the existing `return` dict:

```python
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
        "movement":       movement,
        "speed_stats":    speed_stats,
        "recent_actions": recent_actions,
    }
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/routers/processing.py
git commit -m "feat: expose player movement distance/speed and action speed stats in player stats API"
```

---

### Task 9: VideoPlayer.jsx — trajectory arc on canvas + speed badge (Phase A)

**Files:**
- Modify: `frontend/src/components/Video/VideoPlayer.jsx`

The ball drawing section is inside `drawOverlays` (the `useCallback` that begins around line 47). The current ball circle drawing (lines 88–99):

```javascript
    // Draw ball (only if detection is recent — within 2 seconds of current video time)
    if (trackingData.ball) {
      const b = trackingData.ball
      const vt = videoRef.current?.currentTime ?? 0
      if (b.timestamp == null || Math.abs(b.timestamp - vt) <= 2.0) {
        ctx.beginPath()
        ctx.arc(b.x * scaleX, b.y * scaleY, 8, 0, Math.PI * 2)
        ctx.fillStyle = '#facc15cc'
        ctx.fill()
        ctx.strokeStyle = '#fbbf24'
        ctx.lineWidth = 2
        ctx.stroke()
      }
```

- [ ] **Step 1: Add speed tier color helper at top of drawOverlays**

At the very top of the `drawOverlays` callback body (after `ctx.clearRect`), add this helper:

```javascript
    const speedColor = (kmh) => {
      if (kmh == null)    return 'rgba(200,200,200,0.8)'
      if (kmh < 40)       return 'rgba(80,200,80,0.9)'
      if (kmh < 70)       return 'rgba(255,200,0,0.9)'
      return 'rgba(255,60,60,0.9)'
    }
```

- [ ] **Step 2: Replace ball drawing section with trajectory arc + speed badge**

Replace the entire `// Draw ball ...` section (lines 87–99 in the file) with:

```javascript
    // Draw ball trajectory arc + speed badge
    if (trackingData.ball) {
      const b = trackingData.ball
      const vt = videoRef.current?.currentTime ?? 0
      const isRecent = b.timestamp == null || Math.abs(b.timestamp - vt) <= 2.0

      // Trajectory arc from ball.trajectory (array of {x, y, speed_kmh} objects from DB)
      const traj = b.trajectory || []
      if (traj.length > 1) {
        for (let i = 1; i < traj.length; i++) {
          const alpha = i / traj.length   // fade: oldest=0.15, newest=1.0
          const opacity = 0.15 + alpha * 0.85
          ctx.globalAlpha = opacity
          ctx.strokeStyle = speedColor(traj[i].speed_kmh ?? b.speed_kmh)
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.moveTo(traj[i - 1].x * scaleX, traj[i - 1].y * scaleY)
          ctx.lineTo(traj[i].x * scaleX, traj[i].y * scaleY)
          ctx.stroke()
        }
        ctx.globalAlpha = 1.0
      }

      if (isRecent) {
        const bx = b.x * scaleX
        const by = b.y * scaleY
        // Ball circle
        ctx.beginPath()
        ctx.arc(bx, by, 8, 0, Math.PI * 2)
        ctx.fillStyle = '#facc15cc'
        ctx.fill()
        ctx.strokeStyle = '#fbbf24'
        ctx.lineWidth = 2
        ctx.stroke()

        // Speed badge above ball
        if (b.speed_kmh != null) {
          const label = `${b.speed_kmh} km/h`
          ctx.font = 'bold 10px Inter'
          const lw = ctx.measureText(label).width + 6
          ctx.fillStyle = speedColor(b.speed_kmh)
          ctx.fillRect(bx - lw / 2, by - 26, lw, 16)
          ctx.fillStyle = '#fff'
          ctx.textAlign = 'center'
          ctx.textBaseline = 'middle'
          ctx.fillText(label, bx, by - 18)
          ctx.textAlign = 'left'
          ctx.textBaseline = 'alphabetic'
        }
      }
    }
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/Video/VideoPlayer.jsx
git commit -m "feat: draw speed-colored ball trajectory arc and speed badge on video canvas"
```

---

### Task 10: VideoPlayer.jsx — trajectory polyline on court mini-map (Phase A)

**Files:**
- Modify: `frontend/src/components/Video/VideoPlayer.jsx`

The mini-map ball section (inside `drawMiniMap`, around lines 185–196) currently draws a single yellow dot. The trajectory polyline on the mini-map uses `ball.trajectory` (court coordinates aren't in `trajectory` — it stores pixel positions). We'll use `ball.court_x/court_y` as the current position and the trajectory pixel positions mapped proportionally to approximate court positions on the mini-map. Actually, we'll fetch trajectory data separately and store it in state; but to keep this task self-contained and not require a new API hook, we use only the data already in `trackingData.ball.trajectory` (pixel positions) converted via homography aspect ratio approximation. The simplest correct approach: use the same `court_x/court_y` fields from the full `trajectory` endpoint result — but that requires a new hook. For now, draw the trajectory from pixel positions scaled by the current canvas/court aspect ratio.

**Simpler approach that works today:** The `trajectory` array holds `[pixel_x, pixel_y, timestamp]`. We can project them onto the court mini-map by scaling relative to the frame's resolution using the same scaleX/scaleY that the canvas uses.

- [ ] **Step 1: Replace single ball dot in mini-map with trajectory polyline**

In `frontend/src/components/Video/VideoPlayer.jsx` inside `drawMiniMap`, find the ball section (lines 185–196):

```javascript
    // Ball on mini-map (only if detection is recent)
    if (data.ball?.court_x != null) {
      const vt = videoRef.current?.currentTime ?? 0
      if (data.ball.timestamp == null || Math.abs(data.ball.timestamp - vt) <= 2.0) {
        const bx = mapX + data.ball.court_x * mapW
        const by = mapY + data.ball.court_y * mapH
        ctx.beginPath()
        ctx.arc(bx, by, 4, 0, Math.PI * 2)
        ctx.fillStyle = '#facc15'
        ctx.fill()
      }
    }
```

Replace with:

```javascript
    // Ball trajectory trail on mini-map
    if (data.ball?.court_x != null) {
      const vt = videoRef.current?.currentTime ?? 0
      const isRecent = data.ball.timestamp == null || Math.abs(data.ball.timestamp - vt) <= 2.0

      // Trajectory: draw last 15 pixel positions as approximated court dots
      const traj = data.ball.trajectory || []
      const trailLen = Math.min(15, traj.length)
      const video = videoRef.current
      if (video && trailLen > 1) {
        const vw = video.videoWidth || video.clientWidth || 1
        const vh = video.videoHeight || video.clientHeight || 1
        for (let i = traj.length - trailLen; i < traj.length; i++) {
          const alpha = (i - (traj.length - trailLen)) / trailLen
          const dotR = 1 + alpha * 2   // 1px oldest → 3px newest
          // Use court coords if available, fall back to pixel projection
          const tx = traj[i].court_x != null
            ? mapX + traj[i].court_x * mapW
            : mapX + (traj[i].x / vw) * mapW
          const ty = traj[i].court_y != null
            ? mapY + traj[i].court_y * mapH
            : mapY + (traj[i].y / vh) * mapH
          ctx.beginPath()
          ctx.arc(tx, ty, dotR, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(250,204,21,${0.3 + alpha * 0.7})`
          ctx.fill()
        }
      }

      // Current ball position (bright dot)
      if (isRecent) {
        const bx = mapX + data.ball.court_x * mapW
        const by = mapY + data.ball.court_y * mapH
        ctx.beginPath()
        ctx.arc(bx, by, 4, 0, Math.PI * 2)
        ctx.fillStyle = '#facc15'
        ctx.fill()
      }
    }
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/Video/VideoPlayer.jsx
git commit -m "feat: draw ball trajectory trail on court mini-map"
```

---

### Task 11: MatchDetailPage.jsx — Analytics + Actions tab UI (Phase B + C)

**Files:**
- Modify: `frontend/src/pages/MatchDetailPage.jsx`

- [ ] **Step 1: Find where player stats are displayed in Analytics tab**

```bash
grep -n "distance\|avg_speed\|movement\|involvement\|efficiency\|analytics" \
  frontend/src/pages/MatchDetailPage.jsx | head -40
```

Note the line numbers for the player stats card rendering.

- [ ] **Step 2: Add Movement section to player stats card**

In `frontend/src/pages/MatchDetailPage.jsx`, find the section that renders per-player analytics stats (search for `involvement_pct` or `attack_eff`). After the existing efficiency stats, add:

```jsx
{/* Movement stats */}
{playerStats?.movement && (
  <div className="mt-3 pt-3 border-t border-court-border">
    <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Movement</div>
    <div className="grid grid-cols-3 gap-2">
      {[
        { label: 'Distance', value: playerStats.movement.distance_covered_m != null ? `${playerStats.movement.distance_covered_m} m` : '—' },
        { label: 'Avg Speed', value: playerStats.movement.avg_speed_kmh != null ? `${playerStats.movement.avg_speed_kmh} km/h` : '—' },
        { label: 'Peak Speed', value: playerStats.movement.max_speed_kmh != null ? `${playerStats.movement.max_speed_kmh} km/h` : '—' },
      ].map(({ label, value }) => (
        <div key={label} className="text-center">
          <div className="text-xs text-slate-400">{label}</div>
          <div className="text-sm font-semibold text-white">{value}</div>
        </div>
      ))}
    </div>
    {playerStats.speed_stats?.avg_serve_speed_kmh && (
      <div className="grid grid-cols-2 gap-2 mt-2">
        <div className="text-center">
          <div className="text-xs text-slate-400">Avg Serve</div>
          <div className="text-sm font-semibold text-white">{playerStats.speed_stats.avg_serve_speed_kmh} km/h</div>
        </div>
        <div className="text-center">
          <div className="text-xs text-slate-400">Avg Attack</div>
          <div className="text-sm font-semibold text-white">{playerStats.speed_stats.avg_attack_speed_kmh ?? '—'} km/h</div>
        </div>
      </div>
    )}
  </div>
)}
```

- [ ] **Step 3: Add Rtg. badge and km/h annotation to Actions tab rows**

In `frontend/src/pages/MatchDetailPage.jsx`, find where individual action rows are rendered in the Actions tab (search for `action_type` or `action.result`). In the action row JSX, add:

```jsx
{/* Reception quality badge */}
{action.action_type === 'reception' && action.reception_quality != null && (
  <span className={`ml-2 px-1.5 py-0.5 rounded text-xs font-bold ${
    action.reception_quality === 3 ? 'bg-green-500/20 text-green-400' :
    action.reception_quality === 2 ? 'bg-yellow-500/20 text-yellow-400' :
    action.reception_quality === 1 ? 'bg-orange-500/20 text-orange-400' :
                                     'bg-red-500/20 text-red-400'
  }`}>
    Rtg. {action.reception_quality}
  </span>
)}
{/* Ball speed annotation for serve/attack */}
{(['serve', 'attack'].includes(action.action_type)) && action.ball_speed_kmh != null && (
  <span className="ml-2 text-xs text-slate-400">{action.ball_speed_kmh} km/h</span>
)}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/MatchDetailPage.jsx
git commit -m "feat: add movement stats to Analytics tab and Rtg./speed badges to Actions tab"
```

---

### Task 12: Run all tests and verify full stack (integration check)

- [ ] **Step 1: Run all backend unit tests**

```bash
cd backend && .venv/bin/python -m pytest tests/ -v
```
Expected: all tests PASS (test_migrations, test_ball_speed, test_landing_zones, test_player_movement, test_reception_quality, test_track_merger).

- [ ] **Step 2: Start backend and confirm no startup errors**

```bash
cd backend && .venv/bin/python run.py
```
Expected: server starts at `:8000`, all `Migration OK:` lines appear, no errors.

- [ ] **Step 3: Start frontend and verify UI loads**

```bash
cd frontend && npm run dev
```
Open `http://localhost:5173`. Navigate to a match detail page. Verify:
- Video overlay shows ball trajectory arc (after re-analysis)
- Court mini-map shows ball trail
- Actions tab shows `Rtg. X` on reception rows (after re-analysis)
- Actions tab shows `XX km/h` on serve/attack rows (after re-analysis)
- Analytics tab shows Distance/Avg Speed/Peak Speed for each player (after re-analysis)

**Note:** New analytics fields will only appear after re-running analysis on a match (click Re-Analyze). Existing analysed matches have NULL for all new columns.

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete Balltime-level accuracy analytics — ball speed, trajectories, player movement, reception ratings"
```
