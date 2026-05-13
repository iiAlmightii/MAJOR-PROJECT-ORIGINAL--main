# Balltime-Level Accuracy & Analytics Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.  

**Goal:** Close the accuracy and analytics gap between VolleyVision and Balltime AI across three areas: ball trajectory + speed, player movement analytics, and action outcome ratings.

**Architecture:** Three additive phases built in order. Phase A adds new DB columns + pipeline computations + a trajectory API endpoint + frontend arc visualization. Phase B computes player movement (distance, speed) from existing court coords and surfaces them in stats. Phase C adds granular action quality ratings (reception 0–3, serve/attack speed-at-action) by extending the ScoringEngine and existing Action model.

**Tech Stack:** FastAPI async, SQLAlchemy ORM, PostgreSQL incremental migrations (`database.py::_apply_incremental_migrations`), OpenCV + NumPy for math, React canvas for overlay.

**Court real-world dimensions (standard volleyball):**
- Width: 9.0 m (court_x normalized 0→1 = 0→9m)
- Length: 18.0 m (court_y normalized 0→1 = 0→18m)

---

## Phase A — Ball Trajectory + Speed Intelligence

### Files

- Modify: `backend/app/models/tracking.py` — add `speed_kmh`, `vx`, `vy` to `BallTracking`
- Modify: `backend/app/models/actions.py` — add `landing_x`, `landing_y`, `ball_speed_kmh` to `Action`
- Modify: `backend/app/database.py` — add `ALTER TABLE` migrations for new columns
- Modify: `backend/app/services/ball_detector.py` — compute speed from court coords per frame
- Modify: `backend/app/services/cv_pipeline.py` — store speed in DB rows; landing zone post-pass
- Modify: `backend/app/routers/processing.py` — new `GET /{match_id}/ball/trajectory` endpoint
- Modify: `frontend/src/components/Video/VideoPlayer.jsx` — trajectory arc + speed badge on canvas + court map trail

### A1: DB schema — BallTracking new columns

Add to `BallTracking` ORM (tracking.py):
```python
speed_kmh: Mapped[float] = mapped_column(Float, nullable=True)   # instantaneous 2D speed
vx: Mapped[float] = mapped_column(Float, nullable=True)           # court velocity x (norm/s)
vy: Mapped[float] = mapped_column(Float, nullable=True)           # court velocity y (norm/s)
```

Add to `_apply_incremental_migrations` (database.py):
```python
"ALTER TABLE ball_tracking ADD COLUMN IF NOT EXISTS speed_kmh FLOAT",
"ALTER TABLE ball_tracking ADD COLUMN IF NOT EXISTS vx FLOAT",
"ALTER TABLE ball_tracking ADD COLUMN IF NOT EXISTS vy FLOAT",
```

### A2: DB schema — Action new columns

Add to `Action` ORM (actions.py):
```python
landing_x: Mapped[float] = mapped_column(Float, nullable=True)      # court x of ball landing (0-1)
landing_y: Mapped[float] = mapped_column(Float, nullable=True)       # court y of ball landing (0-1)
ball_speed_kmh: Mapped[float] = mapped_column(Float, nullable=True)  # ball speed at action moment
```

Add to `_apply_incremental_migrations`:
```python
"ALTER TABLE actions ADD COLUMN IF NOT EXISTS landing_x FLOAT",
"ALTER TABLE actions ADD COLUMN IF NOT EXISTS landing_y FLOAT",
"ALTER TABLE actions ADD COLUMN IF NOT EXISTS ball_speed_kmh FLOAT",
```

### A3: ball_detector.py — Speed computation

Add constants and state:
```python
import math
COURT_WIDTH_M  = 9.0
COURT_HEIGHT_M = 18.0
```

Add to `BallDetector.__init__`:
```python
self._prev_court_x: Optional[float] = None
self._prev_court_y: Optional[float] = None
self._prev_timestamp: Optional[float] = None
```

Add to `BallDetector.reset()`:
```python
self._prev_court_x = None
self._prev_court_y = None
self._prev_timestamp = None
```

Replace the `return` dict in `detect()` with this extended version:
```python
# Compute instantaneous speed from court coords
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
        speed_ms = math.sqrt(dx_m**2 + dy_m**2) / dt
        speed_kmh = round(speed_ms * 3.6, 1)
        vx_norm = round(dcx / dt, 4)   # normalized court units per second
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

### A4: cv_pipeline.py — Store speed in BallTracking rows

In the section that builds `ball_row` dict for DB insert, add:
```python
"speed_kmh": ball.get("speed_kmh"),
"vx":        ball.get("vx"),
"vy":        ball.get("vy"),
```

### A5: cv_pipeline.py — Landing zone post-processing pass

After the main frame-processing loop completes (before final DB commit), run a landing zone pass.

**Physical reasoning:** In 2D court coordinates, a served or attacked ball travels monotonically from one side to the other — `court_y` never "reverses" during flight, so vy-sign detection is wrong. Instead: in pixel space, a ball descending toward the floor has *increasing* pixel y (camera looks down at the court). The landing frame is the one where pixel y is maximum in the post-action window. That's the lowest physical point, i.e., the landing.

```python
async def _assign_landing_zones(actions_rows, ball_rows):
    """
    actions_rows: list of Action ORM objects for this match
    ball_rows: list of BallTracking ORM objects for this match
    """
    LANDING_WINDOW_SERVE  = 3.0   # serves travel longer
    LANDING_WINDOW_ATTACK = 1.5   # attacks land faster
    SKIP_FIRST = 0.2              # ignore first 200ms (ball still being hit)

    for action in actions_rows:
        if action.action_type not in ("serve", "attack"):
            continue
        t0 = action.timestamp
        window_len = LANDING_WINDOW_SERVE if action.action_type == "serve" else LANDING_WINDOW_ATTACK

        # Collect ball rows in window, skip frames too close to the action moment
        candidates = [
            b for b in ball_rows
            if t0 + SKIP_FIRST <= b.timestamp <= t0 + window_len
            and b.court_x is not None
            and b.x is not None   # pixel x (used as proxy: b.y is pixel y)
        ]
        if not candidates:
            continue

        # Landing = frame with maximum pixel y (ball at lowest point in frame = on floor)
        landing = max(candidates, key=lambda b: b.y)
        action.landing_x = landing.court_x
        action.landing_y = landing.court_y

        # Ball speed at action moment: closest ball row to action.timestamp
        closest = min(
            [b for b in ball_rows
             if abs(b.timestamp - t0) < 0.5 and b.speed_kmh is not None],
            key=lambda b: abs(b.timestamp - t0),
            default=None,
        )
        if closest:
            action.ball_speed_kmh = closest.speed_kmh
```

Call `_assign_landing_zones` at end of analysis, before final commit, passing all Action and BallTracking rows for the match.

### A6: processing.py — Trajectory endpoint

Add new endpoint:

```python
@router.get("/{match_id}/ball/trajectory")
async def get_ball_trajectory(
    match_id: uuid.UUID,
    t: float = 0.0,          # current video time (seconds)
    window: float = 8.0,     # seconds of history to return
    db: AsyncSession = Depends(get_db),
):
    """Return last `window` seconds of ball positions ending at time `t`."""
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
            "timestamp":  r.timestamp,
            "x":          r.x,
            "y":          r.y,
            "court_x":    r.court_x,
            "court_y":    r.court_y,
            "speed_kmh":  r.speed_kmh,
        }
        for r in rows
    ]
```

Also update the existing ball response in `GET /{match_id}/tracking` to include `speed_kmh` from the nearest BallTracking row.

### A7: VideoPlayer.jsx — Trajectory arc on canvas + court map

**Video canvas overlay:** In the ball drawing section, after drawing the ball circle, draw the trajectory arc using the last N positions from `ball.trajectory`. Color each segment by speed tier:
- `speed_kmh < 40` → green `rgb(80,200,80)`
- `40 ≤ speed_kmh < 70` → yellow `rgb(255,200,0)`
- `speed_kmh ≥ 70` → red `rgb(255,60,60)`
- Speed unknown → white `rgb(200,200,200)`

Draw as connected line segments with alpha fading from newest (opacity 1.0) to oldest (opacity 0.15). Line width: 2px.

Show speed badge: if `speed_kmh` is known, draw a small filled label `"${speed_kmh} km/h"` above the ball circle in the matching tier color.

**Court mini-map:** Draw the last 15 ball court positions as a polyline using the same color scheme. Oldest point has radius 1px, newest has radius 3px (graduated).

---

## Phase B — Player Movement Analytics

### Files

- Modify: `backend/app/models/analytics.py` — add `avg_speed_kmh`, `distance_covered_m`, `max_speed_kmh` to `Analytics`
- Modify: `backend/app/database.py` — add `ALTER TABLE` migrations
- Modify: `backend/app/services/cv_pipeline.py` — compute per-player distance/speed in analytics pass
- Modify: `backend/app/routers/processing.py` — include movement stats in player stats response
- Modify: `frontend/src/pages/MatchDetailPage.jsx` — show distance and avg speed in Analytics tab

### B1: DB schema — Analytics new columns

Add to `Analytics` ORM:
```python
distance_covered_m: Mapped[float] = mapped_column(Float, nullable=True)  # total meters moved
avg_speed_kmh: Mapped[float]      = mapped_column(Float, nullable=True)   # avg movement speed
max_speed_kmh: Mapped[float]      = mapped_column(Float, nullable=True)   # peak speed
```

Add to `_apply_incremental_migrations`:
```python
"ALTER TABLE analytics ADD COLUMN IF NOT EXISTS distance_covered_m FLOAT",
"ALTER TABLE analytics ADD COLUMN IF NOT EXISTS avg_speed_kmh FLOAT",
"ALTER TABLE analytics ADD COLUMN IF NOT EXISTS max_speed_kmh FLOAT",
```

### B2: cv_pipeline.py — Compute movement stats

After all frames are processed, for each player, retrieve all their `PlayerTracking` rows ordered by timestamp. Compute:

```python
import math

def compute_player_movement(tracking_rows):
    """
    tracking_rows: list of PlayerTracking sorted by timestamp, with court_x/court_y
    Returns: (distance_m, avg_speed_kmh, max_speed_kmh)
    """
    COURT_W = 9.0
    COURT_H = 18.0
    total_dist = 0.0
    speeds = []
    for i in range(1, len(tracking_rows)):
        prev, cur = tracking_rows[i-1], tracking_rows[i]
        if None in (prev.court_x, prev.court_y, cur.court_x, cur.court_y):
            continue
        dt = cur.timestamp - prev.timestamp
        if dt <= 0 or dt > 2.0:   # skip gaps > 2s (player lost)
            continue
        dx_m = (cur.court_x - prev.court_x) * COURT_W
        dy_m = (cur.court_y - prev.court_y) * COURT_H
        dist = math.sqrt(dx_m**2 + dy_m**2)
        total_dist += dist
        speed_kmh = (dist / dt) * 3.6
        if speed_kmh < 40.0:   # cap at realistic human sprinting speed
            speeds.append(speed_kmh)
    avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0.0
    max_speed = round(max(speeds), 1) if speeds else 0.0
    return round(total_dist, 1), avg_speed, max_speed
```

Store results in the player's `Analytics` row: `distance_covered_m`, `avg_speed_kmh`, `max_speed_kmh`.

### B3: processing.py — Expose in player stats

In `GET /{match_id}/players/{player_id}/stats`, add to the return dict:
```python
"distance_covered_m":  analytics_row.distance_covered_m,
"avg_speed_kmh":       analytics_row.avg_speed_kmh,
"max_speed_kmh":       analytics_row.max_speed_kmh,
```

### B4: MatchDetailPage.jsx — Display movement stats in Analytics tab

In the per-player analytics card, add a "Movement" section:
```
Distance covered: {distance_covered_m} m
Avg speed: {avg_speed_kmh} km/h
Peak speed: {max_speed_kmh} km/h
```

---

## Phase C — Action Outcome Ratings

### Files

- Modify: `backend/app/models/actions.py` — add `reception_quality` to `Action`
- Modify: `backend/app/database.py` — add migration
- Modify: `backend/app/services/scoring_engine.py` — compute reception_quality (0–3 rating)
- Modify: `backend/app/models/analytics.py` — add `reception_quality_avg`
- Modify: `backend/app/services/cv_pipeline.py` — store reception_quality in analytics pass
- Modify: `frontend/src/pages/MatchDetailPage.jsx` — show "Rtg. X" badge on reception actions

### C1: DB schema — reception_quality column

Add to `Action` ORM:
```python
reception_quality: Mapped[int] = mapped_column(Integer, nullable=True)
# 0=error, 1=poor, 2=good, 3=perfect  (only set for reception actions)
```

Add to `Analytics` ORM:
```python
reception_quality_avg: Mapped[float] = mapped_column(Float, nullable=True)
avg_serve_speed_kmh: Mapped[float]   = mapped_column(Float, nullable=True)
avg_attack_speed_kmh: Mapped[float]  = mapped_column(Float, nullable=True)
```

Add to `_apply_incremental_migrations`:
```python
"ALTER TABLE actions ADD COLUMN IF NOT EXISTS reception_quality SMALLINT",
"ALTER TABLE analytics ADD COLUMN IF NOT EXISTS reception_quality_avg FLOAT",
"ALTER TABLE analytics ADD COLUMN IF NOT EXISTS avg_serve_speed_kmh FLOAT",
"ALTER TABLE analytics ADD COLUMN IF NOT EXISTS avg_attack_speed_kmh FLOAT",
```

### C2: scoring_engine.py — Reception quality rating

Add method to ScoringEngine:

```python
@staticmethod
def compute_reception_quality(action_result: str, ball_speed_at_action: Optional[float]) -> int:
    """
    Rate reception quality 0–3 (Balltime Rtg. system).
    0 = ace / error (ball not controlled)
    1 = poor (overpass risk, setter limited to 1 option)
    2 = good (setter has 2+ attack options)
    3 = perfect (setter can run any offense)

    Logic: based on action result + incoming ball speed.
    Fast serves are harder to receive perfectly.
    """
    if action_result == "error":
        return 0
    speed = ball_speed_at_action or 0.0
    if action_result == "success":
        if speed < 35:
            return 3
        elif speed < 60:
            return 2
        else:
            return 1
    # neutral
    return 1
```

Call this in the ScoringEngine action-scoring loop for every `reception` action after `ball_speed_kmh` is assigned. Store result in `action.reception_quality`.

### C3: cv_pipeline.py — Compute speed-based analytics aggregates

In the analytics aggregation pass (where Analytics rows are upserted), add:

```python
# Serve speeds for this player
serve_speeds = [
    a.ball_speed_kmh for a in player_actions
    if a.action_type == "serve" and a.ball_speed_kmh
]
analytics.avg_serve_speed_kmh = round(sum(serve_speeds)/len(serve_speeds), 1) if serve_speeds else None

# Attack speeds
attack_speeds = [
    a.ball_speed_kmh for a in player_actions
    if a.action_type == "attack" and a.ball_speed_kmh
]
analytics.avg_attack_speed_kmh = round(sum(attack_speeds)/len(attack_speeds), 1) if attack_speeds else None

# Reception quality average
rq_vals = [a.reception_quality for a in player_actions if a.reception_quality is not None]
analytics.reception_quality_avg = round(sum(rq_vals)/len(rq_vals), 2) if rq_vals else None
```

### C4: Frontend — "Rtg. X" badge and speed annotations on Actions tab

In the Actions tab action row, for `reception` actions, show `Rtg. {reception_quality}` badge (colors: 0=red, 1=orange, 2=yellow, 3=green).

For `serve` and `attack` actions where `ball_speed_kmh` is set, show `{ball_speed_kmh} km/h` as a secondary label (gray, small).

Expose `reception_quality` and `ball_speed_kmh` in the existing `GET /{match_id}/actions` endpoint response (both fields already exist on the Action model after Phase C1).

---

## Build Order

1. Phase A (ball trajectory + speed) — all backend then frontend
2. Phase B (movement analytics) — depends on existing PlayerTracking data, no new pipeline logic
3. Phase C (action ratings) — depends on `ball_speed_kmh` computed in Phase A

Do not start Phase B or C until Phase A is committed and the DB migration runs cleanly.
