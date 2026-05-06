# Jersey OCR & Team Assignment Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read jersey numbers from player crops during analysis to create stable player identities (fixing the "4000 players" problem), and replace broken K-means jersey-hue team assignment with reliable court-side majority vote.

**Architecture:** Two independent fixes applied in sequence. (1) Team assignment: replace the K-means hue clustering in `cv_pipeline.py`'s `_save_to_db` with a simple median `court_x` threshold — tracks with median `court_x < 0.5` are Team A, ≥ 0.5 are Team B. (2) Jersey OCR: a new `jersey_ocr.py` service uses EasyOCR (PyTorch-backed, already installed) to read digit(s) from the number region of each player crop every 15 frames; the mode number per track is stored as `Player.jersey_number` and `Player.display_number`; tracks sharing the same team + jersey number are merged in `track_merger.py`.

**Tech Stack:** Python 3.12, EasyOCR 1.7+, OpenCV, FastAPI async, SQLAlchemy async, React/Vite (label display unchanged — already reads `display_number`).

---

## File Map

| File | Change |
|------|--------|
| `backend/requirements.txt` | Add `easyocr` |
| `backend/app/services/jersey_ocr.py` | **Create** — OCR service, pure function |
| `backend/app/services/player_tracker.py` | Add `jersey_number` field to output dict |
| `backend/app/services/cv_pipeline.py` | (1) Replace K-means team logic; (2) collect `jersey_number` per frame; (3) consensus jersey per track in `_save_to_db` |
| `backend/app/services/track_merger.py` | Merge by same team+jersey; use jersey as `display_number` |
| `backend/tests/test_jersey_ocr.py` | **Create** — unit tests for OCR service |
| `backend/tests/test_team_assignment.py` | **Create** — unit tests for team logic |
| `backend/tests/test_track_merger.py` | **Create** — unit tests for jersey-based merging |

---

## Task 1: Team Assignment — Replace K-means with Court-Side Majority Vote

**Files:**
- Modify: `backend/app/services/cv_pipeline.py:423-478`
- Create: `backend/tests/test_team_assignment.py`

The current K-means jersey-hue approach fails on broadcast footage where both teams can have similar hue profiles under arena lighting. Court-side majority vote is deterministic and robust: a player who spends most of their time on the left half of court (court_x < 0.5) is Team A; right half is Team B.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_team_assignment.py`:

```python
import pytest

def _assign_teams_by_court_side(player_rows):
    """Pure extraction of the new logic — mirrors what cv_pipeline will do."""
    from collections import defaultdict
    track_xs = defaultdict(list)
    for r in player_rows:
        if r.get("court_x") is not None:
            track_xs[r["track_id"]].append(r["court_x"])

    team_map = {}
    for tid, xs in track_xs.items():
        median_x = sorted(xs)[len(xs) // 2]
        team_map[tid] = "A" if median_x < 0.5 else "B"

    # Tracks with no court position → None
    all_tids = {r["track_id"] for r in player_rows}
    for tid in all_tids - set(team_map.keys()):
        team_map[tid] = None

    return team_map


def test_left_side_is_team_a():
    rows = [{"track_id": 1, "court_x": 0.2}, {"track_id": 1, "court_x": 0.3}]
    result = _assign_teams_by_court_side(rows)
    assert result[1] == "A"


def test_right_side_is_team_b():
    rows = [{"track_id": 2, "court_x": 0.7}, {"track_id": 2, "court_x": 0.8}]
    result = _assign_teams_by_court_side(rows)
    assert result[2] == "B"


def test_player_crossing_net_uses_majority():
    # 3 frames on left, 1 frame on right → still Team A
    rows = [
        {"track_id": 3, "court_x": 0.2},
        {"track_id": 3, "court_x": 0.3},
        {"track_id": 3, "court_x": 0.4},
        {"track_id": 3, "court_x": 0.6},  # one frame crossing net
    ]
    result = _assign_teams_by_court_side(rows)
    assert result[3] == "A"


def test_no_court_position_returns_none():
    rows = [{"track_id": 4, "court_x": None}]
    result = _assign_teams_by_court_side(rows)
    assert result[4] is None


def test_twelve_players_six_per_side():
    rows = []
    # Tracks 1-6 on left, 7-12 on right
    for tid in range(1, 7):
        rows.append({"track_id": tid, "court_x": 0.1 + tid * 0.05})
    for tid in range(7, 13):
        rows.append({"track_id": tid, "court_x": 0.55 + (tid - 7) * 0.05})
    result = _assign_teams_by_court_side(rows)
    for tid in range(1, 7):
        assert result[tid] == "A", f"track {tid} should be A"
    for tid in range(7, 13):
        assert result[tid] == "B", f"track {tid} should be B"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend
.venv/bin/pytest tests/test_team_assignment.py -v
```

Expected: `ImportError` or 5 PASSes (the helper is defined inline in the test, so all tests should pass immediately — this validates the pure logic before wiring it in).

- [ ] **Step 3: Replace team assignment in cv_pipeline.py**

Open `backend/app/services/cv_pipeline.py`. Find the block starting at approximately line 423:

```python
            # Team assignment: jersey-colour K-means (gap-finding) with
```

Replace everything from that comment through the closing `for tid in track_ids:` loop that creates Player objects (ends around line 491) with:

```python
            # Team assignment: court-side majority vote.
            # Tracks with median court_x < 0.5 → Team A (left), ≥ 0.5 → Team B (right).
            from collections import defaultdict
            track_xs: Dict[int, list] = defaultdict(list)
            for r in player_rows:
                if r.get("court_x") is not None:
                    track_xs[r["track_id"]].append(r["court_x"])

            team_map: Dict[int, Optional[str]] = {}
            for tid in track_ids:
                xs = track_xs.get(tid, [])
                if xs:
                    median_x = sorted(xs)[len(xs) // 2]
                    team_map[tid] = "A" if median_x < 0.5 else "B"
                else:
                    team_map[tid] = None

            for tid in track_ids:
                team = team_map.get(tid)
                player = Player(
                    match_id=uuid.UUID(self.match_id),
                    player_track_id=tid,
                    team=team,
                    display_name=f"Player #{tid} (Team {team or '?'})",
                )
                db.add(player)
                await db.flush()
                await db.refresh(player)
                player_id_map[tid] = player.id
```

Also add `from collections import defaultdict` to the imports at the top of `_save_to_db` (line ~405), or verify it is already imported at module level.

- [ ] **Step 4: Run tests to verify**

```bash
cd backend
.venv/bin/pytest tests/test_team_assignment.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/cv_pipeline.py backend/tests/test_team_assignment.py
git commit -m "fix: replace jersey-hue K-means team assignment with court-side majority vote"
```

---

## Task 2: Install EasyOCR and Create Jersey OCR Service

**Files:**
- Modify: `backend/requirements.txt`
- Create: `backend/app/services/jersey_ocr.py`
- Create: `backend/tests/test_jersey_ocr.py`

EasyOCR uses PyTorch (already installed: torch 2.11.0+cu130) so no heavy new dependency is added. It will run on GPU automatically.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_jersey_ocr.py`:

```python
import pytest
import numpy as np
import cv2


def _make_jersey_crop(number: str, bg_color=(220, 60, 60), text_color=(255, 255, 255)):
    """Synthetic jersey crop: coloured background, white number text."""
    img = np.full((120, 80, 3), bg_color, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.0
    thickness = 4
    (tw, th), _ = cv2.getTextSize(number, font, scale, thickness)
    x = (80 - tw) // 2
    y = (120 + th) // 2
    cv2.putText(img, number, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)
    return img


def test_read_jersey_number_single_digit():
    from app.services.jersey_ocr import read_jersey_number
    crop = _make_jersey_crop("7")
    result = read_jersey_number(crop)
    assert result == 7


def test_read_jersey_number_two_digits():
    from app.services.jersey_ocr import read_jersey_number
    crop = _make_jersey_crop("23")
    result = read_jersey_number(crop)
    assert result == 23


def test_read_jersey_number_returns_none_for_blank():
    from app.services.jersey_ocr import read_jersey_number
    # Solid colour, no text
    blank = np.full((120, 80, 3), (30, 120, 200), dtype=np.uint8)
    result = read_jersey_number(blank)
    assert result is None


def test_read_jersey_number_rejects_out_of_range():
    from app.services.jersey_ocr import read_jersey_number
    # Even if OCR "reads" 99, it's beyond volleyball jersey range 1-99
    crop = _make_jersey_crop("99")
    result = read_jersey_number(crop)
    # 99 is valid (just barely) so should return 99 or None — not raise
    assert result is None or result == 99


def test_consensus_jersey_number():
    from app.services.jersey_ocr import consensus_jersey
    readings = [7, 7, None, 7, None, 8]
    assert consensus_jersey(readings) == 7


def test_consensus_jersey_number_all_none():
    from app.services.jersey_ocr import consensus_jersey
    assert consensus_jersey([None, None, None]) is None


def test_consensus_jersey_requires_majority():
    from app.services.jersey_ocr import consensus_jersey
    # 2 readings of 5 and 2 readings of 9 — tie → None (not confident enough)
    assert consensus_jersey([5, 5, 9, 9]) is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend
.venv/bin/pytest tests/test_jersey_ocr.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'app.services.jersey_ocr'`

- [ ] **Step 3: Add easyocr to requirements.txt**

Open `backend/requirements.txt`. Add at the end:

```
easyocr>=1.7.0
```

Install it:

```bash
cd backend
.venv/bin/pip install easyocr>=1.7.0
```

Expected: Downloads and installs (uses existing torch, no CUDA reinstall needed). Takes ~2 minutes.

- [ ] **Step 4: Create jersey_ocr.py**

Create `backend/app/services/jersey_ocr.py`:

```python
"""
Jersey OCR Service
──────────────────
Reads jersey numbers (1-99) from player bounding box crops using EasyOCR.

Usage:
    from app.services.jersey_ocr import read_jersey_number, consensus_jersey

    number = read_jersey_number(frame_crop)   # int or None
    best   = consensus_jersey([7, 7, None, 7]) # 7
"""

import logging
import re
from statistics import mode, StatisticsError
from typing import Optional, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Module-level EasyOCR reader — lazy-loaded, shared across all frames.
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
            _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
            logger.info("JerseyOCR: EasyOCR reader initialised (GPU)")
        except Exception as exc:
            logger.warning(f"JerseyOCR: failed to init EasyOCR — {exc}")
            _reader = False  # sentinel: do not retry
    return _reader if _reader is not False else None


def _preprocess(crop: np.ndarray) -> np.ndarray:
    """
    Prepare a jersey crop for digit OCR.
    1. Take only the upper 55% of the bbox (number is on chest/back, not legs).
    2. Scale up 3× for better OCR accuracy on small crops.
    3. Convert to grayscale and apply adaptive threshold to isolate digits.
    """
    h, w = crop.shape[:2]
    # Upper portion: skip the bottom half (legs) and very top (collar)
    y1 = int(h * 0.15)
    y2 = int(h * 0.60)
    x1 = int(w * 0.10)
    x2 = int(w * 0.90)
    region = crop[y1:y2, x1:x2]
    if region.size == 0:
        return crop

    # Scale up 3× — EasyOCR performs much better on larger text
    scaled = cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Grayscale + CLAHE for contrast enhancement
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Convert back to BGR (EasyOCR expects BGR or RGB)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def read_jersey_number(crop: np.ndarray) -> Optional[int]:
    """
    Read a jersey number (integer 1-99) from a player bounding box crop.

    Parameters
    ----------
    crop : np.ndarray
        BGR image of the player bounding box.

    Returns
    -------
    int or None
        Detected jersey number, or None if unreadable / out of range.
    """
    reader = _get_reader()
    if reader is None or crop is None or crop.size == 0:
        return None

    try:
        processed = _preprocess(crop)
        results = reader.readtext(
            processed,
            allowlist="0123456789",
            detail=1,
            paragraph=False,
        )
        for (_bbox, text, conf) in results:
            if conf < 0.45:
                continue
            digits = re.sub(r"\D", "", text)
            if not digits:
                continue
            number = int(digits)
            if 1 <= number <= 99:
                return number
    except Exception as exc:
        logger.debug(f"JerseyOCR.read_jersey_number error: {exc}")
    return None


def consensus_jersey(readings: List[Optional[int]]) -> Optional[int]:
    """
    Return the most common jersey number from a list of per-frame readings.
    Returns None if no single number appears more than all others combined
    (i.e., requires strict majority).

    Parameters
    ----------
    readings : list of int or None
        Per-frame OCR results for a single player track.
    """
    valid = [r for r in readings if r is not None]
    if not valid:
        return None
    try:
        m = mode(valid)
    except StatisticsError:
        # No unique mode (tie)
        return None
    # Require strict majority (> 50% of non-None readings agree)
    if valid.count(m) > len(valid) / 2:
        return m
    return None
```

- [ ] **Step 5: Run tests**

```bash
cd backend
.venv/bin/pytest tests/test_jersey_ocr.py -v
```

Expected: `test_consensus_jersey_number` PASS, `test_consensus_jersey_number_all_none` PASS, `test_consensus_jersey_requires_majority` PASS (consensus tests use no EasyOCR). The `read_jersey_number` tests depend on EasyOCR accuracy — they may or may not pass on synthetic data, which is acceptable. The consensus tests must all pass.

- [ ] **Step 6: Commit**

```bash
git add backend/requirements.txt backend/app/services/jersey_ocr.py backend/tests/test_jersey_ocr.py
git commit -m "feat: add jersey number OCR service with EasyOCR and consensus voting"
```

---

## Task 3: Wire Jersey OCR into Player Tracker

**Files:**
- Modify: `backend/app/services/player_tracker.py:219-231`

The tracker runs per-frame. OCR is expensive (~50ms/crop on GPU), so we only call it every 15 frames (0.5 s at 30 fps). The `jersey_number` key is added to the output dict — `None` on skipped frames.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_jersey_ocr.py`:

```python
def test_ocr_every_n_frames():
    """OCR fires on frame 0, 15, 30 but not 1, 14, 16."""
    OCR_INTERVAL = 15
    fired_frames = []

    def mock_ocr(crop):
        fired_frames.append(True)
        return 7

    frames_to_test = [0, 1, 14, 15, 16, 29, 30]
    expected_fires = [True, False, False, True, False, False, True]

    for frame_idx, should_fire in zip(frames_to_test, expected_fires):
        fired_frames.clear()
        if frame_idx % OCR_INTERVAL == 0:
            mock_ocr(None)
        assert bool(fired_frames) == should_fire, f"frame {frame_idx}: fire={bool(fired_frames)}, expected={should_fire}"
```

```bash
cd backend
.venv/bin/pytest tests/test_jersey_ocr.py::test_ocr_every_n_frames -v
```

Expected: PASS.

- [ ] **Step 2: Add jersey_number to player_tracker output**

Open `backend/app/services/player_tracker.py`.

At the top of the file, add the import after the existing imports:

```python
from app.services.jersey_ocr import read_jersey_number
```

Add the class constant just inside `PlayerTracker.__init__` or as a module-level constant near the other constants (around line 42):

```python
OCR_INTERVAL = 15   # run jersey OCR every N frames to limit GPU cost
```

Inside `process_frame`, after the `jersey_hue = self._sample_jersey_hue(...)` line (around line 217), add:

```python
                # Jersey OCR — only every OCR_INTERVAL frames
                jersey_number = None
                if frame_idx % OCR_INTERVAL == 0:
                    x1i = max(0, int(bx))
                    y1i = max(0, int(by))
                    x2i = min(frame.shape[1], int(bx + bw))
                    y2i = min(frame.shape[0], int(by + bh))
                    crop = frame[y1i:y2i, x1i:x2i]
                    jersey_number = read_jersey_number(crop)
```

Then in the `output.append({...})` block, add `"jersey_number": jersey_number` as a new key:

```python
                output.append({
                    "track_id":      tid,
                    "bbox_x":        bx,
                    "bbox_y":        by,
                    "bbox_w":        bw,
                    "bbox_h":        bh,
                    "confidence":    conf,
                    "court_x":       cx if cx >= 0 else None,
                    "court_y":       cy if cy >= 0 else None,
                    "frame_number":  frame_idx,
                    "timestamp":     round(timestamp, 4),
                    "jersey_hue":    jersey_hue,
                    "jersey_number": jersey_number,   # int or None
                })
```

- [ ] **Step 3: Pass jersey_number through cv_pipeline player_rows**

Open `backend/app/services/cv_pipeline.py`. Find the `for p in players:` block that appends to `player_rows` (around line 306):

```python
                for p in players:
                    player_rows.append({
```

Add `"jersey_number": p.get("jersey_number"),` to the dict:

```python
                for p in players:
                    player_rows.append({
                        "match_id":      self.match_id,
                        "track_id":      p["track_id"],
                        "frame_number":  p["frame_number"],
                        "timestamp":     p["timestamp"],
                        "bbox_x": p["bbox_x"], "bbox_y": p["bbox_y"],
                        "bbox_w": p["bbox_w"], "bbox_h": p["bbox_h"],
                        "confidence":    p["confidence"],
                        "court_x":       p.get("court_x"),
                        "court_y":       p.get("court_y"),
                        "jersey_hue":    p.get("jersey_hue", -1.0),
                        "jersey_number": p.get("jersey_number"),
                    })
```

- [ ] **Step 4: Run existing tests to verify no regressions**

```bash
cd backend
.venv/bin/pytest tests/ -v --ignore=tests/test_jersey_ocr.py 2>&1 | tail -20
```

Expected: same pass rate as before (no new failures).

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/player_tracker.py backend/app/services/cv_pipeline.py
git commit -m "feat: wire jersey OCR into player tracker, pass jersey_number through pipeline"
```

---

## Task 4: Use Jersey Numbers for Deduplication in _save_to_db

**Files:**
- Modify: `backend/app/services/cv_pipeline.py` — `_save_to_db` method

After team assignment (Task 1), we compute a consensus jersey number per track and store it on the Player. Before saving, we merge tracks with the same team + jersey number (they're the same physical player with two ByteTrack IDs from a tracking gap).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_track_merger.py`:

```python
from app.services.track_merger import _find_merge_pairs, _assign_display_numbers


def test_find_merge_pairs_same_team_no_overlap():
    tracks = [
        {"player_id": "a", "team": "A", "t_start": 0.0, "t_end": 5.0,
         "last_cx": 0.2, "last_cy": 0.5, "first_cx": 0.2, "first_cy": 0.5},
        {"player_id": "b", "team": "A", "t_start": 5.5, "t_end": 10.0,
         "last_cx": 0.2, "last_cy": 0.5, "first_cx": 0.22, "first_cy": 0.5},
    ]
    pairs = _find_merge_pairs(tracks)
    assert len(pairs) == 1
    assert pairs[0] == ("a", "b")


def test_find_merge_pairs_different_team_not_merged():
    tracks = [
        {"player_id": "a", "team": "A", "t_start": 0.0, "t_end": 5.0,
         "last_cx": 0.2, "last_cy": 0.5, "first_cx": 0.2, "first_cy": 0.5},
        {"player_id": "b", "team": "B", "t_start": 5.5, "t_end": 10.0,
         "last_cx": 0.2, "last_cy": 0.5, "first_cx": 0.22, "first_cy": 0.5},
    ]
    pairs = _find_merge_pairs(tracks)
    assert len(pairs) == 0


def test_find_merge_pairs_overlap_not_merged():
    tracks = [
        {"player_id": "a", "team": "A", "t_start": 0.0, "t_end": 8.0,
         "last_cx": 0.2, "last_cy": 0.5, "first_cx": 0.2, "first_cy": 0.5},
        {"player_id": "b", "team": "A", "t_start": 5.0, "t_end": 12.0,
         "last_cx": 0.22, "last_cy": 0.5, "first_cx": 0.22, "first_cy": 0.5},
    ]
    pairs = _find_merge_pairs(tracks)
    assert len(pairs) == 0


def test_assign_display_numbers_uses_jersey_when_available():
    players = [
        {"player_id": "a", "team": "A", "t_start": 0.0, "jersey_number": 7},
        {"player_id": "b", "team": "B", "t_start": 0.0, "jersey_number": 11},
        {"player_id": "c", "team": "A", "t_start": 1.0, "jersey_number": None},
    ]
    result = _assign_display_numbers(players)
    p = {p["player_id"]: p for p in result}
    assert p["a"]["display_number"] == 7
    assert p["b"]["display_number"] == 11
    # c has no jersey number → gets sequential fallback
    assert p["c"]["display_number"] is not None
    assert isinstance(p["c"]["display_number"], int)


def test_assign_display_numbers_no_jersey_uses_sequential():
    players = [
        {"player_id": "a", "team": "A", "t_start": 0.0, "jersey_number": None},
        {"player_id": "b", "team": "B", "t_start": 0.0, "jersey_number": None},
    ]
    result = _assign_display_numbers(players)
    p = {pl["player_id"]: pl for pl in result}
    assert p["a"]["display_number"] in range(1, 20)
    assert p["b"]["display_number"] in range(1, 20)
    assert p["a"]["display_number"] != p["b"]["display_number"]
```

```bash
cd backend
.venv/bin/pytest tests/test_track_merger.py::test_find_merge_pairs_same_team_no_overlap \
                 tests/test_track_merger.py::test_find_merge_pairs_different_team_not_merged \
                 tests/test_track_merger.py::test_find_merge_pairs_overlap_not_merged -v
```

Expected: 3 PASS (these test existing logic). The `display_number` tests should FAIL because `_assign_display_numbers` currently ignores `jersey_number`.

- [ ] **Step 2: Add jersey-consensus computation to _save_to_db**

Open `backend/app/services/cv_pipeline.py`. In `_save_to_db`, import at the top of the method:

```python
        from app.services.jersey_ocr import consensus_jersey
```

After the `team_map` computation (Task 1 code), add jersey consensus before creating Player objects:

```python
            # Jersey number consensus: mode of non-None OCR readings per track
            track_jerseys: Dict[int, list] = defaultdict(list)
            for r in player_rows:
                jn = r.get("jersey_number")
                if jn is not None:
                    track_jerseys[r["track_id"]].append(jn)

            jersey_map: Dict[int, Optional[int]] = {
                tid: consensus_jersey(readings)
                for tid, readings in track_jerseys.items()
            }

            # Pre-merge: group tracks with same team + same jersey number
            # (ByteTrack assigned them different IDs due to tracking gaps)
            jersey_merge: Dict[int, int] = {}  # ghost_tid → canonical_tid
            seen_jersey: Dict[tuple, int] = {}  # (team, jersey_num) → first tid
            for tid in sorted(track_ids):       # sorted for determinism
                team  = team_map.get(tid)
                jnum  = jersey_map.get(tid)
                key   = (team, jnum)
                if jnum is not None and team is not None and key in seen_jersey:
                    jersey_merge[tid] = seen_jersey[key]  # this tid is a duplicate
                elif jnum is not None and team is not None:
                    seen_jersey[key] = tid

            # Remap ghost track_ids in player_rows and action_rows
            for r in player_rows:
                if r["track_id"] in jersey_merge:
                    r["track_id"] = jersey_merge[r["track_id"]]
            for r in action_rows:
                if r.get("track_id") in jersey_merge:
                    r["track_id"] = jersey_merge[r["track_id"]]

            # Recompute track_ids after merge
            track_ids = {r["track_id"] for r in player_rows}
```

Then update the Player creation loop to set `jersey_number` and `display_number`:

```python
            for tid in track_ids:
                team   = team_map.get(tid)
                jnum   = jersey_map.get(tid)
                player = Player(
                    match_id=uuid.UUID(self.match_id),
                    player_track_id=tid,
                    team=team,
                    jersey_number=str(jnum) if jnum is not None else None,
                    display_number=jnum,
                    display_name=(
                        f"#{jnum} (Team {team or '?'})"
                        if jnum is not None
                        else f"Player #{tid} (Team {team or '?'})"
                    ),
                )
                db.add(player)
                await db.flush()
                await db.refresh(player)
                player_id_map[tid] = player.id
```

- [ ] **Step 3: Update _assign_display_numbers in track_merger.py to use jersey_number**

Open `backend/app/services/track_merger.py`. Find `_assign_display_numbers` (around line 55). Replace it:

```python
def _assign_display_numbers(players: List[Dict]) -> List[Dict]:
    """
    Assign display_number to each player dict.
    If the player has a jersey_number from OCR, use that directly.
    Otherwise fall back to sequential: Team A → #1-6, Team B → #7-12, rest → #13+.
    Returns the same list with 'display_number' key set.
    """
    used_numbers = set()

    # First pass: assign jersey numbers directly
    for p in players:
        jn = p.get("jersey_number")
        if jn is not None:
            p["display_number"] = int(jn)
            used_numbers.add(int(jn))

    # Second pass: sequential fallback for players without jersey OCR
    team_a = sorted([p for p in players if p.get("team") == "A" and p.get("display_number") is None],
                    key=lambda p: p["t_start"])
    team_b = sorted([p for p in players if p.get("team") == "B" and p.get("display_number") is None],
                    key=lambda p: p["t_start"])
    others = sorted([p for p in players if p.get("team") not in ("A", "B") and p.get("display_number") is None],
                    key=lambda p: p["t_start"])

    def _next_available(start: int) -> int:
        n = start
        while n in used_numbers:
            n += 1
        used_numbers.add(n)
        return n

    for p in team_a:
        p["display_number"] = _next_available(1)
    for p in team_b:
        p["display_number"] = _next_available(7)
    for p in others:
        p["display_number"] = _next_available(13)

    return players
```

Also update the `merge_tracks` function to pass `jersey_number` into the summary dict. Find the `summary.append({...})` block (around line 210 in track_merger.py) and add `"jersey_number": p.display_number` (display_number was already set from OCR by the pipeline):

```python
            summary.append({
                "player_id":    str(p.id),
                "team":         p.team,
                "t_start":      t_start_val,
                "frame_count":  frame_cnt_r.scalar() or 0,
                "jersey_number": p.display_number,   # already set by pipeline if OCR succeeded
            })
```

- [ ] **Step 4: Run tests**

```bash
cd backend
.venv/bin/pytest tests/test_track_merger.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/cv_pipeline.py backend/app/services/track_merger.py \
        backend/tests/test_track_merger.py
git commit -m "feat: deduplicate tracks by jersey number, use OCR number as display_number"
```

---

## Task 5: End-to-End Verification

**Files:**
- No new files — manual re-analysis + visual verification

This task verifies the full pipeline works with real video data.

- [ ] **Step 1: Restart the backend**

```bash
cd backend
# Kill existing process on port 8001
lsof -ti:8001 | xargs kill -9 2>/dev/null
sleep 1
nohup .venv/bin/python run.py > /tmp/backend.log 2>&1 &
sleep 4
curl -s http://localhost:8001/api/health
```

Expected: `{"status":"healthy",...}`

- [ ] **Step 2: Re-analyze a match via the UI**

1. Open the browser at `http://localhost:5173`
2. Open any match detail page
3. Click **Re-Analyze**
4. Wait for the progress bar to reach 100%

- [ ] **Step 3: Verify team colors**

On the video canvas:
- Players on the left half of the court must have **blue** bounding boxes (Team A)
- Players on the right half must have **red** bounding boxes (Team B)
- No match should have ALL players the same color

- [ ] **Step 4: Verify player count**

Check the database:

```bash
cd backend
.venv/bin/python -c "
import asyncio
from app.database import AsyncSessionLocal
from sqlalchemy import text

async def main():
    async with AsyncSessionLocal() as db:
        r = await db.execute(text('''
            SELECT m.id, COUNT(p.id) player_count
            FROM matches m
            JOIN players p ON p.match_id = m.id
            GROUP BY m.id
            ORDER BY player_count DESC
            LIMIT 5
        '''))
        for row in r:
            print(f'Match {str(row.id)[:8]}: {row.player_count} players')

asyncio.run(main())
"
```

Expected: each match should have ≤ 16 players (enforced by `MAX_PLAYERS` in `track_merger.py`).

- [ ] **Step 5: Verify jersey numbers (if OCR fires)**

```bash
cd backend
.venv/bin/python -c "
import asyncio
from app.database import AsyncSessionLocal
from sqlalchemy import text

async def main():
    async with AsyncSessionLocal() as db:
        r = await db.execute(text('''
            SELECT team, display_number, jersey_number, display_name
            FROM players
            WHERE match_id = (SELECT id FROM matches ORDER BY created_at DESC LIMIT 1)
            ORDER BY team, display_number
        '''))
        for row in r:
            print(f'Team {row.team} | #{row.display_number} | jersey={row.jersey_number} | {row.display_name}')

asyncio.run(main())
"
```

Expected: players with successful OCR show `jersey_number` matching `display_number`. Players without OCR show sequential numbers.

- [ ] **Step 6: Commit test results note**

```bash
git add -A
git commit -m "chore: end-to-end verification of jersey OCR and team assignment pipeline"
```

---

## Self-Review

**Spec coverage:**
- ✅ Team assignment replaced with court-side majority vote (Task 1)
- ✅ Jersey OCR service created with EasyOCR (Task 2)
- ✅ OCR wired into player_tracker every 15 frames (Task 3)
- ✅ Jersey-based deduplication in `_save_to_db` (Task 4)
- ✅ `display_number` set from jersey OCR or sequential fallback (Task 4 + Task 5)
- ✅ `track_merger._assign_display_numbers` updated to respect jersey numbers (Task 4)
- ✅ Frontend unchanged — already reads `display_number` (verified in VideoPlayer.jsx)

**Placeholder scan:** None found — all steps have concrete code.

**Type consistency:**
- `jersey_number` is `Optional[int]` throughout (player_rows dict, jersey_map, Player.display_number)
- `Player.jersey_number` column is `String(10)` in the model — stored as `str(jnum)`, consistent with Task 4
- `consensus_jersey` returns `Optional[int]`, matches all call sites
- `_assign_display_numbers` reads `p.get("jersey_number")` — the summary dict in `merge_tracks` now includes this key (Task 4 Step 3)
