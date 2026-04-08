"""
Computer Vision Pipeline  (Phase 3 — with Action Recognition + Scoring)
────────────────────────────────────────────────────────────────────────
Orchestrates the full analysis pipeline for one match:

  1. Extract video metadata
  2. Auto-calibrate homography (or use provided court corners)
  3. Process every Nth frame:
      a. PlayerTracker  → bounding boxes + stable IDs + court coords
      b. BallDetector   → ball position + trajectory + court coords
      c. ActionService  → Pose+LSTM action recognition per player
      d. RallyDetector  → segment into rallies
  4. Persist all data to PostgreSQL in batches
  5. Run ScoringEngine → compute stats + update Analytics table
  6. Clip rally segments with FFmpeg
  7. Emit progress via async callback / WebSocket
"""

import os
import asyncio
import logging
import subprocess
from typing import Callable, Optional, List, Dict, Any

import cv2
import numpy as np

from app.services.player_tracker    import PlayerTracker
from app.services.ball_detector     import BallDetector
from app.services.rally_detector    import RallyDetector, RallySegment
from app.services.homography_service import HomographyService
from app.services.action_service    import ActionService
from app.services.scoring_engine    import ScoringEngine
from app.config import settings

logger = logging.getLogger(__name__)

PROCESS_EVERY_N = 3      # analyse every Nth frame (3 = one-third FPS)
BATCH_SIZE      = 500    # rows per DB insert batch
MIN_CLIP_GAP    = 2.0    # seconds of padding around rally clips


class CVPipeline:
    """
    Full CV pipeline for one match video.

    Parameters
    ----------
    match_id       : str  — UUID string
    video_path     : str  — absolute path to video file
    progress_cb    : async callable(pct: int, msg: str)
    court_corners  : optional [[x,y]×4] court corners for homography
    """

    def __init__(
        self,
        match_id:      str,
        video_path:    str,
        progress_cb:   Optional[Callable] = None,
        court_corners: Optional[List[List[float]]] = None,
    ):
        self.match_id      = match_id
        self.video_path    = video_path
        self.progress_cb   = progress_cb
        self.court_corners = court_corners

        self.player_tracker = PlayerTracker()
        self.ball_detector  = BallDetector()
        self.homography     = HomographyService()
        self.rally_detector = RallyDetector()
        self.action_service = ActionService()
        self.scoring_engine = ScoringEngine()

        rally_dir = os.path.join(settings.RALLIES_DIR, match_id)
        os.makedirs(rally_dir, exist_ok=True)
        self.rally_dir = rally_dir

        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    # ──────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────────────

    async def run(self) -> Dict[str, Any]:
        await self._emit(0, "Initialising models...")

        loop = asyncio.get_event_loop()

        # Load all models concurrently
        player_ok, ball_ok, action_ok = await asyncio.gather(
            loop.run_in_executor(None, self.player_tracker.load),
            loop.run_in_executor(None, self.ball_detector.load),
            loop.run_in_executor(None, self.action_service.load),
        )

        if not player_ok: logger.warning("Player tracker unavailable")
        if not ball_ok:   logger.warning("Ball detector unavailable")
        if not action_ok: logger.info("Action service unavailable (train first)")

        await self._emit(4, "Opening video...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.rally_detector.reset(fps)
        self.action_service.reset()

        # ── Homography calibration ──────────────────────────────────────────
        await self._emit(6, "Calibrating court homography...")
        if self.court_corners:
            self.homography.calibrate(self.court_corners)
            logger.info("Homography: using provided court corners")
        else:
            sample_pos = min(int(fps * 5), total_frames // 4)
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample_pos)
            ret, sample_frame = cap.read()
            if ret:
                self.homography.auto_calibrate_from_lines(sample_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            logger.info("Homography: auto-calibrated from frame lines")

        await self._emit(8, "Processing frames...")

        # ── Data collectors ─────────────────────────────────────────────────
        player_rows:   List[Dict] = []
        ball_rows:     List[Dict] = []
        action_rows:   List[Dict] = []
        completed_rallies: List[RallySegment] = []

        frame_idx  = 0
        last_pct   = 8

        while True:
            if self._cancelled:
                cap.release()
                return {"status": "cancelled"}

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % PROCESS_EVERY_N == 0:
                ball    = None
                players = []
                actions = []

                if ball_ok:
                    ball = self.ball_detector.detect(
                        frame, frame_idx, fps, self.homography
                    )
                if player_ok:
                    players = self.player_tracker.process_frame(
                        frame, frame_idx, fps, self.homography
                    )
                if action_ok and player_ok and players:
                    actions = self.action_service.process_frame(
                        frame, players, frame_idx, fps
                    )

                # Rally segmentation
                completed = self.rally_detector.update(frame_idx, ball, players)
                if completed:
                    completed_rallies.append(completed)

                # ── Collect rows ────────────────────────────────────────────
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

                for p in players:
                    player_rows.append({
                        "match_id":     self.match_id,
                        "track_id":     p["track_id"],
                        "frame_number": frame_idx,
                        "timestamp":    p["timestamp"],
                        "bbox_x": p["bbox_x"], "bbox_y": p["bbox_y"],
                        "bbox_w": p["bbox_w"], "bbox_h": p["bbox_h"],
                        "confidence": p["confidence"],
                        "court_x": p.get("court_x"),
                        "court_y": p.get("court_y"),
                    })

                for a in actions:
                    action_rows.append({
                        "match_id":     self.match_id,
                        "track_id":     a["track_id"],
                        "action_type":  a["action"],
                        "confidence":   a["confidence"],
                        "timestamp":    a["timestamp"],
                        "frame_number": a["frame_number"],
                    })

            frame_idx += 1

            if total_frames > 0:
                pct = 8 + int((frame_idx / total_frames) * 72)
                if pct > last_pct + 4:
                    last_pct = pct
                    n_actions = len(action_rows)
                    await self._emit(
                        pct,
                        f"Frame {frame_idx}/{total_frames} "
                        f"| {len(completed_rallies)} rallies"
                        f"{f' | {n_actions} actions' if n_actions else ''}",
                    )
                    await asyncio.sleep(0)

        cap.release()

        # Close any open rally
        final = self.rally_detector.finalize(frame_idx - 1, (frame_idx - 1) / fps)
        if final:
            completed_rallies.append(final)

        await self._emit(81, f"Detected {len(completed_rallies)} rallies, "
                            f"{len(action_rows)} actions. Saving to database...")

        # ── Persist to DB ───────────────────────────────────────────────────
        player_id_map = await self._save_to_db(
            player_rows, ball_rows, action_rows, completed_rallies, fps
        )

        await self._emit(90, "Computing match analytics & scoring...")
        await self._run_scoring(player_id_map, action_rows, completed_rallies)

        await self._emit(94, "Clipping rally segments...")
        await self._clip_rallies(completed_rallies)

        summary = {
            "total_frames":       frame_idx,
            "fps":                fps,
            "resolution":         f"{frame_w}×{frame_h}",
            "total_rallies":      len(completed_rallies),
            "rallies":            [r.to_dict() for r in completed_rallies],
            "player_detections":  len(player_rows),
            "ball_detections":    len(ball_rows),
            "action_detections":  len(action_rows),
            "action_ok":          action_ok,
        }

        await self._emit(100, f"Complete! {len(completed_rallies)} rallies, "
                             f"{len(action_rows)} actions detected.")
        return summary

    # ──────────────────────────────────────────────────────────────────────────
    # DB persistence
    # ──────────────────────────────────────────────────────────────────────────

    async def _save_to_db(
        self,
        player_rows:   List[Dict],
        ball_rows:     List[Dict],
        action_rows:   List[Dict],
        rallies:       List[RallySegment],
        fps:           float,
    ) -> Dict[int, Any]:
        """Persist all data. Returns track_id → Player.id mapping."""
        from app.database import AsyncSessionLocal
        from app.models.match import Match, MatchStatus
        from app.models.player import Player
        from app.models.tracking import PlayerTracking, BallTracking
        from app.models.actions import Action, ActionType, ActionResult, Rally
        import uuid

        player_id_map: Dict[int, uuid.UUID] = {}

        async with AsyncSessionLocal() as db:

            # ── Players ─────────────────────────────────────────────────────
            track_ids = {r["track_id"] for r in player_rows}
            for tid in track_ids:
                # Determine team from court_x position (rough: <0.5 = Team A)
                rows_for_tid = [r for r in player_rows if r["track_id"] == tid]
                court_xs = [r["court_x"] for r in rows_for_tid if r.get("court_x") is not None]
                team = None
                if court_xs:
                    avg_x = sum(court_xs) / len(court_xs)
                    team = "A" if avg_x < 0.5 else "B"

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

            # ── Player tracking ─────────────────────────────────────────────
            for i in range(0, len(player_rows), BATCH_SIZE):
                for row in player_rows[i: i + BATCH_SIZE]:
                    pid = player_id_map.get(row["track_id"])
                    if pid is None:
                        continue
                    db.add(PlayerTracking(
                        player_id=pid,
                        match_id=uuid.UUID(self.match_id),
                        frame_number=row["frame_number"],
                        timestamp=row["timestamp"],
                        bbox_x=row["bbox_x"], bbox_y=row["bbox_y"],
                        bbox_w=row["bbox_w"], bbox_h=row["bbox_h"],
                        confidence=row.get("confidence"),
                        court_x=row.get("court_x"),
                        court_y=row.get("court_y"),
                    ))
                await db.flush()

            # ── Ball tracking ────────────────────────────────────────────────
            for i in range(0, len(ball_rows), BATCH_SIZE):
                for row in ball_rows[i: i + BATCH_SIZE]:
                    db.add(BallTracking(
                        match_id=uuid.UUID(self.match_id),
                        frame_number=row["frame_number"],
                        timestamp=row["timestamp"],
                        x=row["x"], y=row["y"],
                        confidence=row.get("confidence"),
                        court_x=row.get("court_x"),
                        court_y=row.get("court_y"),
                    ))
                await db.flush()

            # ── Actions ─────────────────────────────────────────────────────
            valid_action_types = {e.value for e in ActionType}
            for row in action_rows:
                pid  = player_id_map.get(row["track_id"])
                atype_raw = row["action_type"].lower()
                if atype_raw not in valid_action_types:
                    atype_raw = "unknown"
                db.add(Action(
                    match_id=uuid.UUID(self.match_id),
                    player_id=pid,
                    action_type=ActionType(atype_raw),
                    result=ActionResult.neutral,
                    timestamp=row["timestamp"],
                    frame_number=row["frame_number"],
                    confidence=row.get("confidence"),
                ))
            await db.flush()

            # ── Rallies ─────────────────────────────────────────────────────
            for seg in rallies:
                clip_path = os.path.join(self.rally_dir, f"rally_{seg.rally_number}.mp4")
                db.add(Rally(
                    match_id=uuid.UUID(self.match_id),
                    rally_number=seg.rally_number,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    start_frame=seg.start_frame,
                    end_frame=seg.end_frame,
                    video_clip_path=clip_path if os.path.exists(clip_path) else None,
                    winner_team=seg.winner_team,
                    point_reason=seg.point_reason,
                    events=seg.events or [],
                ))

            # ── Update match status ──────────────────────────────────────────
            from sqlalchemy import select
            res = await db.execute(
                select(Match).where(Match.id == uuid.UUID(self.match_id))
            )
            match = res.scalar_one_or_none()
            if match:
                match.status = MatchStatus.completed
                match.processing_progress = 100
                match.total_rallies = len(rallies)

            await db.commit()
            logger.info(f"DB save complete for match {self.match_id}")

        return player_id_map

    # ──────────────────────────────────────────────────────────────────────────
    # Scoring Engine
    # ──────────────────────────────────────────────────────────────────────────

    async def _run_scoring(
        self,
        player_id_map: Dict[int, Any],
        action_rows:   List[Dict],
        rallies:       List[RallySegment],
    ):
        from app.database import AsyncSessionLocal
        from app.models.analytics import Analytics
        from app.models.match import Match
        from app.models.player import Player
        from sqlalchemy import select
        import uuid

        if not action_rows and not rallies:
            return

        async with AsyncSessionLocal() as db:
            # Build data for scoring engine
            rally_dicts  = [r.to_dict() for r in rallies]
            action_dicts = [
                {
                    "player_id":   str(player_id_map.get(r["track_id"], "")),
                    "action_type": r["action_type"],
                    "result":      "neutral",
                    "timestamp":   r["timestamp"],
                    "team":        None,
                }
                for r in action_rows
            ]

            # Load players to get team info
            p_result = await db.execute(
                select(Player).where(Player.match_id == uuid.UUID(self.match_id))
            )
            db_players = p_result.scalars().all()
            player_dicts = [
                {"id": str(p.id), "team": p.team, "player_track_id": p.player_track_id}
                for p in db_players
            ]

            # Fill team info in action dicts
            pid_to_team = {str(p.id): p.team for p in db_players}
            for ad in action_dicts:
                ad["team"] = pid_to_team.get(ad["player_id"])

            # Run scoring
            summary = self.scoring_engine.compute(rally_dicts, action_dicts, player_dicts)

            # Persist Analytics per player
            for pid_str, stats in summary.get("player_stats", {}).items():
                if not pid_str:
                    continue
                try:
                    player_uuid = uuid.UUID(pid_str)
                except ValueError:
                    continue

                player_team = pid_to_team.get(pid_str)

                db.add(Analytics(
                    match_id=uuid.UUID(self.match_id),
                    player_id=player_uuid,
                    team=player_team,
                    total_serves=stats.get("total_serves", 0),
                    serve_errors=stats.get("serve_errors", 0),
                    aces=stats.get("aces", 0),
                    serve_efficiency=stats.get("serve_efficiency", 0.0),
                    total_attacks=stats.get("total_attacks", 0),
                    attack_errors=stats.get("attack_errors", 0),
                    attack_kills=stats.get("kills", 0),
                    attack_efficiency=stats.get("attack_efficiency", 0.0),
                    total_blocks=stats.get("total_blocks", 0),
                    block_errors=stats.get("block_errors", 0),
                    block_points=stats.get("block_points", 0),
                    total_receptions=stats.get("total_receptions", 0),
                    reception_errors=stats.get("reception_errors", 0),
                    reception_efficiency=stats.get("reception_efficiency", 0.0),
                    total_digs=stats.get("total_digs", 0),
                    extra_data={"key_moments": [
                        m for m in summary.get("key_moments", [])
                        if m.get("player_id") == pid_str
                    ]},
                ))

            # Update match scores
            res = await db.execute(
                select(Match).where(Match.id == uuid.UUID(self.match_id))
            )
            match = res.scalar_one_or_none()
            if match:
                match.team_a_score = summary.get("team_a_score", 0)
                match.team_b_score = summary.get("team_b_score", 0)
                match.summary = {
                    "team_stats":    summary.get("team_stats"),
                    "key_moments":   summary.get("key_moments", []),
                    "total_rallies": summary.get("total_rallies", 0),
                    "action_detections": len(action_rows),
                }

            await db.commit()
            logger.info(f"Scoring complete for match {self.match_id}")

    # ──────────────────────────────────────────────────────────────────────────
    # Rally clip generation (FFmpeg)
    # ──────────────────────────────────────────────────────────────────────────

    async def _clip_rallies(self, rallies: List[RallySegment]):
        if not rallies:
            return
        loop  = asyncio.get_event_loop()
        tasks = []
        for seg in rallies:
            out = os.path.join(self.rally_dir, f"rally_{seg.rally_number}.mp4")
            if os.path.exists(out):
                continue
            start    = max(0, seg.start_time - MIN_CLIP_GAP)
            duration = seg.duration + MIN_CLIP_GAP * 2
            tasks.append((start, duration, out))

        for start, dur, out in tasks:
            await loop.run_in_executor(None, self._ffmpeg_clip, start, dur, out)

    def _ffmpeg_clip(self, start: float, duration: float, out_path: str):
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", self.video_path,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            out_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
        except Exception as e:
            logger.warning(f"FFmpeg clip failed: {e}")

    # ──────────────────────────────────────────────────────────────────────────

    async def _emit(self, pct: int, msg: str):
        logger.info(f"[{pct:3d}%] {msg}")
        if self.progress_cb:
            try:
                await self.progress_cb(pct, msg)
            except Exception:
                pass
