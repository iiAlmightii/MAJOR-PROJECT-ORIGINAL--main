"""
Rally Detector
──────────────
Automatically segments a volleyball match video into individual rallies
by analysing ball detection gaps and motion.

Algorithm
---------
1.  Process ball detections frame by frame.
2.  A RALLY is ACTIVE when the ball is detected.
3.  A RALLY ENDS when:
      a. Ball is not detected for > GAP_SECONDS consecutive seconds, OR
      b. Ball is near the floor (y > FLOOR_THRESHOLD × frame_height)
         and velocity is downward, OR
      c. Manual end signal.
4.  Merge very short segments (< MIN_RALLY_SECONDS) into neighbours.
5.  Detect point winner from:
      - Which half of the court the ball hit
      - Last detected action before rally end

Output: List[RallySegment]
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

GAP_SECONDS       = 1.5     # no-ball gap → end rally
MIN_RALLY_SECONDS = 1.0     # ignore micro-segments
FLOOR_THRESHOLD   = 0.85    # normalised y > this = ball near floor
MAX_RALLY_SECONDS = 120.0   # safety cap
VELOCITY_WINDOW   = 5       # frames to compute ball velocity


@dataclass
class RallySegment:
    rally_number:  int
    start_time:    float
    end_time:      float
    start_frame:   int
    end_frame:     int
    winner_team:   Optional[str] = None
    point_reason:  str = ""
    events:        List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rally_number": self.rally_number,
            "start_time":   round(self.start_time, 3),
            "end_time":     round(self.end_time, 3),
            "start_frame":  self.start_frame,
            "end_frame":    self.end_frame,
            "winner_team":  self.winner_team,
            "point_reason": self.point_reason,
            "events":       self.events,
        }


class RallyDetector:
    """
    Stateful rally detector.  Feed it one frame of data at a time via
    update(), then call finalize() at end of video.
    """

    def __init__(self, fps: float = 25.0):
        self._fps          = max(fps, 1.0)
        self._rallies:     List[RallySegment] = []
        self._in_rally     = False
        self._rally_start_frame  = 0
        self._rally_start_time   = 0.0
        self._last_ball_frame    = -1
        self._last_ball_time     = 0.0
        self._gap_frames         = 0
        self._rally_number       = 0
        self._recent_ball_y:     List[float] = []    # for velocity calc

    def reset(self, fps: float = 25.0):
        self.__init__(fps)

    # ──────────────────────────────────────────────────────────────────────────

    def update(
        self,
        frame_idx: int,
        ball: Optional[Dict[str, Any]],
        player_detections: List[Dict[str, Any]],
    ) -> Optional[RallySegment]:
        """
        Feed one frame. Returns a completed RallySegment if a rally just ended,
        otherwise None.
        """
        timestamp = frame_idx / self._fps
        gap_threshold_frames = int(GAP_SECONDS * self._fps)

        if ball is not None:
            # Ball detected
            self._last_ball_frame = frame_idx
            self._last_ball_time  = timestamp

            # Track y for floor detection
            by = ball.get("court_y", -1.0) or -1.0
            self._recent_ball_y.append(by)
            if len(self._recent_ball_y) > VELOCITY_WINDOW:
                self._recent_ball_y.pop(0)

            if not self._in_rally:
                # Start new rally
                self._in_rally          = True
                self._rally_start_frame = frame_idx
                self._rally_start_time  = timestamp
                self._gap_frames        = 0

            # Check floor hit (ball near floor + moving downward)
            if self._in_rally and self._floor_hit_detected(by):
                return self._end_rally(frame_idx, timestamp, reason="floor_hit",
                                       ball_court_y=by)

        else:
            # No ball detected
            if self._in_rally:
                self._gap_frames += 1
                if self._gap_frames >= gap_threshold_frames:
                    # Gap too long → end rally
                    return self._end_rally(
                        self._last_ball_frame,
                        self._last_ball_time,
                        reason="ball_lost",
                    )

        # Safety cap
        if self._in_rally:
            duration = timestamp - self._rally_start_time
            if duration > MAX_RALLY_SECONDS:
                return self._end_rally(frame_idx, timestamp, reason="timeout")

        return None

    def finalize(self, last_frame: int, last_timestamp: float) -> Optional[RallySegment]:
        """Call at end of video to close any open rally."""
        if self._in_rally:
            return self._end_rally(last_frame, last_timestamp, reason="video_end")
        return None

    def get_rallies(self) -> List[RallySegment]:
        return list(self._rallies)

    # ──────────────────────────────────────────────────────────────────────────

    def _floor_hit_detected(self, court_y: float) -> bool:
        if court_y < 0:
            return False
        if court_y < FLOOR_THRESHOLD:
            return False
        if len(self._recent_ball_y) < 3:
            return False
        # Confirm downward trend
        valid = [y for y in self._recent_ball_y if y >= 0]
        if len(valid) < 3:
            return False
        return valid[-1] > valid[-2] > valid[-3]

    def _end_rally(
        self,
        end_frame: int,
        end_time: float,
        reason: str,
        ball_court_y: float = -1.0,
    ) -> Optional[RallySegment]:
        duration = end_time - self._rally_start_time
        if duration < MIN_RALLY_SECONDS:
            # Too short — discard and reset
            self._in_rally    = False
            self._gap_frames  = 0
            self._recent_ball_y.clear()
            return None

        self._rally_number += 1
        winner = self._infer_winner(ball_court_y, reason)

        seg = RallySegment(
            rally_number = self._rally_number,
            start_time   = round(self._rally_start_time, 3),
            end_time     = round(end_time, 3),
            start_frame  = self._rally_start_frame,
            end_frame    = end_frame,
            winner_team  = winner,
            point_reason = reason,
        )
        self._rallies.append(seg)
        self._in_rally    = False
        self._gap_frames  = 0
        self._recent_ball_y.clear()
        return seg

    def _infer_winner(self, ball_court_y: float, reason: str) -> Optional[str]:
        """
        Simple heuristic: if ball hits floor in top half → Team A scores (Team B erred),
        if ball hits floor in bottom half → Team B scores.
        ball_court_y is normalised: 0 = top (team A back court), 1 = bottom (team B back court).
        """
        if reason == "floor_hit" and ball_court_y >= 0:
            return "A" if ball_court_y > 0.5 else "B"
        return None
