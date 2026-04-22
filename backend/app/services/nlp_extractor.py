"""
NLP Event Extractor
────────────────────
Parses Whisper transcription segments and extracts structured volleyball events.

Approach:  Regex pattern matching + fuzzy word correction.
No external NLP library needed — zero additional GPU/CPU cost.

Why not spaCy / transformers?
  • Volleyball commentary is formulaic and domain-specific
  • Regex patterns handle the finite vocabulary precisely and faster
  • Whisper already fixes most ASR errors for common words
  • A sports-domain fine-tuned NER model doesn't exist off-the-shelf

Extracted schema per event
──────────────────────────
{
  "raw_text":    str,          # original segment text
  "start_time":  float,        # segment start (seconds in video)
  "end_time":    float,        # segment end
  "event_type":  str,          # serve|spike|block|receive|dig|set|unknown
  "player_number": int | None, # extracted jersey number
  "team":        str | None,   # "A" | "B" | None
  "result":      str,          # success|error|neutral
  "confidence":  float,        # 0.0–1.0 extraction confidence
}
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Action keyword → canonical action type ────────────────────────────────────
# Ordered longest-first so "spike" doesn't partially match "spike kill"

ACTION_ALIASES: Dict[str, str] = {
    # spike / attack
    "spike": "spike", "spikes": "spike", "spiked": "spike",
    "kill":  "spike", "kills":  "spike", "killed":  "spike",
    "attack": "spike", "attacks": "spike", "hit": "spike", "smash": "spike",

    # serve
    "serve": "serve", "serves": "serve", "served": "serve",
    "service": "serve", "serving": "serve",
    "ace": "serve",  # ace is a serve outcome — treated as serve+success

    # block
    "block": "block", "blocks": "block", "blocked": "block",
    "blocking": "block", "stuff": "block",

    # receive / reception
    "receive": "receive", "receives": "receive", "received": "receive",
    "reception": "receive", "pass": "receive", "passes": "receive",
    "dig":  "dig",  "digs":  "dig",  "digged": "dig", "defense": "dig",

    # set
    "set": "set", "sets": "set", "setter": "set", "setting": "set",
}

# ── Result keywords ────────────────────────────────────────────────────────────
SUCCESS_WORDS = {
    "great", "good", "excellent", "nice", "amazing", "perfect", "ace",
    "kill", "point", "winner", "in", "success", "successful", "scores",
    "score", "incredible", "wonderful", "beautiful", "clean",
}

ERROR_WORDS = {
    "error", "errors", "fault", "out", "missed", "miss", "failed", "fail",
    "net", "out of bounds", "long", "wide", "illegal", "violation",
    "foul", "overpass", "let", "touch",
}

# ── Team reference patterns ────────────────────────────────────────────────────
TEAM_A_WORDS = {"team a", "team one", "team 1", "home team", "home side", "side a"}
TEAM_B_WORDS = {"team b", "team two", "team 2", "away team", "away side", "side b"}

# ── Number words → digits ──────────────────────────────────────────────────────
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20,
}

# Pre-compiled regex patterns
_RE_PLAYER_NUM  = re.compile(
    r"player\s*(?:number\s*)?(?:#\s*)?(\d+)|#\s*(\d+)|jersey\s*(\d+)",
    re.IGNORECASE,
)
_RE_ACTION_WORD = re.compile(
    r"\b(" + "|".join(sorted(ACTION_ALIASES.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_RE_TEAM_A = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in TEAM_A_WORDS) + r")\b",
    re.IGNORECASE,
)
_RE_TEAM_B = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in TEAM_B_WORDS) + r")\b",
    re.IGNORECASE,
)
_RE_NUMBER_WORD = re.compile(
    r"\b(" + "|".join(NUMBER_WORDS.keys()) + r")\b",
    re.IGNORECASE,
)


class NLPExtractor:
    """
    Stateless extractor.  Call extract_events() with Whisper segments.
    """

    def extract_events(
        self,
        segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Parameters
        ----------
        segments : list from SpeechService.transcribe()
                   [{start, end, text}]

        Returns
        -------
        list of event dicts (only segments where an action was detected)
        """
        events = []
        for seg in segments:
            text  = seg.get("text", "")
            start = float(seg.get("start", 0.0))
            end   = float(seg.get("end", start))

            extracted = self._parse_segment(text, start, end)
            if extracted:
                events.append(extracted)
                logger.debug(
                    f"NLP [{start:.1f}s]: {extracted['event_type']} "
                    f"(player {extracted['player_number']}) → {extracted['result']}"
                )

        logger.info(f"NLPExtractor: extracted {len(events)} events from {len(segments)} segments")
        return events

    # ──────────────────────────────────────────────────────────────────────────

    def _parse_segment(
        self,
        text: str,
        start: float,
        end: float,
    ) -> Optional[Dict[str, Any]]:
        """Parse one Whisper segment. Returns event dict or None."""
        text_lower = text.lower()
        # Strip punctuation for word-set matching
        import re as _re
        text_words_only = _re.sub(r'[^\w\s]', ' ', text_lower)

        # ── Detect action type ─────────────────────────────────────────────────
        action_matches = _RE_ACTION_WORD.findall(text_lower)
        if not action_matches:
            return None     # no volleyball action word found → skip segment

        # Use first action found (most prominent in sentence)
        raw_action  = action_matches[0].lower()
        event_type  = ACTION_ALIASES.get(raw_action, "unknown")

        # Multiple distinct actions in one segment → take the first non-trivial one
        # (e.g. "great serve, followed by a spike" → serve takes priority by order)

        # ── Detect player number ───────────────────────────────────────────────
        player_number = self._extract_player_number(text_lower)

        # ── Detect team ────────────────────────────────────────────────────────
        team = None
        if _RE_TEAM_A.search(text_lower):
            team = "A"
        elif _RE_TEAM_B.search(text_lower):
            team = "B"

        # ── Detect result ──────────────────────────────────────────────────────
        result, result_conf = self._detect_result(text_words_only, event_type)

        # ── Compute confidence ─────────────────────────────────────────────────
        confidence = 0.5   # base for finding an action word
        if player_number is not None:
            confidence += 0.2
        if team is not None:
            confidence += 0.1
        if result != "neutral":
            confidence += result_conf * 0.2
        confidence = min(confidence, 1.0)

        return {
            "raw_text":      text,
            "start_time":    start,
            "end_time":      end,
            "event_type":    event_type,
            "player_number": player_number,
            "team":          team,
            "result":        result,
            "confidence":    round(confidence, 3),
        }

    def _extract_player_number(self, text: str) -> Optional[int]:
        """Extract jersey/player number from text."""
        # Digit pattern: "player 7", "player #7", "#7"
        m = _RE_PLAYER_NUM.search(text)
        if m:
            for g in m.groups():
                if g is not None:
                    try:
                        return int(g)
                    except ValueError:
                        pass

        # Word pattern: "player seven"
        player_ctx = re.search(
            r"player\s+([a-z]+)",
            text,
            re.IGNORECASE,
        )
        if player_ctx:
            word = player_ctx.group(1).lower()
            if word in NUMBER_WORDS:
                return NUMBER_WORDS[word]

        # Standalone number word ONLY when preceded by "by", "from", or "player"
        # e.g. "spike by seven" → player 7  (but NOT "position three")
        by_ctx = re.search(
            r"\b(?:by|from)\s+([a-z]+)\b",
            text,
            re.IGNORECASE,
        )
        if by_ctx:
            word = by_ctx.group(1).lower()
            if word in NUMBER_WORDS:
                return NUMBER_WORDS[word]

        return None

    def _detect_result(self, text: str, event_type: str) -> tuple:
        """Returns (result_str, confidence_bonus)."""
        words = set(text.lower().split())

        # "ace" for serve always means success
        if event_type == "serve" and "ace" in words:
            return "success", 1.0

        # Check success
        if words & SUCCESS_WORDS:
            return "success", 0.8

        # Check error
        if words & ERROR_WORDS:
            return "error", 0.8

        return "neutral", 0.0

    # ── Convenience: parse raw text directly (no timestamp) ───────────────────

    def parse_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse a single text string. Returns event or None."""
        return self._parse_segment(text, 0.0, 0.0)
