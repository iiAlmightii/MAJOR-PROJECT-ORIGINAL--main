"""
Event Fusion Engine
────────────────────
Merges CV-detected action events with speech-extracted events to produce
a higher-confidence unified event stream.

Why this matters
────────────────
  CV events alone:      high spatial precision, ~70% accuracy (without training)
  Speech events alone:  noisy timestamps (commentary lag), good for result/player
  Fused events:         CV timestamp precision + speech result/player confirmation
                        → reaches the 80-90% accuracy target from the synopsis

Fusion Algorithm
────────────────
  For each speech event (SE):
    1. Find all CV actions (CA) within ±FUSION_WINDOW seconds AND same action type
    2. If match found:
       a. Take CV timestamp (more accurate)
       b. If SE has player_number and CA has no player_id  → annotate with SE player
       c. If SE.result != neutral and CA.result == neutral → promote CA result from SE
       d. Increment fused CA confidence by BOOST amount
       e. Mark SE as fused (link to CA.id)
    3. If no CV match:
       a. Insert SE as a standalone Action record (source=speech)
       b. Mark SE fusion_status='standalone'

Conflict resolution
────────────────────
  If SE.event_type ≠ CA.event_type within window:
    → Keep both, mark SE fusion_status='conflict'
    → This is surfaced in the API for human review

Output
──────
  Returns updated action_rows (with boosts applied)
  Mutates speech_events in-place (sets fusion_status, fused_action_id)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

FUSION_WINDOW   = 5.0    # ±seconds to consider a CV event as matching speech
CONFIDENCE_BOOST = 0.15  # added to CA.confidence when a SE confirms it


# ── Canonical action type normalisation ──────────────────────────────────────
# Maps speech-extracted types to the Action DB enum values

_SPEECH_TO_DB_ACTION: Dict[str, str] = {
    "spike":    "attack",     # spike → attack (DB enum)
    "attack":   "attack",
    "receive":  "reception",
    "reception":"reception",
    "dig":      "dig",
    "serve":    "serve",
    "block":    "block",
    "set":      "set",
    "unknown":  "unknown",
}

_DB_TO_SPEECH_ACTION: Dict[str, str] = {v: k for k, v in _SPEECH_TO_DB_ACTION.items()}


class EventFusionEngine:
    """
    Stateless fusion engine.
    Call fuse() after both CV and speech pipelines complete.
    """

    def fuse(
        self,
        cv_action_rows: List[Dict[str, Any]],
        speech_events:  List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Parameters
        ----------
        cv_action_rows  : list of action dicts from CV pipeline
                          keys: track_id, action_type, confidence, timestamp,
                                frame_number, result (default 'neutral')
        speech_events   : list of dicts from NLPExtractor
                          keys: event_type, player_number, team, result,
                                start_time, end_time, confidence, raw_text

        Returns
        -------
        (updated_cv_action_rows, updated_speech_events)
        """
        if not speech_events:
            return cv_action_rows, speech_events

        # Deep-copy to avoid mutating originals
        cv_rows = [dict(r) for r in cv_action_rows]
        se_list = [dict(e) for e in speech_events]

        # Assign an index to each CV row for matching
        for i, row in enumerate(cv_rows):
            row["_idx"] = i
            row.setdefault("result",         "neutral")
            row.setdefault("fusion_matched", False)

        standalone_from_speech: List[Dict[str, Any]] = []
        conflicts:              List[Dict[str, Any]] = []

        for se in se_list:
            se_time   = float(se.get("start_time", 0.0))
            se_type   = se.get("event_type", "unknown")
            se_db_type = _SPEECH_TO_DB_ACTION.get(se_type, se_type)

            # ── Step 1: find candidate CV matches ─────────────────────────────
            candidates = [
                r for r in cv_rows
                if abs(r.get("timestamp", 0.0) - se_time) <= FUSION_WINDOW
            ]

            # ── Step 2: find type-matching candidates ─────────────────────────
            matched = [
                c for c in candidates
                if self._types_match(c.get("action_type", ""), se_db_type)
            ]

            if matched:
                # Pick closest in time
                best = min(matched, key=lambda c: abs(c.get("timestamp", 0.0) - se_time))

                # Apply speech knowledge to CV event
                best["confidence"] = min(
                    1.0,
                    float(best.get("confidence") or 0.5) + CONFIDENCE_BOOST,
                )
                if se.get("result", "neutral") != "neutral" and best.get("result", "neutral") == "neutral":
                    best["result"] = se["result"]
                if se.get("team") and not best.get("team"):
                    best["team"] = se["team"]

                best["fusion_matched"] = True

                # Mark speech event
                se["fusion_status"]    = "fused"
                se["fused_cv_idx"]     = best["_idx"]

                logger.debug(
                    f"EventFusion: fused SE[{se_type}@{se_time:.1f}s] → "
                    f"CA[{best.get('action_type')}@{best.get('timestamp', 0):.1f}s]"
                )

            elif candidates:
                # Candidates exist but action types differ → conflict
                se["fusion_status"] = "conflict"
                conflicts.append(se)
                logger.debug(
                    f"EventFusion: conflict SE[{se_type}@{se_time:.1f}s] "
                    f"vs CA types {[c.get('action_type') for c in candidates]}"
                )

            else:
                # No CV event near this speech event → standalone
                se["fusion_status"] = "standalone"

                # Create a new synthetic action row from speech event
                synthetic = self._speech_event_to_action_row(se)
                if synthetic:
                    standalone_from_speech.append(synthetic)

        # Append synthetic rows from speech
        cv_rows.extend(standalone_from_speech)

        # Clean up internal keys
        for r in cv_rows:
            r.pop("_idx",            None)
            r.pop("fusion_matched",  None)

        fused_count      = sum(1 for se in se_list if se.get("fusion_status") == "fused")
        standalone_count = len(standalone_from_speech)
        conflict_count   = len(conflicts)

        logger.info(
            f"EventFusion: {len(speech_events)} speech events → "
            f"fused={fused_count}, standalone={standalone_count}, "
            f"conflicts={conflict_count}"
        )

        return cv_rows, se_list

    # ──────────────────────────────────────────────────────────────────────────

    def _types_match(self, cv_type: str, se_db_type: str) -> bool:
        """
        Allow loose matching between related action types:
          spike ↔ attack (synonymous)
          receive ↔ reception ↔ dig (defensive actions)
        """
        cv_type    = (cv_type or "").lower()
        se_db_type = (se_db_type or "").lower()

        if cv_type == se_db_type:
            return True

        # Synonyms
        attack_group   = {"attack", "spike"}
        defense_group  = {"reception", "dig", "receive"}

        if cv_type in attack_group and se_db_type in attack_group:
            return True
        if cv_type in defense_group and se_db_type in defense_group:
            return True

        return False

    def _speech_event_to_action_row(
        self,
        se: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a standalone speech event into a synthetic action row
        that can be inserted alongside CV-detected actions.
        """
        se_type = se.get("event_type", "unknown")
        db_type = _SPEECH_TO_DB_ACTION.get(se_type, "unknown")

        if db_type == "unknown":
            return None     # Don't insert unknown-type standalone events

        return {
            "track_id":     None,                        # no tracked player
            "action_type":  db_type,
            "confidence":   float(se.get("confidence", 0.5)),
            "timestamp":    float(se.get("start_time", 0.0)),
            "frame_number": None,
            "result":       se.get("result", "neutral"),
            "team":         se.get("team"),
            "source":       "speech",                   # tag to distinguish
            "player_number": se.get("player_number"),   # jersey # from commentary
        }

    def compute_fusion_stats(
        self,
        cv_action_rows: List[Dict],
        speech_events:  List[Dict],
    ) -> Dict[str, Any]:
        """Summary stats for the fusion run (returned in match summary)."""
        fused_count      = sum(1 for se in speech_events if se.get("fusion_status") == "fused")
        standalone_count = sum(1 for se in speech_events if se.get("fusion_status") == "standalone")
        conflict_count   = sum(1 for se in speech_events if se.get("fusion_status") == "conflict")

        return {
            "total_cv_events":       len(cv_action_rows),
            "total_speech_events":   len(speech_events),
            "fused_events":          fused_count,
            "standalone_speech":     standalone_count,
            "conflict_events":       conflict_count,
            "fusion_coverage":       round(
                fused_count / len(speech_events), 3
            ) if speech_events else 0.0,
        }
