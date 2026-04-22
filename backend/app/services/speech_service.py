"""
Speech-to-Knowledge Service
────────────────────────────
Converts spoken match commentary into structured text using Whisper ASR,
then passes to NLPExtractor to produce volleyball event records.

Pipeline
────────
  Audio file  ──► Whisper (openai-whisper)  ──► timestamped segments
  segments    ──► NLPExtractor              ──► List[SpeechEventDict]
  events      ──► DB (speech_events table)

Whisper model selection (auto-selects based on available VRAM / CPU):
  - tiny    : fastest, ~39M params, suitable for CPU
  - base    : good balance for CPU
  - small   : better accuracy, still CPU-feasible
  - medium  : needs GPU for real-time, but fine for post-processing
  - large-v3: best quality, needs GPU

We default to "base" which works fine on CPU for post-match processing.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Default model size — resolved at load time from settings or env
def _default_model_size() -> str:
    try:
        from app.config import settings
        return settings.WHISPER_MODEL
    except Exception:
        return os.getenv("WHISPER_MODEL", "base")

WHISPER_MODEL = _default_model_size()


class SpeechService:
    """
    Handles ASR transcription of match commentary audio/video.

    Usage
    -----
    svc = SpeechService()
    svc.load()
    segments = svc.transcribe(audio_path)  # list of {start, end, text}
    """

    def __init__(self, model_size: str = WHISPER_MODEL):
        self._model      = None
        self._model_size = model_size
        self._loaded     = False

    def load(self) -> bool:
        """Load Whisper model. Safe to call multiple times."""
        if self._loaded:
            return True
        try:
            import whisper
            logger.info(f"SpeechService: loading Whisper '{self._model_size}'...")
            self._model  = whisper.load_model(self._model_size)
            self._loaded = True
            logger.info(f"SpeechService: Whisper '{self._model_size}' loaded")
            return True
        except ImportError:
            logger.error(
                "SpeechService: 'openai-whisper' not installed. "
                "Run: pip install openai-whisper"
            )
            return False
        except Exception as e:
            logger.error(f"SpeechService: load failed: {e}")
            return False

    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio/video file and return timestamped segments.

        Parameters
        ----------
        audio_path : str — path to audio or video file (Whisper can handle both)
        language   : str — ISO 639-1 code, default 'en'

        Returns
        -------
        List of segment dicts:
            [{"start": 0.0, "end": 3.4, "text": "Great spike by player seven"}]
        """
        if not self._loaded:
            if not self.load():
                return []

        if not os.path.exists(audio_path):
            logger.error(f"SpeechService: file not found: {audio_path}")
            return []

        try:
            logger.info(f"SpeechService: transcribing {audio_path}")
            result = self._model.transcribe(
                audio_path,
                language=language,
                verbose=False,
                word_timestamps=False,      # segment-level timestamps are enough
                condition_on_previous_text=True,
                temperature=0.0,            # greedy decoding for determinism
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            )
            segments = []
            for seg in result.get("segments", []):
                text = seg.get("text", "").strip()
                if not text:
                    continue
                segments.append({
                    "start": round(float(seg["start"]), 2),
                    "end":   round(float(seg["end"]),   2),
                    "text":  text,
                })
            logger.info(
                f"SpeechService: transcribed {len(segments)} segments "
                f"from {os.path.basename(audio_path)}"
            )
            return segments
        except Exception as e:
            logger.error(f"SpeechService: transcription failed: {e}")
            return []

    def transcribe_video_audio(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Extract audio from video using ffmpeg, then transcribe.
        Whisper can handle video directly too, so this is just an alias.
        """
        return self.transcribe(video_path)

    def is_ready(self) -> bool:
        return self._loaded

    @property
    def model_size(self) -> str:
        return self._model_size
