"""
Jersey OCR Service
──────────────────
Reads jersey numbers (1-99) from player bounding box crops using EasyOCR.
"""

import logging
import re
import threading
from statistics import mode, StatisticsError
from typing import Optional, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_reader = None
_reader_lock = threading.Lock()
_reader_failed = False


def _get_reader():
    global _reader, _reader_failed
    if _reader_failed:
        return None
    if _reader is not None:
        return _reader
    with _reader_lock:
        # Re-check inside lock (double-checked locking)
        if _reader is not None or _reader_failed:
            return _reader if not _reader_failed else None
        try:
            import easyocr
            _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
            logger.info("JerseyOCR: EasyOCR reader initialised (GPU)")
        except Exception as exc:
            logger.warning("JerseyOCR: failed to init EasyOCR — %s", exc)
            _reader_failed = True
    return _reader


def _preprocess(crop: np.ndarray) -> np.ndarray:
    """Crop to number region, upscale 3×, apply CLAHE. Returns BGR image."""
    h, w = crop.shape[:2]
    y1 = int(h * 0.15)
    y2 = int(h * 0.60)
    x1 = int(w * 0.10)
    x2 = int(w * 0.90)
    region = crop[y1:y2, x1:x2]
    if region.size == 0:
        return crop
    scaled = cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def read_jersey_number(crop: np.ndarray) -> Optional[int]:
    """Read a jersey number (1-99) from a player bounding box crop. Returns None if unreadable."""
    reader = _get_reader()
    if reader is None or crop is None or crop.size == 0:
        return None
    # Reject crops too small for reliable digit recognition
    h, w = crop.shape[:2]
    if h < 20 or w < 10:
        return None
    try:
        processed = _preprocess(crop)
        results = reader.readtext(
            processed,
            allowlist="0123456789",
            detail=1,
            paragraph=False,
        )
        # Sort by confidence descending — pick highest-confidence digit result
        results_sorted = sorted(results, key=lambda r: r[2], reverse=True)
        for (_bbox, text, conf) in results_sorted:
            if conf < 0.45:
                continue
            digits = re.sub(r"\D", "", text)
            if not digits:
                continue
            number = int(digits)
            if 1 <= number <= 99:
                return number
    except MemoryError:
        logger.warning("JerseyOCR: MemoryError reading jersey number")
    except Exception as exc:
        logger.debug("JerseyOCR.read_jersey_number error: %s", exc)
    return None


def consensus_jersey(readings: List[Optional[int]]) -> Optional[int]:
    """
    Return the most common jersey number requiring strict majority (>50%).
    Requires at least 3 non-None readings for confidence. Returns None on tie, all-None, or insufficient data.
    """
    valid = [r for r in readings if r is not None]
    if len(valid) < 3:
        return None
    try:
        m = mode(valid)
    except StatisticsError:
        return None
    if valid.count(m) > len(valid) / 2:
        return m
    return None
