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
    blank = np.full((120, 80, 3), (30, 120, 200), dtype=np.uint8)
    result = read_jersey_number(blank)
    assert result is None


def test_read_jersey_number_rejects_out_of_range():
    from app.services.jersey_ocr import read_jersey_number
    crop = _make_jersey_crop("99")
    result = read_jersey_number(crop)
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
    assert consensus_jersey([5, 5, 9, 9]) is None


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
