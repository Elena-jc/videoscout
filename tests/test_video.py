from __future__ import annotations

import cv2
from conftest import write_tiny_video

from videoscout.video import browser_playable, fourcc_of, make_browser_preview


def test_opencv_mp4v_is_converted_to_a_browser_codec(tmp_path):
    original = write_tiny_video(tmp_path / "clip.mp4", seconds=4)
    assert fourcc_of(original) in {"mp4v", "fmp4"} and not browser_playable(original)  # MPEG-4 Part 2

    preview = make_browser_preview(original, tmp_path / "preview")
    assert browser_playable(preview), fourcc_of(preview)  # H.264 on Windows, VP9/VP8 WebM elsewhere
    a, b = cv2.VideoCapture(str(original)), cv2.VideoCapture(str(preview))
    try:
        assert abs(a.get(cv2.CAP_PROP_FRAME_COUNT) - b.get(cv2.CAP_PROP_FRAME_COUNT)) <= 1
    finally:
        a.release()
        b.release()
