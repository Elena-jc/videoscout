"""Frame access and image helpers built on OpenCV."""

from __future__ import annotations

import base64
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class VideoReader:
    """Read frames by timestamp. Each method opens its own capture, so one reader
    can be shared across threads."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {self.path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self.duration = self.frame_count / self.fps if self.frame_count > 0 else 0.0

    def _clamp_index(self, t: float) -> int:
        return min(max(int(round(t * self.fps)), 0), max(self.frame_count - 1, 0))

    def frames_at(self, times: Sequence[float]) -> list[tuple[float, np.ndarray]]:
        """Random access by seeking. Good for a handful of frames."""
        cap = cv2.VideoCapture(self.path)
        out = []
        try:
            for t in sorted(times):
                idx = self._clamp_index(t)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok:
                    out.append((idx / self.fps, frame))
        finally:
            cap.release()
        return out

    def sample(self, t_start: float, t_end: float, n: int) -> list[tuple[float, np.ndarray]]:
        """n frames spread evenly over [t_start, t_end] (bin centres)."""
        t_start = max(0.0, t_start)
        t_end = min(self.duration, t_end) if self.duration else t_end
        if t_end <= t_start or n <= 1:
            return self.frames_at([(t_start + t_end) / 2])
        step = (t_end - t_start) / n
        return self.frames_at([t_start + (i + 0.5) * step for i in range(n)])

    def iter_frames_at(self, times: Sequence[float]) -> Iterator[tuple[float, np.ndarray]]:
        """Sequential decode for many timestamps: grab() every frame, decode only
        the ones we need. Much faster than seeking for dense sampling."""
        pending = sorted((self._clamp_index(t), t) for t in times)
        cap = cv2.VideoCapture(self.path)
        i = frame_idx = 0
        try:
            while i < len(pending):
                if not cap.grab():
                    break
                if pending[i][0] <= frame_idx:
                    ok, frame = cap.retrieve()
                    while i < len(pending) and pending[i][0] <= frame_idx:
                        if ok:
                            yield pending[i][1], frame
                        i += 1
                frame_idx += 1
        finally:
            cap.release()


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def to_jpeg_b64(img: np.ndarray, max_side: int = 768, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", resize_max_side(img, max_side), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def crop_normalized(
    img: np.ndarray, box: tuple[float, float, float, float], margin: float = 0.25, min_px: int = 96
) -> np.ndarray:
    """Crop a normalized (x1, y1, x2, y2) box with some context around it."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1) * w, (y2 - y1) * h
    cx, cy = (x1 + x2) / 2 * w, (y1 + y2) / 2 * h
    half_w = max(bw * (1 + 2 * margin), min_px) / 2
    half_h = max(bh * (1 + 2 * margin), min_px) / 2
    left, right = int(max(0, cx - half_w)), int(min(w, cx + half_w))
    top, bottom = int(max(0, cy - half_h)), int(min(h, cy + half_h))
    if right <= left or bottom <= top:
        return img
    return img[top:bottom, left:right]
