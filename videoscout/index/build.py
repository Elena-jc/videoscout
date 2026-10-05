"""Offline indexing pipeline: video -> segments, keyframe embeddings, tracks, text."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..config import Config
from ..llm import LLMClient
from ..video import VideoReader
from .store import Segment, write_index
from .subtitles import assign_to_segments, parse_srt

EMBED_BATCH = 32


def make_segments(duration: float, seg_len: float) -> list[Segment]:
    """Fixed-length windows. This is the video analogue of chunk size in text RAG:
    too long and one embedding blurs several events, too short and context is lost.
    The retriever's neighbour-aware MMR and the inspect tool's free time window
    compensate for events that straddle a boundary."""
    n = max(1, math.ceil(duration / seg_len))
    return [Segment(i, round(i * seg_len, 3), round(min((i + 1) * seg_len, duration), 3)) for i in range(n)]


def keyframe_times(segments: list[Segment], every: float) -> list[tuple[int, float]]:
    out = []
    for seg in segments:
        t = seg.t_start + every / 2
        while t < seg.t_end:
            out.append((seg.seg_id, round(t, 3)))
            t += every
        if not out or out[-1][0] != seg.seg_id:  # very short last segment
            out.append((seg.seg_id, round((seg.t_start + seg.t_end) / 2, 3)))
    return out


def build_index(
    video_path: str | Path,
    out_dir: str | Path,
    cfg: Config,
    srt_path: str | Path | None = None,
    with_dense: bool = True,
    with_tracks: bool = True,
    llm: LLMClient | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    icfg = cfg.index
    video_path = str(Path(video_path).resolve())
    reader = VideoReader(video_path)
    segments = make_segments(reader.duration, icfg.segment_seconds)
    log(f"video: {reader.duration:.1f}s @ {reader.fps:.2f} fps, {reader.width}x{reader.height}; {len(segments)} segments")

    # 1. Dense index: SigLIP embeddings of keyframes, decoded sequentially in batches.
    keyframes: list[tuple[int, float]] = []
    embeddings = np.zeros((0, 0), np.float32)
    if with_dense:
        from .embedder import get_embedder

        t0 = time.perf_counter()
        embedder = get_embedder(icfg.siglip_model, icfg.device)
        seg_of = {t: seg_id for seg_id, t in keyframe_times(segments, icfg.keyframe_every)}
        chunks, batch, batch_times = [], [], []
        for t, frame in reader.iter_frames_at(list(seg_of)):
            batch.append(frame)
            batch_times.append(t)
            if len(batch) == EMBED_BATCH:
                chunks.append(embedder.embed_images(batch))
                keyframes += [(seg_of[x], x) for x in batch_times]
                batch, batch_times = [], []
        if batch:
            chunks.append(embedder.embed_images(batch))
            keyframes += [(seg_of[x], x) for x in batch_times]
        embeddings = np.concatenate(chunks) if chunks else embeddings
        log(f"embedded {len(keyframes)} keyframes on {embedder.device} in {time.perf_counter() - t0:.1f}s")

    # 2. Structured memory: detection + tracking.
    tracks, detections = [], []
    if with_tracks:
        from .tracker import run_tracking, tag_segments

        t0 = time.perf_counter()
        result = run_tracking(
            video_path, reader.fps, icfg.track_fps, icfg.yolo_weights, icfg.yolo_conf, icfg.device,
            progress=lambda i: log(f"  tracking frame {i}..."),
            tracker=icfg.tracker,
        )
        tracks, detections = result.tracks, result.detections
        tag_segments(segments, detections, tracks)
        log(f"tracked {len(tracks)} objects ({len(detections)} detections) in {time.perf_counter() - t0:.1f}s")

    # 3. Text channels for the sparse index.
    if srt_path:
        cues = parse_srt(srt_path)
        assign_to_segments(cues, segments)
        log(f"attached {len(cues)} subtitle cues")
    if icfg.caption:
        from .captioner import caption_segments

        if llm is None:
            raise ValueError("captioning needs an LLM client")
        caption_segments(llm, reader, segments, mode=icfg.caption_mode, log=log)
        log(f"captioned segments; captioner usage: {llm.meter.snapshot()}")

    try:
        stored_path = os.path.relpath(video_path, Path(out_dir).resolve())
    except ValueError:  # different drive on Windows
        stored_path = video_path
    meta = {
        "video_path": stored_path,
        "duration": reader.duration,
        "fps": reader.fps,
        "width": reader.width,
        "height": reader.height,
        "segment_seconds": icfg.segment_seconds,
        "keyframe_every": icfg.keyframe_every,
        "track_fps": icfg.track_fps,
        "siglip_model": icfg.siglip_model if with_dense else None,
        "yolo_weights": icfg.yolo_weights if with_tracks else None,
        "has_subtitles": bool(srt_path),
        "has_captions": bool(icfg.caption),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = write_index(out_dir, meta, segments, keyframes, embeddings, tracks, detections)
    log(f"index written to {out}")
    return out
