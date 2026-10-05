"""Offline indexing pipeline: video -> clips, embeddings, tracks, text, events, storyline."""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import Config
from ..llm import LLMClient, text_of
from ..video import VideoReader, resize_max_side
from .memory import cut_events, summarize_events, write_storyline
from .store import Segment, write_index
from .subtitles import assign_to_segments, parse_srt

EMBED_BATCH = 32
CLIP_BATCH = 16
THUMB_SIDE = 448


def make_segments(duration: float, seg_len: float) -> list[Segment]:
    """Fixed-length windows. This is the video analogue of chunk size in text RAG:
    too long and one embedding blurs several events, too short and context is lost.
    The retriever's neighbour-aware MMR and the inspect tool's free time window
    compensate for events that straddle a boundary; events (index/memory.py) group
    clips into variable-length scenes on top."""
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


def clip_thumbnails(reader: VideoReader, segments: list[Segment], per_clip: int) -> dict[int, list[tuple[float, bytes]]]:
    """`per_clip` evenly spaced JPEG thumbnails per clip, in one sequential decode."""
    owner: dict[float, int] = {}
    for seg in segments:
        step = (seg.t_end - seg.t_start) / per_clip
        for k in range(per_clip):
            owner[round(seg.t_start + (k + 0.5) * step, 3)] = seg.seg_id
    thumbs: dict[int, list[tuple[float, bytes]]] = defaultdict(list)
    for t, frame in reader.iter_frames_at(list(owner)):
        ok, jpeg = cv2.imencode(".jpg", resize_max_side(frame, THUMB_SIDE), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            thumbs[owner[t]].append((t, jpeg.tobytes()))
    return dict(thumbs)


def _decode(items: list[tuple[float, bytes]]) -> list[np.ndarray]:
    return [cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) for _, jpeg in items]


def _llm_writer(llm: LLMClient) -> Callable[[str], str]:
    def write(prompt: str) -> str:
        return text_of(llm.create("captioner", [{"role": "user", "content": [{"type": "text", "text": prompt}]}], max_tokens=1024))

    return write


def build_index(
    video_path: str | Path,
    out_dir: str | Path,
    cfg: Config,
    srt_path: str | Path | None = None,
    with_dense: bool = True,
    with_tracks: bool = True,
    llm: LLMClient | None = None,
    log: Callable[[str], None] = print,
    captioner: Any = None,
    clip_embedder: Any = None,
) -> Path:
    """`captioner` (caption_clip, write) and `clip_embedder` (embed_clips) can be passed
    in; otherwise they are loaded as configured (cfg.index.captioner / clip_embedder)."""
    icfg = cfg.index
    video_path = str(Path(video_path).resolve())
    reader = VideoReader(video_path)
    segments = make_segments(reader.duration, icfg.segment_seconds)
    log(f"video: {reader.duration:.1f}s @ {reader.fps:.2f} fps, {reader.width}x{reader.height}; {len(segments)} clips")

    # 1. Frame-level dense index: SigLIP 2 embeddings of keyframes (kept for ablations).
    keyframes: list[tuple[int, float]] = []
    embeddings = np.zeros((0, 0), np.float32)
    if with_dense and icfg.siglip_model:
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
        log(f"embedded {len(keyframes)} keyframes with SigLIP 2 in {time.perf_counter() - t0:.1f}s")

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

    # 3. Text: subtitles, then clip captions.
    if srt_path:
        cues = parse_srt(srt_path)
        assign_to_segments(cues, segments)
        log(f"attached {len(cues)} subtitle cues")

    thumbs = clip_thumbnails(reader, segments, max(1, icfg.clip_frames))
    mode = icfg.captioner if captioner is None else "local"
    if mode == "local":
        t0 = time.perf_counter()
        if captioner is None:
            from .qwen3vl import LocalCaptioner

            captioner = LocalCaptioner(icfg.captioner_model, icfg.device)
        for i, seg in enumerate(segments, start=1):
            frames = _decode(thumbs.get(seg.seg_id, []))
            if frames:
                seg.caption = captioner.caption_clip(frames, seg.t_end - seg.t_start)
            if i % 25 == 0:
                log(f"  captioned {i}/{len(segments)} clips...")
        log(f"captioned {len(segments)} clips locally in {time.perf_counter() - t0:.1f}s")
    elif mode == "llm":
        from .captioner import caption_segments

        if llm is None:
            raise ValueError("captioner=llm needs an LLM client")
        caption_segments(llm, reader, segments, mode=icfg.caption_mode, log=log)
        log(f"captioned clips; captioner usage: {llm.meter.snapshot()}")

    # 4. Clip-level dense index: Qwen3-VL embeddings of (thumbnails + subtitles + caption).
    clip_emb = None
    if with_dense and (icfg.clip_embedder or clip_embedder is not None):
        t0 = time.perf_counter()
        if clip_embedder is None:
            from .qwen3vl import get_qwen_embedder

            clip_embedder = get_qwen_embedder(icfg.clip_embedder, icfg.device)
        rows = []
        for i in range(0, len(segments), CLIP_BATCH):
            batch_segs = segments[i:i + CLIP_BATCH]
            rows.append(clip_embedder.embed_clips(
                [(_decode(thumbs.get(s.seg_id, [])), " ".join(p for p in (s.subtitle, s.caption) if p)) for s in batch_segs]
            ))
        clip_emb = np.concatenate(rows)
        log(f"embedded {len(segments)} clips with {icfg.clip_embedder or 'the clip embedder'} in {time.perf_counter() - t0:.1f}s")

    # 5. Coarse memory: events cut where the content changes, their summaries, the storyline.
    t0 = time.perf_counter()
    write = captioner.write if mode == "local" else _llm_writer(llm) if mode == "llm" else None
    events = cut_events(segments, clip_emb, icfg.event_min_seconds, icfg.event_max_seconds)
    summarize_events(events, segments, write)
    storyline = write_storyline(events, reader.duration, write)
    log(f"memory: {len(events)} events and a storyline in {time.perf_counter() - t0:.1f}s")
    if mode == "local" and hasattr(captioner, "unload"):
        captioner.unload()  # free the GPU for the query-time models

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
        "siglip_model": icfg.siglip_model if keyframes else None,
        "clip_embedder": (icfg.clip_embedder or "custom") if clip_emb is not None else None,
        "yolo_weights": icfg.yolo_weights if with_tracks else None,
        "captioner": icfg.captioner_model if mode == "local" else mode,
        "has_subtitles": bool(srt_path),
        "has_captions": mode in ("local", "llm"),
        "storyline": storyline,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = write_index(out_dir, meta, segments, keyframes, embeddings, tracks, detections,
                      events=events, clip_emb=clip_emb, thumbs=thumbs)
    log(f"index written to {out}")
    return out
