"""Detection + multi-object tracking (YOLO11 + ByteTrack through Ultralytics),
summarised into per-track motion statistics the agent can query with SQL."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .store import Segment

# Tracks seen in fewer sampled frames than this are almost always false positives.
MIN_OBSERVATIONS = 2

DetectionRow = tuple[int, float, float, float, float, float, float]  # track_id, t, x1, y1, x2, y2, conf


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class TrackingResult:
    detections: list[DetectionRow]
    tracks: list[dict[str, Any]]


def run_tracking(
    video_path: str,
    fps: float,
    track_fps: float,
    weights: str,
    conf: float,
    device: str = "auto",
    progress: Callable[[int], None] | None = None,
    tracker: str = "bytetrack.yaml",
) -> TrackingResult:
    from ultralytics import YOLO

    from ..config import weights_path

    model = YOLO(weights_path(weights))
    names = model.names
    stride = max(1, int(round(fps / track_fps)))
    rows: list[DetectionRow] = []
    label_votes: dict[int, Counter] = defaultdict(Counter)

    results = model.track(
        source=video_path,
        stream=True,
        tracker=tracker,
        vid_stride=stride,
        conf=conf,
        device=resolve_device(device),
        verbose=False,
    )
    for i, result in enumerate(results):
        # With vid_stride the loader yields every `stride`-th frame, in order.
        t = round(i * stride / fps, 3)
        boxes = result.boxes
        if progress and i % 200 == 0:
            progress(i)
        if boxes is None or boxes.id is None:
            continue
        for tid, cls, p, (x1, y1, x2, y2) in zip(
            boxes.id.int().tolist(), boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxyn.tolist()
        ):
            rows.append((tid, t, x1, y1, x2, y2, p))
            label_votes[tid][names[cls]] += 1

    tracks = summarize_tracks(rows, {tid: votes.most_common(1)[0][0] for tid, votes in label_votes.items()})
    kept = {tr["track_id"] for tr in tracks}
    return TrackingResult(detections=[r for r in rows if r[0] in kept], tracks=tracks)


def summarize_tracks(rows: list[DetectionRow], labels: dict[int, str]) -> list[dict[str, Any]]:
    """Per-track statistics. Coordinates are normalized, so path_length = 1.0 means
    the object moved a distance equal to the frame width/height."""
    by_track: dict[int, list[DetectionRow]] = defaultdict(list)
    for row in rows:
        by_track[row[0]].append(row)

    tracks = []
    for tid, obs in by_track.items():
        if len(obs) < MIN_OBSERVATIONS:
            continue
        obs.sort(key=lambda r: r[1])
        centers = [((r[2] + r[4]) / 2, (r[3] + r[5]) / 2) for r in obs]
        path = sum(math.dist(a, b) for a, b in zip(centers, centers[1:]))
        tracks.append(
            {
                "track_id": tid,
                "label": labels[tid],
                "t_first": obs[0][1],
                "t_last": obs[-1][1],
                "duration": round(obs[-1][1] - obs[0][1], 3),
                "n_obs": len(obs),
                "mean_conf": round(sum(r[6] for r in obs) / len(obs), 4),
                "cx_first": round(centers[0][0], 4),
                "cy_first": round(centers[0][1], 4),
                "cx_last": round(centers[-1][0], 4),
                "cy_last": round(centers[-1][1], 4),
                "path_length": round(path, 4),
                "net_displacement": round(math.dist(centers[0], centers[-1]), 4),
                "mean_area": round(sum((r[4] - r[2]) * (r[5] - r[3]) for r in obs) / len(obs), 5),
            }
        )
    return tracks


def tag_segments(segments: list[Segment], rows: list[DetectionRow], tracks: list[dict[str, Any]]) -> None:
    """Write 'objects: person x3, car x1' into each segment (distinct tracks per label)."""
    if not segments:
        return
    label_of = {tr["track_id"]: tr["label"] for tr in tracks}
    seg_len = segments[0].t_end - segments[0].t_start
    per_seg: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for tid, t, *_ in rows:
        if tid in label_of:
            idx = min(int(t // seg_len), len(segments) - 1)
            per_seg[idx][label_of[tid]].add(tid)
    for idx, labels in per_seg.items():
        parts = sorted(labels.items(), key=lambda kv: -len(kv[1]))
        segments[idx].objects = "objects: " + ", ".join(f"{label} x{len(ids)}" for label, ids in parts)
