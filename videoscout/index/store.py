"""On-disk video index: one directory per video.

    index_dir/
      meta.json        video metadata, build settings, storyline (L0 memory)
      index.sqlite     segments (clips), events, keyframes, tracks, detections (queried by the agent)
      keyframes.npy    L2-normalised SigLIP 2 frame embeddings, row i <-> keyframes.kf_id = i
      clips.npy        L2-normalised Qwen3-VL clip embeddings, row i <-> segments.seg_id = i
      thumbs.sqlite    a few JPEG thumbnails per clip (reranking, captions, UI)

This is the video equivalent of a RAG document store: segments are the chunks,
clip and keyframe embeddings are the dense index, segment text (subtitles + object
tags + captions) feeds the sparse BM25 index, events and the storyline are the
coarse levels of the memory, and the track tables are structured memory the agent
can query with SQL.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..video import fmt_ts

SCHEMA = """
CREATE TABLE segments (
    seg_id   INTEGER PRIMARY KEY,
    t_start  REAL NOT NULL,
    t_end    REAL NOT NULL,
    subtitle TEXT NOT NULL DEFAULT '',
    objects  TEXT NOT NULL DEFAULT '',
    caption  TEXT NOT NULL DEFAULT ''
);
CREATE TABLE events (
    event_id   INTEGER PRIMARY KEY,
    t_start    REAL NOT NULL,
    t_end      REAL NOT NULL,
    seg_first  INTEGER NOT NULL,
    seg_last   INTEGER NOT NULL,
    summary    TEXT NOT NULL DEFAULT ''
);
CREATE TABLE keyframes (
    kf_id  INTEGER PRIMARY KEY,
    seg_id INTEGER NOT NULL,
    t      REAL NOT NULL
);
CREATE TABLE tracks (
    track_id         INTEGER PRIMARY KEY,
    label            TEXT NOT NULL,
    t_first          REAL NOT NULL,
    t_last           REAL NOT NULL,
    duration         REAL NOT NULL,
    n_obs            INTEGER NOT NULL,
    mean_conf        REAL NOT NULL,
    cx_first         REAL, cy_first REAL,
    cx_last          REAL, cy_last  REAL,
    path_length      REAL,
    net_displacement REAL,
    mean_area        REAL
);
CREATE TABLE detections (
    track_id INTEGER NOT NULL,
    t        REAL NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    conf     REAL
);
CREATE INDEX idx_det_track ON detections(track_id, t);
CREATE INDEX idx_det_t ON detections(t);
CREATE INDEX idx_tracks_label ON tracks(label);
"""

TRACK_COLUMNS = (
    "track_id", "label", "t_first", "t_last", "duration", "n_obs", "mean_conf",
    "cx_first", "cy_first", "cx_last", "cy_last", "path_length", "net_displacement", "mean_area",
)


@dataclass
class Event:
    """A run of consecutive clips that belong together (see index/memory.py)."""

    event_id: int
    t_start: float
    t_end: float
    seg_first: int
    seg_last: int
    summary: str = ""


@dataclass
class Segment:
    seg_id: int
    t_start: float
    t_end: float
    subtitle: str = ""
    objects: str = ""
    caption: str = ""

    @property
    def document(self) -> str:
        """Text used by the sparse (BM25) retriever."""
        return " ".join(p for p in (self.subtitle, self.objects, self.caption) if p)


def write_index(
    out_dir: str | Path,
    meta: dict[str, Any],
    segments: list[Segment],
    keyframes: list[tuple[int, float]],
    embeddings: np.ndarray,
    tracks: list[dict[str, Any]],
    detections: Iterable[tuple[int, float, float, float, float, float, float]],
    events: Sequence[Event] = (),
    clip_emb: np.ndarray | None = None,
    thumbs: dict[int, list[tuple[float, bytes]]] | None = None,
) -> Path:
    """`thumbs` maps seg_id -> [(t, jpeg bytes)] in time order."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "index.sqlite"
    for stale in (db_path, out / "thumbs.sqlite", out / "clips.npy"):
        if stale.exists():
            stale.unlink()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO segments VALUES (?,?,?,?,?,?)",
            [(s.seg_id, s.t_start, s.t_end, s.subtitle, s.objects, s.caption) for s in segments],
        )
        conn.executemany(
            "INSERT INTO events VALUES (?,?,?,?,?,?)",
            [(e.event_id, e.t_start, e.t_end, e.seg_first, e.seg_last, e.summary) for e in events],
        )
        conn.executemany(
            "INSERT INTO keyframes VALUES (?,?,?)",
            [(i, seg_id, t) for i, (seg_id, t) in enumerate(keyframes)],
        )
        conn.executemany(
            f"INSERT INTO tracks VALUES ({','.join('?' * len(TRACK_COLUMNS))})",
            [tuple(tr[c] for c in TRACK_COLUMNS) for tr in tracks],
        )
        conn.executemany("INSERT INTO detections VALUES (?,?,?,?,?,?,?)", detections)
        conn.commit()
    finally:
        conn.close()

    np.save(out / "keyframes.npy", embeddings.astype(np.float16))
    if clip_emb is not None and clip_emb.size:
        np.save(out / "clips.npy", clip_emb.astype(np.float16))
    if thumbs:
        # Kept out of index.sqlite so the agent's SQL tool never sees image blobs.
        tconn = sqlite3.connect(out / "thumbs.sqlite")
        try:
            tconn.execute("CREATE TABLE thumbs (seg_id INTEGER, k INTEGER, t REAL, jpeg BLOB, PRIMARY KEY (seg_id, k))")
            tconn.executemany(
                "INSERT INTO thumbs VALUES (?,?,?,?)",
                [(seg_id, k, t, jpeg) for seg_id, items in thumbs.items() for k, (t, jpeg) in enumerate(items)],
            )
            tconn.commit()
        finally:
            tconn.close()
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


class VideoIndex:
    """Read-only view of a built index, loaded once and shared by all tools."""

    def __init__(self, index_dir: str | Path):
        self.dir = Path(index_dir).resolve()
        self.meta: dict[str, Any] = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.db_path = self.dir / "index.sqlite"
        # Stored relative to the index directory when possible, so a project can be moved.
        video = Path(self.meta["video_path"])
        self.video_path = str(video if video.is_absolute() else (self.dir / video).resolve())
        self.duration = float(self.meta["duration"])

        conn = self.connect()
        try:
            self.segments = [Segment(*row) for row in conn.execute("SELECT * FROM segments ORDER BY seg_id")]
            self.events = [Event(*row) for row in conn.execute("SELECT * FROM events ORDER BY event_id")]
            kf = conn.execute("SELECT seg_id, t FROM keyframes ORDER BY kf_id").fetchall()
            self.label_counts = dict(
                conn.execute("SELECT label, COUNT(*) FROM tracks GROUP BY label ORDER BY COUNT(*) DESC")
            )
        finally:
            conn.close()

        self.kf_seg = np.array([r[0] for r in kf], dtype=np.int64)
        self.kf_times = np.array([r[1] for r in kf], dtype=np.float64)
        emb_path = self.dir / "keyframes.npy"
        self.kf_emb = np.load(emb_path).astype(np.float32) if emb_path.exists() else np.zeros((0, 0), np.float32)
        clip_path = self.dir / "clips.npy"
        self.clip_emb = np.load(clip_path).astype(np.float32) if clip_path.exists() else np.zeros((0, 0), np.float32)
        self.storyline: str = self.meta.get("storyline", "")
        self._thumbs_path = self.dir / "thumbs.sqlite"

    def connect(self) -> sqlite3.Connection:
        """Read-only connection: the OS-level guard behind the SQL tool's authorizer."""
        return sqlite3.connect(f"{self.db_path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)

    def clip_thumbs(self, seg_id: int, limit: int | None = None) -> list[tuple[float, np.ndarray]]:
        """The stored thumbnails of one clip as (t, BGR image), in time order."""
        if not self._thumbs_path.exists():
            return []
        conn = sqlite3.connect(f"{self._thumbs_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT t, jpeg FROM thumbs WHERE seg_id=? ORDER BY k", (seg_id,)).fetchall()
        finally:
            conn.close()
        if limit is not None and len(rows) > limit:  # spread the picks over the clip
            rows = [rows[round(i * (len(rows) - 1) / max(limit - 1, 1))] for i in range(limit)]
        return [(t, cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)) for t, jpeg in rows]

    def track_boxes(self, track_id: int, t_start: float, t_end: float) -> list[tuple[float, tuple[float, float, float, float]]]:
        conn = self.connect()
        try:
            rows = conn.execute(
                "SELECT t, x1, y1, x2, y2 FROM detections WHERE track_id=? AND t BETWEEN ? AND ? ORDER BY t",
                (track_id, t_start, t_end),
            ).fetchall()
        finally:
            conn.close()
        return [(r[0], (r[1], r[2], r[3], r[4])) for r in rows]

    def overview(self) -> str:
        """Short description of what the index contains, given to the planner up front."""
        objects = ", ".join(f"{label} ({n} tracks)" for label, n in list(self.label_counts.items())[:12]) or "none"
        text = (
            f"Video duration: {fmt_ts(self.duration)} ({self.duration:.1f} s), indexed as "
            f"{len(self.segments)} clips of {self.meta['segment_seconds']:g} s grouped into {len(self.events)} events.\n"
            f"Subtitles available: {'yes' if self.meta.get('has_subtitles') else 'no'}. "
            f"Clip captions available: {'yes' if self.meta.get('has_captions') else 'no'}.\n"
            f"Tracked object classes: {objects}."
        )
        if self.storyline:
            text += f"\nStoryline (machine-written from captions and subtitles; a map, not evidence): {self.storyline}"
        return text
