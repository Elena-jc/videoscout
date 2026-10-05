"""Evaluation data: one JSON object per line.

    {"qid": "demo-1", "video_id": "demo", "video_path": "data/demo/demo.mp4",
     "subtitle_path": "data/demo/demo.srt", "question": "...",
     "options": ["A. ...", "B. ..."], "answer": "B",
     "task_type": "OCR", "gt_windows": [[80, 120]]}

`gt_windows` (optional) are time spans that contain the answer. When present the
evaluator also scores whether the agent searched / looked at the right place,
which separates retrieval failures from perception failures.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


@dataclass
class QAItem:
    qid: str
    video_id: str
    question: str
    options: list[str]
    answer: str
    video_path: str | None = None
    subtitle_path: str | None = None
    task_type: str | None = None
    duration_group: str | None = None
    gt_windows: list[list[float]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_items(path: str | Path) -> list[QAItem]:
    base = Path(path).resolve().parent
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        known = {k: raw.pop(k) for k in list(raw) if k in QAItem.__dataclass_fields__ and k != "extra"}
        item = QAItem(**known, extra=raw)
        # Relative media paths are resolved against the JSONL file's directory.
        for attr in ("video_path", "subtitle_path"):
            value = getattr(item, attr)
            if value and not Path(value).is_absolute():
                setattr(item, attr, str((base / value).resolve()))
        items.append(item)
    return items


def save_items(items: list[QAItem], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            row = asdict(item)
            row.update(row.pop("extra") or {})
            f.write(json.dumps({k: v for k, v in row.items() if v is not None}, ensure_ascii=False) + "\n")


def _find_video(video_dir: Path, video_id: str) -> str | None:
    for ext in VIDEO_EXTENSIONS:
        candidate = video_dir / f"{video_id}{ext}"
        if candidate.exists():
            return str(candidate.resolve())
    return None


def convert_videomme(
    parquet: str | Path,
    video_dir: str | Path,
    out: str | Path,
    subtitle_dir: str | Path | None = None,
    duration: str = "long",
    max_videos: int | None = None,
    seed: int = 0,
) -> list[QAItem]:
    """Convert the Video-MME annotation parquet (lmms-lab/Video-MME) to our JSONL.

    Samples whole videos (all questions of a video), so each index is reused.
    """
    import pandas as pd

    df = pd.read_parquet(parquet)
    required = {"videoID", "question_id", "question", "options", "answer", "duration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"parquet is missing columns {sorted(missing)}; found {list(df.columns)}")
    if duration != "all":
        df = df[df["duration"] == duration]

    video_ids = sorted(df["videoID"].unique())
    if max_videos:
        video_ids = sorted(random.Random(seed).sample(video_ids, min(max_videos, len(video_ids))))

    video_dir, items, skipped = Path(video_dir), [], 0
    for vid in video_ids:
        path = _find_video(video_dir, vid)
        if path is None:
            skipped += 1
            continue
        srt = Path(subtitle_dir) / f"{vid}.srt" if subtitle_dir else None
        for _, row in df[df["videoID"] == vid].iterrows():
            items.append(
                QAItem(
                    qid=str(row["question_id"]),
                    video_id=vid,
                    question=str(row["question"]),
                    options=[str(o) for o in row["options"]],
                    answer=str(row["answer"]).strip().upper(),
                    video_path=path,
                    subtitle_path=str(srt.resolve()) if srt and srt.exists() else None,
                    task_type=str(row.get("task_type", "")) or None,
                    duration_group=str(row["duration"]),
                )
            )
    save_items(items, out)
    print(f"wrote {len(items)} questions from {len(video_ids) - skipped} videos to {out} ({skipped} videos not found)")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Video-MME annotations to VideoScout JSONL")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--subtitle-dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", default="long", choices=["short", "medium", "long", "all"])
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--seed", type=int, default=0)
    a = parser.parse_args()
    convert_videomme(a.parquet, a.video_dir, a.out, a.subtitle_dir, a.duration, a.max_videos, a.seed)


if __name__ == "__main__":
    main()
