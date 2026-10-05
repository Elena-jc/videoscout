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
import re
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


_TIMESTAMP = r"\d{1,2}(?::\d{2}){1,2}"
_SPAN = re.compile(rf"({_TIMESTAMP})\s*-\s*({_TIMESTAMP})")
_OPTION = re.compile(r"^\(([A-Z])\)\s*(.*)$")


def _seconds(ts: str) -> float:
    value = 0.0
    for part in ts.split(":"):
        value = value * 60 + float(part)
    return value


def parse_time_reference(text: str) -> list[list[float]]:
    """LVBench time_reference, e.g. '00:15-00:19' or '1:02:03-1:03:00, 1:10:00-1:11:00'."""
    return [[_seconds(a), _seconds(b)] for a, b in _SPAN.findall(text or "")]


def split_lvbench_question(text: str) -> tuple[str, list[str]]:
    """'What year...?\n(A) 1636\n(B) 1366' -> ('What year...?', ['A. 1636', 'B. 1366'])."""
    stem, options = [], []
    for line in text.splitlines():
        m = _OPTION.match(line.strip())
        if m:
            options.append(f"{m.group(1)}. {m.group(2).strip()}")
        elif options:  # an option that wraps onto the next line
            options[-1] += " " + line.strip()
        elif line.strip():
            stem.append(line.strip())
    return " ".join(stem), options


def convert_lvbench(
    meta: str | Path,
    video_dir: str | Path,
    out: str | Path,
    max_videos: int | None = None,
    per_video: int | None = None,
    seed: int = 0,
    keys: list[str] | None = None,
) -> list[QAItem]:
    """Convert LVBench (zai-org/LVBench video_info.meta.jsonl) to our JSONL.

    Videos are YouTube ids (`key`); only videos found in `video_dir` are kept. Each
    question's time_reference becomes gt_windows, so the evaluator can score whether
    the agent searched / looked at the right moment.
    """
    rows = [json.loads(line) for line in Path(meta).read_text(encoding="utf-8").splitlines() if line.strip()]
    if keys:
        wanted = set(keys)
        rows = [r for r in rows if r["key"] in wanted]
    video_dir, rng = Path(video_dir), random.Random(seed)
    available = [r for r in rows if _find_video(video_dir, r["key"])]
    if max_videos:
        available = sorted(rng.sample(available, min(max_videos, len(available))), key=lambda r: r["key"])
    items = []
    for row in available:
        qa = list(row["qa"])
        if per_video and len(qa) > per_video:
            qa = sorted(rng.sample(qa, per_video), key=lambda q: str(q["uid"]))
        for q in qa:
            question, options = split_lvbench_question(q["question"])
            items.append(QAItem(
                qid=f"lvb-{q['uid']}",
                video_id=row["key"],
                question=question,
                options=options,
                answer=str(q["answer"]).strip().upper(),
                video_path=_find_video(video_dir, row["key"]),
                task_type=",".join(q.get("question_type") or []) or None,
                gt_windows=parse_time_reference(q.get("time_reference", "")) or None,
                extra={"video_type": row.get("type")},
            ))
    save_items(items, out)
    print(f"wrote {len(items)} questions from {len(available)} videos to {out} "
          f"({len(rows) - len([r for r in rows if _find_video(video_dir, r['key'])])} videos not downloaded)")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert benchmark annotations to VideoScout JSONL")
    sub = parser.add_subparsers(dest="benchmark", required=True)

    p = sub.add_parser("videomme", help="Video-MME (lmms-lab/Video-MME parquet)")
    p.add_argument("--parquet", required=True)
    p.add_argument("--video-dir", required=True)
    p.add_argument("--subtitle-dir")
    p.add_argument("--out", required=True)
    p.add_argument("--duration", default="long", choices=["short", "medium", "long", "all"])
    p.add_argument("--max-videos", type=int)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("lvbench", help="LVBench (zai-org/LVBench video_info.meta.jsonl)")
    p.add_argument("--meta", required=True)
    p.add_argument("--video-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-videos", type=int)
    p.add_argument("--per-video", type=int, help="sample at most this many questions per video")
    p.add_argument("--seed", type=int, default=0)

    a = parser.parse_args()
    if a.benchmark == "videomme":
        convert_videomme(a.parquet, a.video_dir, a.out, a.subtitle_dir, a.duration, a.max_videos, a.seed)
    else:
        convert_lvbench(a.meta, a.video_dir, a.out, a.max_videos, a.per_video, a.seed)


if __name__ == "__main__":
    main()
