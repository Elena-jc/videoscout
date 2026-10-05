"""Download a seeded subset of LVBench videos and convert the annotations.

LVBench (zai-org/LVBench) ships annotations only; the videos are YouTube ids. This
script fetches the annotation file, picks `--videos` ids with a fixed seed, and
downloads each one with yt-dlp as a single video-only MP4 stream at <= 480p (no
ffmpeg merge needed; the pipeline does not use audio), then writes the QA JSONL.

    pip install yt-dlp
    python scripts/fetch_lvbench.py --videos 10 --per-video 15 --out data/lvbench

Videos that YouTube refuses (removed, region-locked, bot check) are skipped and
listed; re-running resumes and skips files already downloaded.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from videoscout.eval.data import convert_lvbench  # noqa: E402

META_URL = "https://huggingface.co/datasets/zai-org/LVBench/resolve/main/video_info.meta.jsonl"
FORMAT = "bv*[height<=480][ext=mp4][vcodec^=avc1]/bv*[height<=480][ext=mp4]/b[height<=480][ext=mp4]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=int, default=10)
    parser.add_argument("--per-video", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/lvbench")
    parser.add_argument("--dry-run", action="store_true", help="only print the chosen video ids")
    args = parser.parse_args()

    out = Path(args.out)
    videos = out / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    meta = out / "video_info.meta.jsonl"
    if not meta.exists():
        urllib.request.urlretrieve(META_URL, meta)
    rows = [json.loads(line) for line in meta.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = sorted(random.Random(args.seed).sample([r["key"] for r in rows], args.videos))
    print(f"{len(rows)} LVBench videos; chosen: {' '.join(chosen)}")
    if args.dry_run:
        return

    failed = []
    for key in chosen:
        target = videos / f"{key}.mp4"
        if target.exists():
            continue
        print(f"downloading {key} ...", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "-f", FORMAT, "-o", str(videos / "%(id)s.%(ext)s"),
             "--no-playlist", "--no-progress", f"https://www.youtube.com/watch?v={key}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not target.exists():
            failed.append(key)
            print(f"  skipped {key}: {(result.stderr or result.stdout).strip().splitlines()[-1:]}")
    if failed:
        print(f"{len(failed)} videos could not be downloaded: {' '.join(failed)}")
    convert_lvbench(meta, videos, out / "qa.jsonl", per_video=args.per_video, seed=args.seed, keys=chosen)


if __name__ == "__main__":
    main()
