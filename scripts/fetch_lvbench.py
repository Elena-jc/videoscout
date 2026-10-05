"""Download a seeded subset of LVBench videos and convert the annotations.

LVBench (zai-org/LVBench) ships annotations only; the videos are YouTube ids. This
script fetches the annotation file, picks `--videos` ids with a fixed seed, and
downloads each one with yt-dlp as a single video-only MP4 stream at <= 480p (no
ffmpeg merge needed; the pipeline does not use audio), then writes the QA JSONL.

    pip install yt-dlp
    python scripts/fetch_lvbench.py --videos 10 --per-video 15 --out data/lvbench

Videos that YouTube refuses (removed, private, region-locked) are skipped and
replaced by the next ones in the seeded order until --videos are on disk;
re-running resumes and keeps files already downloaded.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
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
    keys = [r["key"] for r in rows]
    # Videos already downloaded count first; then a seeded order fills up to --videos,
    # so videos that were removed from YouTube are replaced by the next ones.
    order = sorted(k for k in keys if (videos / f"{k}.mp4").exists())
    rest = [k for k in keys if k not in order]
    random.Random(args.seed).shuffle(rest)
    order += rest
    if args.dry_run:
        print(f"{len(rows)} LVBench videos; first candidates: {' '.join(order[:args.videos])}")
        return

    failed, chosen = [], []
    # YouTube needs a JavaScript runtime to solve its player challenges; use Node.js if present.
    js = ["--js-runtimes", "node"] if shutil.which("node") else []
    for key in order:
        if len(chosen) >= args.videos:
            break
        target = videos / f"{key}.mp4"
        if target.exists():
            chosen.append(key)
            continue
        print(f"downloading {key} ...", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "-f", FORMAT, "-o", str(videos / "%(id)s.%(ext)s"), *js,
             "--cache-dir", str(ROOT / ".cache" / "yt-dlp"), "--no-playlist", "--no-progress",
             f"https://www.youtube.com/watch?v={key}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not target.exists():
            failed.append(key)
            print(f"  skipped {key}: {(result.stderr or result.stdout).strip().splitlines()[-1:]}", flush=True)
            continue
        chosen.append(key)
    if failed:
        print(f"{len(failed)} videos could not be downloaded: {' '.join(failed)}")
    convert_lvbench(meta, videos, out / "qa.jsonl", per_video=args.per_video, seed=args.seed, keys=chosen)


if __name__ == "__main__":
    main()
