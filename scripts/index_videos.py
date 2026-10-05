"""Index every video in a folder (one index per video, named after the file stem).

    python scripts/index_videos.py --videos data/lvbench/videos --index-root indexes

Resumable: videos that already have an index are skipped. The local models are
loaded once and reused for all videos. A subtitle file next to a video
(<stem>.srt) is used when present.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from videoscout.config import load_config  # noqa: E402
from videoscout.eval.data import VIDEO_EXTENSIONS  # noqa: E402
from videoscout.index.build import build_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--index-root", default="indexes")
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[], dest="overrides", metavar="SECTION.FIELD=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    captioner = None
    if cfg.index.captioner == "local":
        from videoscout.index.qwen3vl import LocalCaptioner

        captioner = LocalCaptioner(cfg.index.captioner_model, cfg.index.device)
    videos = sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    for i, video in enumerate(videos, start=1):
        out = Path(args.index_root) / video.stem
        if (out / "meta.json").exists():
            print(f"[{i}/{len(videos)}] {video.stem}: already indexed", flush=True)
            continue
        started = time.perf_counter()
        srt = video.with_suffix(".srt")
        print(f"[{i}/{len(videos)}] {video.stem}: indexing ...", flush=True)
        try:
            build_index(video, out, cfg, srt_path=srt if srt.exists() else None, captioner=captioner,
                        log=lambda m: print(f"    {m}", flush=True))
        except Exception as err:  # keep going with the other videos
            print(f"[{i}/{len(videos)}] {video.stem}: FAILED {type(err).__name__}: {err}", flush=True)
            continue
        print(f"[{i}/{len(videos)}] {video.stem}: done in {(time.perf_counter() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
