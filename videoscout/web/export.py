"""Export recorded agent runs as a static site (no server, no API calls).

    python -m videoscout.eval.run --config configs/gemini.yaml --data data/demo/qa.jsonl --out runs/demo_gemini --workers 1
    python -m videoscout.web.export --run runs/demo_gemini --index indexes/demo --out site --repo-url https://github.com/you/videoscout

The site replays real runs from the eval traces with the same UI as the local app,
so it can be hosted for free on GitHub Pages without exposing any API key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ..index.store import VideoIndex
from ..video import browser_playable, make_browser_preview

STATIC = Path(__file__).parent / "static"


def run_events(trace: dict[str, Any], model: str, budget: dict[str, int]) -> list[dict[str, Any]]:
    record = trace["record"]
    events: list[dict[str, Any]] = [{"type": "start", "model": model, "budget": budget}]
    events += [{"type": "step", "step": step} for step in trace.get("steps", [])]
    events.append({
        "type": "final",
        "final": trace["final"],
        "budget": {"tool_calls": record.get("tool_calls", 0), "frames": record.get("frames", 0)},
        "usage": {k: record.get(k, 0) for k in ("input_tokens", "output_tokens", "cost_usd")} | {"calls": record.get("llm_calls", 0)},
        "latency_s": record.get("latency_s"),
        "gold": record.get("gold"),
        "correct": record.get("correct"),
    })
    return events


def export_site(run_dir: Path, index_dir: Path, out: Path, repo_url: str = "") -> Path:
    index = VideoIndex(index_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
    model = config["models"]["planner"]
    budget = {"tool_calls": config["agent"]["max_tool_calls"], "frames": config["agent"]["max_frames"]}

    runs = []
    for path in sorted((run_dir / "traces").glob("*.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        if trace.get("error") or not trace.get("final") or trace["record"].get("video_id") != index.dir.name:
            continue  # only complete runs of this video
        run_model = trace["record"].get("model") or model  # each question may have been recorded with a different model
        runs.append({"question": trace["question"], "options": trace["options"], "events": run_events(trace, run_model, budget)})
    if not runs:
        raise SystemExit(f"no completed runs for video {index.dir.name!r} in {run_dir}")

    out.mkdir(parents=True, exist_ok=True)
    media = out / "media"
    if media.exists():
        shutil.rmtree(media)
    media.mkdir()
    video = Path(index.video_path)
    # The site must play in any browser: reuse the index's preview, copy a video that
    # is already H.264/VP9, or re-encode one that is not (e.g. OpenCV's mp4v).
    preview = next((index.dir / n for n in ("preview.mp4", "preview.webm") if (index.dir / n).exists()), None)
    if preview is not None:
        published = media / f"{index.dir.name}{preview.suffix}"
        shutil.copy2(preview, published)
    elif browser_playable(video):
        published = media / f"{index.dir.name}{video.suffix}"
        shutil.copy2(video, published)
    else:
        published = make_browser_preview(video, media / index.dir.name)
    # Content-hashed name: browsers and the Pages CDN cache media aggressively, so a
    # changed video must get a new URL or visitors keep seeing the old one.
    digest = hashlib.sha256(published.read_bytes()).hexdigest()[:10]
    published = published.rename(published.with_name(f"{index.dir.name}-{digest}{published.suffix}"))

    demo = {
        "model": model,
        "repo_url": repo_url,
        "videos": [{
            "id": index.dir.name,
            "name": video.name,
            "duration": index.duration,
            "has_subtitles": index.meta.get("has_subtitles", False),
            "tracker": index.meta.get("yolo_weights"),
            "embedder": index.meta.get("siglip_model"),
            "available": True,
            "url": f"media/{published.name}",
        }],
        "runs": {index.dir.name: runs},
    }
    (out / "demo.json").write_text(json.dumps(demo, ensure_ascii=False, default=str), encoding="utf-8")
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    page = page.replace("<script>", '<script>window.VIDEOSCOUT_STATIC = "demo.json";</script>\n<script>', 1)
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")  # serve files as-is on GitHub Pages
    print(f"exported {len(runs)} runs to {out} (open {out / 'index.html'} through a local web server to preview)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="eval run directory (with traces/ and config.json)")
    parser.add_argument("--index", required=True, help="index directory of the video")
    parser.add_argument("--out", default="site")
    parser.add_argument("--repo-url", default="")
    args = parser.parse_args()
    export_site(Path(args.run), Path(args.index), Path(args.out), args.repo_url)


if __name__ == "__main__":
    main()
