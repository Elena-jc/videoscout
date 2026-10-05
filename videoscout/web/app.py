"""Local web app: upload / index videos, ask questions, watch the agent work.

    python -m videoscout.web            (or double-click start.bat)

API (JSON; long work runs as background jobs):
    GET  /api/health
    GET  /api/providers                  model backends and whether their key is set
    GET  /api/videos                     indexed videos
    POST /api/videos                     multipart upload (video, subtitles) or JSON {path, srt_path} -> index job
    GET  /api/videos/{id}/stream         the video file (supports Range, so the player can seek)
    GET  /api/videos/{id}/questions      suggested questions shipped next to the video (qa.jsonl)
    POST /api/ask                        {video_id, question, options, provider} -> agent job
    GET  /api/jobs/{id}/events           Server-Sent Events (resumable with Last-Event-ID)
    GET  /api/jobs/{id}?after=N          the same events by polling
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ..agent import AgentDeps, run_agent
from ..config import PROJECT_ROOT, load_config, load_dotenv
from ..eval.data import load_items
from ..index.build import build_index
from ..index.store import VideoIndex
from ..llm import make_llm
from ..tools import open_toolset
from ..video import browser_playable, fourcc_of, make_browser_preview
from ..vision import LLMVision
from .jobs import Job, JobManager

STATIC = Path(__file__).parent / "static"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}

PROVIDERS = {
    "gemini": {"label": "Gemini (free tier)", "config": "configs/gemini.yaml", "key_env": "GEMINI_API_KEY"},
    "claude": {"label": "Claude Opus 5.5", "config": "configs/default.yaml", "key_env": "ANTHROPIC_API_KEY"},
    "ollama": {"label": "Ollama (local)", "config": "configs/ollama.yaml", "key_env": None},
}


class AskRequest(BaseModel):
    video_id: str
    question: str = Field(min_length=1, max_length=2000)
    options: list[str] = Field(default_factory=list, max_length=10)
    provider: str = "gemini"


class PathRequest(BaseModel):
    path: str
    srt_path: str | None = None


def _slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(name).stem).strip("-").lower()[:40] or "video"
    return f"{slug}-{uuid.uuid4().hex[:6]}"


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def create_app(
    index_root: Path | None = None,
    upload_root: Path | None = None,
    runs_root: Path | None = None,
    llm_factory=make_llm,
    warmup: bool = False,
    overrides: list[str] | None = None,
) -> Starlette:
    """`overrides` (section.field=value) apply to every config the app loads,
    e.g. to switch local models off in tests."""
    load_dotenv()
    index_root = Path(index_root or PROJECT_ROOT / "indexes")
    upload_root = Path(upload_root or PROJECT_ROOT / "data" / "uploads")
    runs_root = Path(runs_root or PROJECT_ROOT / "runs" / "web")
    jobs = JobManager()
    if warmup:
        threading.Thread(target=_warm_up, args=(index_root,), daemon=True, name="warmup").start()
    index_cache: dict[str, tuple[float, VideoIndex]] = {}
    cache_lock = threading.Lock()

    def index_dir(video_id: str) -> Path | None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
            return None
        path = index_root / video_id
        return path if (path / "meta.json").exists() else None

    def load_index(video_id: str) -> VideoIndex:
        path = index_dir(video_id)
        if path is None:
            raise KeyError(video_id)
        mtime = (path / "meta.json").stat().st_mtime
        with cache_lock:
            cached = index_cache.get(video_id)
            if cached is None or cached[0] != mtime:  # rebuilt indexes are reloaded
                index_cache[video_id] = (mtime, VideoIndex(path))
            return index_cache[video_id][1]

    # ------------------------------------------------------------------ reads
    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def providers(request: Request) -> Response:
        out = []
        for pid, p in PROVIDERS.items():
            if p["key_env"]:
                ready = bool(os.environ.get(p["key_env"]))
                hint = "" if ready else f"Add {p['key_env']} to the .env file in the project folder, then restart."
            else:
                ready = await asyncio.to_thread(_ollama_up)
                hint = "" if ready else "Start Ollama (http://localhost:11434) and pull the model in configs/ollama.yaml."
            models = load_config(PROJECT_ROOT / p["config"]).models
            guard = llm_factory(models, {}).guard  # same ledger the backend writes to
            out.append({
                "id": pid, "label": p["label"], "ready": ready, "hint": hint,
                "today": guard.today(),
                "limits": {"requests": models.daily_request_limit, "usd": models.daily_cost_limit_usd},
            })
        return JSONResponse(out)

    async def list_videos(request: Request) -> Response:
        videos = []
        if index_root.exists():
            for path in sorted(index_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                meta_file = path / "meta.json"
                if not meta_file.exists():
                    continue
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                video = Path(meta["video_path"])
                video = video if video.is_absolute() else (path / video).resolve()
                videos.append({
                    "id": path.name,
                    "name": video.name,
                    "duration": meta["duration"],
                    "segment_seconds": meta["segment_seconds"],
                    "has_subtitles": meta.get("has_subtitles", False),
                    "tracker": meta.get("yolo_weights"),
                    "embedder": meta.get("siglip_model"),
                    "created_at": meta.get("created_at"),
                    "available": video.exists(),
                })
        return JSONResponse(videos)

    async def stream_video(request: Request) -> Response:
        try:
            index = load_index(request.path_params["video_id"])
        except KeyError:
            return _error(404, "unknown video")
        preview = _preview_of(index.dir)  # browser-playable copy, if the original is not
        if preview is not None:
            return FileResponse(preview)
        if not Path(index.video_path).exists():
            return _error(404, "video file has moved or been deleted")
        return FileResponse(index.video_path)

    async def questions(request: Request) -> Response:
        try:
            index = load_index(request.path_params["video_id"])
        except KeyError:
            return _error(404, "unknown video")
        qa = Path(index.video_path).parent / "qa.jsonl"
        if not qa.exists():
            return JSONResponse([])
        items = [it for it in load_items(qa) if it.video_id == index.dir.name]
        return JSONResponse([{"question": it.question, "options": it.options, "answer": it.answer} for it in items])

    # ----------------------------------------------------------------- writes
    def start_index_job(video: Path, srt: Path | None, video_id: str) -> Job:
        def work(job: Job) -> None:
            job.emit("log", message=f"indexing {video.name} ...")
            cfg = load_config(None, overrides)
            out = index_root / video_id
            build_index(video, out, cfg, srt_path=srt, log=lambda m: job.emit("log", message=m))
            if not browser_playable(video):
                job.emit("log", message=f"{fourcc_of(video) or 'this codec'} does not play in browsers; writing a preview...")
                make_browser_preview(video, out / "preview")
            job.emit("done", video_id=video_id)

        return jobs.submit("index", work)

    async def add_video(request: Request) -> Response:
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("video")
            if upload is None or not getattr(upload, "filename", ""):
                return _error(400, "missing video file")
            if Path(upload.filename).suffix.lower() not in VIDEO_EXTENSIONS:
                return _error(400, f"unsupported video type; use one of {sorted(VIDEO_EXTENSIONS)}")
            video_id = _slug(upload.filename)
            folder = upload_root / video_id
            folder.mkdir(parents=True, exist_ok=True)
            video = folder / Path(upload.filename).name
            with open(video, "wb") as f:
                while chunk := await upload.read(1 << 20):
                    f.write(chunk)
            srt = None
            subs = form.get("subtitles")
            if subs is not None and getattr(subs, "filename", ""):
                srt = folder / "subtitles.srt"
                srt.write_bytes(await subs.read())
        else:
            try:
                body = PathRequest.model_validate(await request.json())
            except (ValidationError, ValueError) as err:
                return _error(400, f"invalid request: {err}")
            video = Path(body.path.strip().strip('"'))
            if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
                return _error(400, f"not a video file: {video}")
            srt = Path(body.srt_path.strip().strip('"')) if body.srt_path else None
            if srt is not None and not srt.is_file():
                return _error(400, f"subtitle file not found: {srt}")
            video_id = _slug(video.name)
        job = start_index_job(video, srt, video_id)
        return JSONResponse({"job_id": job.id, "video_id": video_id}, status_code=202)

    async def ask(request: Request) -> Response:
        try:
            body = AskRequest.model_validate(await request.json())
        except (ValidationError, ValueError) as err:
            return _error(400, f"invalid request: {err}")
        provider = PROVIDERS.get(body.provider)
        if provider is None:
            return _error(400, f"unknown provider {body.provider!r}")
        try:
            index = load_index(body.video_id)
        except KeyError:
            return _error(404, "unknown video")
        options = [o.strip() for o in body.options if o.strip()]

        def work(job: Job) -> None:
            cfg = load_config(PROJECT_ROOT / provider["config"], overrides)
            llm = llm_factory(cfg.models, cfg.pricing)
            job.emit("start", provider=body.provider, model=cfg.models.planner,
                     budget={"tool_calls": cfg.agent.max_tool_calls, "frames": cfg.agent.max_frames})
            tools = open_toolset(str(index.dir), cfg, LLMVision(llm), index=index)

            def on_update(chunk: dict[str, Any]) -> None:
                for node, delta in chunk.items():
                    for step in (delta or {}).get("trace", []):
                        job.emit("step", step=step)

            try:
                state = run_agent(AgentDeps(llm, tools, cfg, index.overview()), body.question, options, on_update)
            finally:
                tools.close()
            result = {
                "final": state.get("final"),
                "budget": {"tool_calls": state.get("tool_calls_used", 0), "frames": state.get("frames_used", 0)},
                "usage": llm.meter.snapshot(),
            }
            job.emit("final", **result)
            runs_root.mkdir(parents=True, exist_ok=True)
            record = {"job_id": job.id, "request": body.model_dump(), "result": result, "trace": state.get("trace", [])}
            (runs_root / f"{job.id}.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

        job = jobs.submit("ask", work)
        return JSONResponse({"job_id": job.id}, status_code=202)

    # ------------------------------------------------------------------ events
    async def job_poll(request: Request) -> Response:
        job = jobs.get(request.path_params["job_id"])
        if job is None:
            return _error(404, "unknown job")
        after = int(request.query_params.get("after", -1))
        return JSONResponse({"done": job.done, "events": job.events_after(after)})

    async def job_events(request: Request) -> Response:
        job = jobs.get(request.path_params["job_id"])
        if job is None:
            return _error(404, "unknown job")
        last = int(request.headers.get("last-event-id", -1))

        async def stream():
            nonlocal last
            while True:
                if await request.is_disconnected():
                    return
                for event in job.events_after(last):
                    last = event["id"]
                    yield {"id": str(event["id"]), "data": json.dumps(event, default=str)}
                if job.done and not job.events_after(last):
                    yield {"event": "end", "data": "{}"}
                    return
                await asyncio.sleep(0.2)

        return EventSourceResponse(stream(), ping=15)

    async def index_page(request: Request) -> Response:
        return FileResponse(STATIC / "index.html")

    return Starlette(
        routes=[
            Route("/", index_page),
            Route("/api/health", health),
            Route("/api/providers", providers),
            Route("/api/videos", list_videos, methods=["GET"]),
            Route("/api/videos", add_video, methods=["POST"]),
            Route("/api/videos/{video_id}/stream", stream_video),
            Route("/api/videos/{video_id}/questions", questions),
            Route("/api/ask", ask, methods=["POST"]),
            Route("/api/jobs/{job_id}", job_poll),
            Route("/api/jobs/{job_id}/events", job_events),
            Mount("/static", StaticFiles(directory=STATIC), name="static"),
        ]
    )


def _preview_of(index_dir: Path) -> Path | None:
    for name in ("preview.mp4", "preview.webm"):
        if (index_dir / name).exists():
            return index_dir / name
    return None


def _warm_up(index_root: Path) -> None:
    """Load the embedding model, the reranker and the open-vocabulary detector before
    the first question (each takes 10-30 s to load), so the first answer is not slow."""
    from ..tools import default_detector, default_embedder_factory, default_reranker_factory

    try:
        cfg = load_config()
        indexes = [p for p in index_root.iterdir() if (p / "meta.json").exists()] if index_root.exists() else []
        if indexes:
            index = VideoIndex(max(indexes, key=lambda p: p.stat().st_mtime))
            factory = default_embedder_factory(index, cfg)
            if factory is not None:
                factory().embed_text(["warm up"])
            reranker = default_reranker_factory(index, cfg)
            if reranker is not None:
                reranker()
        detector = default_detector(cfg)
        if detector is not None:
            detector.warm()
        print("models loaded; ready")
    except Exception as err:  # warm-up is an optimisation; failures surface on first use
        print(f"warm-up skipped: {type(err).__name__}: {err}")


def _ollama_up() -> bool:
    try:
        return httpx.get("http://localhost:11434/api/version", timeout=0.5).status_code == 200
    except httpx.HTTPError:
        return False
