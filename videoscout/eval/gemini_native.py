"""Baseline: give the whole video to Gemini and ask once (no index, no agent).

This is the bar a video agent has to clear in 2026: the Gemini API watches a full
uploaded video itself, either

- static:  frames sampled at a fixed rate (default 1 fps, ~70-260 tokens per frame)
           all go into one long context, or
- agentic: Gemini's own agentic video processing navigates the timeline and loads
           only the parts it needs (Interactions API, "processing": "agentic").

Comparing against both answers the obvious review question ("why not just give
the video to Gemini?") with accuracy, tokens and latency on the same questions.
Uploaded files are cached for 47 hours (the File API keeps them 48).
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

from ..agent.graph import normalize_answer
from ..config import PROJECT_ROOT
from ..llm import LLMError
from ..llm_openai import MAX_RETRY_WAIT, QuotaExhaustedError, _retry_delay
from ..spend import SpendGuard

API = "https://generativelanguage.googleapis.com"
FILE_TTL_SECONDS = 47 * 3600
CACHE_PATH = PROJECT_ROOT / ".cache" / "gemini_files.json"

PROMPT = """Watch the video and answer the question.
Question: {question}
{options}
Reply with exactly two lines:
ANSWER: <{answer_hint}>
CONFIDENCE: <your probability, 0-1, that the answer is correct>"""

_ANSWER = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)
_CONF = re.compile(r"CONFIDENCE:\s*([01](?:\.\d+)?)", re.IGNORECASE)


def parse_reply(text: str, options: list[str]) -> tuple[str, float]:
    m = _ANSWER.search(text)
    raw = (m.group(1) if m else text).strip().strip("*").strip()
    c = _CONF.search(text)
    confidence = min(max(float(c.group(1)), 0.0), 1.0) if c else 0.5
    return normalize_answer(raw, options), confidence


def output_text(response: dict[str, Any]) -> str:
    """Text of the model_output steps of an Interactions API response."""
    parts = []
    for step in response.get("steps", []):
        if step.get("type") == "model_output":
            parts += [c.get("text", "") for c in step.get("content", []) if c.get("type") == "text"]
    if not parts:  # other response shapes: outputs[].text
        parts = [o.get("text", "") for o in response.get("outputs", []) if isinstance(o, dict)]
    return "\n".join(p for p in parts if p).strip()


class GeminiNative:
    def __init__(self, model: str, processing: str = "agentic", fps: float | None = None,
                 api_key: str | None = None, client: httpx.Client | None = None,
                 daily_request_limit: int = 0, cache_path: Path = CACHE_PATH):
        if processing not in ("agentic", "static"):
            raise ValueError("processing must be 'agentic' or 'static'")
        self.model = model
        self.processing = processing
        self.fps = fps
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set: add it to the .env file in the project folder")
        self.http = client or httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0))
        self.guard = SpendGuard("generativelanguage.googleapis.com", daily_request_limit, 0.0)
        self.cache_path = cache_path

    # ------------------------------------------------------------- files
    def _cache(self) -> dict[str, Any]:
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def upload(self, video_path: str) -> dict[str, str]:
        """Upload once per video (resumable File API), wait until ACTIVE, cache the uri."""
        path = Path(video_path).resolve()
        key = f"{path}|{path.stat().st_size}"
        cache = self._cache()
        hit = cache.get(key)
        if hit and time.time() - hit["at"] < FILE_TTL_SECONDS:
            return hit
        mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
        headers = {"x-goog-api-key": self.api_key}
        start = self.http.post(
            f"{API}/upload/v1beta/files",
            headers={**headers, "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                     "X-Goog-Upload-Header-Content-Length": str(path.stat().st_size),
                     "X-Goog-Upload-Header-Content-Type": mime, "Content-Type": "application/json"},
            json={"file": {"display_name": path.stem}},
        )
        self._raise(start, "file upload (start)")
        upload_url = start.headers["x-goog-upload-url"]
        with open(path, "rb") as f:
            done = self.http.post(upload_url, headers={"X-Goog-Upload-Command": "upload, finalize",
                                                        "X-Goog-Upload-Offset": "0"}, content=f.read())
        self._raise(done, "file upload")
        file = done.json()["file"]
        deadline = time.time() + 900
        while file.get("state") == "PROCESSING" and time.time() < deadline:
            time.sleep(5)
            poll = self.http.get(f"{API}/v1beta/{file['name']}", headers=headers)
            self._raise(poll, "file status")
            file = poll.json()
        if file.get("state") != "ACTIVE":
            raise LLMError(f"uploaded video is not usable (state {file.get('state')})")
        entry = {"uri": file["uri"], "mime_type": file.get("mimeType", mime), "at": time.time()}
        cache[key] = entry
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        return entry

    def _raise(self, response: httpx.Response, what: str) -> None:
        if response.status_code == 429:
            wait = _retry_delay(response)
            if wait is None or wait > MAX_RETRY_WAIT:
                raise QuotaExhaustedError(f"Gemini quota exhausted ({what}, model {self.model}); try again later")
            raise LLMError(f"Gemini rate limit ({what}); retry in {wait:.0f}s")
        if response.status_code >= 400:
            raise LLMError(f"Gemini {what} failed: HTTP {response.status_code}: {response.text[:300]}")

    # ------------------------------------------------------------- ask
    def ask(self, video_path: str, question: str, options: list[str]) -> dict[str, Any]:
        file = self.upload(video_path)
        video: dict[str, Any] = {"type": "video", "uri": file["uri"], "mime_type": file["mime_type"]}
        if self.processing == "agentic":
            video["processing"] = "agentic"
        elif self.fps:
            video["processing"] = {"type": "static", "fps": self.fps}
        prompt = PROMPT.format(
            question=question,
            options=("Options:\n" + "\n".join(options)) if options else "",
            answer_hint="option letter" if options else "short answer",
        )
        self.guard.check()
        started = time.perf_counter()
        response = self.http.post(f"{API}/v1beta/interactions", headers={"x-goog-api-key": self.api_key},
                                  json={"model": self.model, "input": [video, {"type": "text", "text": prompt}]})
        self.guard.record(0.0)
        self._raise(response, "interaction")
        body = response.json()
        text = output_text(body)
        answer, confidence = parse_reply(text, options)
        usage = body.get("usage") or body.get("usage_metadata") or {}
        return {
            "answer": answer, "raw": text, "confidence": confidence, "self_confidence": confidence,
            "latency_s": round(time.perf_counter() - started, 3),
            "input_tokens": int(usage.get("input_tokens") or usage.get("total_input_tokens") or usage.get("promptTokenCount") or 0),
            "output_tokens": int(usage.get("output_tokens") or usage.get("total_output_tokens") or usage.get("candidatesTokenCount") or 0),
            "processing": self.processing,
        }
