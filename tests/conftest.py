"""Test doubles and fixtures. No test here calls the real API or needs model weights,
except the ones marked `integration`."""

from __future__ import annotations

import copy
import itertools
import json
import os

# Tests must never reach a real API: ignore the project's .env and drop any keys
# already in the environment.
os.environ["VIDEOSCOUT_NO_DOTENV"] = "1"
for _key in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
    os.environ.pop(_key, None)
# ...and must not touch the real daily spending ledger.
import tempfile  # noqa: E402

os.environ["VIDEOSCOUT_USAGE_DIR"] = tempfile.mkdtemp(prefix="videoscout-usage-")
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from videoscout.config import load_config
from videoscout.tools.registry import ToolOutput

_ids = itertools.count(1)


def tool_use(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": f"toolu_{next(_ids):04d}", "name": name, "input": args}


def text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


THINKING = {"type": "thinking", "thinking": "", "signature": "sig-abc123"}


def _content(message: dict[str, Any]) -> list[dict[str, Any]]:
    c = message["content"]
    return c if isinstance(c, list) else [{"type": "text", "text": c}]


def check_protocol(messages: list[dict[str, Any]]) -> None:
    """The invariants the Messages API enforces on a tool-use transcript."""
    assert messages and messages[0]["role"] == "user"
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        ids = [b["id"] for b in _content(message) if b["type"] == "tool_use"]
        if not ids or i + 1 == len(messages):
            continue
        reply = messages[i + 1]
        assert reply["role"] == "user", "tool_use must be followed by a user turn"
        blocks = _content(reply)
        results = [b for b in blocks if b.get("type") == "tool_result"]
        assert sorted(b["tool_use_id"] for b in results) == sorted(ids), "every tool_use needs a tool_result"
        assert all(b.get("type") == "tool_result" for b in blocks[: len(results)]), "tool_result blocks come first"


def fake_response(content: list[dict[str, Any]], stop_reason: str | None = None, model: str = "claude-opus-5-5"):
    if stop_reason is None:
        stop_reason = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0, cache_read_input_tokens=500)
    return SimpleNamespace(content=content, stop_reason=stop_reason, model=model, usage=usage, stop_details=None)


class FakeAnthropic:
    """Scripted stand-in for anthropic.Anthropic, routed by request shape:
    tools present -> planner script; output_config.format -> structured script;
    otherwise -> a vision reply."""

    def __init__(self, planner=None, structured=None, vision_reply: str = "The sign reads GATE B CLOSED."):
        self.planner = list(planner or [])
        self.structured = list(structured or [])
        self.vision_reply = vision_reply
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._handle)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._handle))

    def _handle(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        check_protocol(kwargs["messages"])
        if kwargs.get("tools"):
            item = self.planner.pop(0)
            return item if isinstance(item, SimpleNamespace) else fake_response(item)
        if kwargs.get("output_config", {}).get("format"):
            return fake_response([text(json.dumps(self.structured.pop(0)))])
        return fake_response([text(self.vision_reply)])

    def planner_requests(self):
        return [r for r in self.requests if r.get("tools")]

    def structured_requests(self):
        return [r for r in self.requests if r.get("output_config", {}).get("format")]


class FakeTools:
    NAMES = ("search_segments", "query_tracks", "inspect_clip")

    def __init__(self):
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def schemas(self):
        return [{"name": n, "description": n, "input_schema": {"type": "object", "properties": {}}} for n in self.NAMES]

    def call(self, name, args):
        self.calls.append((name, dict(args)))
        if name == "inspect_clip":
            return ToolOutput(f"Observation (t={args['t_start']:.1f}-{args['t_end']:.1f}s): the sign reads GATE B CLOSED")
        return ToolOutput(f"{name} result: [1] 01:20-01:30 (t=80.0-90.0s)")

    def close(self):
        pass


@pytest.fixture
def cfg():
    return load_config()


def write_tiny_video(path: Path, seconds: int = 25, fps: int = 5, size: int = 64) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    for i in range(seconds * fps):
        frame = np.zeros((size, size, 3), np.uint8)
        frame[:, :, i * 3 // (seconds * fps)] = 200  # blue, then green, then red thirds
        writer.write(frame)
    writer.release()
    return path


TINY_SRT = """1
00:00:01,000 --> 00:00:04,000
A delivery truck arrives at the warehouse.

2
00:00:12,000 --> 00:00:15,500
The <i>forklift</i> driver waves hello.
"""


@pytest.fixture
def tiny_index(tmp_path, cfg):
    """A real index built from a synthetic video: subtitles only, no model weights needed."""
    from videoscout.index.build import build_index

    video = write_tiny_video(tmp_path / "tiny.mp4")
    srt = tmp_path / "tiny.srt"
    srt.write_text(TINY_SRT, encoding="utf-8")
    out = build_index(video, tmp_path / "index", cfg, srt_path=srt, with_dense=False, with_tracks=False, log=lambda _: None)
    return out
