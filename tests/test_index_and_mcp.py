"""Index building on a synthetic video, the inspect tool with a fake vision
backend, and the MCP server/client round trip (real subprocess, stdio)."""

from __future__ import annotations

import sys

from conftest import FakeAnthropic

from videoscout.index.build import keyframe_times, make_segments
from videoscout.index.store import VideoIndex
from videoscout.llm import ClaudeLLM
from videoscout.tools import build_tools
from videoscout.tools.mcp_client import MCPTools
from videoscout.vision import LLMVision


def test_segments_and_keyframes():
    segs = make_segments(25.0, 10.0)
    assert [(s.t_start, s.t_end) for s in segs] == [(0, 10), (10, 20), (20, 25)]
    kf = keyframe_times(segs, 4.0)
    assert kf[:3] == [(0, 2.0), (0, 6.0), (1, 12.0)] and kf[-1] == (2, 22.0)


def test_build_index_attaches_subtitles(tiny_index):
    index = VideoIndex(tiny_index)
    assert len(index.segments) == 3 and 24 < index.duration <= 25.2
    assert "delivery truck" in index.segments[0].subtitle
    assert index.segments[1].subtitle == "The forklift driver waves hello."  # tags stripped
    assert index.kf_emb.size == 0 and "Subtitles available: yes" in index.overview()


def test_tools_on_real_index_with_fake_vision(tiny_index, cfg):
    index = VideoIndex(tiny_index)
    client = FakeAnthropic(vision_reply="Frame 00:12 shows a green screen.")
    tools = {t.name: t for t in build_tools(index, cfg, LLMVision(ClaudeLLM(cfg.models, cfg.pricing, client=client)))}
    assert set(tools) == {"browse_timeline", "search_segments", "inspect_clip"}  # no tracks were built

    out = tools["search_segments"].run({"query": "forklift driver"})
    assert not out.is_error and "00:10-00:20" in out.text and "text #1" in out.text

    seen = tools["inspect_clip"].run({"t_start": 10, "t_end": 20, "question": "What color?", "num_frames": 3})
    assert not seen.is_error and "green screen" in seen.text
    (request,) = client.requests
    images = [b for b in request["messages"][0]["content"] if b["type"] == "image"]
    assert len(images) == 3 and request["model"] == cfg.models.vision

    late = tools["inspect_clip"].run({"t_start": 500, "t_end": 510, "question": "?"})
    assert late.is_error and "after the end" in late.text


class FakeDetector:
    """Stands in for YOLOE: finds a forklift in every other frame."""

    def detect(self, images, names):
        return [[("forklift", 0.9, (0.1, 0.2, 0.3, 0.4))] if i % 2 == 0 else [] for i in range(len(images))]


def test_find_objects_and_box_zoom(tiny_index, cfg):
    index = VideoIndex(tiny_index)
    client = FakeAnthropic(vision_reply="A forklift carrying a pallet.")
    vision = LLMVision(ClaudeLLM(cfg.models, cfg.pricing, client=client))
    tools = {t.name: t for t in build_tools(index, cfg, vision, detector=FakeDetector())}
    assert list(tools) == ["browse_timeline", "search_segments", "find_objects", "inspect_clip"]

    found = tools["find_objects"].run({"names": ["forklift", "pallet"], "t_start": 0, "t_end": 20, "num_frames": 4})
    assert not found.is_error
    assert "forklift: in 2/4 frames, up to 1 at once" in found.text and "pallet: not found" in found.text
    assert "box [0.10, 0.20, 0.30, 0.40]" in found.text

    zoomed = tools["inspect_clip"].run(
        {"t_start": 0, "t_end": 10, "question": "What is it carrying?", "num_frames": 2, "box": [0.1, 0.2, 0.3, 0.4]}
    )
    assert not zoomed.is_error and "cropped to box [0.10, 0.20, 0.30, 0.40]" in zoomed.text
    assert tools["inspect_clip"].run({"t_start": 0, "t_end": 10, "question": "?", "box": [0.5, 0.5, 0.2, 0.9]}).is_error
    both = tools["inspect_clip"].run({"t_start": 0, "t_end": 10, "question": "?", "box": [0, 0, 1, 1], "track_id": 3})
    assert both.is_error and "not both" in both.text


def test_mcp_round_trip(tiny_index):
    no_models = ["--set", "retrieval.rerank=false", "--set", "grounding.backend=yoloe"]
    tools = MCPTools([sys.executable, "-m", "videoscout.tools.mcp_server", "--index", str(tiny_index), *no_models])
    try:
        names = [s["name"] for s in tools.schemas()]
        assert names == ["browse_timeline", "search_segments", "find_objects", "inspect_clip"]  # detector loads lazily
        search_schema = tools.schemas()[1]["input_schema"]
        assert search_schema["required"] == ["query"]

        out = tools.call("search_segments", {"query": "delivery truck", "top_k": 2})
        assert not out.is_error and "00:00-00:10" in out.text

        bad = tools.call("search_segments", {"top_k": 2})
        assert bad.is_error
    finally:
        tools.close()
