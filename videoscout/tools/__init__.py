"""Video tools and the factories that assemble them for one index."""

from __future__ import annotations

import importlib.util
import sys
import threading
from collections.abc import Callable

from ..config import Config
from ..index.store import VideoIndex
from ..retrieval import HybridRetriever
from ..video import VideoReader
from ..vision import VisionBackend
from .grounding import OpenVocabDetector, make_find_objects_tool
from .inspect import make_inspect_tool
from .registry import InProcessTools, Tool, ToolError, ToolOutput, ToolRegistry
from .search import make_search_tool
from .tracks_sql import make_query_tracks_tool

__all__ = [
    "InProcessTools", "Tool", "ToolError", "ToolOutput", "ToolRegistry",
    "build_tools", "default_detector", "default_embedder_factory", "open_toolset",
]


def default_embedder_factory(index: VideoIndex, cfg: Config) -> Callable[[], object] | None:
    """Lazily load the SigLIP model the index was built with (None if it has no dense index)."""
    name = index.meta.get("siglip_model")
    if not name:
        return None

    def factory():
        from ..index.embedder import get_embedder

        return get_embedder(name, cfg.index.device)

    return factory


_DETECTORS: dict[tuple[str, str, float], OpenVocabDetector] = {}
_DETECTORS_LOCK = threading.Lock()


def default_detector(cfg: Config) -> OpenVocabDetector | None:
    """One open-vocabulary detector per process (the model loads on first use)."""
    g = cfg.grounding
    if not g.weights or importlib.util.find_spec("ultralytics") is None:
        return None
    key = (g.weights, cfg.index.device, g.conf)
    with _DETECTORS_LOCK:
        if key not in _DETECTORS:
            _DETECTORS[key] = OpenVocabDetector(g.weights, cfg.index.device, g.conf)
        return _DETECTORS[key]


def build_tools(
    index: VideoIndex,
    cfg: Config,
    vision: VisionBackend,
    embedder_factory: Callable[[], object] | None = None,
    detector: OpenVocabDetector | None = None,
) -> list[Tool]:
    reader = VideoReader(index.video_path)
    retriever = HybridRetriever(index, cfg.retrieval, embedder_factory)
    tools = [make_search_tool(retriever)]
    if index.meta.get("yolo_weights"):
        tools.append(make_query_tracks_tool(index.connect, float(index.meta.get("track_fps", 2.0))))
    if detector is not None:
        tools.append(
            make_find_objects_tool(index, reader, detector, cfg.grounding.max_frames, cfg.grounding.default_frames)
        )
    tools.append(
        make_inspect_tool(index, reader, vision, cfg.agent.max_frames_per_inspect, cfg.agent.default_frames_per_inspect)
    )
    disabled = {name.strip() for name in cfg.agent.disabled_tools.split(",") if name.strip()}
    return [t for t in tools if t.name not in disabled]


def open_toolset(index_dir: str, cfg: Config, vision: VisionBackend, index: VideoIndex | None = None) -> ToolRegistry:
    """Tools for the agent, either as in-process calls or through an MCP server
    subprocess (same tools, same schemas, different transport)."""
    if cfg.agent.tools == "mcp":
        from .mcp_client import MCPTools

        command = [sys.executable, "-m", "videoscout.tools.mcp_server", "--index", str(index_dir)]
        for section in ("models", "index", "retrieval", "grounding", "agent"):
            for key, value in vars(getattr(cfg, section)).items():
                command += ["--set", f"{section}.{key}={value}"]
        return MCPTools(command)

    index = index or VideoIndex(index_dir)
    tools = build_tools(index, cfg, vision, default_embedder_factory(index, cfg), default_detector(cfg))
    return InProcessTools(tools, strict=cfg.agent.strict_tools)
