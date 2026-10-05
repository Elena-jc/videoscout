"""Video tools and the factories that assemble them for one index."""

from __future__ import annotations

import importlib.util
import sys
import threading
from collections.abc import Callable

from ..config import Config
from ..index.store import VideoIndex
from ..retrieval import HybridRetriever, dense_kind
from ..video import VideoReader
from ..vision import VisionBackend
from .concepts import Sam3ConceptTracker, sam3_weights_cached
from .grounding import OpenVocabDetector, make_find_objects_tool
from .inspect import make_inspect_tool
from .registry import InProcessTools, Tool, ToolError, ToolOutput, ToolRegistry
from .search import make_search_tool
from .timeline import make_browse_tool
from .tracks_sql import make_query_tracks_tool

__all__ = [
    "InProcessTools", "Tool", "ToolError", "ToolOutput", "ToolRegistry",
    "build_tools", "default_detector", "default_embedder_factory", "default_reranker_factory", "open_toolset",
]


def default_embedder_factory(index: VideoIndex, cfg: Config) -> Callable[[], object] | None:
    """Lazily load the text encoder that matches the index's dense embeddings:
    Qwen3-VL-Embedding for clip vectors, SigLIP 2 for keyframe vectors."""
    kind = dense_kind(index, cfg.retrieval)
    if kind == "clip":
        name = index.meta["clip_embedder"]

        def clip_factory():
            from ..index.qwen3vl import get_qwen_embedder

            return get_qwen_embedder(name, cfg.index.device)

        return clip_factory
    if kind == "keyframe":
        name = index.meta["siglip_model"]

        def keyframe_factory():
            from ..index.embedder import get_embedder

            return get_embedder(name, cfg.index.device)

        return keyframe_factory
    return None


def default_reranker_factory(index: VideoIndex, cfg: Config) -> Callable[[], object] | None:
    """The cross-encoder reranker, if enabled and the index has clip thumbnails to show it."""
    r = cfg.retrieval
    if not r.rerank or not r.reranker_model or not (index.dir / "thumbs.sqlite").exists():
        return None

    def factory():
        from ..index.qwen3vl import get_qwen_reranker

        return get_qwen_reranker(r.reranker_model, cfg.index.device)

    return factory


_DETECTORS: dict[tuple, object] = {}
_DETECTORS_LOCK = threading.Lock()


def default_detector(cfg: Config):
    """One open-vocabulary detector per process (the model loads on first use):
    SAM 3 when configured and its weights are cached, else YOLOE."""
    g = cfg.grounding
    use_sam3 = g.backend == "sam3" or (g.backend == "auto" and sam3_weights_cached(g.sam3_model))
    with _DETECTORS_LOCK:
        if use_sam3:
            key = ("sam3", g.sam3_model, cfg.index.device)
            if key not in _DETECTORS:
                _DETECTORS[key] = Sam3ConceptTracker(g.sam3_model, cfg.index.device)
            return _DETECTORS[key]
        if not g.weights or importlib.util.find_spec("ultralytics") is None:
            return None
        key = ("yoloe", g.weights, cfg.index.device, g.conf)
        if key not in _DETECTORS:
            _DETECTORS[key] = OpenVocabDetector(g.weights, cfg.index.device, g.conf)
        return _DETECTORS[key]


def build_tools(
    index: VideoIndex,
    cfg: Config,
    vision: VisionBackend,
    embedder_factory: Callable[[], object] | None = None,
    detector=None,
    reranker_factory: Callable[[], object] | None = None,
) -> list[Tool]:
    reader = VideoReader(index.video_path)
    retriever = HybridRetriever(index, cfg.retrieval, embedder_factory, reranker_factory)
    tools = []
    if index.events or index.storyline:
        tools.append(make_browse_tool(index))
    tools.append(make_search_tool(retriever))
    if index.meta.get("yolo_weights"):
        tools.append(make_query_tracks_tool(index.connect, float(index.meta.get("track_fps", 2.0))))
    if detector is not None:
        sam3 = getattr(detector, "kind", "") == "sam3"
        g = cfg.grounding
        tools.append(make_find_objects_tool(
            index, reader, detector,
            g.sam3_max_frames if sam3 else g.max_frames, 16 if sam3 else g.default_frames,
        ))
    tools.append(
        make_inspect_tool(index, reader, vision, cfg.agent.max_frames_per_inspect, cfg.agent.default_frames_per_inspect)
    )
    disabled = {name.strip() for name in cfg.agent.disabled_tools.split(",") if name.strip()}
    return [t for t in tools if t.name not in disabled]


def default_tools(index: VideoIndex, cfg: Config, vision: VisionBackend) -> list[Tool]:
    return build_tools(index, cfg, vision, default_embedder_factory(index, cfg), default_detector(cfg),
                       default_reranker_factory(index, cfg))


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
    return InProcessTools(default_tools(index, cfg, vision), strict=cfg.agent.strict_tools)
