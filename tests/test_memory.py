"""Multi-granular memory (events + storyline), clip embeddings, the reranker stage,
browse_timeline and the SAM 3 output format, with fake local models (no weights)."""

from __future__ import annotations

import numpy as np
from conftest import write_tiny_video

from videoscout.index.build import build_index
from videoscout.index.memory import cut_events, write_storyline
from videoscout.index.store import Segment, VideoIndex
from videoscout.retrieval import HybridRetriever, dense_kind
from videoscout.tools.concepts import format_tracks
from videoscout.tools.grounding import make_find_objects_tool
from videoscout.tools.timeline import make_browse_tool
from videoscout.video import VideoReader

COLORS = ("blue", "green", "red")  # BGR channel order of the synthetic video


def _color_vector(frames) -> np.ndarray:
    v = np.mean([f.reshape(-1, 3).mean(axis=0) for f in frames], axis=0)
    return v / np.linalg.norm(v)


class FakeCaptioner:
    def __init__(self):
        self.prompts = []

    def caption_clip(self, frames, seconds):
        return f"a {COLORS[int(np.argmax(_color_vector(frames)))]} screen"

    def write(self, prompt):
        self.prompts.append(prompt)
        found = [c for c in COLORS if c in prompt]
        return ("Storyline: " if prompt.startswith("Below are summaries") else "Event: ") + ", ".join(found)


class FakeClipEmbedder:
    def embed_clips(self, clips):
        return np.stack([_color_vector(frames) for frames, _ in clips]).astype(np.float32)

    def embed_text(self, texts):
        return np.stack([np.eye(3, dtype=np.float32)[[c in t for c in COLORS].index(True)] for t in texts])


class FakeReranker:
    """Prefers the very first clip whatever the query, to make the reordering visible."""

    def __init__(self):
        self.seen = []

    def score(self, query, clips):
        self.seen.append((query, len(clips), len(clips[0][0])))
        return [1.0 if i == 0 else 0.1 for i in range(len(clips))]


def _three_scene_index(tmp_path, cfg):
    video = write_tiny_video(tmp_path / "scenes.mp4", seconds=60)  # blue 0-20 s, green 20-40 s, red 40-60 s
    cfg.index.event_min_seconds = 5
    out = build_index(video, tmp_path / "index", cfg, with_tracks=False, log=lambda _: None,
                      captioner=FakeCaptioner(), clip_embedder=FakeClipEmbedder())
    return VideoIndex(out)


def test_build_writes_events_storyline_clip_embeddings_and_thumbnails(tmp_path, cfg):
    index = _three_scene_index(tmp_path, cfg)
    assert [s.caption for s in index.segments] == ["a blue screen"] * 2 + ["a green screen"] * 2 + ["a red screen"] * 2
    assert [(e.t_start, e.t_end) for e in index.events] == [(0, 20), (20, 40), (40, 60)]  # cut at the scene changes
    assert [e.summary for e in index.events] == ["Event: blue", "Event: green", "Event: red"]
    assert index.storyline == "Storyline: blue, green, red" and index.storyline in index.overview()
    assert index.clip_emb.shape == (6, 3) and index.meta["clip_embedder"]
    thumbs = index.clip_thumbs(2, limit=2)
    assert len(thumbs) == 2 and 20 <= thumbs[0][0] < thumbs[1][0] < 30 and thumbs[0][1].ndim == 3


def test_browse_timeline_overview_and_zoom(tmp_path, cfg):
    index = _three_scene_index(tmp_path, cfg)
    tool = make_browse_tool(index)
    overview = tool.run({}).text
    assert "Storyline: blue, green, red" in overview and "[E1] 00:20-00:40 (t=20.0-40.0s): Event: green" in overview
    zoom = tool.run({"t_start": 20, "t_end": 40})
    assert not zoom.is_error and zoom.text.count("a green screen") == 2 and "red" not in zoom.text
    assert tool.run({"t_start": 20}).is_error


def test_clip_dense_retrieval_and_rerank_stage(tmp_path, cfg):
    index = _three_scene_index(tmp_path, cfg)
    cfg.retrieval.use_bm25 = False
    assert dense_kind(index, cfg.retrieval) == "clip"
    retriever = HybridRetriever(index, cfg.retrieval, embedder_factory=lambda: FakeClipEmbedder())
    top = retriever.search("something green", top_k=1)[0]
    assert top.seg_id in (2, 3) and top.dense_rank == 1 and top.rerank_rank is None

    cfg.retrieval.rerank = True
    reranker = FakeReranker()
    retriever = HybridRetriever(index, cfg.retrieval, lambda: FakeClipEmbedder(), lambda: reranker)
    top = retriever.search("something green", top_k=1)[0]
    assert top.rerank_rank == 1 and top.rerank_score == 1.0
    query, n_clips, n_frames = reranker.seen[0]
    assert query == "something green" and n_clips == 6 and n_frames == cfg.retrieval.rerank_frames


def test_events_without_embeddings_and_hierarchical_storyline():
    segments = [Segment(i, i * 10.0, (i + 1) * 10.0, caption=f"clip {i}") for i in range(30)]
    events = cut_events(segments, None, min_seconds=20, max_seconds=180)
    assert [(e.t_start, e.t_end) for e in events] == [(0, 90), (90, 180), (180, 270), (270, 300)]

    calls = []
    many = [type("E", (), {"t_start": i * 10.0, "t_end": i * 10.0 + 10, "summary": f"part {i}"})() for i in range(50)]
    story = write_storyline(many, 500.0, lambda p: calls.append(p) or f"summary {len(calls)}")
    assert len(calls) == 4 and story == "summary 4"  # three groups of 20, then one pass over the group summaries


class FakeSam3:
    kind = "sam3"

    def track(self, frames, names):
        # Two forklifts: #7 in every frame, #9 only in the second half.
        n = len(frames)
        return {name: ({7: [(i, 0.9, (0.1, 0.2, 0.3, 0.4)) for i in range(n)],
                        9: [(i, 0.7, (0.5, 0.5, 0.7, 0.9)) for i in range(n // 2, n)]} if name == "forklift" else {})
                for name in names}


def test_find_objects_with_sam3_tracks_distinct_instances(tiny_index):
    index = VideoIndex(tiny_index)
    tool = make_find_objects_tool(index, VideoReader(index.video_path), FakeSam3(), 32, 16)
    out = tool.run({"names": ["forklift", "pallet"], "t_start": 0, "t_end": 20, "num_frames": 8})
    assert not out.is_error
    assert "forklift: 2 distinct instance(s)" in out.text and "up to 2 visible at once" in out.text
    assert "in 8/8 frames" in out.text and "in 4/8 frames" in out.text and "pallet: not found" in out.text
    assert "Tracked with SAM 3" in format_tracks([0.0], {"x": {}}, ["x"], 0, 1)
