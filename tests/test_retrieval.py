from __future__ import annotations

import numpy as np
import pytest

from videoscout.agent.state import SubmitAnswer
from videoscout.config import RetrievalConfig
from videoscout.index.store import Segment, VideoIndex, write_index
from videoscout.retrieval import BM25, HybridRetriever, mmr, rrf, tokenize
from videoscout.schema import to_api_schema
from videoscout.tools.search import SearchArgs


def test_tokenize_handles_english_and_chinese():
    assert tokenize("The Bus stops at Gate-B") == ["bus", "stops", "gate", "b"]
    assert tokenize("红色的车") == ["红", "色", "的", "车"]


def test_bm25_prefers_rare_matching_terms():
    bm25 = BM25(["a person walks", "a person and a bus", "person person person", "empty street"])
    scores = bm25.scores("bus")
    assert scores.argmax() == 1 and scores[0] == 0
    # "person" is in 3 of 4 docs, so it barely discriminates; tf saturates.
    assert bm25.scores("person")[2] < 2 * bm25.scores("person")[0]


def test_rrf_sums_reciprocal_ranks():
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert max(fused, key=fused.get) == 1


def test_mmr_spreads_results_in_time():
    centers = {0: 5, 1: 15, 2: 25, 3: 305}
    relevance = {0: 1.0, 1: 0.95, 2: 0.9, 3: 0.6}
    # After picking segment 0, segment 1 (10 s away) is penalised more than it gains in relevance.
    assert mmr([0, 1, 2, 3], relevance, centers, k=2, lam=1.0, tau=30) == [0, 1]  # pure relevance
    assert mmr([0, 1, 2, 3], relevance, centers, k=2, lam=0.7, tau=30) == [0, 2]  # skips the neighbour
    assert mmr([0, 1, 2, 3], relevance, centers, k=2, lam=0.5, tau=30) == [0, 3]  # favours a distant moment


def _text_index(tmp_path):
    texts = ["opening credits", "a red bus arrives objects: bus x1, person x2", "people talk in an office",
             "the sign says gate b closed", "a bus leaves the depot"]
    segments = [Segment(i, i * 10.0, (i + 1) * 10.0, subtitle=t) for i, t in enumerate(texts)]
    meta = {"video_path": str(tmp_path / "none.mp4"), "duration": 50.0, "segment_seconds": 10.0}
    write_index(tmp_path, meta, segments, [], np.zeros((0, 0), np.float32), [], [])
    return VideoIndex(tmp_path)


def test_hybrid_retriever_sparse_only_with_time_filter(tmp_path):
    index = _text_index(tmp_path)
    retriever = HybridRetriever(index, RetrievalConfig(), embedder_factory=None)
    hits = retriever.search("bus", top_k=3)
    assert {h.seg_id for h in hits[:2]} == {1, 4}
    assert all(h.dense_rank is None and h.text_rank for h in hits)
    late = retriever.search("bus", top_k=3, t_start=30)
    assert [h.seg_id for h in late] == [4]


class _FakeEmbedder:
    def embed_text(self, texts):
        return np.array([[0.0, 1.0]], np.float32)


def test_hybrid_retriever_fuses_dense_maxsim(tmp_path):
    index = _text_index(tmp_path)
    # Two keyframes per segment; only segment 2's second keyframe matches the query direction.
    index.kf_seg = np.repeat(np.arange(5), 2)
    index.kf_times = np.arange(10) * 5.0 + 2.5
    index.kf_emb = np.tile(np.array([[1.0, 0.0]], np.float32), (10, 1))
    index.kf_emb[5] = [0.0, 1.0]
    index.meta["siglip_model"] = "fake-siglip"
    retriever = HybridRetriever(index, RetrievalConfig(), embedder_factory=lambda: _FakeEmbedder())
    top = retriever.search("xyzzy", visual_query="two people talking", top_k=1)[0]
    assert top.seg_id == 2 and top.dense_rank == 1 and top.best_frame_t == 27.5


def test_api_schema_is_strict_compatible():
    schema = to_api_schema(SearchArgs)
    assert schema["additionalProperties"] is False
    assert "title" not in schema and all("default" not in p for p in schema["properties"].values())
    nested = to_api_schema(SubmitAnswer)
    assert nested["$defs"]["EvidenceItem"]["additionalProperties"] is False
