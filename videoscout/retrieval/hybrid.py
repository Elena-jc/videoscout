"""Hybrid segment retrieval: dense (SigLIP) + sparse (BM25), fused with RRF,
diversified with temporal MMR.

Why each piece exists:
- Dense text->image similarity finds visual content nobody talked about.
- BM25 over subtitles / object tags / captions finds exact words, names and text
  that embeddings blur.
- Reciprocal Rank Fusion combines the two rankings without calibrating their
  scores against each other (cosine similarity and BM25 live on different scales).
- MMR stops the top-k from being five adjacent 10-second windows of the same event.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..config import RetrievalConfig
from ..index.store import VideoIndex
from .bm25 import BM25


@dataclass
class Hit:
    seg_id: int
    t_start: float
    t_end: float
    score: float
    dense_rank: int | None
    text_rank: int | None
    best_frame_t: float | None
    text: str


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (k + rank_r(d)), ranks from 1."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused


def mmr(
    candidates: Sequence[int],
    relevance: dict[int, float],
    centers: dict[int, float],
    k: int,
    lam: float = 0.7,
    tau: float = 30.0,
) -> list[int]:
    """Maximal Marginal Relevance with a temporal similarity kernel
    sim(i, j) = exp(-|center_i - center_j| / tau)."""
    if not candidates:
        return []
    top = max(relevance[c] for c in candidates) or 1.0
    remaining = list(candidates)
    selected: list[int] = []
    while remaining and len(selected) < k:
        def gain(c: int) -> float:
            redundancy = max((math.exp(-abs(centers[c] - centers[s]) / tau) for s in selected), default=0.0)
            return lam * relevance[c] / top - (1 - lam) * redundancy

        best = max(remaining, key=gain)
        selected.append(best)
        remaining.remove(best)
    return selected


class HybridRetriever:
    def __init__(self, index: VideoIndex, cfg: RetrievalConfig, embedder_factory: Callable[[], object] | None = None):
        self.index = index
        self.cfg = cfg
        self._embedder_factory = embedder_factory
        self.bm25 = BM25([seg.document for seg in index.segments])
        self.centers = {s.seg_id: (s.t_start + s.t_end) / 2 for s in index.segments}

    @property
    def dense_available(self) -> bool:
        return self.cfg.use_dense and self.index.kf_emb.size > 0 and self._embedder_factory is not None

    def _dense_ranking(self, text: str, allowed: np.ndarray) -> tuple[list[int], dict[int, float]]:
        q = self._embedder_factory().embed_text([text])[0]
        sims = self.index.kf_emb @ q
        # Late interaction: a segment scores as its best-matching keyframe (MaxSim).
        best = np.full(len(self.index.segments), -np.inf)
        best_t: dict[int, float] = {}
        for kf, seg_id in enumerate(self.index.kf_seg):
            if sims[kf] > best[seg_id]:
                best[seg_id] = sims[kf]
                best_t[int(seg_id)] = float(self.index.kf_times[kf])
        order = [int(i) for i in np.argsort(-best) if allowed[i] and np.isfinite(best[i])]
        return order[: self.cfg.candidate_pool], best_t

    def _text_ranking(self, text: str, allowed: np.ndarray) -> list[int]:
        scores = self.bm25.scores(text)
        order = [int(i) for i in np.argsort(-scores, kind="stable") if allowed[i] and scores[i] > 0]
        return order[: self.cfg.candidate_pool]

    def search(
        self,
        query: str,
        visual_query: str | None = None,
        top_k: int = 5,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> list[Hit]:
        segs = self.index.segments
        lo = -math.inf if t_start is None else t_start
        hi = math.inf if t_end is None else t_end
        allowed = np.array([s.t_end > lo and s.t_start < hi for s in segs], dtype=bool)

        rankings: list[list[int]] = []
        dense_order: list[int] = []
        best_t: dict[int, float] = {}
        if self.dense_available:
            dense_order, best_t = self._dense_ranking(visual_query or query, allowed)
            rankings.append(dense_order)
        text_order: list[int] = []
        if self.cfg.use_bm25:
            text_order = self._text_ranking(query, allowed)
            rankings.append(text_order)
        if not rankings:
            raise RuntimeError("no retriever enabled (dense unavailable and BM25 disabled)")

        fused = rrf(rankings, k=self.cfg.rrf_k)
        pool = sorted(fused, key=fused.get, reverse=True)[: self.cfg.candidate_pool]
        chosen = mmr(pool, fused, self.centers, k=top_k, lam=self.cfg.mmr_lambda, tau=self.cfg.mmr_tau_seconds)

        dense_rank = {s: r for r, s in enumerate(dense_order, start=1)}
        text_rank = {s: r for r, s in enumerate(text_order, start=1)}
        return [
            Hit(
                seg_id=s,
                t_start=segs[s].t_start,
                t_end=segs[s].t_end,
                score=fused[s],
                dense_rank=dense_rank.get(s),
                text_rank=text_rank.get(s),
                best_frame_t=best_t.get(s),
                text=segs[s].document,
            )
            for s in chosen
        ]
