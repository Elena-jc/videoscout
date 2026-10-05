"""Retrieval-only evaluation: does search_segments find the moment a question is
about? Uses the time annotations of the benchmark (LVBench time_reference ->
gt_windows) and runs entirely locally: no LLM calls, no API cost, so it can cover
every question of every downloaded video.

A hit at rank r means the r-th returned clip overlaps a ground-truth window.
Reports Recall@1/@5, MRR@k and search latency per retrieval configuration:

    python -m videoscout.eval.retrieval --data data/lvbench/qa.jsonl --index-root indexes \\
        --configs bm25 keyframe clip keyframe+bm25 clip+bm25 clip+bm25+rerank --out runs/retrieval.md

Configuration tokens: bm25, keyframe (SigLIP 2 frames, MaxSim), clip (Qwen3-VL clip
embeddings), rerank (Qwen3-VL-Reranker over the fused top-N).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

from ..config import RetrievalConfig, load_config
from ..index.store import VideoIndex
from ..retrieval import HybridRetriever
from ..tools import default_embedder_factory, default_reranker_factory
from .data import QAItem, load_items

TOKENS = {"bm25", "keyframe", "clip", "rerank"}


def retrieval_config(spec: str, base: RetrievalConfig) -> RetrievalConfig:
    tokens = set(spec.split("+"))
    if not tokens <= TOKENS or not tokens & {"bm25", "keyframe", "clip"}:
        raise ValueError(f"bad configuration {spec!r}: combine {sorted(TOKENS)} with '+', e.g. clip+bm25+rerank")
    if {"keyframe", "clip"} <= tokens:
        raise ValueError("pick one dense encoder: keyframe or clip")
    cfg = copy.deepcopy(base)
    cfg.use_bm25 = "bm25" in tokens
    cfg.use_dense = bool(tokens & {"keyframe", "clip"})
    cfg.dense = "clip" if "clip" in tokens else "keyframe"
    cfg.rerank = "rerank" in tokens
    return cfg


def first_hit_rank(windows: list[tuple[float, float]], gt: list[list[float]]) -> int | None:
    for rank, (a, b) in enumerate(windows, start=1):
        if any(a < g1 and g0 < b for g0, g1 in gt):
            return rank
    return None


def summarize_ranks(ranks: list[int | None], k: int) -> dict[str, float]:
    n = len(ranks) or 1
    return {
        "n": len(ranks),
        "recall@1": round(sum(1 for r in ranks if r == 1) / n, 4),
        f"recall@{k}": round(sum(1 for r in ranks if r is not None and r <= k) / n, 4),
        f"mrr@{k}": round(sum(1 / r for r in ranks if r is not None and r <= k) / n, 4),
    }


def evaluate(items: list[QAItem], index_root: Path, specs: list[str], top_k: int = 5,
             factories: dict[str, Any] | None = None, log=print) -> dict[str, dict[str, Any]]:
    """`factories` (tests) maps 'embedder:<kind>' / 'reranker' to factories; by
    default the real models are loaded."""
    base = load_config()
    by_video: dict[str, list[QAItem]] = {}
    for item in items:
        if item.gt_windows:
            by_video.setdefault(item.video_id, []).append(item)
    results: dict[str, dict[str, Any]] = {}
    for spec in specs:
        rcfg = retrieval_config(spec, base.retrieval)
        ranks: list[int | None] = []
        latency = 0.0
        for video_id, qs in by_video.items():
            index_dir = index_root / video_id
            if not (index_dir / "meta.json").exists():
                log(f"  skipping {video_id}: no index")
                continue
            index = VideoIndex(index_dir)
            cfg = copy.deepcopy(base)
            cfg.retrieval = rcfg
            if factories is not None:
                kind = "clip" if rcfg.dense == "clip" else "keyframe"
                embedder = factories.get(f"embedder:{kind}") if rcfg.use_dense else None
                reranker = factories.get("reranker") if rcfg.rerank else None
            else:
                embedder = default_embedder_factory(index, cfg)
                reranker = default_reranker_factory(index, cfg)
            if rcfg.use_dense and embedder is None:
                raise RuntimeError(f"{spec}: index {video_id} has no {rcfg.dense} embeddings")
            retriever = HybridRetriever(index, rcfg, embedder, reranker)
            for item in qs:
                started = time.perf_counter()
                hits = retriever.search(item.question, top_k=top_k)
                latency += time.perf_counter() - started
                ranks.append(first_hit_rank([(h.t_start, h.t_end) for h in hits], item.gt_windows))
        summary = summarize_ranks(ranks, top_k)
        summary["avg_search_s"] = round(latency / max(len(ranks), 1), 3)
        results[spec] = summary
        log(f"{spec}: {json.dumps(summary)}")
    return results


def table(results: dict[str, dict[str, Any]], k: int) -> str:
    lines = [f"| retrieval | n | R@1 % | R@{k} % | MRR@{k} | s / search |", "|---|---|---|---|---|---|"]
    for spec, s in results.items():
        lines.append(f"| {spec} | {s['n']} | {100 * s['recall@1']:.1f} | {100 * s[f'recall@{k}']:.1f} | "
                     f"{s[f'mrr@{k}']:.3f} | {s['avg_search_s']:.2f} |")
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--index-root", default="indexes")
    parser.add_argument("--configs", nargs="+", default=["bm25", "keyframe", "clip", "clip+bm25", "clip+bm25+rerank"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", help="write the markdown table here")
    args = parser.parse_args()
    results = evaluate(load_items(args.data), Path(args.index_root), args.configs, args.top_k)
    text = table(results, args.top_k)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
