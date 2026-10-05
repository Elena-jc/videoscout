"""Evaluate a method on a JSONL eval set. Writes one record per question and is
resumable: re-running the same command skips questions already in results.jsonl.

    python -m videoscout.eval.run --data data/demo/qa.jsonl --method agent --out runs/agent
    python -m videoscout.eval.run --data data/demo/qa.jsonl --method agent --out runs/agent_noverify \
        --set agent.verify=false
    python -m videoscout.eval.run --data data/demo/qa.jsonl --method uniform --frames 32 --out runs/uniform32
    python -m videoscout.eval.run --config configs/gemini.yaml --data data/lvbench/qa.jsonl \
        --method gemini-native --processing agentic --out runs/lvb_gemini_agentic
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..agent import AgentDeps, run_agent
from ..config import Config, load_config
from ..index.build import build_index
from ..index.store import VideoIndex
from ..llm import make_llm
from ..tools import open_toolset
from ..vision import LLMVision
from .baselines import run_uniform
from .data import QAItem, load_items
from .report import load_records, summarize

_WINDOW = re.compile(r"t=(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)s")


def _overlaps(a: tuple[float, float], b: list[float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def grounding(evidence: list[dict[str, Any]], gt_windows: list[list[float]]) -> dict[str, bool]:
    """Did the agent retrieve / look at a window that contains the answer?"""
    searched = [
        (float(a), float(b))
        for e in evidence
        if e["tool"] == "search_segments" and not e["is_error"]
        for a, b in _WINDOW.findall(e["result"])
    ]
    inspected = [
        (float(e["args"]["t_start"]), float(e["args"]["t_end"]))
        for e in evidence
        if e["tool"] == "inspect_clip" and not e["is_error"]
    ]
    hit = lambda windows: any(_overlaps(w, g) for w in windows for g in gt_windows)  # noqa: E731
    return {"search_hit": hit(searched), "inspect_hit": hit(inspected)}


class Evaluator:
    def __init__(self, cfg: Config, method: str, index_root: str, n_frames: int, with_subtitles: bool, build_missing: bool,
                 processing: str = "agentic", fps: float | None = None):
        self.cfg = cfg
        self.native = None
        if method == "gemini-native":
            from .gemini_native import GeminiNative

            self.native = GeminiNative(cfg.models.planner, processing, fps,
                                       daily_request_limit=cfg.models.daily_request_limit)
        self.method = method
        self.index_root = Path(index_root)
        self.n_frames = n_frames
        self.with_subtitles = with_subtitles
        self.build_missing = build_missing
        self.base_llm = make_llm(cfg.models, cfg.pricing)
        self._indexes: dict[str, VideoIndex] = {}
        self._lock = threading.Lock()

    def index_for(self, item: QAItem) -> tuple[str, VideoIndex]:
        index_dir = self.index_root / item.video_id
        with self._lock:
            if item.video_id not in self._indexes:
                if not (index_dir / "meta.json").exists():
                    if not self.build_missing:
                        raise FileNotFoundError(
                            f"no index for video {item.video_id!r} at {index_dir}; build it with "
                            "`python -m videoscout index` or pass --build-missing"
                        )
                    build_index(item.video_path, index_dir, self.cfg, srt_path=item.subtitle_path, llm=self.base_llm)
                self._indexes[item.video_id] = VideoIndex(index_dir)
            return str(index_dir), self._indexes[item.video_id]

    def evaluate(self, item: QAItem) -> tuple[dict[str, Any], dict[str, Any]]:
        llm = self.base_llm.fresh()  # per-question usage meter
        started = time.perf_counter()
        record: dict[str, Any] = {
            "qid": item.qid, "video_id": item.video_id, "task_type": item.task_type,
            "method": self.method, "gold": item.answer,
            "model": self.cfg.models.vision if self.method == "uniform" else self.cfg.models.planner,
        }
        trace: dict[str, Any] = {"question": item.question, "options": item.options, "gold": item.answer}
        try:
            if self.method == "agent":
                index_dir, index = self.index_for(item)
                tools = open_toolset(index_dir, self.cfg, LLMVision(llm), index=index)
                try:
                    state = run_agent(AgentDeps(llm, tools, self.cfg, index.overview()), item.question, item.options)
                finally:
                    tools.close()
                final = state.get("final") or {}
                evidence = state.get("evidence", [])
                record.update(
                    pred=final.get("answer"),
                    confidence=final.get("confidence"),
                    self_confidence=final.get("self_confidence"),
                    accepted=bool(final.get("accepted")),
                    forced=bool(final.get("forced")),
                    tool_calls=state.get("tool_calls_used", 0),
                    frames=state.get("frames_used", 0),
                    llm_turns=state.get("llm_turns", 0),
                    verify_rounds=state.get("verify_rounds", 0),
                    used_inspect=any(e["tool"] == "inspect_clip" and not e["is_error"] for e in evidence),
                )
                if item.gt_windows:
                    record.update(grounding(evidence, item.gt_windows))
                trace.update(final=final, steps=state.get("trace", []), evidence=evidence)
            elif self.method == "uniform":
                result = run_uniform(
                    llm, item.video_path, item.question, item.options, n_frames=self.n_frames,
                    subtitle_path=item.subtitle_path if self.with_subtitles else None,
                )
                record.update(
                    pred=result["answer"],
                    confidence=result["confidence"],
                    self_confidence=result["self_confidence"],
                    accepted=result["confidence"] >= self.cfg.agent.confidence_threshold,
                    forced=False,
                    tool_calls=0,
                    frames=result["frames_used"],
                    llm_turns=1,
                )
                trace.update(result=result)
            elif self.method == "gemini-native":
                result = self.native.ask(item.video_path, item.question, item.options)
                record.update(
                    pred=result["answer"],
                    confidence=result["confidence"],
                    self_confidence=result["self_confidence"],
                    accepted=result["confidence"] >= self.cfg.agent.confidence_threshold,
                    forced=False,
                    tool_calls=0,
                    frames=None,  # chosen by Gemini itself
                    llm_turns=1,
                    processing=result["processing"],
                )
                trace.update(result=result)
            else:
                raise ValueError(f"unknown method {self.method!r}")
        except Exception as err:
            record["error"] = f"{type(err).__name__}: {err}"
            trace["error"] = traceback.format_exc()

        pred = str(record.get("pred") or "").strip().upper()
        record["correct"] = bool(pred) and pred == item.answer.strip().upper()
        record["latency_s"] = round(time.perf_counter() - started, 3)
        usage = llm.meter.snapshot()
        record.update(
            llm_calls=usage["calls"], cost_usd=usage["cost_usd"], input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"], cache_read_tokens=usage["cache_read_tokens"],
        )
        if self.method == "gemini-native" and "result" in trace:  # not metered by the LLM client
            native = trace["result"]
            record.update(llm_calls=1, input_tokens=native["input_tokens"], output_tokens=native["output_tokens"])
        trace["record"] = record
        return record, trace


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", default="agent", choices=["agent", "uniform", "gemini-native"])
    parser.add_argument("--processing", default="agentic", choices=["agentic", "static"],
                        help="gemini-native: Gemini's agentic video processing or static frame sampling")
    parser.add_argument("--fps", type=float, help="gemini-native static: frames per second (default: Gemini's 1 fps)")
    parser.add_argument("--index-root", default="indexes")
    parser.add_argument("--frames", type=int, default=32, help="frames for the uniform baseline")
    parser.add_argument("--with-subtitles", action="store_true", help="give the uniform baseline the subtitles")
    parser.add_argument("--build-missing", action="store_true", help="index videos that have no index yet")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[], dest="overrides", metavar="SECTION.FIELD=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    items = load_items(args.data)[: args.limit]
    out = Path(args.out)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({"method": args.method, "frames": args.frames, "processing": args.processing, "fps": args.fps,
                    "overrides": args.overrides, "config": cfg.to_dict()}, indent=2),
        encoding="utf-8",
    )

    results_path = out / "results.jsonl"
    # Questions that ended in an error (quota, network) are retried on the next run.
    done = {r["qid"] for r in load_records(out) if not r.get("error")}
    pending = [it for it in items if it.qid not in done]
    print(f"{len(items)} questions, {len(done)} already done, {len(pending)} to run")

    evaluator = Evaluator(cfg, args.method, args.index_root, args.frames, args.with_subtitles, args.build_missing,
                          args.processing, args.fps)
    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(evaluator.evaluate, it): it for it in pending}
        for i, future in enumerate(as_completed(futures), start=1):
            record, trace = future.result()
            with write_lock:
                with open(results_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                (out / "traces" / f"{record['qid']}.json").write_text(
                    json.dumps(trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
                )
            status = "ERROR " + record["error"] if record.get("error") else ("correct" if record["correct"] else "wrong")
            print(f"[{i}/{len(pending)}] {record['qid']}: pred={record.get('pred')} gold={record['gold']} "
                  f"{status} ({record['latency_s']:.1f}s, ${record['cost_usd']:.4f})")

    summary = summarize(load_records(out))
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
