"""Metrics, failure taxonomy and run comparison.

    python -m videoscout.eval.report runs/uniform32 runs/agent runs/agent_noverify --out runs/report.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_records(run_dir: str | Path) -> list[dict[str, Any]]:
    """Latest record per question (retried questions are appended again)."""
    path = Path(run_dir) / "results.jsonl"
    if not path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["qid"]] = record
    return list(latest.values())


def expected_calibration_error(pairs: list[tuple[float, bool]], bins: int = 10) -> float | None:
    """ECE: average |accuracy - confidence| over confidence bins, weighted by bin size.
    0 means "when it says 70%, it is right 70% of the time"."""
    if not pairs:
        return None
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for conf, correct in pairs:
        buckets[min(int(conf * bins), bins - 1)].append((conf, correct))
    total = len(pairs)
    return round(
        sum(
            len(b) / total * abs(sum(c for _, c in b) / len(b) - sum(p for p, _ in b) / len(b))
            for b in buckets.values()
        ),
        4,
    )


def failure_category(r: dict[str, Any]) -> str | None:
    """Root-cause bucket for a wrong answer. With ground-truth windows we can tell
    "never found the moment" from "found it but misread it"."""
    if r.get("correct"):
        return None
    if r.get("error"):
        return "error"
    if r.get("forced"):
        return "forced_answer"
    if r.get("search_hit") is False and r.get("inspect_hit") is False:
        return "retrieval_miss"
    if r.get("inspect_hit") is False:
        return "looked_in_wrong_place"
    if r.get("method") == "agent" and not r.get("used_inspect"):
        return "no_visual_check"
    if r.get("inspect_hit") is True:
        return "perception_or_reasoning"
    return "wrong_answer"


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0}

    def mean(key: str) -> float:
        return round(sum(r.get(key) or 0 for r in records) / n, 4)

    correct = sum(bool(r.get("correct")) for r in records)
    accepted = [r for r in records if r.get("accepted")]
    wrong = [r for r in records if not r.get("correct")]
    summary: dict[str, Any] = {
        "n": n,
        "accuracy": round(correct / n, 4),
        "errors": sum(1 for r in records if r.get("error")),
        # Selective answering: how often we trust the answer, and how good trusted answers are.
        "coverage": round(len(accepted) / n, 4),
        "selective_accuracy": round(sum(bool(r.get("correct")) for r in accepted) / len(accepted), 4) if accepted else None,
        "false_accept_rate": round(sum(1 for r in wrong if r.get("accepted")) / len(wrong), 4) if wrong else None,
        "ece": expected_calibration_error(
            [(float(r["confidence"]), bool(r.get("correct"))) for r in records if r.get("confidence") is not None]
        ),
        "forced_rate": round(sum(1 for r in records if r.get("forced")) / n, 4),
        "avg_tool_calls": mean("tool_calls"),
        "avg_frames": mean("frames"),
        "avg_llm_calls": mean("llm_calls"),
        "avg_latency_s": mean("latency_s"),
        "avg_cost_usd": mean("cost_usd"),
        "total_cost_usd": round(sum(r.get("cost_usd") or 0 for r in records), 4),
        "failures": dict(Counter(c for r in records if (c := failure_category(r)))),
    }
    grounded = [r for r in records if "inspect_hit" in r]
    if grounded:
        summary["search_hit_rate"] = round(sum(bool(r["search_hit"]) for r in grounded) / len(grounded), 4)
        summary["inspect_hit_rate"] = round(sum(bool(r["inspect_hit"]) for r in grounded) / len(grounded), 4)

    by_type: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        by_type[r.get("task_type") or "-"].append(bool(r.get("correct")))
    summary["by_task_type"] = {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for k, v in sorted(by_type.items())}
    return summary


def write_failures_csv(run_dir: str | Path, records: list[dict[str, Any]]) -> Path:
    """One row per wrong answer, with a blank `notes` column for manual review."""
    path = Path(run_dir) / "failures.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "task_type", "gold", "pred", "confidence", "accepted", "category", "trace", "notes"])
        for r in records:
            category = failure_category(r)
            if category:
                writer.writerow([
                    r["qid"], r.get("task_type"), r.get("gold"), r.get("pred"), r.get("confidence"),
                    r.get("accepted"), category, f"traces/{r['qid']}.json", "",
                ])
    return path


def _pct(x: Any) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def comparison_table(runs: dict[str, dict[str, Any]]) -> str:
    header = (
        "| run | n | acc % | selective acc % | coverage % | ECE | avg tool calls | avg frames "
        "| avg latency s | avg cost $ |\n|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = [
        f"| {name} | {s['n']} | {_pct(s.get('accuracy'))} | {_pct(s.get('selective_accuracy'))} | "
        f"{_pct(s.get('coverage'))} | {s.get('ece') if s.get('ece') is not None else '-'} | "
        f"{s.get('avg_tool_calls', 0):.1f} | {s.get('avg_frames', 0):.1f} | {s.get('avg_latency_s', 0):.1f} | "
        f"{s.get('avg_cost_usd', 0):.4f} |"
        for name, s in runs.items()
        if s.get("n")
    ]
    return "\n".join([header, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare evaluation runs")
    parser.add_argument("runs", nargs="+", help="run directories written by videoscout.eval.run")
    parser.add_argument("--out", help="write the markdown report here")
    args = parser.parse_args()

    summaries, sections = {}, []
    for run_dir in args.runs:
        records = load_records(run_dir)
        summary = summarize(records)
        summaries[Path(run_dir).name] = summary
        if records:
            write_failures_csv(run_dir, records)
            sections.append(f"### {Path(run_dir).name}\nfailures: {json.dumps(summary.get('failures', {}))}")

    report = "## Results\n\n" + comparison_table(summaries) + "\n\n## Failure taxonomy\n\n" + "\n\n".join(sections)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
