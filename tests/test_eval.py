from __future__ import annotations

import json

import pytest

from videoscout.eval.data import load_items
from videoscout.eval.report import expected_calibration_error, failure_category, summarize
from videoscout.eval.run import grounding


def test_ece_perfect_and_overconfident():
    assert expected_calibration_error([(0.95, True)] * 19 + [(0.95, False)]) == pytest.approx(0.0)
    assert expected_calibration_error([(0.9, False)] * 10) == pytest.approx(0.9)


def test_grounding_from_evidence_ledger():
    evidence = [
        {"tool": "search_segments", "is_error": False, "result": "[1] 01:20-01:30 (t=80.0-90.0s) matched by: text #1"},
        {"tool": "inspect_clip", "is_error": False, "args": {"t_start": 0, "t_end": 30}, "result": "..."},
    ]
    assert grounding(evidence, [[85, 100]]) == {"search_hit": True, "inspect_hit": False}


def test_failure_taxonomy_and_summary():
    records = [
        {"qid": "1", "correct": True, "accepted": True, "confidence": 0.9, "method": "agent", "tool_calls": 4},
        {"qid": "2", "correct": False, "accepted": True, "confidence": 0.8, "method": "agent",
         "used_inspect": True, "search_hit": True, "inspect_hit": True},
        {"qid": "3", "correct": False, "accepted": False, "confidence": 0.3, "method": "agent",
         "used_inspect": True, "search_hit": False, "inspect_hit": False},
        {"qid": "4", "correct": False, "forced": True, "method": "agent"},
    ]
    assert [failure_category(r) for r in records] == [None, "perception_or_reasoning", "retrieval_miss", "forced_answer"]
    s = summarize(records)
    assert s["accuracy"] == 0.25 and s["coverage"] == 0.5 and s["selective_accuracy"] == 0.5
    assert s["false_accept_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert s["inspect_hit_rate"] == 0.5


def test_load_items_resolves_relative_paths(tmp_path):
    row = {"qid": "q1", "video_id": "v", "question": "?", "options": ["A. x"], "answer": "A",
           "video_path": "v.mp4", "difficulty": "hard"}
    (tmp_path / "qa.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (item,) = load_items(tmp_path / "qa.jsonl")
    assert item.video_path == str((tmp_path / "v.mp4").resolve()) and item.extra == {"difficulty": "hard"}
