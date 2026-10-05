"""LVBench conversion, the retrieval-only evaluation, and the Gemini-native
baseline against a mocked HTTP API (no network, no key)."""

from __future__ import annotations

import json

import httpx
import pytest
from test_memory import FakeClipEmbedder, FakeReranker, _three_scene_index

from videoscout.eval.data import QAItem, convert_lvbench, load_items
from videoscout.eval.gemini_native import GeminiNative, parse_reply
from videoscout.eval.retrieval import evaluate, first_hit_rank, retrieval_config, table


def test_convert_lvbench_keeps_downloaded_videos_and_time_references(tmp_path):
    meta = tmp_path / "video_info.meta.jsonl"
    rows = [
        {"key": "abc", "type": "cartoon", "qa": [
            {"uid": 55, "question": "What year appears?\n(A) 1636\n(B) 1366", "answer": "b",
             "question_type": ["key information retrieval"], "time_reference": "00:15-00:19"}]},
        {"key": "missing", "type": "sports", "qa": [{"uid": 1, "question": "?\n(A) x", "answer": "A"}]},
    ]
    meta.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "abc.mp4").write_bytes(b"")
    convert_lvbench(meta, tmp_path / "videos", tmp_path / "qa.jsonl")
    (item,) = load_items(tmp_path / "qa.jsonl")
    assert item.qid == "lvb-55" and item.video_id == "abc" and item.answer == "B"
    assert item.question == "What year appears?" and item.options == ["A. 1636", "B. 1366"]
    assert item.gt_windows == [[15.0, 19.0]] and item.extra["video_type"] == "cartoon"


def test_retrieval_eval_scores_configurations(tmp_path, cfg):
    index = _three_scene_index(tmp_path, cfg)
    items = [QAItem("q1", "index", "something green", [], "A", gt_windows=[[22.0, 25.0]]),
             QAItem("q2", "index", "a red thing", [], "A", gt_windows=[[45.0, 50.0]]),
             QAItem("q3", "index", "no annotation", [], "A")]
    factories = {"embedder:clip": lambda: FakeClipEmbedder(), "reranker": lambda: FakeReranker()}
    results = evaluate(items, index.dir.parent, ["clip", "clip+rerank"], top_k=5, factories=factories, log=lambda _: None)
    assert results["clip"]["n"] == 2 and results["clip"]["recall@5"] == 1.0
    # The fake reranker always puts the first (blue) clip on top: recall@1 drops to 0.
    assert results["clip+rerank"]["recall@1"] == 0.0
    assert "| clip+rerank | 2 |" in table(results, 5)
    assert first_hit_rank([(0, 10), (20, 30)], [[25, 26]]) == 2 and first_hit_rank([(0, 10)], [[50, 60]]) is None
    with pytest.raises(ValueError):
        retrieval_config("clip+keyframe", cfg.retrieval)


def _mock_gemini(requests):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/upload/v1beta/files":
            return httpx.Response(200, headers={"x-goog-upload-url": "https://upload.example/session/1"})
        if request.url.host == "upload.example":
            return httpx.Response(200, json={"file": {"name": "files/f1", "uri": "https://files/f1",
                                                      "mimeType": "video/mp4", "state": "ACTIVE"}})
        if request.url.path == "/v1beta/interactions":
            return httpx.Response(200, json={"steps": [
                {"type": "thought", "summary": [{"type": "text", "text": "..."}]},
                {"type": "processing_call", "id": "c1"},
                {"type": "model_output", "content": [{"type": "text", "text": "ANSWER: (B)\nCONFIDENCE: 0.8"}]}],
                "usage": {"input_tokens": 1234, "output_tokens": 9}})
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_gemini_native_agentic_uploads_once_and_parses_the_answer(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00" * 100)
    requests = []
    native = GeminiNative("gemini-x", "agentic", api_key="test-key", client=_mock_gemini(requests),
                          cache_path=tmp_path / "files.json")
    out = native.ask(str(video), "Which?", ["A. one", "B. two"])
    assert out["answer"] == "B" and out["confidence"] == 0.8 and out["input_tokens"] == 1234
    body = json.loads(requests[-1].content)
    assert body["model"] == "gemini-x" and body["input"][0]["processing"] == "agentic"
    assert body["input"][0]["uri"] == "https://files/f1" and "B. two" in body["input"][1]["text"]

    native.ask(str(video), "Again?", ["A. one", "B. two"])  # the upload is cached
    assert sum(r.url.path == "/upload/v1beta/files" for r in requests) == 1

    static = GeminiNative("gemini-x", "static", fps=0.5, api_key="test-key", client=_mock_gemini(requests),
                          cache_path=tmp_path / "files.json")
    static.ask(str(video), "Which?", ["A. one", "B. two"])
    assert json.loads(requests[-1].content)["input"][0]["processing"] == {"type": "static", "fps": 0.5}
    assert parse_reply("no structure, maybe A", ["A. x", "B. y"])[1] == 0.5
