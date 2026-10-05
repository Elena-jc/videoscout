"""Web API: listing, streaming with Range, upload + indexing job, and an agent run
driven by the scripted model (no API calls)."""

from __future__ import annotations

import json
import time

import pytest
from conftest import TINY_SRT, FakeAnthropic, tool_use, write_tiny_video
from starlette.testclient import TestClient

from videoscout.index.build import build_index
from videoscout.llm import ClaudeLLM
from videoscout.web.app import create_app

# Keep the 2B local models (captioner, clip embedder, reranker) and SAM 3 out of the tests.
NO_LOCAL_MODELS = ["index.captioner=none", "index.clip_embedder=", "retrieval.rerank=false", "grounding.backend=yoloe"]

SUBMIT = {"answer": "B", "confidence": 0.8, "rationale": "The subtitles mention a forklift.",
          "evidence": [{"t_start": 10, "t_end": 20, "observation": "forklift driver waves"}]}


def wait_for(client, job_id, timeout=60):
    events, deadline = [], time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}", params={"after": len(events) - 1}).json()
        events += body["events"]
        if body["done"]:
            return events
        time.sleep(0.05)
    raise TimeoutError(job_id)


@pytest.fixture
def env(tmp_path, cfg):
    media = tmp_path / "media"
    media.mkdir()
    video = write_tiny_video(media / "tiny.mp4")
    (media / "tiny.srt").write_text(TINY_SRT, encoding="utf-8")
    (media / "qa.jsonl").write_text(json.dumps({
        "qid": "t1", "video_id": "tiny", "question": "Who waves?", "options": ["A. nobody", "B. the forklift driver"],
        "answer": "B"}) + "\n", encoding="utf-8")
    index_root = tmp_path / "indexes"
    build_index(video, index_root / "tiny", cfg, srt_path=media / "tiny.srt", with_dense=False, with_tracks=False,
                log=lambda _: None)
    clients = []

    def llm_factory(models, pricing):
        client = FakeAnthropic(
            planner=[[tool_use("search_segments", {"query": "forklift"})], [tool_use("submit_answer", SUBMIT)]],
            structured=[{"supported": True, "confidence": 0.9, "issues": "", "next_step": ""}],
        )
        clients.append(client)
        return ClaudeLLM(models, pricing, client=client)

    app = create_app(index_root, tmp_path / "uploads", tmp_path / "runs", llm_factory=llm_factory, overrides=NO_LOCAL_MODELS)
    return TestClient(app), tmp_path


def test_listing_streaming_and_questions(env):
    client, _ = env
    assert client.get("/api/health").json() == {"status": "ok"}
    assert {p["id"] for p in client.get("/api/providers").json()} == {"gemini", "claude", "ollama"}
    (video,) = client.get("/api/videos").json()
    assert video["id"] == "tiny" and video["has_subtitles"] and video["available"]
    partial = client.get("/api/videos/tiny/stream", headers={"Range": "bytes=0-99"})
    assert partial.status_code == 206 and len(partial.content) == 100  # the player can seek
    assert client.get("/api/videos/tiny/questions").json()[0]["question"] == "Who waves?"
    assert client.get("/api/videos/..%2F..%2Fetc/stream").status_code == 404
    assert client.get("/").status_code == 200


def test_agent_run_streams_steps_and_saves_a_record(env):
    client, tmp_path = env
    job = client.post("/api/ask", json={"video_id": "tiny", "question": "Who waves?",
                                        "options": ["A. nobody", "B. the forklift driver"], "provider": "claude"})
    assert job.status_code == 202
    events = wait_for(client, job.json()["job_id"])
    types = [e["type"] for e in events]
    assert types[0] == "start" and types[-1] == "final" and "error" not in types
    steps = [e["step"]["node"] for e in events if e["type"] == "step"]
    assert steps == ["agent", "tools", "agent", "verify"]
    final = events[-1]
    assert final["final"]["answer"] == "B" and final["final"]["accepted"] and final["budget"]["tool_calls"] == 1
    assert (tmp_path / "runs" / f"{job.json()['job_id']}.json").exists()


def test_missing_key_becomes_an_error_event(env, monkeypatch):
    from videoscout.llm import make_llm

    _, tmp_path = env
    # The real Gemini backend: with no key it fails before any network call.
    app = create_app(tmp_path / "indexes", tmp_path / "uploads", tmp_path / "runs", llm_factory=make_llm,
                     overrides=NO_LOCAL_MODELS)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # create_app may have loaded one from .env
    real = TestClient(app)
    job = real.post("/api/ask", json={"video_id": "tiny", "question": "?", "provider": "gemini"}).json()
    events = wait_for(real, job["job_id"])
    assert events[-1]["type"] == "error" and "GEMINI_API_KEY" in events[-1]["message"]


def test_bad_requests(env):
    client, _ = env
    assert client.post("/api/ask", json={"video_id": "nope", "question": "?"}).status_code == 404
    assert client.post("/api/ask", json={"video_id": "tiny", "question": ""}).status_code == 400
    assert client.post("/api/ask", json={"video_id": "tiny", "question": "?", "provider": "x"}).status_code == 400
    assert client.post("/api/videos", json={"path": "C:/definitely/not/here.mp4"}).status_code == 400


def test_static_export_replays_recorded_runs(env):
    from videoscout.web.export import export_site

    _, tmp_path = env
    run = tmp_path / "run"
    (run / "traces").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"config": {"models": {"planner": "gemini-x"},
                                                            "agent": {"max_tool_calls": 12, "max_frames": 48}}}))
    steps = [{"node": "agent", "text": "", "tool_calls": [{"name": "search_segments", "input": {"query": "x"}}]},
             {"node": "tools", "name": "search_segments", "args": {}, "is_error": False, "latency_s": 0.1, "result": "..."}]
    record = {"video_id": "tiny", "gold": "B", "correct": True, "tool_calls": 1, "frames": 0, "llm_calls": 3,
              "latency_s": 12.5, "cost_usd": 0.0}
    (run / "traces" / "t1.json").write_text(json.dumps({"question": "Who waves?", "options": ["A. x", "B. y"],
                                                         "final": {"answer": "B"}, "steps": steps, "record": record}))
    (run / "traces" / "t2.json").write_text(json.dumps({"question": "?", "options": [], "error": "quota", "record": record}))
    site = export_site(run, tmp_path / "indexes" / "tiny", tmp_path / "site", repo_url="https://example.com/repo")

    demo = json.loads((site / "demo.json").read_text(encoding="utf-8"))
    (recorded,) = demo["runs"]["tiny"]  # the failed run is left out
    assert [e["type"] for e in recorded["events"]] == ["start", "step", "step", "final"]
    assert recorded["events"][-1]["correct"] is True
    # The tiny test video is OpenCV mp4v, which browsers cannot play: the export re-encodes it.
    from videoscout.video import browser_playable

    published = site / demo["videos"][0]["url"]
    assert published.exists() and browser_playable(published) and (site / ".nojekyll").exists()
    assert 'window.VIDEOSCOUT_STATIC = "demo.json"' in (site / "index.html").read_text(encoding="utf-8")


def test_upload_indexes_the_video(env, monkeypatch, cfg):
    client, tmp_path = env
    # Skip the heavy models for this test: index with subtitles only.
    import videoscout.web.app as web

    real_build = web.build_index
    monkeypatch.setattr(web, "build_index", lambda *a, **k: real_build(*a, **{**k, "with_dense": False, "with_tracks": False}))
    video = write_tiny_video(tmp_path / "clip.mp4", seconds=6)
    with open(video, "rb") as v:
        response = client.post("/api/videos", files={"video": ("My Clip.mp4", v, "video/mp4"),
                                                      "subtitles": ("s.srt", TINY_SRT.encode(), "text/plain")})
    assert response.status_code == 202
    body = response.json()
    events = wait_for(client, body["job_id"])
    assert events[-1] == {**events[-1], "type": "done", "video_id": body["video_id"]}
    assert body["video_id"].startswith("my-clip-")
    assert body["video_id"] in {v["id"] for v in client.get("/api/videos").json()}
    # The upload was mp4v, so indexing also wrote a browser preview, and that is what gets streamed.
    preview = next((tmp_path / "indexes" / body["video_id"]).glob("preview.*"))
    streamed = client.get(f"/api/videos/{body['video_id']}/stream")
    assert streamed.status_code == 200 and streamed.content == preview.read_bytes()
