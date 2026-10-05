"""The OpenAI-compatible backend (Gemini / Ollama / ...) against a mock HTTP server."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from conftest import FakeTools

from videoscout.agent import AgentDeps, run_agent
from videoscout.agent.state import SubmitAnswer
from videoscout.config import load_config
from videoscout.llm import ClaudeLLM, LLMError, make_llm
from videoscout.llm_openai import RAW_BLOCK, OpenAICompatLLM, from_openai_response, simplify_schema, to_openai_messages
from videoscout.schema import to_api_schema
from videoscout.tools.search import SearchArgs

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = ["A. GATE A OPEN", "B. GATE B CLOSED"]


def gemini_cfg():
    cfg = load_config(ROOT / "configs" / "gemini.yaml")
    cfg.models.min_request_interval = 0.0
    return cfg


def chat(content=None, tool_calls=None, finish="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"model": "gemini-test", "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def call(call_id, name, args, **extra):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}, **extra}


class MockServer:
    def __init__(self, planner=(), structured=(), fail_first=0):
        self.planner, self.structured, self.fail_first = list(planner), list(structured), fail_first
        self.bodies, self.headers = [], []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        self.headers.append(request.headers)
        if self.fail_first:
            self.fail_first -= 1
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "quota exceeded"})
        if "tools" in body:
            return httpx.Response(200, json=self.planner.pop(0))
        if "response_format" in body:
            return httpx.Response(200, json=chat("```json\n" + json.dumps(self.structured.pop(0)) + "\n```"))
        return httpx.Response(200, json=chat("The sign reads GATE B CLOSED."))

    def llm(self, cfg):
        return OpenAICompatLLM(cfg.models, cfg.pricing, http=httpx.Client(transport=httpx.MockTransport(self.handler)))


def test_simplified_schema_has_no_refs_nullable_unions_or_additional_properties():
    search = simplify_schema(to_api_schema(SearchArgs))
    assert search["properties"]["visual_query"]["type"] == "string" and "anyOf" not in search["properties"]["visual_query"]
    submit = json.dumps(simplify_schema(to_api_schema(SubmitAnswer)))
    assert "$ref" not in submit and "$defs" not in submit and "additionalProperties" not in submit
    assert '"t_start"' in submit  # nested model was inlined


def test_transcript_translation():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "", "signature": "s"},
            {"type": "text", "text": "Searching."},
            {"type": "tool_use", "id": "t1", "name": "search_segments", "input": {"query": "sign"}},
            {"type": "tool_use", "id": "t2", "name": "query_tracks", "input": {"sql": "SELECT 1"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "3 segments"},
            {"type": "tool_result", "tool_use_id": "t2", "content": "bad sql", "is_error": True},
            {"type": "text", "text": "[budget] 2/12"},
        ]},
        {"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}},
            {"type": "text", "text": "what is this?"},
        ]},
    ]
    out = to_openai_messages("sys", messages)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "tool", "user", "user"]
    assert out[2]["content"] == "Searching." and [c["id"] for c in out[2]["tool_calls"]] == ["t1", "t2"]
    assert out[4] == {"role": "tool", "tool_call_id": "t2", "content": "ERROR: bad sql"}
    assert out[5]["content"] == "[budget] 2/12"
    assert out[6]["content"][0] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}


def test_response_translation_handles_missing_ids_bad_json_and_signatures():
    data = chat(tool_calls=[
        {"type": "function", "function": {"name": "query_tracks", "arguments": "{not json"}},
        call("c9", "search_segments", {"query": "x"}, extra_content={"google": {"thought_signature": "sig"}}),
    ])  # finish_reason "stop" despite tool calls, as some providers do
    response = from_openai_response(data)
    assert response.stop_reason == "tool_use"
    first, second, raw = response.content
    assert first["id"].startswith("call_") and first["input"] == {"_invalid_json": "{not json"}
    assert second["input"] == {"query": "x"}
    assert raw["type"] == RAW_BLOCK and raw["message"]["tool_calls"][1]["extra_content"]["google"]["thought_signature"] == "sig"
    assert raw["message"]["tool_calls"][0]["id"] == first["id"]  # generated id is used on both sides


def test_full_agent_over_openai_compatible_api(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg = gemini_cfg()
    submit = {"answer": "B", "confidence": 0.8, "rationale": "Sign reads GATE B CLOSED.",
              "evidence": [{"t_start": 80, "t_end": 90, "observation": "GATE B CLOSED"}]}
    server = MockServer(
        planner=[
            chat("Let me search.", [call("c1", "search_segments", {"query": "sign"},
                                         extra_content={"google": {"thought_signature": "sig-1"}})], "tool_calls"),
            chat(None, [call("c2", "submit_answer", submit)], "tool_calls"),
        ],
        structured=[{"supported": True, "confidence": 0.9, "issues": "", "next_step": ""}],
        fail_first=1,  # first request gets a 429 and is retried
    )
    llm = server.llm(cfg)
    state = run_agent(AgentDeps(llm, FakeTools(), cfg, overview="Video duration: 02:00."), "What does the sign say?", OPTIONS)

    assert state["final"]["answer"] == "B" and state["final"]["accepted"] and state["final"]["confidence"] == 0.9
    assert server.headers[0]["authorization"] == "Bearer test-key"
    planner_bodies = [b for b in server.bodies if "tools" in b]
    assert len(planner_bodies) == 3  # 429, retry, second turn
    second = planner_bodies[2]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool", "user"]
    # The provider's assistant message (with its thought signature) is sent back verbatim.
    assert second[2]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "sig-1"
    assert planner_bodies[0]["tools"][0]["function"]["name"] == "search_segments"
    assert planner_bodies[0]["reasoning_effort"] == "medium"  # from configs/gemini.yaml
    verify_body = next(b for b in server.bodies if "response_format" in b)
    assert verify_body["reasoning_effort"] == "low"
    assert verify_body["response_format"]["type"] == "json_schema" and "JSON schema" in verify_body["messages"][0]["content"]
    assert llm.meter.snapshot()["cost_usd"] == 0.0  # no pricing configured for free models


def test_daily_quota_fails_fast_instead_of_retrying(monkeypatch):
    import time

    from videoscout.llm_openai import QuotaExhaustedError

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    def handler(request):  # the shape Gemini returns when the free tier's daily quota is used up
        calls.append(1)
        return httpx.Response(429, json=[{"error": {
            "code": 429, "message": "You exceeded your current quota.\n* limit: 20", "status": "RESOURCE_EXHAUSTED",
            "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "66734s"}]}}])

    cfg = gemini_cfg()
    llm = OpenAICompatLLM(cfg.models, cfg.pricing, http=httpx.Client(transport=httpx.MockTransport(handler)))
    started = time.monotonic()
    with pytest.raises(QuotaExhaustedError, match="resets in about 18.5 h"):
        llm.create("planner", [{"role": "user", "content": "hi"}])
    assert len(calls) == 1 and time.monotonic() - started < 5


def test_missing_api_key_is_a_clear_error(monkeypatch):
    cfg = gemini_cfg()  # may load a real key from .env, so remove it afterwards
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    llm = MockServer().llm(cfg)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        llm.create("planner", [{"role": "user", "content": "hi"}])


def test_make_llm_selects_backend():
    assert isinstance(make_llm(load_config().models, {}), ClaudeLLM)
    assert isinstance(make_llm(gemini_cfg().models, {}), OpenAICompatLLM)
    assert isinstance(make_llm(load_config(ROOT / "configs" / "ollama.yaml").models, {}), OpenAICompatLLM)
