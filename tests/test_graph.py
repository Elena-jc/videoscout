"""Agent graph behaviour with a scripted model: routing, budget enforcement,
verification loop, and the tool-use protocol invariants (checked on every call
by FakeAnthropic)."""

from __future__ import annotations

import pytest
from conftest import THINKING, FakeAnthropic, FakeTools, fake_response, text, tool_use

from videoscout.agent import AgentDeps, run_agent
from videoscout.agent.graph import normalize_answer
from videoscout.agent.prompts import NUDGE
from videoscout.llm import FALLBACK_BETA, ClaudeLLM, RefusalError

OPTIONS = ["A. GATE A OPEN", "B. GATE B CLOSED", "C. EXIT ONLY", "D. NO PARKING"]
SUPPORTED = {"supported": True, "confidence": 0.9, "issues": "", "next_step": ""}


def submit(answer="B", confidence=0.8):
    return tool_use(
        "submit_answer",
        {
            "answer": answer,
            "confidence": confidence,
            "rationale": "The sign reads GATE B CLOSED.",
            "evidence": [{"t_start": 80.0, "t_end": 90.0, "observation": "sign reads GATE B CLOSED"}],
        },
    )


def make(cfg, planner, structured=()):
    client = FakeAnthropic(planner=planner, structured=list(structured))
    tools = FakeTools()
    deps = AgentDeps(ClaudeLLM(cfg.models, cfg.pricing, client=client), tools, cfg, overview="Video duration: 02:00.")
    return deps, client, tools


def test_happy_path_parallel_calls_clamping_and_verification(cfg):
    planner = [
        [THINKING, text("Search two ways."), tool_use("search_segments", {"query": "sign"}),
         tool_use("search_segments", {"query": "gate", "visual_query": "a sign board"})],
        [tool_use("inspect_clip", {"t_start": 80, "t_end": 90, "question": "What does the sign say?", "num_frames": 20})],
        [submit("B. GATE B CLOSED")],
    ]
    deps, client, tools = make(cfg, planner, [SUPPORTED])
    state = run_agent(deps, "What does the sign say?", OPTIONS)

    final = state["final"]
    assert final["answer"] == "B" and final["accepted"] and not final["forced"]
    assert final["confidence"] == 0.9 and final["self_confidence"] == 0.8
    assert state["tool_calls_used"] == 3
    # num_frames=20 was clamped in code to max_frames_per_inspect before the tool ran.
    assert tools.calls[2] == ("inspect_clip", {"t_start": 80, "t_end": 90, "question": "What does the sign say?", "num_frames": 8})
    assert state["frames_used"] == 8

    first, second, _ = client.planner_requests()
    assert first["betas"] == [FALLBACK_BETA] and first["fallbacks"] == "default"
    assert first["cache_control"] == {"type": "ephemeral"}
    assert first["output_config"]["effort"] == cfg.models.planner_effort
    submit_spec = next(t for t in first["tools"] if t["name"] == "submit_answer")
    assert submit_spec["strict"] is True and submit_spec["input_schema"]["additionalProperties"] is False
    # The thinking block goes back byte-for-byte, and both parallel results share ONE user message.
    assert second["messages"][1]["content"][0] == THINKING
    reply = second["messages"][2]["content"]
    assert [b["type"] for b in reply] == ["tool_result", "tool_result", "text"]
    assert reply[-1]["text"].startswith("[budget] tool calls used 2/")

    (verify_request,) = client.structured_requests()
    assert verify_request["model"] == cfg.models.verifier
    ledger = verify_request["messages"][0]["content"]
    assert "Step 3: inspect_clip" in ledger and "Proposed answer: B. GATE B CLOSED" in ledger


def test_rejected_answer_gets_feedback_and_can_retry(cfg):
    rejected = {"supported": False, "confidence": 0.3, "issues": "No visual confirmation.", "next_step": "inspect 80-90s"}
    planner = [
        [tool_use("search_segments", {"query": "sign"})],
        [submit("A", 0.6)],
        [tool_use("inspect_clip", {"t_start": 80, "t_end": 90, "question": "Read the sign"})],
        [submit("B", 0.85)],
    ]
    deps, client, _ = make(cfg, planner, [rejected, SUPPORTED])
    state = run_agent(deps, "What does the sign say?", OPTIONS)

    assert state["final"]["answer"] == "B" and state["final"]["accepted"]
    assert state["verify_rounds"] == 2
    feedback = client.planner_requests()[2]["messages"][-1]["content"][0]
    assert feedback["type"] == "tool_result" and "not accepted" in feedback["content"]
    assert "No visual confirmation." in feedback["content"]


def test_budget_is_enforced_in_code_then_forced_answer(cfg):
    cfg.agent.max_tool_calls = 2
    planner = [[tool_use("search_segments", {"query": f"q{i}"})] for i in range(4)]
    forced = {"answer": "C", "confidence": 0.4, "rationale": "Best guess from the log."}
    deps, client, tools = make(cfg, planner, [forced])
    state = run_agent(deps, "What does the sign say?", OPTIONS)

    assert len(tools.calls) == 2  # calls 3 and 4 were refused, never executed
    assert state["final"]["forced"] and state["final"]["answer"] == "C" and not state["final"]["accepted"]
    assert len(client.planner_requests()) == 4  # no extra planner call after the second refusal
    refused = client.planner_requests()[3]["messages"][-1]["content"][0]
    assert refused["is_error"] and "budget exhausted" in refused["content"]


def test_turn_without_tool_call_is_nudged(cfg):
    deps, client, _ = make(cfg, [[text("Hmm, let me think.")], [submit("B")]], [SUPPORTED])
    state = run_agent(deps, "What does the sign say?", OPTIONS)
    assert state["nudges"] == 1 and state["final"]["answer"] == "B"
    assert client.planner_requests()[1]["messages"][-1]["content"] == NUDGE


def test_invalid_submit_arguments_are_returned_as_error(cfg):
    deps, client, _ = make(cfg, [[tool_use("submit_answer", {"answer": "B"})], [submit("B")]], [SUPPORTED])
    state = run_agent(deps, "What does the sign say?", OPTIONS)
    error = client.planner_requests()[1]["messages"][-1]["content"][0]
    assert error["is_error"] and "Invalid submit_answer arguments" in error["content"]
    assert state["final"]["answer"] == "B"


def test_verification_can_be_disabled_for_ablation(cfg):
    cfg.agent.verify = False
    deps, client, _ = make(cfg, [[submit("(b)", 0.7)]])
    state = run_agent(deps, "What does the sign say?", OPTIONS)
    assert state["final"] == {**state["final"], "answer": "B", "accepted": True, "confidence": 0.7}
    assert client.structured_requests() == []


def test_refusal_is_raised_not_silently_used(cfg):
    client = FakeAnthropic(planner=[fake_response([text("")], stop_reason="refusal")])
    llm = ClaudeLLM(cfg.models, cfg.pricing, client=client)
    with pytest.raises(RefusalError):
        llm.create("planner", [{"role": "user", "content": "hi"}], tools=[{"name": "x"}])


def test_usage_meter_prices_cache_reads(cfg):
    deps, _, _ = make(cfg, [[submit("B")]], [SUPPORTED])
    run_agent(deps, "What does the sign say?", OPTIONS)
    usage = deps.llm.meter.snapshot()
    # 2 calls x (1000 input * $4 + 200 output * $20 + 500 cache-read * $0.20) / 1M
    assert usage["calls"] == 2 and usage["cost_usd"] == pytest.approx(2 * (0.004 + 0.004 + 0.0001))


@pytest.mark.parametrize(
    "raw, expected",
    [("B", "B"), ("(c)", "C"), ("B. GATE B CLOSED", "B"), ("gate b closed", "B"), ("d: no parking", "D"), ("E", "E")],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw, OPTIONS) == expected
