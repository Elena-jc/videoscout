r"""The agent as a LangGraph state machine.

                +---------------- tools <--------------+
                v                   |                  | tool calls
    START --> agent ----------------|----------------- +
                |  \                +-- budget refused twice --> force_answer --> END
                |   \--- no tool call ---> nudge --> agent
                |    \-- turn limit ---> force_answer --> END
                v submit_answer
              verify --- accepted, or out of rounds/budget ---> END
                \------- rejected: feedback as tool_result ---> agent

Why a graph instead of a plain while-loop: the control flow has more than one
exit (verified answer, forced answer), a budget that is enforced by code rather
than by asking the model nicely, and a verifier that runs in a fresh context.
Each of those is a node or an edge you can test, log and ablate on its own.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from ..config import Config
from ..llm import LLMClient, LLMError, blocks_to_dicts
from ..schema import format_validation_error, to_api_schema
from ..tools.registry import ToolOutput, ToolRegistry
from . import prompts
from .state import AgentState, ForcedAnswer, SubmitAnswer, Verdict

SUBMIT = "submit_answer"
LEDGER_RESULT_CHARS = 1500


@dataclass
class AgentDeps:
    llm: LLMClient
    tools: ToolRegistry
    cfg: Config
    overview: str = ""


def tool_uses(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]


def assistant_text(message: dict[str, Any]) -> str:
    return "\n".join(b["text"] for b in message.get("content", []) if isinstance(b, dict) and b.get("type") == "text")


def normalize_answer(answer: str, options: list[str]) -> str:
    """Map free-form answers like 'B', '(B)', 'B. a bus' or 'a bus' to an option letter."""
    answer = answer.strip()
    if not options:
        return answer
    letters = [opt.strip()[0].upper() for opt in options if opt.strip()]
    match = re.match(r"^\(?([A-Za-z])(?:[\).:\s]|$)", answer)
    if match and match.group(1).upper() in letters:
        return match.group(1).upper()
    lowered = answer.lower()
    for letter, opt in zip(letters, options):
        body = re.sub(r"^\(?[A-Za-z][\).:]\s*", "", opt.strip()).lower()
        if body and (body == lowered or body in lowered):
            return letter
    return answer


def _clip01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def render_ledger(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "(no tool calls were made)"
    parts = []
    for e in evidence:
        flag = " [ERROR]" if e["is_error"] else ""
        args = json.dumps(e["args"], ensure_ascii=False)
        parts.append(f"### Step {e['step']}: {e['tool']}({args}){flag}\n{_truncate(e['result'], LEDGER_RESULT_CHARS)}")
    return "\n\n".join(parts)


def _question_block(state: AgentState) -> str:
    block = f"Question: {state['question']}"
    if state.get("options"):
        block += "\nOptions:\n" + prompts.render_options(state["options"])
    return block


def build_graph(deps: AgentDeps, checkpointer: Any = None):
    acfg = deps.cfg.agent
    tool_schemas = deps.tools.schemas()
    submit_schema: dict[str, Any] = {
        "name": SUBMIT,
        "description": prompts.SUBMIT_DESCRIPTION,
        "input_schema": to_api_schema(SubmitAnswer),
    }
    if acfg.strict_tools:
        submit_schema["strict"] = True
    all_tools = tool_schemas + [submit_schema]

    def budget_line(calls: int, frames: int) -> str:
        return (
            f"[budget] tool calls used {calls}/{acfg.max_tool_calls}, "
            f"frames used {frames}/{acfg.max_frames}"
        )

    # ---------------------------------------------------------------- planner
    def agent_node(state: AgentState) -> dict[str, Any]:
        new: list[dict[str, Any]] = []
        if not state.get("messages"):
            task = prompts.render_task(
                state["question"], state.get("options", []), deps.overview, acfg.max_tool_calls, acfg.max_frames
            )
            new.append({"role": "user", "content": task})
        started = time.perf_counter()
        response = deps.llm.create(
            "planner",
            state.get("messages", []) + new,
            system=prompts.PLANNER_SYSTEM,
            tools=all_tools,
            cache=True,
        )
        message = {"role": "assistant", "content": blocks_to_dicts(response.content)}
        new.append(message)
        return {
            "messages": new,
            "llm_turns": state.get("llm_turns", 0) + 1,
            "trace": [
                {
                    "node": "agent",
                    "latency_s": round(time.perf_counter() - started, 3),
                    "stop_reason": response.stop_reason,
                    "text": assistant_text(message),
                    "tool_calls": [{"name": u["name"], "input": u["input"]} for u in tool_uses(message)],
                }
            ],
        }

    def route_after_agent(state: AgentState) -> str:
        uses = tool_uses(state["messages"][-1])
        if any(u["name"] == SUBMIT for u in uses):
            return "verify"
        if state.get("llm_turns", 0) >= acfg.max_llm_turns:
            return "force_answer"
        if uses:
            return "tools"
        return "nudge" if state.get("nudges", 0) < 2 else "force_answer"

    def route_after_tools(state: AgentState) -> str:
        # Two turns in a row where every call was refused for budget: the model is not
        # going to stop by itself, so answer from the evidence gathered so far.
        return "force_answer" if state.get("over_budget_turns", 0) >= 2 else "agent"

    # ------------------------------------------------------------------ tools
    def tools_node(state: AgentState) -> dict[str, Any]:
        uses = tool_uses(state["messages"][-1])
        calls = state.get("tool_calls_used", 0)
        frames = state.get("frames_used", 0)

        # 1) Budget check and argument clamping, sequentially and in code.
        plans: list[tuple[dict, dict | None, str | None, int]] = []
        rejected_for_budget = 0
        for use in uses:
            if calls >= acfg.max_tool_calls:
                plans.append((use, None, "Tool-call budget exhausted. Call submit_answer now with your best answer.", 0))
                rejected_for_budget += 1
                continue
            args = dict(use.get("input") or {})
            reserved = 0
            if use["name"] == "inspect_clip":
                left = acfg.max_frames - frames
                if left <= 0:
                    plans.append((use, None, "Frame budget exhausted. Use other tools or submit_answer.", 0))
                    rejected_for_budget += 1
                    continue
                requested = args.get("num_frames") or acfg.default_frames_per_inspect
                reserved = max(1, min(int(requested), acfg.max_frames_per_inspect, left))
                args["num_frames"] = reserved
            calls += 1
            frames += reserved
            plans.append((use, args, None, reserved))

        # 2) Execute. Every tool is read-only, so parallel calls are safe to run concurrently.
        def execute(plan):
            use, args, error, _ = plan
            if error is not None:
                return ToolOutput(error, True), 0.0
            started = time.perf_counter()
            return deps.tools.call(use["name"], args), time.perf_counter() - started

        with ThreadPoolExecutor(max_workers=max(1, min(4, len(plans)))) as pool:
            outputs = list(pool.map(execute, plans))

        results, evidence, trace = [], [], []
        step = len(state.get("evidence", []))
        for (use, args, error, reserved), (out, latency) in zip(plans, outputs):
            if out.is_error and reserved:
                frames -= reserved  # don't charge frames for a failed inspection
            block = {"type": "tool_result", "tool_use_id": use["id"], "content": out.text}
            if out.is_error:
                block["is_error"] = True
            results.append(block)
            step += 1
            evidence.append(
                {"step": step, "tool": use["name"], "args": args or use.get("input"), "result": out.text, "is_error": out.is_error}
            )
            trace.append(
                {
                    "node": "tools",
                    "name": use["name"],
                    "args": args or use.get("input"),
                    "is_error": out.is_error,
                    "latency_s": round(latency, 3),
                    "result": _truncate(out.text, 2000),
                }
            )

        # All results go back in ONE user message (tool_result blocks first, then text).
        # Splitting them across messages teaches the model to stop making parallel calls.
        content = results + [{"type": "text", "text": budget_line(calls, frames)}]
        over = state.get("over_budget_turns", 0) + 1 if uses and rejected_for_budget == len(uses) else 0
        return {
            "messages": [{"role": "user", "content": content}],
            "tool_calls_used": calls,
            "frames_used": frames,
            "over_budget_turns": over,
            "evidence": evidence,
            "trace": trace,
        }

    def nudge_node(state: AgentState) -> dict[str, Any]:
        return {
            "messages": [{"role": "user", "content": prompts.NUDGE}],
            "nudges": state.get("nudges", 0) + 1,
            "trace": [{"node": "nudge"}],
        }

    # ----------------------------------------------------------------- verify
    def _reply_to_submit(uses: list[dict], submit_id: str, text: str, is_error: bool = False) -> dict[str, Any]:
        """Answer every tool_use of the last assistant turn (the API requires it)."""
        blocks = []
        for use in uses:
            if use["id"] == submit_id:
                block = {"type": "tool_result", "tool_use_id": use["id"], "content": text}
                if is_error:
                    block["is_error"] = True
            else:
                block = {
                    "type": "tool_result",
                    "tool_use_id": use["id"],
                    "content": "Not executed: submit_answer was called in the same turn.",
                    "is_error": True,
                }
            blocks.append(block)
        return {"role": "user", "content": blocks}

    def verify_node(state: AgentState) -> dict[str, Any]:
        uses = tool_uses(state["messages"][-1])
        submit = next(u for u in uses if u["name"] == SUBMIT)
        try:
            proposal = SubmitAnswer.model_validate(submit.get("input") or {})
        except ValidationError as err:
            text = f"Invalid submit_answer arguments: {format_validation_error(err)}"
            return {"messages": [_reply_to_submit(uses, submit["id"], text, True)], "trace": [{"node": "verify", "error": text}]}

        options = state.get("options", [])
        answer = normalize_answer(proposal.answer, options)
        base = {
            "answer": answer,
            "raw_answer": proposal.answer,
            "self_confidence": _clip01(proposal.confidence),
            "rationale": proposal.rationale,
            "evidence": [e.model_dump() for e in proposal.evidence],
            "forced": False,
        }
        if not acfg.verify:
            final = {**base, "confidence": base["self_confidence"], "accepted": True, "verdict": None}
            return {"final": final, "trace": [{"node": "verify", "skipped": True, "answer": answer}]}

        cited = "\n".join(f"- {e.t_start:.1f}-{e.t_end:.1f}s: {e.observation}" for e in proposal.evidence) or "- (none)"
        request = (
            f"{_question_block(state)}\n\nProposed answer: {proposal.answer}\n"
            f"Agent's rationale: {proposal.rationale}\nEvidence cited by the agent:\n{cited}\n\n"
            f"Complete tool observation log:\n{render_ledger(state.get('evidence', []))}"
        )
        rounds = state.get("verify_rounds", 0) + 1
        started = time.perf_counter()
        try:
            verdict = deps.llm.create_json(
                "verifier", [{"role": "user", "content": request}], Verdict, system=prompts.VERIFIER_SYSTEM
            )
        except (LLMError, ValidationError, ValueError) as err:
            final = {**base, "confidence": base["self_confidence"], "accepted": False, "verdict": None}
            return {"final": final, "verify_rounds": rounds, "trace": [{"node": "verify", "error": str(err)}]}

        confidence = _clip01(verdict.confidence)
        accepted = verdict.supported and confidence >= acfg.confidence_threshold
        out_of_budget = state.get("tool_calls_used", 0) >= acfg.max_tool_calls
        out_of_turns = state.get("llm_turns", 0) >= acfg.max_llm_turns
        trace = {
            "node": "verify",
            "round": rounds,
            "answer": answer,
            "verdict": verdict.model_dump(),
            "accepted": accepted,
            "latency_s": round(time.perf_counter() - started, 3),
        }
        if accepted or rounds >= acfg.max_verify_rounds or out_of_budget or out_of_turns:
            final = {**base, "confidence": confidence, "accepted": accepted, "verdict": verdict.model_dump()}
            return {"final": final, "verify_rounds": rounds, "trace": [trace]}

        feedback = (
            f"Answer not accepted by the independent verifier (confidence {confidence:.2f}).\n"
            f"Issues: {verdict.issues}\nSuggested next step: {verdict.next_step}\n"
            "Gather the missing evidence, then call submit_answer again.\n"
            + budget_line(state.get("tool_calls_used", 0), state.get("frames_used", 0))
        )
        return {
            "messages": [_reply_to_submit(uses, submit["id"], feedback)],
            "verify_rounds": rounds,
            "trace": [trace],
        }

    def route_after_verify(state: AgentState) -> str:
        return END if state.get("final") else "agent"

    # ----------------------------------------------------------- force answer
    def force_answer_node(state: AgentState) -> dict[str, Any]:
        """Out of turns or budget: answer from the evidence ledger in a fresh context."""
        request = f"{_question_block(state)}\n\nTool observation log:\n{render_ledger(state.get('evidence', []))}"
        started = time.perf_counter()
        forced = deps.llm.create_json(
            "verifier", [{"role": "user", "content": request}], ForcedAnswer, system=prompts.FORCE_ANSWER_SYSTEM
        )
        answer = normalize_answer(forced.answer, state.get("options", []))
        final = {
            "answer": answer,
            "raw_answer": forced.answer,
            "confidence": _clip01(forced.confidence),
            "self_confidence": _clip01(forced.confidence),
            "rationale": forced.rationale,
            "evidence": [],
            "forced": True,
            "accepted": False,
            "verdict": None,
        }
        return {"final": final, "trace": [{"node": "force_answer", "answer": answer, "latency_s": round(time.perf_counter() - started, 3)}]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("nudge", nudge_node)
    graph.add_node("verify", verify_node)
    graph.add_node("force_answer", force_answer_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "verify": "verify", "nudge": "nudge", "force_answer": "force_answer"}
    )
    graph.add_conditional_edges("tools", route_after_tools, {"agent": "agent", "force_answer": "force_answer"})
    graph.add_edge("nudge", "agent")
    graph.add_conditional_edges("verify", route_after_verify, {"agent": "agent", END: END})
    graph.add_edge("force_answer", END)
    return graph.compile(checkpointer=checkpointer)


def initial_state(question: str, options: list[str] | None = None) -> AgentState:
    return {
        "question": question,
        "options": list(options or []),
        "messages": [],
        "evidence": [],
        "trace": [],
        "tool_calls_used": 0,
        "frames_used": 0,
        "llm_turns": 0,
        "nudges": 0,
        "over_budget_turns": 0,
        "verify_rounds": 0,
        "final": None,
    }


def run_agent(
    deps: AgentDeps,
    question: str,
    options: list[str] | None = None,
    on_update: Callable[[dict[str, Any]], None] | None = None,
) -> AgentState:
    """Run one question to completion and return the final graph state."""
    app = build_graph(deps)
    limit = 4 * deps.cfg.agent.max_llm_turns + 10
    state: AgentState = initial_state(question, options)
    for mode, chunk in app.stream(state, {"recursion_limit": limit}, stream_mode=["updates", "values"]):
        if mode == "updates" and on_update is not None:
            on_update(chunk)
        elif mode == "values":
            state = chunk
    return state
