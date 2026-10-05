"""The same agent without LangGraph: the bare tool-use loop.

Read this first. Everything in graph.py is this loop plus budget enforcement,
an independent verifier, a forced-answer fallback and tracing.
"""

from __future__ import annotations

from ..llm import LLMClient, blocks_to_dicts
from ..tools.registry import ToolRegistry
from .prompts import PLANNER_SYSTEM, render_task

SUBMIT_TOOL = {
    "name": "submit_answer",
    "description": "Submit the final answer.",
    "input_schema": {
        "type": "object",
        "properties": {"answer": {"type": "string", "description": "Option letter or short phrase."}},
        "required": ["answer"],
        "additionalProperties": False,
    },
}


def react_loop(
    llm: LLMClient,
    tools: ToolRegistry,
    question: str,
    options: list[str],
    overview: str,
    max_turns: int = 10,
) -> str:
    schemas = tools.schemas() + [SUBMIT_TOOL]
    messages = [{"role": "user", "content": render_task(question, options, overview, max_turns, 48)}]

    for _ in range(max_turns):
        # 1. Reason + act: the model reads the whole transcript and emits tool calls.
        response = llm.create("planner", messages, system=PLANNER_SYSTEM, tools=schemas, cache=True)
        # 2. Append its turn unchanged (thinking blocks included).
        messages.append({"role": "assistant", "content": blocks_to_dicts(response.content)})
        uses = [b for b in messages[-1]["content"] if b.get("type") == "tool_use"]
        if not uses:
            messages.append({"role": "user", "content": "Use the tools, or call submit_answer."})
            continue
        # 3. A terminal tool ends the loop.
        for use in uses:
            if use["name"] == "submit_answer":
                return use["input"]["answer"]
        # 4. Observe: run every call, return all results in one user message.
        results = []
        for use in uses:
            out = tools.call(use["name"], use["input"])
            block = {"type": "tool_result", "tool_use_id": use["id"], "content": out.text}
            if out.is_error:
                block["is_error"] = True
            results.append(block)
        messages.append({"role": "user", "content": results})

    return "(no answer within the turn limit)"
