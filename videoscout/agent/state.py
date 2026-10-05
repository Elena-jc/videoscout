"""Graph state and the structured-output models used by the agent."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class AgentState(TypedDict, total=False):
    question: str
    options: list[str]
    # Append-only lists (the reducer concatenates what each node returns). The
    # planner transcript must never be edited in place: prompt caching is a prefix
    # match, and thinking blocks are only valid in the exact history that produced them.
    messages: Annotated[list[dict[str, Any]], operator.add]
    evidence: Annotated[list[dict[str, Any]], operator.add]  # tool observation ledger
    trace: Annotated[list[dict[str, Any]], operator.add]
    tool_calls_used: int
    frames_used: int
    llm_turns: int
    nudges: int
    over_budget_turns: int
    verify_rounds: int
    final: Optional[dict[str, Any]]


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t_start: float = Field(description="Window start in seconds.")
    t_end: float = Field(description="Window end in seconds.")
    observation: str = Field(description="What a tool result showed in this window.")


class SubmitAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="For multiple-choice questions, only the option letter (e.g. 'B'). Otherwise a short phrase.")
    confidence: float = Field(description="Your probability (0-1) that the answer is correct, given the evidence.")
    rationale: str = Field(description="1-3 sentences on how the evidence supports this answer and rules out the alternatives.")
    evidence: list[EvidenceItem] = Field(description="The key observations, each with its time window.")


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supported: bool = Field(description="True only if the observation log supports the proposed answer.")
    confidence: float = Field(description="Probability (0-1) that the proposed answer is correct.")
    issues: str = Field(description="Missing or contradictory evidence; empty string if none.")
    next_step: str = Field(description="The cheapest tool call that would resolve the issues; empty string if supported.")


class ForcedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="For multiple-choice questions, only the option letter. Otherwise a short phrase.")
    confidence: float = Field(description="Probability (0-1) that the answer is correct.")
    rationale: str = Field(description="One or two sentences.")
