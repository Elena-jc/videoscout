"""Claude client wrapper: per-role model and effort, refusal fallback, prompt
caching, and token/cost accounting.

Every LLM call in the project goes through `ClaudeLLM.create`, so there is one
place that decides the model, the effort level, and how a response is checked.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from .config import ModelConfig
from .schema import to_api_schema

FALLBACK_BETA = "server-side-fallback-2026-07-01"

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class RefusalError(LLMError):
    pass


def block_to_dict(block: Any) -> dict[str, Any]:
    """Convert an SDK content block to the dict we store in conversation history.

    `to_dict()` keeps exactly the fields the API returned (including thinking
    signatures), so the block can be sent back unchanged on the next turn.
    """
    if isinstance(block, dict):
        return block
    return block.to_dict()


def blocks_to_dicts(content: list[Any]) -> list[dict[str, Any]]:
    return [block_to_dict(b) for b in content]


def text_of(response: Any) -> str:
    parts = []
    for block in response.content:
        block = block_to_dict(block)
        if block.get("type") == "text":
            parts.append(block["text"])
    return "\n".join(parts).strip()


@dataclass
class UsageMeter:
    """Accumulates token usage and cost across calls (thread-safe)."""

    pricing: dict[str, dict[str, float]]
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    by_role: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, role: str, model: str, usage: Any, price_scale: float = 1.0) -> float:
        """Record one response; returns its cost. `price_scale=0.5` for Batches API results."""
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cr = getattr(usage, "cache_read_input_tokens", 0) or 0
        # `model` is the model that served the response; after a refusal fallback
        # it differs from the requested one and is billed at its own rates.
        price = self.pricing.get(model, {})
        cost = price_scale * (
            inp * price.get("input", 0.0)
            + out * price.get("output", 0.0)
            + cw * price.get("cache_write", 0.0)
            + cr * price.get("cache_read", 0.0)
        ) / 1e6
        with self._lock:
            self.calls += 1
            self.input_tokens += inp
            self.output_tokens += out
            self.cache_write_tokens += cw
            self.cache_read_tokens += cr
            self.cost_usd += cost
            self.by_role[role] = self.by_role.get(role, 0) + 1
        return cost

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_write_tokens": self.cache_write_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "cost_usd": round(self.cost_usd, 6),
                "calls_by_role": dict(self.by_role),
            }


class LLMClient(Protocol):
    """What the agent needs from a model backend (ClaudeLLM, OpenAICompatLLM)."""

    meter: UsageMeter

    def create(self, role: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...

    def create_json(self, role: str, messages: list[dict[str, Any]], output_model: type[T], **kwargs: Any) -> T: ...

    def fresh(self) -> "LLMClient": ...

    def model_for(self, role: str) -> str: ...


def make_llm(models: ModelConfig, pricing: dict[str, dict[str, float]]) -> LLMClient:
    if models.provider == "anthropic":
        return ClaudeLLM(models, pricing)
    if models.provider == "openai_compat":
        from .llm_openai import OpenAICompatLLM

        return OpenAICompatLLM(models, pricing)
    raise ValueError(f"unknown models.provider {models.provider!r}")


class ClaudeLLM:
    """Role-aware Claude client. Roles: planner, verifier, vision, captioner."""

    supports_batches = True

    def __init__(self, models: ModelConfig, pricing: dict[str, dict[str, float]], client: Any = None):
        self._client = client
        self._client_lock = threading.Lock()
        self.models = models
        self.pricing = pricing
        self.meter = UsageMeter(pricing)
        from .spend import SpendGuard

        self.guard = SpendGuard("anthropic", models.daily_request_limit, models.daily_cost_limit_usd)

    @property
    def client(self) -> Any:
        # Created on first use, so code paths that never call the API (BM25-only
        # search, SQL queries, tests) don't need credentials.
        with self._client_lock:
            if self._client is None:
                import anthropic

                # The SDK already retries 408/409/429/5xx and connection errors with backoff.
                self._client = anthropic.Anthropic(max_retries=4)
            return self._client

    def fresh(self) -> "ClaudeLLM":
        """Same client and settings, new usage meter (one per evaluated question)."""
        return ClaudeLLM(self.models, self.pricing, client=self.client)

    def model_for(self, role: str) -> str:
        return getattr(self.models, role)

    def create(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        cache: bool = False,
        max_tokens: int | None = None,
    ) -> Any:
        model = self.model_for(role)
        output_config: dict[str, Any] = {"effort": getattr(self.models, f"{role}_effort")}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or self.models.max_tokens,
            "messages": messages,
            "output_config": output_config,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if cache:
            # Automatic prompt caching: the growing agent transcript is re-read at
            # ~0.1x input price on every turn. Only worth it for multi-turn loops;
            # one-shot calls would pay the 1.25x cache-write premium for nothing.
            kwargs["cache_control"] = {"type": "ephemeral"}

        self.guard.check()  # local daily caps, before any money is spent
        if self.models.use_fallbacks:
            response = self.client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **kwargs
            )
        else:
            response = self.client.messages.create(**kwargs)

        cost = self.meter.add(role, getattr(response, "model", model) or model, response.usage)
        self.guard.record(cost)

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise RefusalError(f"{role} request was declined (category={category})")
        if response.stop_reason == "max_tokens":
            # A tool_use block cut off here may still parse as valid JSON, so never
            # run tools from a truncated turn.
            raise LLMError(f"{role} output hit max_tokens; raise models.max_tokens")
        return response

    def create_json(
        self,
        role: str,
        messages: list[dict[str, Any]],
        output_model: type[T],
        *,
        system: str | None = None,
    ) -> T:
        """Structured output: the API constrains the reply to the model's schema,
        and pydantic validates it again on our side."""
        response = self.create(role, messages, system=system, json_schema=to_api_schema(output_model))
        return output_model.model_validate_json(text_of(response))
