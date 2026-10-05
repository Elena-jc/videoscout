"""OpenAI-compatible chat-completions backend: Gemini (free tier), Ollama (local),
OpenRouter, Groq, DashScope, vLLM, ...

The agent keeps its transcript in the Anthropic message format. This adapter
translates every request to POST {base_url}/chat/completions and every response
back into Anthropic-style content blocks, so the graph, the tools and the tests
are the same for every provider.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from .config import ModelConfig
from .llm import LLMError, RefusalError, UsageMeter, text_of
from .schema import to_api_schema

T = TypeVar("T", bound=BaseModel)

# The provider's own assistant message is kept in the transcript next to the
# translated blocks and sent back verbatim on the next turn (some providers attach
# signatures to tool calls that must round-trip, e.g. Gemini thought signatures).
RAW_BLOCK = "openai_message"

FINISH_TO_STOP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
MAX_RETRIES = 6
MAX_RETRY_WAIT = 90.0  # seconds; a longer requested wait means a daily quota


class QuotaExhaustedError(LLMError):
    pass


def _retry_delay(response: httpx.Response) -> float | None:
    """Seconds the server asks us to wait: Retry-After header, or Google's RetryInfo."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    try:
        payload = response.json()
    except ValueError:
        return None
    errors = payload if isinstance(payload, list) else [payload]
    for item in errors:
        for detail in (item.get("error") or {}).get("details") or []:
            if str(detail.get("@type", "")).endswith("RetryInfo"):
                match = re.fullmatch(r"([\d.]+)s", str(detail.get("retryDelay", "")))
                if match:
                    return float(match.group(1))
    return None


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
        item = payload[0] if isinstance(payload, list) else payload
        return str(item["error"]["message"]).split("\n")[0][:300]
    except (ValueError, KeyError, IndexError, TypeError):
        return response.text[:300]

_throttle_lock = threading.Lock()
_last_request = 0.0


def _throttle(min_interval: float) -> None:
    """Space requests out to stay under free-tier requests-per-minute limits."""
    global _last_request
    if min_interval <= 0:
        return
    with _throttle_lock:
        wait = _last_request + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()


def simplify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Lowest-common-denominator JSON schema: inline $refs, turn optional
    `anyOf: [X, null]` into X, drop `additionalProperties`. Providers differ in
    how much JSON Schema they accept; this subset works across the common ones."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(deepcopy(defs[node["$ref"].split("/")[-1]]))
        any_of = node.get("anyOf")
        if any_of and any(a.get("type") == "null" for a in any_of):
            rest = [a for a in any_of if a.get("type") != "null"]
            if len(rest) == 1:
                merged = {**{k: v for k, v in node.items() if k != "anyOf"}, **rest[0]}
                return walk(merged)
        out = {}
        for key, value in node.items():
            if key in ("$defs", "additionalProperties", "strict"):
                continue
            out[key] = {n: walk(s) for n, s in value.items()} if key == "properties" else walk(value)
        return out

    return walk(schema)


def _result_text(block: dict[str, Any]) -> str:
    content = block.get("content", "")
    if isinstance(content, list):
        content = "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
    return f"ERROR: {content}" if block.get("is_error") else content


def _assistant_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    raw = next((b["message"] for b in content if b.get("type") == RAW_BLOCK), None)
    if raw is not None:
        return raw
    text = "\n".join(b["text"] for b in content if b.get("type") == "text")
    calls = [
        {"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
        for b in content
        if b.get("type") == "tool_use"
    ]
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = calls
    return message


def to_openai_messages(system: str | None, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic transcript -> chat-completions messages. A user turn that carries
    tool_result blocks becomes one `tool` message per result (they must directly
    follow the assistant message that made the calls), then a user message for
    any remaining text or images."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for message in messages:
        content = message["content"]
        if message["role"] == "assistant":
            out.append(_assistant_message(content))
            continue
        if isinstance(content, str):
            out.append({"role": "user", "content": content})
            continue
        parts: list[dict[str, Any]] = []
        for block in content:
            kind = block.get("type")
            if kind == "tool_result":
                out.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": _result_text(block)})
            elif kind == "text":
                parts.append({"type": "text", "text": block["text"]})
            elif kind == "image":
                src = block["source"]
                parts.append({"type": "image_url", "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"}})
        if len(parts) == 1 and parts[0]["type"] == "text":
            out.append({"role": "user", "content": parts[0]["text"]})
        elif parts:
            out.append({"role": "user", "content": parts})
    return out


def from_openai_response(data: dict[str, Any]) -> SimpleNamespace:
    """chat-completions response -> object shaped like an Anthropic Message."""
    choice = data["choices"][0]
    message = choice.get("message") or {}
    text = message.get("content")
    if isinstance(text, list):  # some providers return content parts
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))

    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    raw_calls = []
    for call in message.get("tool_calls") or []:
        call = dict(call)
        call.setdefault("id", f"call_{uuid.uuid4().hex[:12]}")  # some local servers omit ids
        call.setdefault("type", "function")
        args = call["function"].get("arguments") or "{}"
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            # Fails pydantic validation in the tool, so the model sees an error and retries.
            parsed = {"_invalid_json": args}
        blocks.append({"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": parsed})
        raw_calls.append(call)

    raw: dict[str, Any] = {"role": "assistant", "content": text or None}
    if raw_calls:
        raw["tool_calls"] = raw_calls
    if message.get("extra_content") is not None:
        raw["extra_content"] = message["extra_content"]
    blocks.append({"type": RAW_BLOCK, "message": raw})

    stop = FINISH_TO_STOP.get(choice.get("finish_reason") or "stop", "end_turn")
    if raw_calls and stop == "end_turn":  # some providers report "stop" even with tool calls
        stop = "tool_use"
    usage = data.get("usage") or {}
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop,
        stop_details=None,
        model=data.get("model"),
        usage=SimpleNamespace(
            input_tokens=usage.get("prompt_tokens", 0) or 0,
            output_tokens=usage.get("completion_tokens", 0) or 0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


def extract_json(text: str) -> str:
    """Tolerate ```json fences or prose around the object."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


class OpenAICompatLLM:
    """Same interface as ClaudeLLM (create / create_json / fresh / meter)."""

    supports_batches = False

    def __init__(self, models: ModelConfig, pricing: dict[str, dict[str, float]], http: httpx.Client | None = None):
        if not models.base_url:
            raise ValueError("models.base_url is required for provider openai_compat")
        self.models = models
        self.pricing = pricing
        self.meter = UsageMeter(pricing)
        self._http = http
        self._lock = threading.Lock()
        from urllib.parse import urlparse

        from .spend import SpendGuard

        host = urlparse(models.base_url).hostname or "openai-compat"
        self.guard = SpendGuard(host, models.daily_request_limit, models.daily_cost_limit_usd)

    @property
    def http(self) -> httpx.Client:
        with self._lock:
            if self._http is None:
                self._http = httpx.Client(timeout=httpx.Timeout(300.0, connect=15.0))
            return self._http

    def fresh(self) -> "OpenAICompatLLM":
        return OpenAICompatLLM(self.models, self.pricing, http=self.http)

    def model_for(self, role: str) -> str:
        return getattr(self.models, role)

    def _headers(self) -> dict[str, str]:
        env = self.models.api_key_env
        key = os.environ.get(env, "") if env else ""
        if env and not key:
            raise LLMError(f"environment variable {env} is not set (needed for {self.models.base_url})")
        return {"Authorization": f"Bearer {key or 'unused'}"}

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        url = self.models.base_url.rstrip("/") + "/chat/completions"
        headers = self._headers()
        delay = 2.0
        for attempt in range(MAX_RETRIES + 1):
            _throttle(self.models.min_request_interval)
            try:
                response = self.http.post(url, json=body, headers=headers)
            except httpx.TransportError as err:
                if attempt == MAX_RETRIES:
                    raise LLMError(f"cannot reach {url}: {err}") from err
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            if response.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
                wait = _retry_delay(response)
                if wait is not None and wait > MAX_RETRY_WAIT:
                    # A per-day quota, not a per-minute one: retrying now only burns time.
                    raise QuotaExhaustedError(
                        f"{self.models.base_url} quota exhausted for {body['model']}; it resets in about "
                        f"{wait / 3600:.1f} h. Use another model or provider meanwhile. "
                        f"Details: {_error_message(response)}"
                    )
                # Per-minute limits: wait as instructed (Retry-After / RetryInfo), else back off.
                time.sleep(min(max(wait if wait is not None else delay, 0.0), MAX_RETRY_WAIT))
                delay = min(delay * 2, 60.0)
                continue
            if response.status_code >= 400:
                raise LLMError(f"HTTP {response.status_code} from {url}: {response.text[:800]}")
            return response.json()
        raise LLMError(f"giving up on {url} after {MAX_RETRIES} retries")

    def create(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        cache: bool = False,  # providers that cache do it automatically
        max_tokens: int | None = None,
    ) -> SimpleNamespace:
        model = self.model_for(role)
        body: dict[str, Any] = {
            "model": model,
            "messages": to_openai_messages(system, messages),
            "max_tokens": max_tokens or self.models.max_tokens,
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": simplify_schema(t["input_schema"]),
                    },
                }
                for t in tools
            ]
        if self.models.send_reasoning_effort:
            # Thinking models default to long reasoning; reading frames or judging one
            # answer does not need it. OpenAI-style levels: low | medium | high.
            effort = getattr(self.models, f"{role}_effort", "medium")
            body["reasoning_effort"] = {"xhigh": "high", "max": "high"}.get(effort, effort)
        if json_schema is not None:
            if self.models.json_mode == "schema":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "output", "schema": simplify_schema(json_schema)},
                }
            else:
                body["response_format"] = {"type": "json_object"}

        self.guard.check()  # local daily caps, before the request is sent
        response = from_openai_response(self._post(body))
        self.guard.record(self.meter.add(role, response.model or model, response.usage))
        if response.stop_reason == "refusal":
            raise RefusalError(f"{role} request was declined by the provider's content filter")
        if response.stop_reason == "max_tokens":
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
        schema = to_api_schema(output_model)
        # Also state the schema in words: not every provider enforces response_format.
        instruction = "Reply with only a JSON object that matches this JSON schema:\n" + json.dumps(simplify_schema(schema))
        system = f"{system}\n\n{instruction}" if system else instruction
        response = self.create(role, messages, system=system, json_schema=schema)
        return output_model.model_validate_json(extract_json(text_of(response)))
