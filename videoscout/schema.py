"""Helpers for turning pydantic models into JSON schemas the Claude API accepts.

Pydantic is the single source of truth for tool arguments and structured outputs:
the same model generates the schema we send to the API *and* validates what comes
back. Strict tool use / structured outputs accept a subset of JSON Schema: every
object needs `additionalProperties: false`, and numeric/string constraints
(minimum, maxLength, ...) are not supported, so we strip them here and enforce
ranges in our own code instead.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

_DROP_KEYS = {
    "title",
    "default",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "pattern",
}


def _clean(node: Any) -> Any:
    if isinstance(node, list):
        return [_clean(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROP_KEYS:
            continue
        if key in ("properties", "$defs"):
            # Keys here are field / definition names, not schema keywords.
            out[key] = {name: _clean(sub) for name, sub in value.items()}
        else:
            out[key] = _clean(value)
    if out.get("type") == "object":
        out.setdefault("additionalProperties", False)
    return out


def to_api_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for `model`, restricted to what strict mode supports."""
    return _clean(model.model_json_schema())


def format_validation_error(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "(root)"
        parts.append(f"{loc}: {e['msg']}")
    return "; ".join(parts)
