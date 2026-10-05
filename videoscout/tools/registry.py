"""Tool abstraction shared by the in-process registry and the MCP server.

A tool is (name, description, pydantic args model, function). The args model is
the single source of truth: it produces the JSON schema the model sees and
validates the arguments the model sends back before anything runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from ..schema import format_validation_error, to_api_schema


class ToolError(Exception):
    """An expected failure the model should see and recover from (bad window,
    unknown track, SQL error...). Returned as a tool_result with is_error=true."""


@dataclass
class ToolOutput:
    text: str
    is_error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any], str]

    def schema(self, strict: bool = False) -> dict[str, Any]:
        spec = {"name": self.name, "description": self.description, "input_schema": to_api_schema(self.args_model)}
        if strict:
            spec["strict"] = True
        return spec

    def run(self, args: dict[str, Any]) -> ToolOutput:
        try:
            parsed = self.args_model.model_validate(args)
        except ValidationError as err:
            return ToolOutput(f"Invalid arguments for {self.name}: {format_validation_error(err)}", True)
        try:
            return ToolOutput(self.fn(parsed))
        except ToolError as err:
            return ToolOutput(str(err), True)
        except Exception as err:  # keep the agent loop alive; the trace records the failure
            return ToolOutput(f"{self.name} failed unexpectedly: {type(err).__name__}: {err}", True)


class ToolRegistry(Protocol):
    def schemas(self) -> list[dict[str, Any]]: ...

    def call(self, name: str, args: dict[str, Any]) -> ToolOutput: ...

    def close(self) -> None: ...


class InProcessTools:
    """Tools executed as plain function calls in this process."""

    def __init__(self, tools: list[Tool], strict: bool = True):
        self._tools = {t.name: t for t in tools}
        self._strict = strict

    def schemas(self) -> list[dict[str, Any]]:
        # Stable order matters: the tool list is the first part of the cached prompt prefix.
        return [t.schema(self._strict) for t in self._tools.values()]

    def call(self, name: str, args: dict[str, Any]) -> ToolOutput:
        tool = self._tools.get(name)
        if tool is None:
            return ToolOutput(f"Unknown tool {name!r}. Available: {', '.join(self._tools)}", True)
        return tool.run(args)

    def close(self) -> None:
        pass
