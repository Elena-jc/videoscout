"""Use tools served by an MCP server as a (synchronous) ToolRegistry.

The MCP SDK is async and its stdio transport has to be opened and closed inside
the same task, so one long-lived coroutine owns the connection on a background
event loop, and synchronous callers (the LangGraph tools node, possibly from
several threads) submit tool calls to that loop.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from mcp import Client, StdioServerParameters

from .registry import ToolOutput

CONNECT_TIMEOUT = 120.0  # the server loads the index (and maybe SigLIP) before it answers
CALL_TIMEOUT = 300.0


class MCPTools:
    def __init__(self, command: list[str]):
        self._params = StdioServerParameters(command=command[0], args=command[1:])
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._ready = threading.Event()
        self._client: Client | None = None
        self._schemas: list[dict[str, Any]] = []
        self._stop: asyncio.Event | None = None
        self._main = asyncio.run_coroutine_threadsafe(self._run(), self._loop)
        if not self._ready.wait(CONNECT_TIMEOUT):
            self.close()
            raise TimeoutError("MCP server did not become ready")
        if self._main.done():  # the server failed during startup
            self._main.result()

    async def _run(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with Client(self._params, read_timeout_seconds=CALL_TIMEOUT) as client:
                listed = await client.list_tools()
                # MCP tool definitions map 1:1 onto Claude tool definitions.
                self._schemas = [
                    {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                    for t in listed.tools
                ]
                self._client = client
                self._ready.set()
                await self._stop.wait()
        finally:
            self._client = None
            self._ready.set()

    def schemas(self) -> list[dict[str, Any]]:
        return list(self._schemas)

    def call(self, name: str, args: dict[str, Any]) -> ToolOutput:
        if self._client is None:
            return ToolOutput("MCP server is not connected", True)
        future = asyncio.run_coroutine_threadsafe(self._client.call_tool(name, args), self._loop)
        result = future.result(CALL_TIMEOUT)
        text = "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")
        return ToolOutput(text or "(empty result)", bool(result.is_error))

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        try:
            self._main.result(10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)
