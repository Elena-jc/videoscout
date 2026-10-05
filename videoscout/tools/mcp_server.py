"""Serve the video tools over MCP (stdio transport).

Any MCP client can then investigate an indexed video: this project's agent
(agent.tools=mcp), Claude Code (`claude mcp add videoscout -- python -m
videoscout.tools.mcp_server --index indexes/demo`), or Claude Desktop.

We use the low-level `Server` (handlers passed to the constructor, MCP SDK 2.x)
rather than the decorator-based MCPServer, so the MCP schemas come from the same
pydantic models the in-process registry uses.

    python -m videoscout.tools.mcp_server --index indexes/demo
"""

from __future__ import annotations

import argparse
from typing import Any

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from ..config import load_config
from ..index.store import VideoIndex
from ..llm import make_llm
from ..schema import to_api_schema
from ..vision import LLMVision
from . import build_tools, default_detector, default_embedder_factory


def _text_result(text: str, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=is_error)


def create_server(index_dir: str, overrides: list[str] | None = None, config_path: str | None = None) -> Server:
    cfg = load_config(config_path, overrides)
    index = VideoIndex(index_dir)
    vision = LLMVision(make_llm(cfg.models, cfg.pricing))
    tools = {
        t.name: t
        for t in build_tools(index, cfg, vision, default_embedder_factory(index, cfg), default_detector(cfg))
    }
    listing = types.ListToolsResult(
        tools=[
            types.Tool(name=t.name, description=t.description, input_schema=to_api_schema(t.args_model))
            for t in tools.values()
        ]
    )

    async def on_list_tools(ctx: Any, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
        return listing

    async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        tool = tools.get(params.name)
        if tool is None:
            return _text_result(f"Unknown tool {params.name!r}", is_error=True)
        # Tools block (SQLite, OpenCV, HTTP to the vision model), so run them in a
        # worker thread and keep the event loop free for other requests.
        out = await anyio.to_thread.run_sync(tool.run, dict(params.arguments or {}))
        return _text_result(out.text, out.is_error)

    return Server("videoscout", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def serve_stdio(server: Server) -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, help="index directory built by `videoscout index`")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", action="append", default=[], dest="overrides", metavar="SECTION.FIELD=VALUE")
    args = parser.parse_args()
    anyio.run(serve_stdio, create_server(args.index, args.overrides, args.config))


if __name__ == "__main__":
    main()
