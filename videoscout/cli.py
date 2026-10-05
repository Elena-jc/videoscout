"""Command line entry point.

    python -m videoscout index data/demo/demo.mp4 --srt data/demo/demo.srt --out indexes/demo
    python -m videoscout ask --index indexes/demo "What does the sign say?" -o "A. GATE A OPEN" -o "B. GATE B CLOSED"
    python -m videoscout serve-mcp --index indexes/demo
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import load_config


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--set", action="append", default=[], dest="overrides", metavar="SECTION.FIELD=VALUE")


def _print_update(update: dict[str, Any]) -> None:
    for node, delta in update.items():
        for step in (delta or {}).get("trace", []):
            if step["node"] == "agent":
                if step.get("text"):
                    print(f"\n[agent] {step['text'].strip()[:400]}")
                for call in step.get("tool_calls", []):
                    print(f"[agent] -> {call['name']} {json.dumps(call['input'], ensure_ascii=False)[:300]}")
            elif step["node"] == "tools":
                flag = " (error)" if step["is_error"] else ""
                print(f"[tool]  <- {step['name']}{flag} in {step['latency_s']:.1f}s")
                for line in step["result"].splitlines()[:8]:
                    print(f"          {line[:160]}")
            elif step["node"] == "verify" and "verdict" in step:
                v = step["verdict"]
                print(f"[verify] answer={step['answer']} supported={v['supported']} conf={v['confidence']:.2f} "
                      f"accepted={step['accepted']}")
                if not step["accepted"] and v.get("issues"):
                    print(f"         issues: {v['issues'][:300]}")
            elif step["node"] in ("nudge", "force_answer"):
                print(f"[{step['node']}]")


def cmd_index(args: argparse.Namespace) -> None:
    from .index.build import build_index
    from .llm import make_llm

    cfg = load_config(args.config, args.overrides)
    if args.caption:
        cfg.index.caption = True
    llm = make_llm(cfg.models, cfg.pricing) if cfg.index.caption else None
    build_index(args.video, args.out, cfg, srt_path=args.srt, with_dense=not args.no_dense,
                with_tracks=not args.no_tracks, llm=llm)


def cmd_ask(args: argparse.Namespace) -> None:
    from .agent import AgentDeps, run_agent
    from .agent.react_minimal import react_loop
    from .index.store import VideoIndex
    from .llm import make_llm
    from .tools import open_toolset
    from .vision import LLMVision

    cfg = load_config(args.config, args.overrides)
    index = VideoIndex(args.index)
    llm = make_llm(cfg.models, cfg.pricing)
    tools = open_toolset(args.index, cfg, LLMVision(llm), index=index)
    try:
        if args.minimal:
            answer = react_loop(llm, tools, args.question, args.option, index.overview())
            print(f"\nanswer: {answer}")
        else:
            state = run_agent(AgentDeps(llm, tools, cfg, index.overview()), args.question, args.option,
                              on_update=None if args.quiet else _print_update)
            final = state["final"]
            print("\n" + "=" * 60)
            print(f"answer:      {final['answer']}")
            print(f"confidence:  {final['confidence']:.2f} (self-reported {final['self_confidence']:.2f})")
            print(f"accepted:    {final['accepted']}   forced: {final['forced']}")
            print(f"rationale:   {final['rationale']}")
            print(f"budget used: {state['tool_calls_used']} tool calls, {state['frames_used']} frames")
    finally:
        tools.close()
    print(f"usage:       {json.dumps(llm.meter.snapshot())}")


def cmd_serve_mcp(args: argparse.Namespace) -> None:
    import anyio

    from .tools.mcp_server import create_server, serve_stdio

    anyio.run(serve_stdio, create_server(args.index, args.overrides, args.config))


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="videoscout", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="build the index for one video")
    p.add_argument("video")
    p.add_argument("--out", required=True)
    p.add_argument("--srt", help="subtitle file")
    p.add_argument("--caption", action="store_true", help="add VLM captions (costs API calls)")
    p.add_argument("--no-dense", action="store_true", help="skip SigLIP embeddings")
    p.add_argument("--no-tracks", action="store_true", help="skip detection and tracking")
    _common(p)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("ask", help="ask a question about an indexed video")
    p.add_argument("--index", required=True)
    p.add_argument("question")
    p.add_argument("-o", "--option", action="append", default=[], help="multiple-choice option, e.g. 'A. a bus'")
    p.add_argument("--minimal", action="store_true", help="use the bare ReAct loop instead of the graph")
    p.add_argument("--quiet", action="store_true")
    _common(p)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("serve-mcp", help="serve the video tools over MCP (stdio)")
    p.add_argument("--index", required=True)
    _common(p)
    p.set_defaults(func=cmd_serve_mcp)

    args = parser.parse_args(argv)
    from .llm import LLMError

    try:
        args.func(args)
    except LLMError as err:
        # Configuration and API problems (missing key, HTTP errors, refusals): one clear line, no traceback.
        print(f"\nerror: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
