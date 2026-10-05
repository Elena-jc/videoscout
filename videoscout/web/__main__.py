"""python -m videoscout.web [--port 8000] [--no-browser]"""

from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoScout local web app")
    # Loopback only: the app reads local files and spends your API quota, so it is
    # deliberately not reachable from other machines.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(f"VideoScout is running at {url}  (close this window to stop it)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(warmup=True), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
