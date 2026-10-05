"""Local daily spending caps, enforced before every model request.

Provider-side limits (free-tier quotas, billing alerts) can change or be turned off;
this guard is ours. Each backend keeps a per-day ledger in .cache/usage/ with the
number of requests and the estimated cost, and refuses new requests once a cap from
the config is reached:

    models:
      daily_request_limit: 60      # 0 = no cap
      daily_cost_limit_usd: 5.0    # 0 = no cap
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import date
from pathlib import Path

from .config import PROJECT_ROOT
from .llm import LLMError

_LOCK = threading.Lock()


class SpendLimitError(LLMError):
    pass


def usage_dir() -> Path:
    return Path(os.environ.get("VIDEOSCOUT_USAGE_DIR") or PROJECT_ROOT / ".cache" / "usage")


class SpendGuard:
    def __init__(self, name: str, max_requests: int = 0, max_usd: float = 0.0):
        self.name = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "default"
        self.max_requests = max_requests
        self.max_usd = max_usd

    def _path(self) -> Path:
        return usage_dir() / f"{date.today().isoformat()}-{self.name}.json"

    def today(self) -> dict:
        path = self._path()
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        return {"requests": 0, "cost_usd": 0.0}

    def check(self) -> None:
        with _LOCK:
            used = self.today()
        if self.max_requests and used["requests"] >= self.max_requests:
            raise SpendLimitError(
                f"daily safety limit reached: {used['requests']} requests to {self.name} today "
                f"(models.daily_request_limit = {self.max_requests}). It resets at midnight; raise it in the config only if you are sure."
            )
        if self.max_usd and used["cost_usd"] >= self.max_usd:
            raise SpendLimitError(
                f"daily safety limit reached: ${used['cost_usd']:.2f} spent on {self.name} today "
                f"(models.daily_cost_limit_usd = {self.max_usd}). It resets at midnight."
            )

    def record(self, cost_usd: float) -> None:
        with _LOCK:
            used = self.today()
            used["requests"] += 1
            used["cost_usd"] = round(used["cost_usd"] + cost_usd, 6)
            path = self._path()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(used), encoding="utf-8")
            tmp.replace(path)  # atomic: a crash never leaves a half-written ledger
