"""Background jobs with an append-only event log.

Indexing a long video takes minutes and an agent run takes about a minute, so
HTTP requests only *start* work and return a job id. The work runs in a thread
pool and appends events; clients follow them over Server-Sent Events (resuming
from Last-Event-ID after a reconnect) or by polling.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

MAX_FINISHED_JOBS = 200


class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.created = time.time()
        self.events: list[dict[str, Any]] = []
        self.done = False
        self._lock = threading.Lock()

    def emit(self, type_: str, **data: Any) -> None:
        with self._lock:
            self.events.append({"id": len(self.events), "type": type_, "time": round(time.time(), 3), **data})

    def events_after(self, last_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return self.events[last_id + 1 :]


class JobManager:
    def __init__(self, index_workers: int = 1, ask_workers: int = 2):
        # Indexing is GPU-bound, so one at a time; agent runs mostly wait on the API.
        self._pools = {
            "index": ThreadPoolExecutor(index_workers, thread_name_prefix="index"),
            "ask": ThreadPoolExecutor(ask_workers, thread_name_prefix="ask"),
        }
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, kind: str, work: Callable[[Job], None]) -> Job:
        job = Job(kind)
        with self._lock:
            self._jobs[job.id] = job
            finished = [j for j in self._jobs.values() if j.done]
            for old in sorted(finished, key=lambda j: j.created)[: max(0, len(finished) - MAX_FINISHED_JOBS)]:
                del self._jobs[old.id]

        def run() -> None:
            try:
                work(job)
            except Exception as err:
                job.emit("error", message=f"{type(err).__name__}: {err}", detail=traceback.format_exc(limit=3))
            finally:
                job.done = True

        self._pools[kind].submit(run)
        return job
