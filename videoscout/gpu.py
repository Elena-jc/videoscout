"""Share one small GPU between several local models.

An 8 GB laptop GPU holds one 2B-parameter model at a time, not three (embedder,
reranker, captioner) plus SAM 3. Each model runs inside `POOL.use(...)`: the pool
moves it onto the GPU and parks the other large models in CPU RAM until they are
needed again. Moving a 4 GB model takes about a second, far less than reloading it
from disk, and the lock also stops two threads from running large models at once.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def _device_of(module: Any) -> str:
    try:
        return str(next(module.parameters()).device)
    except (StopIteration, AttributeError):
        return "cpu"


class GpuPool:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._models: dict[str, Any] = {}  # name -> torch module, registered on first use

    @staticmethod
    def cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    @contextmanager
    def use(self, name: str, module: Any, device: str = "cuda") -> Iterator[None]:
        """Run `module` on `device` for the duration of the block. CPU-only setups
        (device 'cpu' or no CUDA) run in place."""
        with self._lock:
            self._models[name] = module
            if device.startswith("cuda") and self.cuda_available():
                import torch

                for other, mod in self._models.items():
                    if other != name and _device_of(mod).startswith("cuda"):
                        mod.to("cpu")
                torch.cuda.empty_cache()
                if not _device_of(module).startswith("cuda"):
                    module.to(device)
            yield

    def release(self, name: str) -> None:
        """Forget a model (e.g. the captioner after indexing) and free its GPU memory."""
        with self._lock:
            module = self._models.pop(name, None)
            if module is not None and _device_of(module).startswith("cuda"):
                module.to("cpu")
                import torch

                torch.cuda.empty_cache()


POOL = GpuPool()
