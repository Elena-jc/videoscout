"""Vision backends used by inspect_clip.

The planner never sees pixels. inspect_clip sends frames to a vision model with a
focused question and returns only a short text observation, so dozens of images
never enter the planner's context. This is the sub-agent pattern: an expensive,
context-heavy subtask runs in its own call and hands back a summary.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .llm import LLMClient, text_of
from .video import fmt_ts, to_jpeg_b64


class VisionBackend(Protocol):
    def ask(self, frames: list[tuple[float, np.ndarray]], prompt: str) -> str: ...


class LLMVision:
    def __init__(self, llm: LLMClient, max_side: int = 768):
        self.llm = llm
        self.max_side = max_side

    def ask(self, frames: list[tuple[float, np.ndarray]], prompt: str) -> str:
        content = []
        for t, img in frames:
            content.append({"type": "text", "text": f"Frame at {fmt_ts(t)} ({t:.1f}s):"})
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": to_jpeg_b64(img, self.max_side)},
                }
            )
        content.append({"type": "text", "text": prompt})
        return text_of(self.llm.create("vision", [{"role": "user", "content": content}]))
