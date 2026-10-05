"""Non-agentic baseline: sample N frames uniformly over the whole video and ask
the vision model once. This is how most VLM long-video numbers are produced, and
it is the bar the agent has to beat on accuracy per frame and per dollar."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..agent.graph import normalize_answer
from ..index.subtitles import parse_srt
from ..llm import LLMClient
from ..video import VideoReader, fmt_ts, to_jpeg_b64

MAX_SUBTITLE_CHARS = 8000


class BaselineAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="For multiple-choice questions, only the option letter.")
    confidence: float = Field(description="Probability (0-1) that the answer is correct.")


def run_uniform(
    llm: LLMClient,
    video_path: str,
    question: str,
    options: list[str],
    n_frames: int = 32,
    max_side: int = 512,
    subtitle_path: str | None = None,
) -> dict:
    reader = VideoReader(video_path)
    frames = reader.sample(0.0, reader.duration, n_frames)
    content = []
    for t, img in frames:
        content.append({"type": "text", "text": f"Frame at {fmt_ts(t)}:"})
        content.append(
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": to_jpeg_b64(img, max_side)}}
        )
    prompt = f"These {len(frames)} frames are sampled uniformly from a {fmt_ts(reader.duration)} video.\n"
    if subtitle_path:
        subs = " ".join(body for _, _, body in parse_srt(subtitle_path))[:MAX_SUBTITLE_CHARS]
        prompt += f"Subtitles (possibly truncated): {subs}\n"
    prompt += f"\nQuestion: {question}\n"
    if options:
        prompt += "Options:\n" + "\n".join(options) + "\nAnswer with the option letter."
    content.append({"type": "text", "text": prompt})

    result = llm.create_json("vision", [{"role": "user", "content": content}], BaselineAnswer)
    confidence = min(max(result.confidence, 0.0), 1.0)
    return {
        "answer": normalize_answer(result.answer, options),
        "raw_answer": result.answer,
        "confidence": confidence,
        "self_confidence": confidence,
        "frames_used": len(frames),
    }
