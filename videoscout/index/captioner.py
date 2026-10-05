"""Optional dense captions per segment, written by a vision model.

Captions give the sparse retriever real text to match against (subtitles only
cover speech, object tags only cover COCO classes). Captioning is offline and not
latency-sensitive, so the default path is the Message Batches API: half price,
results usually within the hour.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..llm import LLMClient, block_to_dict, text_of
from ..video import VideoReader, fmt_ts, to_jpeg_b64
from .store import Segment

CAPTION_PROMPT = (
    "These {n} frames come from one segment of a video ({start}-{end}). Write one dense "
    "sentence (at most 40 words) describing what is visible: people, objects, actions, "
    "setting and any readable text. Describe only what you can see."
)


def _requests(reader: VideoReader, segments: list[Segment], frames_per_segment: int, max_side: int):
    times: list[float] = []
    owner: dict[float, int] = {}
    for seg in segments:
        step = (seg.t_end - seg.t_start) / frames_per_segment
        for k in range(frames_per_segment):
            t = round(seg.t_start + (k + 0.5) * step, 3)
            times.append(t)
            owner[t] = seg.seg_id

    images: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for t, frame in reader.iter_frames_at(times):
        images[owner[t]].append(
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": to_jpeg_b64(frame, max_side)}}
        )

    for seg in segments:
        blocks = images.get(seg.seg_id)
        if not blocks:
            continue
        prompt = CAPTION_PROMPT.format(n=len(blocks), start=fmt_ts(seg.t_start), end=fmt_ts(seg.t_end))
        yield seg, [{"role": "user", "content": [*blocks, {"type": "text", "text": prompt}]}]


def caption_segments(
    llm: LLMClient,
    reader: VideoReader,
    segments: list[Segment],
    mode: str = "batch",
    frames_per_segment: int = 3,
    max_side: int = 512,
    poll_seconds: float = 30.0,
    log: Callable[[str], None] = print,
) -> None:
    if mode == "batch" and not getattr(llm, "supports_batches", False):
        mode = "sync"  # only the Claude backend has a Batches API path
    requests = list(_requests(reader, segments, frames_per_segment, max_side))
    by_id = {seg.seg_id: seg for seg, _ in requests}

    if mode == "sync":
        def run(item):
            seg, messages = item
            return seg, text_of(llm.create("captioner", messages, max_tokens=1024))

        with ThreadPoolExecutor(max_workers=4) as pool:
            for seg, caption in pool.map(run, requests):
                seg.caption = caption
        return

    model = llm.model_for("captioner")
    batch = llm.client.messages.batches.create(
        requests=[
            {
                "custom_id": f"seg-{seg.seg_id}",
                "params": {
                    "model": model,
                    "max_tokens": 1024,
                    "messages": messages,
                    "output_config": {"effort": llm.models.captioner_effort},
                },
            }
            for seg, messages in requests
        ]
    )
    log(f"caption batch {batch.id} submitted ({len(requests)} segments); polling...")
    while True:
        status = llm.client.messages.batches.retrieve(batch.id)
        if status.processing_status == "ended":
            break
        time.sleep(poll_seconds)

    failed = 0
    for result in llm.client.messages.batches.results(batch.id):
        seg = by_id[int(result.custom_id.split("-", 1)[1])]
        if result.result.type != "succeeded":
            failed += 1
            continue
        message = result.result.message
        llm.meter.add("captioner", model, message.usage, price_scale=0.5)
        seg.caption = " ".join(
            b["text"] for b in map(block_to_dict, message.content) if b.get("type") == "text"
        ).strip()
    if failed:
        log(f"warning: {failed} caption requests did not succeed; those segments have no caption")
