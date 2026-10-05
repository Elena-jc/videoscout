"""inspect_clip: look at a time window with a vision model, optionally zoomed
onto one tracked object."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from ..index.store import VideoIndex
from ..video import VideoReader, crop_normalized, fmt_ts
from ..vision import VisionBackend
from .registry import Tool, ToolError

DESCRIPTION = """Look at a time window of the video: frames are sampled evenly from [t_start, t_end] and a \
vision model answers your question about them, citing frame timestamps. This is the only tool that actually \
sees pixels, so use it to confirm candidates from search_segments / query_tracks before answering.
Tips: ask a specific question; keep windows tight (10-60 s) so frames are dense. When details are small (text, \
held objects, clothing), zoom in: pass track_id to follow one tracked object, or box (from find_objects) to crop \
a fixed region. Frames count against the frame budget."""

VISION_PROMPT = """You are the visual inspector for a video question-answering agent. The frames above are \
sampled in order from {start}-{end} of the video{zoom}.
Task: {question}
Report only what is visible, citing frame timestamps. If the frames are not enough to answer, say so \
explicitly and describe what is visible instead. At most 120 words."""


class InspectArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t_start: float = Field(description="Window start in seconds.")
    t_end: float = Field(description="Window end in seconds. Longer windows spread the frames thinner.")
    question: str = Field(
        description="What to find out from these frames, e.g. 'What is written on the sign?' rather than 'describe'."
    )
    num_frames: Optional[int] = Field(
        None, description="Frames to sample from the window, 1-8 (default 6). Counts against the frame budget."
    )
    track_id: Optional[int] = Field(
        None, description="Zoom in: crop each frame around this tracked object (ids come from query_tracks)."
    )
    box: Optional[list[float]] = Field(
        None, description="Zoom in: crop each frame to this normalized [x1, y1, x2, y2] box (e.g. from find_objects)."
    )


def _nearest_box(boxes, t: float):
    return min(boxes, key=lambda b: abs(b[0] - t))[1]


def make_inspect_tool(
    index: VideoIndex,
    reader: VideoReader,
    vision: VisionBackend,
    max_frames_per_call: int,
    default_frames: int,
) -> Tool:
    def run(args: InspectArgs) -> str:
        t0 = max(0.0, min(args.t_start, args.t_end))
        t1 = min(index.duration, max(args.t_start, args.t_end))
        if t0 >= index.duration:
            raise ToolError(f"The window starts after the end of the video ({index.duration:.1f}s).")
        if t1 - t0 < 0.5:
            t1 = min(index.duration, t0 + 1.0)
        n = min(max(args.num_frames or default_frames, 1), max_frames_per_call)
        frames = reader.sample(t0, t1, n)
        if not frames:
            raise ToolError("Could not decode frames in this window.")

        zoom = ""
        if args.track_id is not None and args.box is not None:
            raise ToolError("Pass either track_id or box, not both.")
        if args.box is not None:
            if len(args.box) != 4:
                raise ToolError("box must be [x1, y1, x2, y2] with values between 0 and 1.")
            x1, y1, x2, y2 = (min(max(float(v), 0.0), 1.0) for v in args.box)
            if x2 <= x1 or y2 <= y1:
                raise ToolError("box must satisfy x1 < x2 and y1 < y2 (normalized coordinates).")
            frames = [(t, crop_normalized(img, (x1, y1, x2, y2))) for t, img in frames]
            zoom = f", cropped to box [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]"
        elif args.track_id is not None:
            boxes = index.track_boxes(args.track_id, t0 - 1.0, t1 + 1.0)
            if not boxes:
                raise ToolError(
                    f"Track {args.track_id} is not visible between {t0:.1f}s and {t1:.1f}s; "
                    "use query_tracks to find when it appears."
                )
            frames = [(t, crop_normalized(img, _nearest_box(boxes, t))) for t, img in frames]
            zoom = f", cropped around tracked object #{args.track_id}"

        prompt = VISION_PROMPT.format(start=fmt_ts(t0), end=fmt_ts(t1), zoom=zoom, question=args.question)
        answer = vision.ask(frames, prompt)
        return f"Observation {fmt_ts(t0)}-{fmt_ts(t1)} (t={t0:.1f}-{t1:.1f}s, {len(frames)} frames{zoom}):\n{answer}"

    return Tool("inspect_clip", DESCRIPTION, InspectArgs, run)
