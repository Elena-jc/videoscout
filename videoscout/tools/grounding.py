"""find_objects: open-vocabulary detection at question time.

The offline index only knows the 80 COCO classes its closed-set detector was
trained on: a cheap, always-on first tier. Questions name anything ("red jacket",
"forklift", "price tag"), so this tool runs a heavier open-vocabulary model on
demand, on one time window: SAM 3 concept tracking when its weights are available
(identities across frames, see concepts.py), else YOLOE-26 per-frame detection with
text prompts encoded by MobileCLIP2. Both run locally, so the tool costs a tool call
but no API tokens or frame budget, and the boxes can be handed to inspect_clip.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..index.store import VideoIndex
from ..video import VideoReader, fmt_ts
from .registry import Tool, ToolError

Detection = tuple[str, float, tuple[float, float, float, float]]  # name, confidence, normalized box

TEXT_ENCODER_WEIGHTS = {"mobileclip2:b": "mobileclip2_b.ts", "mobileclip:blt": "mobileclip_blt.ts"}
MAX_NAMES = 5
MAX_LINES_PER_NAME = 6


class OpenVocabDetector:
    """YOLOE behind a lock: set_classes mutates the model, so calls are serialized."""

    def __init__(self, weights: str, device: str = "auto", conf: float = 0.25):
        self.weights = weights
        self.device = device
        self.conf = conf
        self._model = None
        self._names: list[str] | None = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            import torch
            from ultralytics import YOLOE
            from ultralytics.nn.text_model import MobileCLIPTS

            from ..config import weights_path
            from ..index.tracker import resolve_device

            self.device = resolve_device(self.device)
            model = YOLOE(weights_path(self.weights))
            model.to(self.device)
            # Load the text encoder once, from the project's weights/ folder. (By default
            # Ultralytics rebuilds it on every prompt change and saves it to the cwd.)
            encoder = TEXT_ENCODER_WEIGHTS.get(getattr(model.model, "text_model", ""))
            if encoder:
                model.model.clip_model = MobileCLIPTS(torch.device(self.device), weight=weights_path(encoder))
            self._model = model
        return self._model

    def warm(self) -> None:
        """Load the model now instead of on the first find_objects call."""
        with self._lock:
            self._load()

    def detect(self, images: Sequence[np.ndarray], names: list[str]) -> list[list[Detection]]:
        with self._lock:
            model = self._load()
            if names != self._names:
                embeddings = model.model.get_text_pe(names, cache_clip_model=True)
                model.set_classes(names, embeddings)
                self._names = list(names)
            results = model.predict(list(images), conf=self.conf, device=self.device, verbose=False)
        out = []
        for result in results:
            boxes = result.boxes
            dets: list[Detection] = []
            if boxes is not None and len(boxes):
                for cls, p, box in zip(boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxyn.tolist()):
                    dets.append((names[cls], float(p), tuple(round(v, 3) for v in box)))
            out.append(dets)
        return out


DESCRIPTION = """Detect objects described in words (open-vocabulary) in a time window, e.g. names=['red jacket', \
'forklift', 'stop sign']. Frames are sampled evenly from [t_start, t_end] and a local detector (YOLOE) returns, \
per name, in how many frames it was found, how many at once, and boxes as normalized [x1, y1, x2, y2].
Use it for objects outside the tracked COCO classes, for counting specific things in a moment, and to get a box \
to pass to inspect_clip for a zoomed-in look. It does not use the frame budget. Detections are candidates: \
confirm important ones with inspect_clip, and treat "not found" as weak evidence (small or unusual objects are missed)."""


class FindObjectsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: list[str] = Field(description="1-5 short noun phrases to detect, e.g. ['red jacket', 'forklift'].")
    t_start: float = Field(description="Window start in seconds.")
    t_end: float = Field(description="Window end in seconds.")
    num_frames: Optional[int] = Field(None, description="Frames to scan, 1-16 (default 8).")


def format_detections(times: list[float], per_frame: list[list[Detection]], names: list[str], t0: float, t1: float) -> str:
    lines = [f"Scanned {len(times)} frames in {fmt_ts(t0)}-{fmt_ts(t1)} (t={t0:.1f}-{t1:.1f}s) for: {', '.join(names)}"]
    for name in names:
        hits = [(t, [d for d in dets if d[0] == name]) for t, dets in zip(times, per_frame)]
        hits = [(t, dets) for t, dets in hits if dets]
        if not hits:
            lines.append(f"- {name}: not found (the detector can miss small or unusual objects)")
            continue
        peak = max(len(dets) for _, dets in hits)
        lines.append(f"- {name}: in {len(hits)}/{len(times)} frames, up to {peak} at once")
        for t, dets in hits[:MAX_LINES_PER_NAME]:
            best = max(dets, key=lambda d: d[1])
            box = ", ".join(f"{v:.2f}" for v in best[2])
            lines.append(f"    t={t:.1f}s: {len(dets)} found, best conf {best[1]:.2f}, box [{box}]")
    lines.append("Pass a box to inspect_clip (box=[x1, y1, x2, y2]) to zoom in.")
    return "\n".join(lines)


SAM3_DESCRIPTION = """Find and TRACK objects described in words (open-vocabulary) in a time window, e.g. \
names=['man in a suit', 'red car', 'forklift']. Frames are sampled evenly from [t_start, t_end] and SAM 3 \
segments every instance of each phrase and keeps its identity across frames. Returns, per name, the number of \
distinct instances, the most visible at once, when each instance is visible, and a box as normalized [x1, y1, x2, y2].
Use it for counting distinct objects, for how long or when something is visible, for objects outside the tracked \
COCO classes, and to get a box to pass to inspect_clip. It runs locally and does not use the frame budget, but it \
is slower than other tools: keep windows short (under about a minute) and use 1-3 precise noun phrases. \
Results are candidates: confirm important ones with inspect_clip."""


class TrackObjectsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: list[str] = Field(description="1-3 short noun phrases to track, e.g. ['man in a suit', 'red car'].")
    t_start: float = Field(description="Window start in seconds.")
    t_end: float = Field(description="Window end in seconds (keep windows short).")
    num_frames: Optional[int] = Field(None, description="Frames to track through, 2-32 (default 16).")


def make_find_objects_tool(
    index: VideoIndex, reader: VideoReader, detector, max_frames: int, default_frames: int
) -> Tool:
    """`detector` is an OpenVocabDetector (YOLOE, per-frame boxes) or a
    concepts.Sam3ConceptTracker (SAM 3, instances tracked across frames)."""
    sam3 = getattr(detector, "kind", "") == "sam3"
    max_names = 3 if sam3 else MAX_NAMES

    def run(args) -> str:
        names = [n.strip() for n in args.names if n.strip()][:max_names]
        if not names:
            raise ToolError("Give at least one object name.")
        t0 = max(0.0, min(args.t_start, args.t_end))
        t1 = min(index.duration, max(args.t_start, args.t_end))
        if t0 >= index.duration:
            raise ToolError(f"The window starts after the end of the video ({index.duration:.1f}s).")
        n = min(max(args.num_frames or default_frames, 2 if sam3 else 1), max_frames)
        frames = reader.sample(t0, t1, n)
        if not frames:
            raise ToolError("Could not decode frames in this window.")
        times = [t for t, _ in frames]
        if sam3:
            from .concepts import format_tracks

            return format_tracks(times, detector.track([img for _, img in frames], names), names, t0, t1)
        per_frame = detector.detect([img for _, img in frames], names)
        return format_detections(times, per_frame, names, t0, t1)

    if sam3:
        return Tool("find_objects", SAM3_DESCRIPTION, TrackObjectsArgs, run)
    return Tool("find_objects", DESCRIPTION, FindObjectsArgs, run)
