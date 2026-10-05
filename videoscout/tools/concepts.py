"""SAM 3 concept tracking behind find_objects.

SAM 3 (Meta, Nov 2025) does Promptable Concept Segmentation: given a short noun
phrase ("man in a suit", "red car") it detects, segments and *tracks* every instance
of that concept through a video. Compared with the per-frame YOLOE detector it
replaces, identities are kept across frames, so "how many distinct people" and
"how long was the forklift visible" come from the tracker instead of being guessed
from per-frame counts. It is heavier (~0.85B parameters), so it runs on demand on a
short window, never over the whole video.

Weights are gated on Hugging Face (facebook/sam3). The tool only uses SAM 3 when the
weights are already in the local cache; nothing is downloaded at question time.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Sequence

import cv2
import numpy as np

from ..gpu import POOL
from ..video import fmt_ts

SAM3_INPUT_SIDE = 1008  # SAM 3's own working resolution; larger frames only cost decode time

# name -> instance id -> [(frame index, score, normalized xyxy box)]
Tracks = dict[str, dict[int, list[tuple[int, float, tuple[float, float, float, float]]]]]


def sam3_weights_cached(model_name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(model_name, "model.safetensors"), str)


class Sam3ConceptTracker:
    kind = "sam3"

    def __init__(self, model_name: str = "facebook/sam3", device: str = "auto", conf: float = 0.4):
        self.model_name = model_name
        self.device = device
        self.conf = conf
        self.name = f"sam3:{model_name}"
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            import torch
            from transformers import Sam3VideoModel, Sam3VideoProcessor
            from transformers.utils import logging as hf_logging

            from ..index.tracker import resolve_device

            hf_logging.disable_progress_bar()
            self.device = resolve_device(self.device)
            self._dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
            self._model = Sam3VideoModel.from_pretrained(self.model_name, dtype=self._dtype).eval()
            self._processor = Sam3VideoProcessor.from_pretrained(self.model_name)
        return self._model, self._processor

    def warm(self) -> None:
        with self._lock:
            self._load()

    def track(self, frames: Sequence[np.ndarray], names: list[str]) -> Tracks:
        from PIL import Image

        tracks: Tracks = {name: defaultdict(list) for name in names}
        if not frames:
            return tracks
        h, w = frames[0].shape[:2]
        scale = min(1.0, SAM3_INPUT_SIDE / max(h, w))
        video = []
        for img in frames:
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            video.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        vw, vh = video[0].size
        with self._lock:
            model, processor = self._load()
            with POOL.use(self.name, model, self.device):
                session = processor.init_video_session(
                    video=video, inference_device=self.device, processing_device="cpu",
                    video_storage_device="cpu", dtype=self._dtype,
                )
                processor.add_text_prompt(session, names)
                for out in model.propagate_in_video_iterator(session, max_frame_num_to_track=len(video)):
                    result = processor.postprocess_outputs(session, out)
                    owner = {oid: prompt for prompt, ids in result.get("prompt_to_obj_ids", {}).items() for oid in ids}
                    for oid, score, box in zip(result["object_ids"].tolist(), result["scores"].tolist(),
                                               result["boxes"].float().tolist()):
                        name = owner.get(oid, names[0] if len(names) == 1 else None)
                        if name not in tracks or score < self.conf:
                            continue
                        x1, y1, x2, y2 = box
                        norm = (round(x1 / vw, 3), round(y1 / vh, 3), round(x2 / vw, 3), round(y2 / vh, 3))
                        tracks[name][int(oid)].append((int(out.frame_idx), float(score), norm))
        return tracks


MAX_INSTANCES_SHOWN = 6


def format_tracks(times: list[float], tracks: Tracks, names: list[str], t0: float, t1: float) -> str:
    lines = [f"Tracked with SAM 3 over {fmt_ts(t0)}-{fmt_ts(t1)} (t={t0:.1f}-{t1:.1f}s, {len(times)} frames) "
             f"for: {', '.join(names)}"]
    for name in names:
        instances = tracks.get(name) or {}
        if not instances:
            lines.append(f"- {name}: not found (small, distant or unusual objects can be missed)")
            continue
        per_frame: dict[int, int] = defaultdict(int)
        for obs in instances.values():
            for f, _, _ in obs:
                per_frame[f] += 1
        lines.append(f"- {name}: {len(instances)} distinct instance(s) (identity kept across frames), "
                     f"up to {max(per_frame.values())} visible at once")
        ranked = sorted(instances.items(), key=lambda kv: -len(kv[1]))
        for k, (_, obs) in enumerate(ranked[:MAX_INSTANCES_SHOWN], start=1):
            obs = sorted(obs)
            first, last = times[obs[0][0]], times[obs[-1][0]]
            mid = obs[len(obs) // 2]
            box = ", ".join(f"{v:.2f}" for v in mid[2])
            mean = sum(s for _, s, _ in obs) / len(obs)
            lines.append(f"    #{k}: visible t={first:.1f}-{last:.1f}s in {len(obs)}/{len(times)} frames, "
                         f"score {mean:.2f}, box at t={times[mid[0]]:.1f}s [{box}]")
        if len(ranked) > MAX_INSTANCES_SHOWN:
            lines.append(f"    ... {len(ranked) - MAX_INSTANCES_SHOWN} more")
    lines.append("Pass a box to inspect_clip (box=[x1, y1, x2, y2]) to zoom in.")
    return "\n".join(lines)
