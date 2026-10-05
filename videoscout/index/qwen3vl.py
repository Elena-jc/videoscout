"""Local Qwen3-VL models (Apache-2.0), all built on the same 2B vision-language base.

- Qwen3VLEmbedder (Qwen3-VL-Embedding-2B): one vector per *clip* (a few frames plus
  its subtitles) and per text query, in a shared space. Unlike a frame-level image
  encoder (SigLIP), it sees a clip as a short video, so motion and order count.
- Qwen3VLReranker (Qwen3-VL-Reranker-2B): a cross-encoder that reads the query and a
  candidate clip together. Too slow to score every clip, accurate on the top 20:
  the usual retrieve-then-rerank split of search systems.
- LocalCaptioner (Qwen3-VL-2B-Instruct): writes clip captions and event summaries
  while indexing, on the local GPU, so building the memory costs no API calls.

All three share the GPU through videoscout.gpu.POOL.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

import cv2
import numpy as np

from ..gpu import POOL
from .tracker import resolve_device

EMBEDDER_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
CAPTIONER_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

QUERY_INSTRUCTION = "Retrieve video clips that show or discuss what the user's query describes."
CLIP_TEXT_LIMIT = 400

Clip = tuple[Sequence[np.ndarray], str]  # BGR frames in time order, text (subtitles, captions)


def to_pil(frames: Sequence[np.ndarray], max_side: int):
    from PIL import Image

    out = []
    for img in frames:
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        out.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return out


def _quiet_transformers() -> None:
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()


def _dtype(device: str):
    import torch

    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def _no_frame_resampling(model) -> None:
    """Frames arrive already sampled from the clip; stop the video processor from
    resampling them by fps (it would duplicate or drop frames and warn)."""
    for module in model.modules():
        processor = getattr(module, "processor", None)
        video_processor = getattr(processor, "video_processor", None)
        if video_processor is not None:
            video_processor.do_sample_frames = False


def clip_document(frames: Sequence[np.ndarray], text: str, max_side: int) -> dict:
    doc: dict = {"video": to_pil(frames, max_side)}
    text = " ".join(text.split())[:CLIP_TEXT_LIMIT]
    if text:
        doc["text"] = text
    return doc


class Qwen3VLEmbedder:
    """Clips are embedded on the GPU while indexing. Queries are a dozen tokens: at
    question time they are encoded by an fp32 copy on the CPU (~0.9 s), which is
    faster than swapping the 4 GB model back onto an 8 GB GPU (~3 s each way) and
    leaves the GPU to the reranker."""

    def __init__(self, model_name: str = EMBEDDER_MODEL, device: str = "auto", max_side: int = 448, batch_size: int = 4):
        _quiet_transformers()
        self.model_name = model_name
        self.name = f"embedder:{model_name}"
        self.device = resolve_device(device)
        self.max_side = max_side
        self.batch_size = batch_size
        self._clip_model = None  # GPU dtype, loaded for indexing only
        self._query_model = None  # fp32 on the CPU
        self._lock = threading.Lock()

    @staticmethod
    def _load(model_name: str, dtype):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device="cpu", model_kwargs={"torch_dtype": dtype})
        _no_frame_resampling(model)
        return model

    def embed_text(self, texts: list[str]) -> np.ndarray:
        import torch

        with self._lock:
            if self._query_model is None:
                self._query_model = self._load(self.model_name, torch.float32)
            out = self._query_model.encode(list(texts), prompt=QUERY_INSTRUCTION, normalize_embeddings=True,
                                           convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(out, dtype=np.float32)

    def embed_clips(self, clips: Sequence[Clip]) -> np.ndarray:
        docs = [clip_document(frames, text, self.max_side) for frames, text in clips]
        with self._lock:
            if self._clip_model is None:
                self._clip_model = self._load(self.model_name, _dtype(self.device))
            with POOL.use(self.name, self._clip_model, self.device):
                out = self._clip_model.encode(docs, batch_size=self.batch_size, normalize_embeddings=True,
                                              convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(out, dtype=np.float32)


class Qwen3VLReranker:
    def __init__(self, model_name: str = RERANKER_MODEL, device: str = "auto", max_side: int = 448, batch_size: int = 4):
        from sentence_transformers import CrossEncoder

        _quiet_transformers()
        self.name = f"reranker:{model_name}"
        self.device = resolve_device(device)
        self.model = CrossEncoder(model_name, device="cpu", model_kwargs={"torch_dtype": _dtype(self.device)})
        _no_frame_resampling(self.model)
        self.max_side = max_side
        self.batch_size = batch_size

    def score(self, query: str, clips: Sequence[Clip]) -> list[float]:
        """Relevance in [0, 1] of each clip to the query."""
        import torch

        if not clips:
            return []
        pairs = [(query, clip_document(frames, text, self.max_side)) for frames, text in clips]
        with POOL.use(self.name, self.model, self.device):
            scores = self.model.predict(pairs, prompt=QUERY_INSTRUCTION, batch_size=self.batch_size,
                                        activation_fn=torch.nn.Sigmoid(), show_progress_bar=False)
        return [float(s) for s in np.asarray(scores, dtype=np.float32).reshape(-1)]


CLIP_CAPTION_PROMPT = (
    "Describe this {seconds:.0f}-second video clip in one sentence of at most 35 words: people, objects, "
    "actions, setting, and any readable on-screen text. Describe only what is visible."
)


class LocalCaptioner:
    def __init__(self, model_name: str = CAPTIONER_MODEL, device: str = "auto", max_side: int = 448):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        _quiet_transformers()
        self.name = f"captioner:{model_name}"
        self.device = resolve_device(device)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, dtype=_dtype(self.device)).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.max_side = max_side

    def _generate(self, content: list[dict], max_new_tokens: int, **processor_kwargs) -> str:
        import torch

        messages = [{"role": "user", "content": content}]
        with POOL.use(self.name, self.model, self.device):
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                **processor_kwargs,
            ).to(self.model.device)
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return " ".join(text.split())

    def caption_clip(self, frames: Sequence[np.ndarray], seconds: float, max_new_tokens: int = 72) -> str:
        n = len(frames)
        metadata = [{"fps": n / max(seconds, 1e-3), "total_num_frames": n, "frames_indices": list(range(n))}]
        content = [{"type": "video", "video": to_pil(frames, self.max_side)},
                   {"type": "text", "text": CLIP_CAPTION_PROMPT.format(seconds=seconds)}]
        return self._generate(content, max_new_tokens, do_sample_frames=False, video_metadata=metadata)

    def write(self, prompt: str, max_new_tokens: int = 160) -> str:
        """Text-only generation (event summaries, storyline)."""
        return self._generate([{"type": "text", "text": prompt}], max_new_tokens)

    def unload(self) -> None:
        POOL.release(self.name)


_CACHE: dict[tuple[str, str, str], object] = {}
_CACHE_LOCK = threading.Lock()


def _cached(kind: str, cls, model_name: str, device: str):
    key = (kind, model_name, device)
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = cls(model_name, device)
        return _CACHE[key]


def get_qwen_embedder(model_name: str = EMBEDDER_MODEL, device: str = "auto") -> Qwen3VLEmbedder:
    return _cached("embedder", Qwen3VLEmbedder, model_name, device)


def get_qwen_reranker(model_name: str = RERANKER_MODEL, device: str = "auto") -> Qwen3VLReranker:
    return _cached("reranker", Qwen3VLReranker, model_name, device)
