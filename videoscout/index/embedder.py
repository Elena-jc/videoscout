"""SigLIP 2 image/text embeddings for cross-modal (text -> frame) retrieval.

Works with SigLIP and SigLIP 2 checkpoints through the transformers Auto classes.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from .tracker import resolve_device


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


class SiglipEmbedder:
    def __init__(self, model_name: str, device: str = "auto"):
        import torch
        from transformers import AutoModel, AutoProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()  # SigLIP 2 configs trigger a harmless token-id warning
        self._torch = torch
        self.device = resolve_device(device)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(model_name).to(self.device, dtype=dtype).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self._lock = threading.Lock()

    @staticmethod
    def _as_tensor(out):
        # Newer transformers versions may return a model output instead of a tensor.
        return getattr(out, "pooler_output", out)

    def _to_device(self, inputs) -> dict:
        moved = {}
        for key, value in inputs.items():
            value = value.to(self.device)
            moved[key] = value.to(self.model.dtype) if value.is_floating_point() else value
        return moved

    def embed_images(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        pil = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in images_bgr]
        with self._lock, self._torch.no_grad():
            inputs = self._to_device(self.processor(images=pil, return_tensors="pt"))
            feats = self._as_tensor(self.model.get_image_features(**inputs))
        return _l2norm(feats.float().cpu().numpy())

    def embed_text(self, texts: list[str]) -> np.ndarray:
        with self._lock, self._torch.no_grad():
            # SigLIP / SigLIP 2 text towers were trained on lowercased text padded to
            # exactly 64 tokens; dynamic padding silently degrades retrieval.
            inputs = self.processor(
                text=[t.lower() for t in texts], padding="max_length", max_length=64,
                truncation=True, return_tensors="pt",
            )
            feats = self._as_tensor(self.model.get_text_features(**self._to_device(inputs)))
        return _l2norm(feats.float().cpu().numpy())


_CACHE: dict[tuple[str, str], SiglipEmbedder] = {}
_CACHE_LOCK = threading.Lock()


def get_embedder(model_name: str, device: str = "auto") -> SiglipEmbedder:
    """Load each model once per process; loading takes seconds."""
    key = (model_name, device)
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = SiglipEmbedder(model_name, device)
        return _CACHE[key]
