"""CLIP image encoder (experimental — not wired into any pipeline).

This module implements the "Path B" cross-modal direction from the original
project proposal. **It is not used by any training, evaluation or inference
path in this repository** — the experiments use the precomputed 768-d UPFD BERT
features instead. It is kept as a documented starting point for future work
and needs `clip` installed plus a model download to run.

See ``features/text_encoder.py`` and ``features/cross_modal.py`` for the rest
of that path.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self._model = None
        self._preprocess = None
        self.dim = 512  # ViT-B/32 output dim

    def _load_model(self):
        import clip
        self._model, self._preprocess = clip.load(self.model_name, device=self.device)

    def forward(self, images: List) -> torch.Tensor:
        # images: list of PIL images or image paths
        # Requires CLIP preprocessing per image
        if self._model is None:
            self._load_model()
        from PIL import Image
        processed = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            processed.append(self._preprocess(img))
        batch = torch.stack(processed).to(self.device)
        with torch.no_grad():
            embeddings = self._model.encode_image(batch)
        return embeddings
