"""Sentence-Transformers text encoder (experimental — not wired into any pipeline).

This module implements the "Path B" cross-modal direction from the original
project proposal. **It is not used by any training, evaluation or inference
path in this repository** — the experiments use the precomputed 768-d UPFD BERT
features instead. It is kept as a documented starting point for future work
and needs `sentence-transformers` installed plus a model download to run.

See ``features/image_encoder.py`` and ``features/cross_modal.py`` for the
rest of that path.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model_name = model_name
        # Sentence-Bert model loaded lazily
        self._model = None
        self.dim = 384  # all-MiniLM-L6-v2 output dim

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def forward(self, texts: List[str]) -> torch.Tensor:
        if self._model is None:
            self._load_model()
        embeddings = self._model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return embeddings


class TextEmbeddingProjector(nn.Module):
    """Projects variable-dim text embeddings to a fixed hidden dim."""
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
