"""Cross-modal alignment (experimental — not wired into any pipeline).

Implements the "Path B" cross-modal direction from the original project
proposal: a learned projection of text and image embeddings into a shared
space with a contrastive alignment score.

**This module is not used by any training, evaluation or inference path in
this repository.** The reported experiments use the precomputed 768-d UPFD
BERT features only. It is kept as a documented starting point for future work.

See ``features/text_encoder.py`` and ``features/image_encoder.py`` for the
encoders it is designed to align.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAligner(nn.Module):
    """Computes and projects cross-modal alignment scores."""
    def __init__(self, text_dim: int = 384, image_dim: int = 512, hidden_dim: int = 64):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        return_scores: bool = True,
    ) -> torch.Tensor:
        t = F.normalize(self.text_proj(text_emb), dim=-1)
        i = F.normalize(self.image_proj(image_emb), dim=-1)
        sim = (t * i).sum(dim=-1) / self.temperature
        return torch.sigmoid(sim) if return_scores else sim


def cross_modal_similarity(
    text_emb: torch.Tensor, image_emb: torch.Tensor
) -> torch.Tensor:
    """Cosine similarity between text and image embeddings."""
    t = F.normalize(text_emb, dim=-1)
    i = F.normalize(image_emb, dim=-1)
    return (t * i).sum(dim=-1)
