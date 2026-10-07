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
