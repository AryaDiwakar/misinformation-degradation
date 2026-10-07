import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineLSTM(nn.Module):
    """
    Simple LSTM over node sequence (BFS order from root).
    Ignores graph structure — treats propagation as an ordered sequence.
    """
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        order = data.hop_depth.argsort()
        x = x[order].unsqueeze(0)
        _, (h_n, _) = self.lstm(x)
        if self.lstm.bidirectional:
            h_fwd = h_n[-2]
            h_bwd = h_n[-1]
            h = torch.cat([h_fwd, h_bwd], dim=-1)
        else:
            h = h_n[-1]
        logits = self.classifier(h)
        return logits
