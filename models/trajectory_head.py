import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryPredictor(nn.Module):
    """
    Predicts the degradation trajectory for unseen future hops.

    Given observed drift for first k hops, predicts drift for
    hops k+1 through N using a Transformer encoder.

    This enables early detection: flag content as fake when
    the projected degradation trajectory crosses a threshold.
    """
    def __init__(
        self,
        drift_dim: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 100,
    ):
        super().__init__()
        self.input_proj = nn.Linear(drift_dim, d_model)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(d_model, drift_dim)

    def forward(self, observed_drift, observed_mask=None):
        # observed_drift: [batch, seq_len_observed, drift_dim]
        # observed_mask: [batch, seq_len_observed] — True where padded

        seq_len = observed_drift.shape[1]
        x = self.input_proj(observed_drift)
        positions = self.pos_encoder(
            torch.arange(seq_len, device=x.device).unsqueeze(0)
        )
        x = x + positions

        if observed_mask is not None:
            # Transformer expects True for masked positions
            x = self.transformer(x, src_key_padding_mask=observed_mask)
        else:
            x = self.transformer(x)

        drift_pred = self.output_head(x)
        return drift_pred


def predict_degradation_curve(
    model: TrajectoryPredictor,
    initial_drift: torch.Tensor,
    num_future_steps: int,
) -> torch.Tensor:
    """
    Autoregressive rollout: predict drift for future hops one at a time,
    feeding each prediction back as input.
    """
    model.eval()
    predictions = []
    current = initial_drift.clone()

    with torch.no_grad():
        for _ in range(num_future_steps):
            pred = model(current)
            next_drift = pred[:, -1:, :]
            predictions.append(next_drift)
            current = torch.cat([current, next_drift], dim=1)

    return torch.cat(predictions, dim=1)
