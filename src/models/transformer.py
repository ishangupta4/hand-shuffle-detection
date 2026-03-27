"""Lightweight Transformer encoder for hand-shuffle classification.

Kept intentionally small (2 layers max) to avoid overfitting on
the tiny dataset. Uses sinusoidal positional encoding.

Input: (batch, T, F) feature sequences.
Output: (batch, 2) class probabilities.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """Small Transformer encoder for sequence classification.

    Args:
        input_dim: Number of input features per timestep.
        d_model: Internal model dimension. If different from input_dim,
                 a linear projection is added.
        nhead: Number of attention heads.
        dim_feedforward: Feed-forward hidden dim in each encoder layer.
        num_layers: Number of TransformerEncoder layers (1-2 recommended).
        num_classes: Number of output classes.
        dropout: Dropout rate throughout.
    """

    def __init__(
        self,
        input_dim: int = 39,
        d_model: int | None = None,
        nhead: int = 4,
        dim_feedforward: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        # d_model must be divisible by nhead
        if d_model is None:
            # Round input_dim up to nearest multiple of nhead
            d_model = input_dim + (nhead - input_dim % nhead) % nhead

        self.d_model = d_model
        self.input_dim = input_dim

        # Project input to d_model if needed
        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, lengths=None, mask=None):
        """Forward pass.

        Args:
            x: (batch, T, F) input features.
            lengths: (batch,) actual lengths (used to build mask if mask is None).
            mask: (batch, T) binary mask, 1=real, 0=padding.

        Returns:
            logits: (batch, num_classes) raw scores.
        """
        B, T, F = x.shape

        # Build padding mask if not provided
        if mask is None and lengths is not None:
            mask = torch.zeros(B, T, device=x.device)
            for i, l in enumerate(lengths):
                mask[i, :l] = 1.0

        # Project and add positional encoding
        out = self.input_proj(x)
        out = self.pos_enc(out)

        # Transformer expects src_key_padding_mask: True = ignore
        if mask is not None:
            key_padding_mask = (mask == 0)
        else:
            key_padding_mask = None

        out = self.encoder(out, src_key_padding_mask=key_padding_mask)

        # Masked global average pooling
        if mask is not None:
            m = mask.unsqueeze(2)  # (B, T, 1)
            pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            pooled = out.mean(dim=1)

        logits = self.fc(pooled)
        return logits


if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, F = 4, 50, 39
    x = torch.randn(B, T, F)
    mask = torch.ones(B, T)
    mask[0, 30:] = 0

    model = TransformerClassifier(input_dim=F)
    print(f"d_model: {model.d_model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    logits = model(x, mask=mask)
    print(f"Output shape: {logits.shape}")