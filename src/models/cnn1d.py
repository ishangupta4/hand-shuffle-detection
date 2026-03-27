"""1D CNN classifier for hand-shuffle sequence classification.

Input: (batch, T, F) feature sequences.
Output: (batch, 2) class probabilities.
"""

import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    """Two-layer 1D CNN with global average pooling and dense head.

    Convolutions operate along the time axis. Padding='same' preserves
    temporal resolution so global average pooling captures the full sequence.

    Args:
        input_dim: Number of input features per timestep.
        filters: List of filter counts for each conv layer.
        kernel_sizes: List of kernel sizes for each conv layer.
        num_classes: Number of output classes.
        dropout_conv: Dropout after each conv block.
        dropout_fc: Dropout before final classification layer.
    """

    def __init__(
        self,
        input_dim: int = 39,
        filters: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        num_classes: int = 2,
        dropout_conv: float = 0.3,
        dropout_fc: float = 0.3,
    ):
        super().__init__()
        if filters is None:
            filters = [32, 64]
        if kernel_sizes is None:
            kernel_sizes = [5, 3]

        assert len(filters) == len(kernel_sizes), "filters and kernel_sizes must match"

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Build conv blocks: Conv1D -> BatchNorm -> ReLU -> Dropout
        layers = []
        in_channels = input_dim
        for filt, ks in zip(filters, kernel_sizes):
            padding = ks // 2  # 'same' padding
            layers.extend([
                nn.Conv1d(in_channels, filt, kernel_size=ks, padding=padding),
                nn.BatchNorm1d(filt),
                nn.ReLU(),
                nn.Dropout(dropout_conv),
            ])
            in_channels = filt

        self.conv = nn.Sequential(*layers)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 16),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(16, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, lengths=None, mask=None):
        """Forward pass.

        Args:
            x: (batch, T, F) input features.
            lengths: (batch,) actual sequence lengths (unused, for API compat).
            mask: (batch, T) binary mask. If provided, zeroes out padded
                  positions before pooling.

        Returns:
            logits: (batch, num_classes) raw scores.
        """
        # Conv1d expects (batch, channels, time)
        out = x.transpose(1, 2)
        out = self.conv(out)  # (batch, filters[-1], T)

        # Masked global average pooling
        if mask is not None:
            # Expand mask to match conv output: (batch, 1, T)
            m = mask.unsqueeze(1)
            out = out * m
            pooled = out.sum(dim=2) / m.sum(dim=2).clamp(min=1)
        else:
            pooled = out.mean(dim=2)

        logits = self.fc(pooled)
        return logits


if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, F = 4, 50, 39
    x = torch.randn(B, T, F)
    mask = torch.ones(B, T)
    mask[0, 30:] = 0
    mask[1, 40:] = 0

    model = CNN1DClassifier(input_dim=F)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    logits = model(x, mask=mask)
    print(f"Output shape: {logits.shape}")