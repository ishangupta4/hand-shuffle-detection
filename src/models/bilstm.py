"""Bi-LSTM classifier for hand-shuffle sequence classification.

Input: (batch, T, F) feature sequences with optional padding masks.
Output: (batch, 2) class probabilities.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMClassifier(nn.Module):
    """Two-layer bidirectional LSTM with dense classification head.

    Args:
        input_dim: Number of input features per timestep.
        hidden_dims: List of hidden sizes for each LSTM layer.
        num_classes: Number of output classes.
        dropout_lstm: Dropout between LSTM layers and after each layer.
        dropout_fc: Dropout before final classification layer.
        use_packing: Use pack_padded_sequence for variable-length inputs.
    """

    def __init__(
        self,
        input_dim: int = 39,
        hidden_dims: list[int] | None = None,
        num_classes: int = 2,
        dropout_lstm: float = 0.5,
        dropout_fc: float = 0.3,
        use_packing: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.use_packing = use_packing

        # Build stacked LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        in_size = input_dim
        for h_dim in hidden_dims:
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=h_dim,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.dropout_layers.append(nn.Dropout(dropout_lstm))
            in_size = h_dim * 2  # bidirectional doubles output

        # Classification head
        final_dim = hidden_dims[-1] * 2
        self.fc = nn.Sequential(
            nn.Linear(final_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(16, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot initialization for all parameters."""
        for lstm in self.lstm_layers:
            for name, param in lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1 for better gradient flow
                    hidden_size = param.shape[0] // 4
                    param.data[hidden_size:2 * hidden_size].fill_(1.0)

        for module in self.fc:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, lengths=None, mask=None):
        """Forward pass.

        Args:
            x: (batch, T, F) input features.
            lengths: (batch,) actual sequence lengths for packing.
            mask: (batch, T) binary mask, 1 for real frames.

        Returns:
            logits: (batch, num_classes) raw scores (pre-softmax).
        """
        batch_size = x.size(0)

        # Infer lengths from mask if not provided
        if lengths is None and mask is not None:
            lengths = mask.sum(dim=1).long()

        out = x
        for lstm, drop in zip(self.lstm_layers, self.dropout_layers):
            if self.use_packing and lengths is not None:
                # Pack for efficient computation on variable-length sequences
                lengths_cpu = lengths.cpu().clamp(min=1)
                packed = pack_padded_sequence(
                    out, lengths_cpu, batch_first=True, enforce_sorted=False
                )
                packed_out, _ = lstm(packed)
                out, _ = pad_packed_sequence(packed_out, batch_first=True)
            else:
                out, _ = lstm(out)
            out = drop(out)

        # Extract the last valid hidden state for each sequence
        if lengths is not None:
            # Gather the output at the last real timestep
            idx = (lengths - 1).clamp(min=0).long()
            idx = idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2))
            last_hidden = out.gather(1, idx).squeeze(1)
        else:
            last_hidden = out[:, -1, :]

        logits = self.fc(last_hidden)
        return logits


if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, F = 4, 50, 39
    x = torch.randn(B, T, F)
    lengths = torch.tensor([50, 40, 30, 20])
    mask = torch.zeros(B, T)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0

    model = BiLSTMClassifier(input_dim=F)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    logits = model(x, lengths=lengths)
    print(f"Output shape: {logits.shape}")

    logits_mask = model(x, mask=mask)
    print(f"Output (mask mode): {logits_mask.shape}")