# src/pinn_model.py
"""
CNN + LSTM architecture for reconstructing deck strain from sparse
acceleration channels.

Input:  (batch, sequence_length, n_input_channels)
Output: (batch, sequence_length, n_output_channels)
"""

import torch
import torch.nn as nn


class StrainReconstructionNet(nn.Module):
    def __init__(
        self,
        n_input_channels: int = 4,
        n_output_channels: int = 8,
        cnn_channels: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
    ):
        super().__init__()

        # Conv1d expects (batch, channels, sequence_length) -- input will be
        # permuted accordingly in forward().
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_input_channels, cnn_channels, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=15, padding=7),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.output_layer = nn.Linear(lstm_hidden * 2, n_output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, sequence_length, n_input_channels)
        x = x.permute(0, 2, 1)          # -> (batch, n_input_channels, sequence_length)
        x = self.conv_block(x)          # -> (batch, cnn_channels, sequence_length)
        x = x.permute(0, 2, 1)          # -> (batch, sequence_length, cnn_channels)
        x, _ = self.lstm(x)             # -> (batch, sequence_length, lstm_hidden*2)
        out = self.output_layer(x)      # -> (batch, sequence_length, n_output_channels)
        return out


if __name__ == "__main__":
    model = StrainReconstructionNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {n_params:,} parameters")

    # Sanity check: verify forward pass shape with a realistic batch,
    # using a SHORT sequence here purely to keep this check fast --
    # the model is fully convolutional/recurrent, so it works at any length,
    # including the real WINDOW_LENGTH of 36000 used in training.
    dummy_input = torch.randn(2, 1000, 4)  # (batch=2, seq_len=1000, channels=4)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 1000, 8), f"Unexpected output shape: {output.shape}"
    print("Shape check passed.")
