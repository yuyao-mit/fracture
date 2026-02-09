
# https://www.nature.com/articles/323533a0

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        hidden_dim: int = 1024,
        num_layers: int = 10,
    ):
        super().__init__()

        input_dim = 5 * channels * height * width  # 5 input frames concatenated
        output_dim = channels * height * width     # one predicted frame

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2 (1 hidden + 1 output layer)")

        layers = []

        # First layer: input to hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # Middle hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # Final layer: hidden to output
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)
        self.channels = channels
        self.height = height
        self.width = width

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization for all Linear layers
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = x.view(b, -1)
        x = self.mlp(x)
        x = x.view(b, self.channels, self.height, self.width)
        x = x.unsqueeze(1)  # Add time dimension: [B, 1, C, H, W]
        return x
