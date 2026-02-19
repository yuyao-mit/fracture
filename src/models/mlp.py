
# https://www.nature.com/articles/323533a0

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        hidden_dim: int = 1024,
        num_layers: int = 10,
    ):
        super().__init__()

        _, T_in, C_in, H, W = input_shape
        _, T_out, C_out, H_out, W_out = output_shape

        assert H == H_out and W == W_out

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        self.T_in = T_in
        self.C_in = C_in
        self.H = H
        self.W = W

        self.T_out = T_out
        self.C_out = C_out

        input_dim = T_in * C_in * H * W
        output_dim = T_out * C_out * H * W

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.view(B, -1)
        x = self.mlp(x)
        x = x.view(B, self.T_out, self.C_out, self.H, self.W)
        return x
