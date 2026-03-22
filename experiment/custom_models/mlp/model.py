import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual block
        residual = x
        x = self.norm(x)
        x = torch.relu(self.linear(x))
        x = self.dropout(x)
        return x + residual


class MLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_blocks: int,
        hidden_dim: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__()
        self.input_layer = nn.Linear(num_features, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        self.blocks = nn.ModuleList([
            MLPBlock(hidden_dim, dropout)
            for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(hidden_dim, num_classes)  # num_classes=1 for regression
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_norm(self.input_layer(x)))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)