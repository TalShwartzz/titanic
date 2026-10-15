"""Compact MLP model for small tabular datasets."""

from __future__ import annotations

import torch
from torch import nn


class TitanicMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (32, 16), dropout: float = 0.15):
        super().__init__()
        if input_dim < 1 or not hidden_dims or any(n < 1 for n in hidden_dims):
            raise ValueError("Invalid neural network dimensions")
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers += [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)]
            previous = width
        layers.append(nn.Linear(previous, 1))  # logits; sigmoid applied only for prediction
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
