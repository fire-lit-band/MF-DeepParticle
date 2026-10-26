"""The existing MF identity and near-identity DP construction in any supported dimension."""
from __future__ import annotations
import math
import torch
from torch import nn


class MeanFlow(nn.Module):
    def __init__(self, dimension, width=64, condition_mode="protocol", hidden_layers=3):
        super().__init__()
        if min(dimension, width, hidden_layers) < 1:
            raise ValueError("MF dimension, width and hidden-layer count must be positive")
        self.condition_mode = condition_mode
        self.width, self.hidden_layers = width, hidden_layers
        layers = [nn.Linear(dimension + 3, width), nn.SELU()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(width, width), nn.SELU()))
        layers.append(nn.Linear(width, dimension))
        self.net = nn.Sequential(*layers)

    def forward(self, x, r, t, condition):
        return self.net(torch.cat((x, t, r, condition), -1))


class Corrector(nn.Module):
    def __init__(self, dimension, periodic, width=64, blocks=2, alpha=.1):
        super().__init__()
        self.periodic, self.alpha = periodic, alpha
        self.input = nn.Linear((2 * dimension if periodic else dimension) + 1, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                                nn.SiLU(), nn.Linear(width, width)) for _ in range(blocks))
        self.output = nn.Linear(width, dimension)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, condition):
        position = torch.cat((x.sin(), x.cos()), -1) if self.periodic else x
        hidden = self.input(torch.cat((position, condition), -1))
        for block in self.blocks:
            hidden = hidden + .2 * block(hidden)
        return x + self.alpha * self.output(hidden)


@torch.no_grad()
def generate(method, source, condition, protocol, mf=None, dp=None, mf_condition=None):
    if method not in ("mf", "dp", "mf_dp"):
        raise ValueError("Unknown learned initialization")
    result = source
    if method in ("mf", "mf_dp"):
        if mf is None:
            raise ValueError("MeanFlow model is missing")
        r, t = torch.zeros_like(condition), torch.ones_like(condition)
        result = result - mf(result, r, t, condition if mf_condition is None else mf_condition)
        if protocol.periodic:
            result = torch.remainder(result, 2 * math.pi)
    if method in ("dp", "mf_dp"):
        if dp is None:
            raise ValueError("Corrector model is missing")
        result = dp(result, condition)
    return torch.remainder(result, 2 * math.pi) if protocol.periodic else result
