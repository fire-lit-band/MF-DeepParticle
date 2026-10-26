"""Position/condition-only DP correction, with no constant time channels."""
from __future__ import annotations

import math
import torch
from torch import nn


class CompactCorrector(nn.Module):
    def __init__(self, dimension, *, width=64, hidden_layers=3, alpha=10.,
                 condition_mode="protocol", zero_head=True):
        super().__init__()
        self.dimension, self.width = dimension, width
        self.hidden_layers, self.alpha = hidden_layers, alpha
        self.condition_mode = condition_mode
        layers = [nn.Linear(dimension + 1, width), nn.SELU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(width, width), nn.SELU()])
        layers.append(nn.Linear(width, dimension))
        self.net = nn.Sequential(*layers)
        if zero_head:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, condition):
        if self.condition_mode == "log10_sigma":
            condition = condition + math.log10(15.)
        return x + self.alpha * self.net(torch.cat((x, condition), dim=-1))


def describe(dimension=2, alpha=10.):
    return {"kind": "position_condition_dp", "dimension": dimension, "width": 64,
            "hidden_layers": 3, "activation": "SELU", "alpha": alpha,
            "condition_mode": "protocol", "input_order": ["x", "condition"],
            "input_features": dimension + 1, "head_initialization": "zero",
            "internal_residual_blocks": 0, "layer_norms": 0,
            "periodic_position_encoding": False,
            "output": "x + alpha * MLP(x,condition)"}
