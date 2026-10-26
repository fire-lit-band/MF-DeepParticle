"""An independently weighted DP map with exactly the MeanFlow MLP backbone."""
from __future__ import annotations

import math
import torch
from torch import nn
from particle_pipeline.models import MeanFlow


class MatchedCorrector(nn.Module):
    def __init__(self, dimension, *, width=64, hidden_layers=3, alpha=.1,
                 condition_mode="protocol", zero_head=True):
        super().__init__()
        self.dimension, self.width = dimension, width
        self.hidden_layers, self.alpha = hidden_layers, alpha
        self.condition_mode = condition_mode
        self.net = MeanFlow(dimension, width, condition_mode, hidden_layers).net
        if zero_head:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, condition):
        # The shared DP trainer supplies the protocol's scalar condition. KS MF
        # uses log10(sigma), so translate its centered logarithm to that feature.
        if self.condition_mode == "log10_sigma":
            condition = condition + math.log10(15.)
        one, zero = torch.ones_like(condition), torch.zeros_like(condition)
        correction = self.net(torch.cat((x, one, zero, condition), dim=-1))
        return x + self.alpha * correction


def describe(dimension, alpha, condition_mode):
    return {"kind": "mf_matched_dp", "dimension": dimension, "width": 64,
            "hidden_layers": 3, "activation": "SELU", "alpha": alpha,
            "condition_mode": condition_mode, "input_order": ["x", "t=1", "r=0", "condition"],
            "head_initialization": "zero", "internal_residual_blocks": 0,
            "layer_norms": 0, "periodic_position_encoding": False,
            "output": "x + alpha * MLP(x,1,0,condition)",
            "KS_feature_note": "centered log10(sigma/15) converted to log10(sigma); float32 rounding may differ"}
