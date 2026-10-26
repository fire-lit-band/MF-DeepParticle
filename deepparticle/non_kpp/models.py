"""The manuscript's current d+1, three-hidden-layer SELU DP corrector."""
import torch
from torch import nn


class CompactCorrector(nn.Module):
    def __init__(self, dimension=3, width=64, hidden_layers=3, alpha=.1):
        super().__init__()
        self.alpha = alpha
        self.architecture = {'kind': 'position_condition_dp', 'dimension': dimension,
            'width': width, 'hidden_layers': hidden_layers, 'alpha': alpha,
            'input_features': dimension + 1, 'activation': 'SELU', 'blocks': 0,
            'head_initialization': 'zero', 'constant_condition': 0.}
        layers = [nn.Linear(dimension + 1, width), nn.SELU()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(width, width), nn.SELU()))
        layers.append(nn.Linear(width, dimension))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, condition):
        return x + self.alpha * self.net(torch.cat((x, condition), -1))
