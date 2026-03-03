"""
PINN Model for Inverse Heat Source Identification
Architecture: fully-connected network with tanh activations
Outputs: T(x) — temperature field
The source term f(x) is a *trainable parameter field* (discrete or network-based)
"""

import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Physics-Informed Neural Network.
    Input  : x in [0, 1]  (1D domain)
    Output : T(x)          (temperature)
    """

    def __init__(self, layers: list[int] = [1, 64, 64, 64, 1], activation=nn.Tanh()):
        super().__init__()
        self.activation = activation
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                net.append(activation)
        self.net = nn.Sequential(*net)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SourceNetwork(nn.Module):
    """
    Auxiliary network that parametrises the unknown source f(x).
    Separate network so we can control its complexity independently.
    """

    def __init__(self, layers: list[int] = [1, 32, 32, 1]):
        super().__init__()
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                net.append(nn.Tanh())
        self.net = nn.Sequential(*net)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
