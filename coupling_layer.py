import torch
import torch.nn as nn
import torch.nn.functional as F

class CouplingLayer(nn.Module):
    """
    2D Affine Coupling Layer for (x1, x2) -> (y1, y2)
    Forward: y1 = x1, y2 = x2 * exp(s(x1)) + t(x1)
    s is normalized with tanh, t is not normalized.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        # s(x1) and t(x1) are small MLPs mapping 1D -> 1D
        self.s_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(), # Relu for first thing
            nn.Linear(hidden_dim, 1),
            nn.Tanh()   # normalize s with tanh
        )
        self.t_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
            # t is not normalized
        )

    def forward(self, x: torch.Tensor):
        """
        x: (N,2)
        Returns y, log_det_J
        """
        x1, x2 = x[:,0:1], x[:,1:2]
        s = self.s_net(x1)
        t = self.t_net(x1)
        y1 = x1
        y2 = x2 * torch.exp(s) + t
        y = torch.cat([y1, y2], dim=1)
        log_det_J = s.sum(dim=1)
        return y, log_det_J

    def inverse(self, y: torch.Tensor):
        """
        y: (N,2)
        Returns x
        """
        y1, y2 = y[:,0:1], y[:,1:2]
        s = self.s_net(y1)  # forward s uses tanh already
        t = self.t_net(y1)
        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)
        x = torch.cat([x1, x2], dim=1)
        return x


def f(x: torch.Tensor, layer: CouplingLayer):
    """
    Apply coupling layer forward to Nx2 tensor x.
    Returns transformed Nx2 tensor.
    """
    y, _ = layer.forward(x)
    return y


