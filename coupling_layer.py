import torch
import torch.nn as nn
import torch.nn.functional as F

class CouplingLayer(nn.Module):
    """
    2D Affine Coupling Layer for (x1, x2) -> (y1, y2)
    Forward: y_i = x_i, y_j = x_j * exp(s(x_i)) + t(x_i)
    s is normalized with tanh, t is not normalized.
    The column to transform is determined by 'index' (0 or 1)
    """

    def __init__(self, hidden_dim: int = 32, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')
        # s(x) and t(x) are small MLPs mapping 1D -> 1D
        self.s_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        ).to(self.device)
        self.t_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)

    def forward(self, x: torch.Tensor, index: int = 1):
        """
        x: (N,2)
        index: 0 or 1, the column to transform (y_j)
        Returns y, log_det_J
        """
        x = x.to(self.device)
        x_i, x_j = x[:, 1-index:2-index], x[:, index:index+1]  # i is conditioned, j is transformed
        s = self.s_net(x_i)
        t = self.t_net(x_i)
        y_j = x_j * torch.exp(s) + t
        y_i = x_i
        # reconstruct y in original order
        if index == 0:
            y = torch.cat([y_j, y_i], dim=1)
        else:
            y = torch.cat([y_i, y_j], dim=1)
        log_det_J = s.sum(dim=1)
        return y, log_det_J

    def inverse(self, y: torch.Tensor, index: int = 1):
        """
        y: (N,2)
        index: column transformed in forward
        Returns x
        """
        y = y.to(self.device)
        y_i, y_j = y[:, 1-index:2-index], y[:, index:index+1]
        s = self.s_net(y_i)
        t = self.t_net(y_i)
        x_j = (y_j - t) * torch.exp(-s)
        x_i = y_i
        if index == 0:
            x = torch.cat([x_j, x_i], dim=1)
        else:
            x = torch.cat([x_i, x_j], dim=1)
        return x

