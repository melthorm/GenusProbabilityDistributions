import torch
import torch.nn as nn
from maps import disk_to_R2, R2_to_disk
from coupling_layer import CouplingLayer

class Flow(nn.Module):
    """
    Flow consisting of multiple coupling layers.
    Forward: disk -> R2 -> coupling layers -> R2 -> disk
    Computes total log-det including disk_to_R2 and R2_to_disk
    """

    def __init__(self, num_layers: int = 4, hidden_dim: int = 32, device=None):
        super(Flow, self).__init__()
        self.device = device or torch.device('cpu')
        # creates the stuff in a modulelist which holds parameters special
        self.layers = nn.ModuleList([CouplingLayer(hidden_dim, device=self.device) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, index_cycle=None):
        """
        x: (N,2), points in disk
        index_cycle: optional list to choose which column to transform per layer
        Returns: x_out, logdet
        """
        x = x.to(self.device)
        # map to R^2 with logdet
        y, ld_to = disk_to_R2(x, device=self.device)
        logDet = ld_to

        # coupling layers
        for i, layer in enumerate(self.layers):
            # might be some world where we provide a cycle of indices, otherwise
            # just modulo 2 to alternate between first and second columns
            index = index_cycle[i] if index_cycle is not None else i % 2
            y, ld = layer.forward(y, index=index)
            logDet += ld

        # map back to disk with logdet
        x_out, ld_back = R2_to_disk(y, device=self.device)
        logDet += ld_back
        # coudl also do logdet -= disk_to_R2(x_out)

        return x_out, logDet

    def inverse(self, x_out: torch.Tensor, index_cycle=None):
        """
        Apply inverse of the flow to x_out (disk points)
        """
        x_out = x_out.to(self.device)
        # Map to R^2
        y, _ = disk_to_R2(x_out, device=self.device)

        # Apply inverse coupling layers in reverse order
        for i, layer in reversed(list(enumerate(self.layers))):
            index = index_cycle[i] if index_cycle is not None else i % 2
            y = layer.inverse(y, index=index)

        # Map back to disk
        x_recovered, _ = R2_to_disk(y, device=self.device)
        return x_recovered

def flow_loss(flow, base, target, n):
    # Sample from base
    z = base.sample(n).to(flow.device)           # (n, 2)
    log_pz = base.log_prob(z)    # (n,)

    # Pushforward through flow
    x_out, logDet = flow.forward(z)   # (n,2), (n,)

    # Compute pushforward log-density
    log_q = log_pz - logDet

    # Evaluate target log-density at pushed points
    log_target = target.log_prob(x_out)

    # KL divergence: E_q[log q - log target] ≈ mean over samples
    kl = torch.mean(log_q - log_target)
    return kl

