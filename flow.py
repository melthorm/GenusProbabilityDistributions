import torch
import torch.nn as nn
import maps
import coupling_layer

class Flow(nn.Module):
    """
    Flow consisting of multiple coupling layers.
    Forward: disk -> R2 -> coupling layers -> R2 -> disk
    Computes total log-det including disk_to_R2 and R2_to_disk
    """

    def __init__(self, num_layers: int = 4, hidden_dim: int = 32):
        super(Flow, self).__init__()
        # creates the stuff in a modulelist which holds parameters special
        self.layers = nn.ModuleList([coupling_layer.CouplingLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, index_cycle=None):
        """
        x: (N,2), points in disk
        index_cycle: optional list to choose which column to transform per layer
        Returns: x_out, logdet
        """
        # map to R^2 with logdet
        y, ld_to = maps.disk_to_R2(x)
        logDet = ld_to

        # coupling layers
        for i, layer in enumerate(self.layers):
            # might be some world where we provide a cycle of indices, otherwise
            # just modulo 2 to alternate between first and second columns
            index = index_cycle[i] if index_cycle is not None else i % 2
            y, ld = layer.forward(y, index=index)
            logDet += ld

        # map back to disk with logdet
        x_out, ld_back = maps.R2_to_disk(y)
        logDet += ld_back
        # coudl also do logdet -= disk_to_R2(x_out)

        return x_out, logDet


    def inverse(self, x_out: torch.Tensor, index_cycle=None):
        """
        Apply inverse of the flow to x_out (disk points)
        """
        # Map to R^2
        y, _ = maps.disk_to_R2(x_out)

        # Apply inverse coupling layers in reverse order
        for i, layer in reversed(list(enumerate(self.layers))):
            index = index_cycle[i] if index_cycle is not None else i % 2
            y = layer.inverse(y, index=index)

        # Map back to disk
        x_recovered, _ = maps.R2_to_disk(y)
        return x_recovered

        
    def sample(self, n_samples: int, device="cpu", index_cycle=None, scale=2.0):
        """
        Generate new samples from the learned flow using a scaled base distribution.

        Steps:
        1. Sample from standard normal in R².
        2. Scale to roughly match typical R² magnitude of forward-mapped disk points.
        3. Apply inverse coupling layers in reverse order.
        4. Map back to disk.

        Args:
            n_samples: number of points to generate
            device: torch device
            index_cycle: optional list of column indices for coupling layers
            scale: scaling factor for base normal

        Returns:
            x_disk: (n_samples, 2) tensor in disk space
        """
        # Step 1 & 2: sample and scale
        z_base = torch.randn(n_samples, 2, device=device) * scale

        # Step 3: inverse through coupling layers
        x_disk = z_base
        for i, layer in reversed(list(enumerate(self.layers))):
            index = index_cycle[i] if index_cycle is not None else i % 2
            x_disk = layer.inverse(x_disk, index=index)

        # Step 4: final map to disk
        x_disk, _ = maps.R2_to_disk(x_disk)
        return x_disk


