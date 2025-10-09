import torch

def disk_to_R2(x: torch.Tensor):
    """
    x: (N,2), points in unit disk ||x|| <= 1
    Returns y in R^2 and logdet of the Jacobian
    """
    norms = x.norm(dim=1, keepdim=True)
    y = x / (1 - norms)
    logdet = -2 * torch.log(1 - norms).squeeze(1)
    return y, logdet

def R2_to_disk(y: torch.Tensor):
    """
    y: (N,2), points in R^2
    Returns x in unit disk and logdet of the Jacobian
    """
    norms = y.norm(dim=1, keepdim=True)
    x = y / (1 + norms)
    logdet = -2 * torch.log(1 + norms).squeeze(1)
    return x, logdet

