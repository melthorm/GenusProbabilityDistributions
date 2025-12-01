import torch
import matplotlib.pyplot as plt
from normal_sampler import NormalSampler
from beta_sampler import BetaSampler

class MixtureSampler:
    def __init__(self, probabilityChoose: float, sigma: float, a: float, b: float, device=None):
        self.probabilityChoose = probabilityChoose
        self.device = device or torch.device('cpu')
        self.normal_sampler = NormalSampler(sigma, device=self.device)
        self.beta_sampler = BetaSampler(a, b, device=self.device)

    def sample(self, n: int):
        # Decide which distribution each sample comes from
        mix_mask = torch.bernoulli(torch.full((n, 1), self.probabilityChoose, device=self.device))  # 1 = Normal, 0 = Beta
        mix_mask = mix_mask.bool().squeeze(-1)  # convert to boolean mask for indexing

        # Preallocate output tensor
        samples = torch.empty((n, 2), dtype=torch.float32, device=self.device)

        # Sample only from Normal where mix_mask is True
        n_normal = mix_mask.sum().item()
        if n_normal > 0:
            samples[mix_mask] = self.normal_sampler.sample(n_normal)

        # Sample only from Beta where mix_mask is False
        n_beta = (~mix_mask).sum().item()
        if n_beta > 0:
            samples[~mix_mask] = self.beta_sampler.sample(n_beta)

        return samples

    def sample_boundary(self, n: int):
        # projects onto unit circle byt aking unti vector
        samples = self.sample(n)
        norms = samples.norm(dim=1, keepdim=True)
        return samples / norms

    def log_prob(self, z):
        z = z.to(self.device)
        log_N = self.normal_sampler.log_prob(z)  # (n,)
        log_B = self.beta_sampler.log_prob(z)    # (n,)
        # mixture log-prob
        log_mix = torch.log(self.probabilityChoose * torch.exp(log_N) + (1 - self.probabilityChoose) * torch.exp(log_B))
        return log_mix

    def plot(self, n: int = 1000):
        samples = self.sample(n).cpu()
        fig, ax = plt.subplots()
        circle = plt.Circle((0, 0), 1.0, color='gray', alpha=0.1)
        ax.add_patch(circle)
        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
        ax.set_aspect('equal')
        ax.set_title(f"Mixture (α={self.probabilityChoose}, σ={self.normal_sampler.sigma}, a={self.beta_sampler.a}, b={self.beta_sampler.b})")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        plt.show()

