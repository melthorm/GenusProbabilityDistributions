import torch
import matplotlib.pyplot as plt
from normal_sampler import NormalSampler
from beta_sampler import BetaSampler

class MixtureSampler:
    def __init__(self, alpha: float, sigma: float, a: float, b: float):
        self.alpha = alpha
        self.normal_sampler = NormalSampler(sigma)
        self.beta_sampler = BetaSampler(a, b)

    def sample(self, n: int):
        # n samples from each distribution type: Truncated Gaussian + Beta
        samples_N = self.normal_sampler.sample(n)
        samples_Beta = self.beta_sampler.sample(n)
        # For each index, select either Truncated Gaussian or Beta
        mix_mask = torch.bernoulli(torch.full((n, 1), self.alpha))
        # Stack and return those samples
        return mix_mask * samples_N + (1 - mix_mask) * samples_Beta

    def sample_boundary(self, n: int):
        samples = self.sample(n)
        norms = samples.norm(dim=1, keepdim=True)
        return samples / norms

    def plot(self, n: int = 1000):
        samples = self.sample(n)
        fig, ax = plt.subplots()
        circle = plt.Circle((0, 0), 1.0, color='gray', alpha=0.1)
        ax.add_patch(circle)
        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
        ax.set_aspect('equal')
        ax.set_title(f"Mixture (α={self.alpha}, σ={self.normal_sampler.sigma}, a={self.beta_sampler.a}, b={self.beta_sampler.b})")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        plt.show()

