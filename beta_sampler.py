# beta_sampler.py
import torch
import matplotlib.pyplot as plt

class BetaSampler:
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def sample(self, n: int):
        # Samples from the beta distribution with parameters a, b
        r = torch.distributions.Beta(self.a, self.b).sample((n,))

        # Samples theta uniformly
        theta = 2 * torch.pi * torch.rand(n)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack((x, y), dim=1)

    def sample_boundary(self, n: int):
        samples = self.sample(n)
        norms = samples.norm(dim=1, keepdim=True)
        return samples / norms  # project to unit circle

    def log_prob(self, z):
        r = torch.norm(z, dim=1)
        # compute log Beta function using lgamma
        log_B = torch.lgamma(torch.tensor(self.a)) + torch.lgamma(torch.tensor(self.b)) - torch.lgamma(torch.tensor(self.a + self.b))
        log_pr = (self.a - 1) * torch.log(r) + (self.b - 1) * torch.log(1 - r) - log_B
        log_theta = -torch.log(torch.tensor(2 * torch.pi))
        return log_pr + log_theta

    def plot(self, n: int = 1000):
        samples = self.sample(n)
        fig, ax = plt.subplots()
        circle = plt.Circle((0, 0), 1.0, color='gray', alpha=0.1)
        ax.add_patch(circle)
        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
        ax.set_aspect('equal')
        ax.set_title(f"Beta Disk (a={self.a}, b={self.b})")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        plt.show()

