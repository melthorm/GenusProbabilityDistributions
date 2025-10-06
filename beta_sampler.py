import torch
import matplotlib.pyplot as plt

class BetaSampler:
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def sample(self, n: int):
        # Samples from the beta distribution with a b parameters
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

