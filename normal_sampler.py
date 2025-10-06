import torch
import matplotlib.pyplot as plt

class NormalSampler:
    def __init__(self, sigma: float):
        """
        Samples from a 2D Gaussian N(0, sigma^2 I) truncated to unit Disk
        """
        self.sigma = sigma

    def sample(self, n: int):
        # Draws uniform distributions for radius and theta values
        u = torch.rand(n)
        theta = 2 * torch.pi * torch.rand(n)

        # Computes the maximum possible cdf values
        max_cdf = 1 - torch.exp(torch.tensor(-1.0 / (2 * self.sigma ** 2)))

        # Inverts CDF to obtain radius values
        r = self.sigma * torch.sqrt(-2 * torch.log(1 - u * max_cdf))

        # Theta is just a uniform distribution, so we get x y
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
        ax.set_title(f"Truncated Gaussian on Unit Disk (σ={self.sigma})")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        plt.show()

