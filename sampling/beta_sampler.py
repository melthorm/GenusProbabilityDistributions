# beta_sampler.py
import torch
import matplotlib.pyplot as plt

class BetaSampler:
    def __init__(self, a: float, b: float, device=None):
        self.a = a
        self.b = b
        self.device = device or torch.device('cpu')

    def sample(self, n: int):
        # Samples from the beta distribution with parameters a, b
        r = torch.distributions.Beta(self.a, self.b).sample((n,)).to(self.device)

        # Samples theta uniformly
        theta = 2 * torch.pi * torch.rand(n, device=self.device)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack((x, y), dim=1)

    def sample_boundary(self, n: int):
        samples = self.sample(n)
        norms = samples.norm(dim=1, keepdim=True)
        return samples / norms  # project to unit circle

    def log_prob(self, z):
        z = z.to(self.device)
        # finds r values of points in R^2
        r = torch.norm(z, dim=1) 
        # (\alpha - 1) * log(z) + (beta - 1) * log(1-x) - lgamma(alpha) - lgamma(beta) + lgamma(alpha + beta) is full formula
        
        # finds log beta function
        log_B = torch.lgamma(torch.tensor(self.a, device=self.device)) + \
                torch.lgamma(torch.tensor(self.b, device=self.device)) - \
                torch.lgamma(torch.tensor(self.a + self.b, device=self.device))

        # actual values to find 'radial beta density'
        log_pr = (self.a - 1) * torch.log(r) + (self.b - 1) * torch.log(1 - r)

        # jacobian factor, do not understand this, is composed of a uniform angle and r
        log_jacobian = torch.log(torch.tensor(2 * torch.pi, device=self.device)) + torch.log(r)
        return log_pr - log_B - log_jacobian

    def plot(self, n: int = 1000):
        samples = self.sample(n).cpu()
        fig, ax = plt.subplots()
        circle = plt.Circle((0, 0), 1.0, color='gray', alpha=0.1)
        ax.add_patch(circle)
        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
        ax.set_aspect('equal')
        ax.set_title(f"Beta Disk (a={self.a}, b={self.b})")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        plt.show()

