# normal_sampler.py
import torch
import matplotlib.pyplot as plt

class NormalSampler:
    def __init__(self, sigma: float, device=None):
        """
        Samples from a 2D Gaussian N(0, sigma^2 I) truncated to the unit disk.
        """
        self.sigma = sigma
        self.device = device or torch.device('cpu')

    def sample(self, n: int):
        # apparently, the rayleigh distribution represents a 2d Gaussian in polar coordinates and is given by
        # \frac{r}{\sigma ^ 2} e ^ {\frac{-r^2}{2\sigma ^ 2}}, wikipedia no explain
        # We use inverse cdf sampling to first sample the uniform, and eventually find r by inverse CDF
        u = torch.rand(n, device=self.device)

        # compute by cdf = 1 - e ^ {-\frac{r ^ 2}{2\sigma ^ 2}}, maximum possible cdf is at r = 1 since D^2
        max_cdf = 1 - torch.exp(torch.tensor(-1.0 / (2 * self.sigma**2), device=self.device))
        
        # We do uniform = F(r) / F(1) since max r is 1, we solve for r and get the equation:
        # r = \sigma \sqrt{-2 \log {1 - uniform * (1 - e ^ \frac{-1}{2\sigma ^ 2})}}
        r = self.sigma * torch.sqrt(-2 * torch.log(1 - u * max_cdf))

        # Sample theta uniformly
        theta = 2 * torch.pi * torch.rand(n, device=self.device)

        # Convert to Cartesian coordinates
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack((x, y), dim=1)

    def sample_boundary(self, n: int):
        samples = self.sample(n)
        norms = samples.norm(dim=1, keepdim=True)
        return samples / norms  # project to unit circle

    def log_prob(self, z):
        z = z.to(self.device)
        # Gives x^2 + y^2 terms 
        r2 = torch.sum(z**2, dim=1)
        # unnormalized 2D Gaussian log-density that is found by taking the logarithm of the probability density function for Gaussian
        # Which gives ln (1 / 2pi\sigma ^ 2) + ln (e ^ -\frac{x^2 + y^2}{2 \sigma ^ 2})
        log_gaussian = - (r2 / (2 * self.sigma**2)) - torch.log(torch.tensor(2 * torch.pi * self.sigma**2, dtype=torch.float32, device=self.device))

        # normalization for truncation to unit disk which is found by finding the probability amss in unit disk
        log_norm = torch.log(1 - torch.exp(torch.tensor(-1/(2*self.sigma**2), dtype=torch.float32, device=self.device)))

        return log_gaussian - log_norm

    def plot(self, n: int = 1000):
        samples = self.sample(n).cpu()
        fig, ax = plt.subplots()
        circle = plt.Circle((0, 0), 1.0, color='gray', alpha=0.1)
        ax.add_patch(circle)
        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
        ax.set_aspect('equal')
        ax.set_title(f"Truncated Gaussian on Unit Disk (σ={self.sigma})")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        plt.show()

