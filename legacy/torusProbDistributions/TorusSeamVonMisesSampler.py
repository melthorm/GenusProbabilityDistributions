import numpy as np
import torch
from torch.distributions import VonMises
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TorusSeamVonMisesSampler:
    """
    Samples points along the seam of two tori (R=2r, D=2R) using
    von Mises distributions on the y-coordinate of the seam branches
    to produce a non-uniform PDF along the seam.
    """
    def __init__(self, R=2.0, r=1.0, num_points=1000,
                 mu_top=0.0, kappa_top=4.0,
                 mu_bottom=0.0, kappa_bottom=4.0,
                 device="cpu"):
        self.R = R
        self.r = r
        self.num_points = num_points
        self.device = device

        # von Mises distributions for top and bottom branches
        self.dist_top = VonMises(torch.tensor(mu_top, device=device),
                                 torch.tensor(kappa_top, device=device))
        self.dist_bottom = VonMises(torch.tensor(mu_bottom, device=device),
                                    torch.tensor(kappa_bottom, device=device))

        self._compute_seam()
        self._compute_pdf()

    def _compute_seam(self):
        epsilon = 1e-6
        y_min = -np.sqrt((self.R + self.r)**2 - self.R**2) + epsilon
        y_max = np.sqrt((self.R + self.r)**2 - self.R**2) - epsilon
        y_vals = np.linspace(y_min, y_max, self.num_points)

        radicand = -y_vals**2 - 7 + 4*np.sqrt(y_vals**2 + 4)
        z_top = np.sqrt(radicand)
        z_bottom = -np.sqrt(radicand)

        self.y_full = np.concatenate([y_vals, y_vals])
        self.z_full = np.concatenate([z_top, z_bottom])
        self.x_full = np.full_like(self.y_full, self.R)

        self.y_top = torch.tensor(y_vals, device=self.device)
        self.y_bottom = torch.tensor(y_vals, device=self.device)

    def _compute_pdf(self):
        # Evaluate von Mises PDFs on top and bottom branches
        p_top = torch.exp(self.dist_top.log_prob(self.y_top))
        p_bottom = torch.exp(self.dist_bottom.log_prob(self.y_bottom))
        p_full = torch.cat([p_top, p_bottom])

        # Arc-length derivatives
        phi_vals = np.linspace(0, 2*np.pi, self.num_points)
        dz_dphi_top = np.gradient(np.sqrt(-self.y_full[:self.num_points]**2 - 7 + 4*np.sqrt(self.y_full[:self.num_points]**2 + 4)), phi_vals)
        dz_dphi_bottom = np.gradient(-np.sqrt(-self.y_full[:self.num_points]**2 - 7 + 4*np.sqrt(self.y_full[:self.num_points]**2 + 4)), phi_vals)
        dz_dphi_full = np.concatenate([dz_dphi_top, dz_dphi_bottom])
        dy_dphi_full = np.gradient(self.y_full, np.linspace(0, 4*np.pi, len(self.y_full)))
        dl_dphi = np.sqrt(dy_dphi_full**2 + dz_dphi_full**2)
        dl_dphi = torch.tensor(dl_dphi, device=self.device)

        # Induced PDF along seam
        pdf = p_full * dl_dphi
        self.pdf = pdf / torch.sum(pdf)
        self.cdf = torch.cumsum(self.pdf, dim=0)

    def sample(self, n_samples=1000):
        u = torch.rand(n_samples, device=self.device)
        indices = torch.searchsorted(self.cdf, u)
        x = self.x_full[indices.cpu().numpy()]
        y = self.y_full[indices.cpu().numpy()]
        z = self.z_full[indices.cpu().numpy()]
        return x, y, z

    def plot(self, n_samples=1000, show_seam=True):
        # Torus surfaces for reference
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, 2*np.pi, 100)
        theta, phi = np.meshgrid(theta, phi)
        x1 = (self.R + self.r * np.cos(theta)) * np.cos(phi)
        y1 = (self.R + self.r * np.cos(theta)) * np.sin(phi)
        z1 = self.r * np.sin(theta)
        x2 = 2*self.R + (self.R + self.r * np.cos(theta)) * np.cos(phi)
        y2 = (self.R + self.r * np.cos(theta)) * np.sin(phi)
        z2 = self.r * np.sin(theta)

        sample_x, sample_y, sample_z = self.sample(n_samples)

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x1, y1, z1, color='cyan', alpha=0.05)
        ax.plot_surface(x2, y2, z2, color='orange', alpha=0.05)
        if show_seam:
            ax.plot(self.x_full, self.y_full, self.z_full, 'r', linewidth=2, label='Seam')
        ax.scatter(sample_x, sample_y, sample_z, color='blue', s=50, label='Sampled Points')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,1])
        ax.legend()
        plt.show()




