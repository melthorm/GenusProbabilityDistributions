import torch
from torch.distributions import VonMises
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class NormalTorusSampler:
    def __init__(self, R = 2.0, r = 1.0,
                       mu1 = 0.0, kappa1 = 4.0,
                       mu2 = 0.0, kappa2 = 4.0,
                       device = "cpu"):
        """
            initializes a torus to sample,
                R - major radius
                r - minor radius
                mu1 - mean angle (preferred angle) for S1 that defines major radius
                kappa1 - concentration for mu1 of S1 that defines major radius
                mu2 - mean angle (preferred angle) for S1 that defines minor radius
                kappa2 - concentration for mu2 of S1 that defines minor radius
                device - device to use (cpu, gpu)
        """
        self.R = R;
        self.r = r;
        self.mu1 = mu1;
        self.kappa1 = kappa1;
        self.mu2 = mu2;
        self.kappa2 = kappa2;
        
        self.device = device;


        self.dist1 = VonMises(torch.tensor(mu1, device=device), 
                                torch.tensor(kappa1, device=device))
        self.dist2 = VonMises(torch.tensor(mu2, device=device), 
                                torch.tensor(kappa2, device=device))

    def sample_angles(self, n_samples):
        """Sample (theta1, theta2) from S1 x S1."""
        theta = self.dist1.sample((n_samples,))
        phi = self.dist2.sample((n_samples,))
        return theta, phi

    def sample_points(self, n_samples):
        """Return samples embedded in R^3 as (x, y, z)."""
        theta, phi = self.sample_angles(n_samples)

        x = (self.R + self.r * torch.cos(phi)) * torch.cos(theta)
        y = (self.R + self.r * torch.cos(phi)) * torch.sin(theta)
        z = self.r * torch.sin(phi)

        return torch.stack([x, y, z], dim = 1)

    def plot_samples(self, n_samples = 1000, s = 5, 
                           show_surface = False, return_points = False):
        """
        Plot sampled points on the torus in 3D.
        
        Args:
            n_samples - number of sampled points
            s - point size for scatter
            show_surface - whether to also plot the torus surface mesh
            return_points - whether to return sampled points
        
        Returns:
            torch.Tensor or None: (n_samples, 3) points if return_points = True
        """
        pts = self.sample_points(n_samples).cpu().numpy()

        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection="3d")

        # Optional: plot underlying torus surface mesh
        if show_surface:
            u = np.linspace(0, 2*np.pi, 60)
            v = np.linspace(0, 2*np.pi, 30)
            U, V = np.meshgrid(u, v)
            X = (self.R + self.r*np.cos(V)) * np.cos(U)
            Y = (self.R + self.r*np.cos(V)) * np.sin(U)
            Z = self.r * np.sin(V)

            ax.plot_surface(X, Y, Z, color="lightblue", alpha=0.3, rstride=2, cstride=2, linewidth=0)

        # Plot sampled points
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=s, c="red", alpha=0.7)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Torus samples (von Mises)")

        plt.show()

        return torch.tensor(pts) if return_points else None
