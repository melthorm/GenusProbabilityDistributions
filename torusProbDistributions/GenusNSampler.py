import torch
import numpy as np
import matplotlib.pyplot as plt
from NormalTorusSampler import NormalTorusSampler

def torus_intersection_mask(theta, phi, R, r, x_offset):
    """
    Boolean mask for points inside a torus centered at x_offset along x-axis.
    """
    x = (R + r*torch.cos(theta)) * torch.cos(phi)
    y = (R + r*torch.cos(theta)) * torch.sin(phi)
    z = r * torch.sin(theta)
    rho = torch.sqrt((x - x_offset)**2 + y**2)
    return ((rho - R)**2 + z**2) <= r**2


class GenusNSampler(NormalTorusSampler):
    def __init__(self, n, R=2.0, r=0.7, mu_list=None, kappa_list=None, x_spacing=1.0, sigma=0.2, intersect_map=None, *args, **kwargs):
        """
        Continuous genus-n torus sampler directly from NormalTorusSampler.

        Args:
            n (int): number of tori
            R, r (float): torus radii
            mu_list: list of (mu_theta, mu_phi) per torus
            kappa_list: list of (kappa_theta, kappa_phi) per torus
            x_spacing: spacing along x-axis
            sigma: width for PoU blending
            intersect_map: list of lists, intersect_map[i] = list of neighbor x_offsets to avoid for torus i
        """
        super().__init__(R=R, r=r, *args, **kwargs)
        self.n = n
        self.R = R
        self.r = r
        self.x_offsets = np.arange(n) * x_spacing
        self.sigma = sigma

        if mu_list is None:
            mu_list = [(np.pi/4, np.pi/4)] * n
        if kappa_list is None:
            kappa_list = [(4.0, 4.0)] * n

        # Create NormalTorusSampler for each torus
        self.samplers = []
        for i in range(n):
            mu_theta, mu_phi = mu_list[i]
            kappa_theta, kappa_phi = kappa_list[i]
            sampler = NormalTorusSampler(R=R, r=r, mu1=mu_theta, kappa1=kappa_theta, mu2=mu_phi, kappa2=kappa_phi)
            self.samplers.append(sampler)

        # intersect_map[i] = list of x_offsets of other tori to avoid
        self.intersect_map = intersect_map if intersect_map else [[] for _ in range(n)]

    def compute_weights(self, s):
        """
        PoU blending weights along normalized axis s in [0,1].
        """
        centers = torch.linspace(0.0, 1.0, self.n)
        s = s.unsqueeze(1)
        w = torch.exp(-0.5 * ((s - centers)**2) / self.sigma**2)
        w = w / w.sum(dim=1, keepdim=True)
        return w

    def sample(self, n_samples):
        s = torch.rand(n_samples)
        weights = self.compute_weights(s)
        points = []

        for i, sampler in enumerate(self.samplers):
            probs = weights[:, i]
            mask = torch.bernoulli(probs).bool()
            count = mask.sum().item()
            if count == 0:
                continue

            # Sample points for this torus
            angles = sampler.sample_angles(count)  # shape [count,2]
            theta, phi = angles

            # Remove points intersecting any neighbor tori in intersect_map[i]
            if self.intersect_map[i]:
                intersect_mask = torch.zeros(count, dtype=torch.bool)
                for x_off in self.intersect_map[i]:
                    intersect_mask |= torus_intersection_mask(theta, phi, self.R, self.r, x_off)
                keep_mask = ~intersect_mask
                theta, phi = theta[keep_mask], phi[keep_mask]
                if theta.numel() == 0:
                    continue  # skip if all points rejected

            # Convert to 3D
            x = (self.R + self.r*torch.cos(theta)) * torch.cos(phi) + self.x_offsets[i]
            y = (self.R + self.r*torch.cos(theta)) * torch.sin(phi)
            z = self.r * torch.sin(theta)
            points.append(torch.stack([x,y,z], dim=1))

        return torch.cat(points, dim=0)

    def plot(self, n_samples=2000, s=3, show_surface=True):
        pts = self.sample(n_samples).cpu().numpy()
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')

        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, 2*np.pi, 25)
        U, V = np.meshgrid(u, v)
        X_base = (self.R + self.r*np.cos(U)) * np.cos(V)
        Y_base = (self.R + self.r*np.cos(U)) * np.sin(V)
        Z_base = self.r * np.sin(U)

        if show_surface:
            for x_off in self.x_offsets:
                ax.plot_surface(X_base + x_off, Y_base, Z_base, color='lightblue', alpha=0.3)

        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=s, c='red', alpha=0.7)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'Genus-{self.n} Torus with Continuous PoU Sampling')
        plt.show()

