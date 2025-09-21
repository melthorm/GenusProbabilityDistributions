import torch
import numpy as np
import matplotlib.pyplot as plt
from TorusNoIntersectSampler import TorusNoIntersectSampler

class Genus2TorusSampler:
    def __init__(self, R, r,
                 mu_list=[(0.0,0.0),(np.pi,np.pi)],
                 kappa_list=[(4.0,4.0),(4.0,4.0)],
                 remove_radii=[0.5], torus_spacing=0.5):
        """
        Continuous genus-2 torus sampler using partition-of-unity blending.

        Args:
            R, r: torus radii
            mu_list: [(mu_theta1, mu_phi1), (mu_theta2, mu_phi2)]
            kappa_list: [(kappa_theta1, kappa_phi1), (kappa_theta2, kappa_phi2)]
            remove_radii: radius of removed disks for each torus
            torus_spacing: spacing along x-axis between tori
        """
        self.R = R
        self.r = r
        self.torus_spacing = torus_spacing

        self.samplers = []
        for i in range(2):
            mu_theta, mu_phi = mu_list[i]
            kappa_theta, kappa_phi = kappa_list[i]
            x_offset = i * (2*R + torus_spacing)  # offset along x-axis for the second torus
            sampler = TorusNoIntersectSampler(
                x_offset=x_offset,
                R=R, r=r,
                mu1=mu_theta, kappa1=kappa_theta,
                mu2=mu_phi, kappa2=kappa_phi
            )
            self.samplers.append(sampler)

    def compute_weights(self, s):
        """
        Soft partition-of-unity weights along stitched axis.
        s: continuous axis coordinate in [0,1] (normalized)
        """
        centers = torch.tensor([0.25, 0.75])
        sigma = 0.2
        w = torch.exp(-0.5 * ((s.unsqueeze(1) - centers)**2) / sigma**2)
        w = w / w.sum(dim=1, keepdim=True)
        return w  # shape [n_samples, 2]

    def sample(self, n_samples):
        """
        Sample points continuously across the genus-2 torus.
        """
        s = torch.rand(n_samples)
        weights = self.compute_weights(s)

        points = []
        for i in range(2):
            probs = weights[:, i]
            mask = torch.bernoulli(probs).bool()
            count = mask.sum().item()
            if count == 0:
                continue
            pts = self.samplers[i].sample_points(count)
            points.append(pts)

        return torch.cat(points, dim=0)

    def plot(self, n_samples=2000, s=3, show_surface=True):
        """
        Plot genus-2 torus with sampled points and optional torus surfaces.
        """
        points = self.sample(n_samples).cpu().numpy()
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')

        # Torus surface mesh
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, 2*np.pi, 25)
        U, V = np.meshgrid(u, v)
        X_base = (self.R + self.r*np.cos(U)) * np.cos(V)
        Y_base = (self.R + self.r*np.cos(U)) * np.sin(V)
        Z_base = self.r * np.sin(U)

        if show_surface:
            for sampler in self.samplers:
                ax.plot_surface(
                    X_base + sampler.x_offset, Y_base, Z_base,
                    color='lightblue', alpha=0.3, rstride=2, cstride=2
                )

        # Sampled points
        ax.scatter(points[:,0], points[:,1], points[:,2], s=s, c='red', alpha=0.7)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Genus-2 Torus with Partition-of-Unity von Mises Sampling')
        plt.show()

