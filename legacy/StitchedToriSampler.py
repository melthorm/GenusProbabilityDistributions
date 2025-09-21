import TorusNoDiskSampler
import torch
from torch.distributions import VonMises
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt

class StitchedToriSampler:
    def __init__(self, num_tori, R, r, torus_spacing=0.5,
                 mu_theta=0.0, mu_phi=0.0, mu_s=0.0,
                 kappa_theta=4.0, kappa_phi=4.0, kappa_s=2.0,
                 decay=0.5):
        """
        Continuous stitched tori sampler using von Mises bumps over (theta, phi, s).

        Args:
            num_tori: number of tori to stitch
            R, r: torus radii
            torus_spacing: spacing along x-axis between tori
            mu_theta, mu_phi: base mean directions for angles
            mu_s: mean torus index (continuous)
            kappa_theta, kappa_phi, kappa_s: base concentrations
            decay: how quickly angular kappas decay with |s - mu_s|
        """
        self.num_tori = num_tori
        self.R = R
        self.r = r
        self.torus_spacing = torus_spacing

        self.mu_theta = mu_theta
        self.mu_phi = mu_phi
        self.mu_s = mu_s
        self.kappa_theta = kappa_theta
        self.kappa_phi = kappa_phi
        self.kappa_s = kappa_s
        self.decay = decay

    def sample_axis(self, n_samples):
        """Sample torus axis positions s from [0, num_tori)."""
        vm = torch.distributions.VonMises(
            loc=torch.tensor(self.mu_s),
            concentration=torch.tensor(self.kappa_s)
        )
        s = vm.sample((n_samples,))
        # wrap s into [0, num_tori)
        return (s % self.num_tori)

    def sample_angles(self, s):
        """Sample theta, phi conditioned on axis position s."""
        # angular kappas decay with distance from mu_s
        scale = torch.exp(-self.decay * torch.abs(s - self.mu_s))
        kappa_theta_s = self.kappa_theta * scale
        kappa_phi_s = self.kappa_phi * scale

        # means stay fixed, but could shift if desired
        mu_theta_s = torch.full_like(s, self.mu_theta)
        mu_phi_s = torch.full_like(s, self.mu_phi)

        vm_theta = torch.distributions.VonMises(mu_theta_s, kappa_theta_s)
        vm_phi = torch.distributions.VonMises(mu_phi_s, kappa_phi_s)

        theta = vm_theta.sample()
        phi = vm_phi.sample()
        return theta, phi

    def sample_points(self, n_samples):
        """Sample 3D points across stitched tori."""
        s = self.sample_axis(n_samples)
        theta, phi = self.sample_angles(s)

        x_torus = (self.R + self.r * torch.cos(theta)) * torch.cos(phi)
        y = (self.R + self.r * torch.cos(theta)) * torch.sin(phi)
        z = self.r * torch.sin(theta)

        # shift along x by s * (2R + spacing)
        x = x_torus + s * (2*self.R + self.torus_spacing)

        return torch.stack([x, y, z], dim=1)

    def plot(self, n_samples=2000, s=3, show_surface=True):
        """Plot sampled stitched tori with optional surface rendering."""
        points = self.sample_points(n_samples).cpu().numpy()

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot surfaces
        if show_surface:
            u = np.linspace(0, 2*np.pi, 50)
            v = np.linspace(0, 2*np.pi, 25)
            U, V = np.meshgrid(u, v)
            X_base = (self.R + self.r*np.cos(U)) * np.cos(V)
            Y_base = (self.R + self.r*np.cos(U)) * np.sin(V)
            Z_base = self.r * np.sin(U)

            for i in range(self.num_tori):
                x_shift = i * (2*self.R + self.torus_spacing)
                ax.plot_surface(X_base + x_shift, Y_base, Z_base,
                                color='lightblue', alpha=0.2, rstride=2, cstride=2)

        # Plot points
        ax.scatter(points[:,0], points[:,1], points[:,2], s=s, c='red', alpha=0.7)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Continuous Stitched Tori with von Mises Sampling")
        plt.show()


