import NormalTorusSampler
import torch
import matplotlib.pyplot as plt
import numpy as np

class TorusNoIntersectSampler(NormalTorusSampler.NormalTorusSampler):
    def __init__(self, x_offset, *args, **kwargs):
        """
        Initializes a torus sampler that avoids the intersection region with another torus.

        Args:
            x_offset (float): displacement of the second torus along the x-axis
            *args, **kwargs: passed to NormalTorusSampler
        """
        super().__init__(*args, **kwargs)
        self.x_offset = x_offset

    def sample_angles(self, n_samples):
        """
        Sample angles (theta, phi) from von Mises distributions, rejecting points inside
        the intersection with the other torus.
        """
        angles = torch.zeros(n_samples, 2, dtype=torch.float64)
        samples_gotten = 0

        while samples_gotten < n_samples:
            theta = self.dist1.sample((n_samples,))
            phi = self.dist2.sample((n_samples,))

            # Mask points inside the intersection
            mask = ~torus_intersection_mask(
                theta, phi, 
                self.R, self.r, self.x_offset
            )

            theta, phi = theta[mask], phi[mask]

            num_to_add = min(theta.numel(), n_samples - samples_gotten)
            if num_to_add > 0:
                angles[samples_gotten:samples_gotten + num_to_add] = torch.stack(
                    (theta[:num_to_add], phi[:num_to_add]), dim=1
                )
                samples_gotten += num_to_add
            print(f"{samples_gotten} samples gotten out of {n_samples} needed.")

        return angles

    def sample_points(self, n_samples):
        """
        Convert sampled angles to 3D points on the torus.
        """
        angles = self.sample_angles(n_samples)
        theta, phi = angles[:,0], angles[:,1]

        x = (self.R + self.r * torch.cos(theta)) * torch.cos(phi)
        y = (self.R + self.r * torch.cos(theta)) * torch.sin(phi)
        z = self.r * torch.sin(theta)

        return torch.stack([x, y, z], dim=1)

    def plot_points(self, n_samples=1000, s=5, 
                    show_surface=True, show_second_torus=True, 
                    return_points=False):
        """
        Plot sampled points along with optional torus surfaces.
        
        Args:
            n_samples (int): number of points to sample
            s (float): point size
            show_surface (bool): whether to show the first torus surface
            show_second_torus (bool): whether to show the second torus surface
            return_points (bool): whether to return sampled points as torch.Tensor
        """
        pts = self.sample_points(n_samples).cpu().numpy()
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection="3d")

        # First torus surface
        if show_surface:
            u = np.linspace(0, 2*np.pi, 60)
            v = np.linspace(0, 2*np.pi, 30)
            U, V = np.meshgrid(u, v)
            X1 = (self.R + self.r*np.cos(V)) * np.cos(U)
            Y1 = (self.R + self.r*np.cos(V)) * np.sin(U)
            Z1 = self.r * np.sin(V)
            ax.plot_surface(X1, Y1, Z1, color="lightblue", alpha=0.3, rstride=2, cstride=2, linewidth=0)

        # Second torus surface
        if show_second_torus:
            X2 = (self.R + self.r*np.cos(V)) * np.cos(U) + self.x_offset
            Y2 = (self.R + self.r*np.cos(V)) * np.sin(U)
            Z2 = self.r * np.sin(V)
            ax.plot_surface(X2, Y2, Z2, color="lightgreen", alpha=0.3, rstride=2, cstride=2, linewidth=0)

        # Sampled points
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=s, c="red", alpha=0.7)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Torus Samples with Avoided Intersection")
        plt.show()

        return torch.tensor(pts) if return_points else None



def torus_intersection_mask(theta, phi, R, r, x_offset, tol=1e-5):
    x = (R + r * torch.cos(theta)) * torch.cos(phi)
    y = (R + r * torch.cos(theta)) * torch.sin(phi)
    z = r * torch.sin(theta)

    # Distance to second torus (offset in x)
    rho = torch.sqrt((x - x_offset)**2 + y**2)
    mask = ((rho - R)**2 + z**2) <= r**2
    return mask
