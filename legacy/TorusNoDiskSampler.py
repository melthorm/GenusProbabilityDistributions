import NormalTorusSampler
import torch
from torch.distributions import VonMises
import matplotlib.pyplot as plt
import numpy as np
import warnings

class TorusNoDiskSampler(NormalTorusSampler.NormalTorusSampler):
    def __init__(self, remove_centers=[(0.0, 0.0)], remove_radii=[0.5], *args, **kwargs):
        """
            initializes a torus with disks cut out to sample from
                remove_centers (list of tuples): [(theta0, phi0), ...] 
                                                 centers of patches in radians
                remove_radii (list of floats): [r0, r1, ...] 
                                               angular radii of each patch
                *args, **kwargs: passed to TorusSampler
        """
        super().__init__(*args, **kwargs)
        assert len(remove_centers) == len(remove_radii), "centers and radii must match"
        self.remove_centers = remove_centers
        self.remove_radii = remove_radii

        # Check if patches are near the von Mises peaks
        for (theta0, phi0), radius in zip(remove_centers, remove_radii):
            d_theta = abs((theta0 - self.mu1 + np.pi) % (2*np.pi) - np.pi)
            d_phi   = abs((phi0   - self.mu2 + np.pi) % (2*np.pi) - np.pi)

            # Consider “too close” if within ~2 standard deviations of the von Mises peak
            # std ~ 1/sqrt(kappa) for large kappa approximation
            std_theta = 1/np.sqrt(self.kappa1) if self.kappa1 > 0 else np.pi
            std_phi   = 1/np.sqrt(self.kappa2) if self.kappa2 > 0 else np.pi

            if d_theta < 2*std_theta and d_phi < 2*std_phi:
                warnings.warn(
                    f"Removed patch at ({theta0:.2f}, {phi0:.2f}) is close to "
                    f"von Mises peak (mu1={self.mu1:.2f}, mu2={self.mu2:.2f}). "
                    "Rejection sampling may be inefficient."
                )

    def sample_angles(self, n_samples):
        """
        Sample angles (theta, phi) from von Mises, rejecting all removed patches.
        """
        angles = torch.zeros(n_samples, 2, dtype = torch.float64)

        samplesGotten = 0
        while samplesGotten < n_samples:
            theta = self.dist1.sample((n_samples,))
            phi = self.dist2.sample((n_samples,))

            # Mask points outside all patches
            mask = torch.ones(n_samples, dtype=torch.bool, device=theta.device)
            for (center, radius) in zip(self.remove_centers, self.remove_radii):
                theta0, phi0 = center
                delta2 = radius
                
                thetaDiff = (theta-theta0) % (2*np.pi)
                phiDiff = (phi-phi0) % (2*np.pi)

                d2 = np.sqrt((thetaDiff * self.R) ** 2 +
                             (phiDiff * self.r) ** 2)
                mask &= (theta < 0.2) | (phi < 0.2)  # keep points outside this patch

            theta = theta[mask]
            phi = phi[mask]
            
            newSamplesGotten = samplesGotten + theta.numel()
            addedTensors = torch.stack((theta, phi), dim = 1)
            if (not newSamplesGotten > n_samples):
                angles[samplesGotten:newSamplesGotten] = addedTensors
            else:
                angles[samplesGotten:n_samples] = addedTensors[:n_samples-samplesGotten]


            samplesGotten = newSamplesGotten

        return angles

    def sample_points(self, n_samples):
        angles = self.sample_angles(n_samples)
        theta = angles[:,0]
        phi = angles[:,1]

        x = (self.R + self.r * torch.cos(theta)) * torch.cos(phi)
        y = (self.R + self.r * torch.cos(theta)) * torch.sin(phi)
        z = self.r * torch.sin(theta)

        return torch.stack([x, y, z], dim=1)

    def plot_points(self, n_samples=1000, s=5, 
                          show_surface=True, show_patches=True, 
                          return_points=False):
        """
        Plot sampled points with optional torus surface and multiple patches as filled surfaces.
        """
        pts = self.sample_points(n_samples).cpu().numpy()
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection="3d")

        # Torus surface
        if show_surface:
            u = np.linspace(0, 2*np.pi, 60)
            v = np.linspace(0, 2*np.pi, 30)
            U, V = np.meshgrid(u, v)
            X = (self.R + self.r*np.cos(V)) * np.cos(U)
            Y = (self.R + self.r*np.cos(V)) * np.sin(U)
            Z = self.r * np.sin(V)
            ax.plot_surface(X, Y, Z, color="lightblue", alpha=0.3, rstride=2, cstride=2, linewidth=0)

        # Sampled points
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=s, c="red", alpha=0.7)




        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Torus Samples")
        plt.show()

        return torch.tensor(pts) if return_points else None


    def is_on_torus_intersection(theta, phi, torus_index, R, r, x_offset, tol=1e-6):
        """
        Determine if a point on one of two tori lies on their intersection.

        Parameters
        ----------
        theta : float
            Angular coordinate along the central circle of the torus.
        phi : float
            Angular coordinate around the tube of the torus.
        torus_index : int
            0 for the first torus (centered at x=0), 1 for the second torus (centered at x=x_offset)
        R : float
            Major radius of both tori.
        r : float
            Minor radius of both tori.
        x_offset : float
            Horizontal offset along x-axis between the two torus centers.
        tol : float
            Numerical tolerance for "on the surface" check.
        
        Returns
        -------
        bool
        True if the point lies on the intersection.
        """
        # Coordinates of the point on its torus
        if torus_index == 0:
            x0 = (R + r * np.cos(phi)) * np.cos(theta)
            y0 = (R + r * np.cos(phi)) * np.sin(theta)
            z0 = r * np.sin(phi)
        else:
            x0 = x_offset + (R + r * np.cos(phi)) * np.cos(theta)
            y0 = (R + r * np.cos(phi)) * np.sin(theta)
            z0 = r * np.sin(phi)

        # Distance squared to the other torus's central circle
        x_center = 0 if torus_index == 1 else x_offset
        rho = np.sqrt((x0 - x_center)**2 + y0**2)
        dist_sq = (rho - R)**2 + z0**2

        # Check if distance squared equals r^2 within tolerance
        return abs(dist_sq - r**2) < tol

