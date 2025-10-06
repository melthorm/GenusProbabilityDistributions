from normal_sampler import NormalSampler
from beta_sampler import BetaSampler
from mixture_sampler import MixtureSampler
import coupling_layer
from coupling_layer import CouplingLayer
import matplotlib.pyplot as plt

import maps
import torch


if __name__ == "__main__":
    n = 2000
    alpha = 0.5
    sigma = 0.3
    a, b = 5.0, 2.0

    normal_sampler = NormalSampler(sigma)
    beta_sampler = BetaSampler(a, b)
    mixture_sampler = MixtureSampler(alpha, sigma, a, b)

    samples_normal = normal_sampler.sample(n)
    samples_beta = beta_sampler.sample(n)
    samples_mixture = mixture_sampler.sample(n)
        
    """
    print("First 10 Truncated Gaussian samples:\n", samples_normal[:10])
    print("First 10 Beta samples:\n", samples_beta[:10])
    print("First 10 Mixture samples:\n", samples_mixture[:10])
    """

    normal_sampler.plot(n)
    beta_sampler.plot(n)
    mixture_sampler.plot(n)

    # Map mixture samples to R2
    mixture_R2 = maps.disk_to_R2(samples_mixture)

    # Apply coupling layer
    layer = CouplingLayer()
    mixture_transformed_R2 = coupling_layer.f(mixture_R2, layer)

    # Map back to disk (warped)
    mixture_transformed_disk = maps.R2_to_disk(mixture_transformed_R2)

    # Apply inverse to transformed R2 points
    mixture_recovered_R2 = layer.inverse(mixture_transformed_R2)
    mixture_recovered_disk = maps.R2_to_disk(mixture_recovered_R2)
    mixture_recovered_disk_np = mixture_recovered_disk.detach().numpy()

    # Detach and convert to numpy for plotting
    mixture_transformed_R2_np = mixture_transformed_R2.detach().numpy()
    mixture_transformed_disk_np = mixture_transformed_disk.detach().numpy()

    # Plot original, transformed, back-to-disk, and inverse-recovered
    fig, axs = plt.subplots(1,4, figsize=(24,6))

    # Original mixture samples
    axs[0].scatter(samples_mixture[:,0], samples_mixture[:,1], s=5, alpha=0.5)
    circle0 = plt.Circle((0,0),1.0,color='gray',alpha=0.1)
    axs[0].add_patch(circle0)
    axs[0].set_title("Original Mixture Samples")
    axs[0].set_aspect('equal')

    # Transformed in R2
    axs[1].scatter(mixture_transformed_R2_np[:,0], mixture_transformed_R2_np[:,1], s=5, alpha=0.5)
    axs[1].set_title("Coupling Layer Output in R2")
    axs[1].set_aspect('equal')

    # Back to disk (warped)
    axs[2].scatter(mixture_transformed_disk_np[:,0], mixture_transformed_disk_np[:,1], s=5, alpha=0.5)
    circle2 = plt.Circle((0,0),1.0,color='gray',alpha=0.1)
    axs[2].add_patch(circle2)
    axs[2].set_title("Back to Disk")
    axs[2].set_aspect('equal')

    # Inverse recovered disk
    axs[3].scatter(mixture_recovered_disk_np[:,0], mixture_recovered_disk_np[:,1], s=5, alpha=0.5)
    circle3 = plt.Circle((0,0),1.0,color='gray',alpha=0.1)
    axs[3].add_patch(circle3)
    axs[3].set_title("Inverse Recovered Disk")
    axs[3].set_aspect('equal')

    plt.show()


