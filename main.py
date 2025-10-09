import torch
import matplotlib.pyplot as plt
import maps
import flow
from mixture_sampler import MixtureSampler

if __name__ == "__main__":
    n = 2000
    alpha = 0.5
    sigma = 0.3
    a, b = 5.0, 2.0

    # Create the mixture sampler and sample points
    mixture_sampler = MixtureSampler(alpha, sigma, a, b)
    samples_mixture = mixture_sampler.sample(n)  # shape (N,2)

    # Create a Flow with 4 coupling layers
    my_flow = flow.Flow(num_layers=4, hidden_dim=32)

    # Forward through the flow
    x_out, logdet = my_flow.forward(samples_mixture)

    # Apply inverse using the Flow method
    x_recovered = my_flow.inverse(x_out)

    # Convert to numpy for plotting
    samples_mixture_np = samples_mixture.numpy()
    x_out_np = x_out.detach().numpy()
    x_recovered_np = x_recovered.detach().numpy()

    # Plot original, transformed, and inverse-recovered
    fig, axs = plt.subplots(1,3, figsize=(18,6))

    axs[0].scatter(samples_mixture_np[:,0], samples_mixture_np[:,1], s=5, alpha=0.5)
    circle0 = plt.Circle((0,0),1.0,color='gray',alpha=0.1)
    axs[0].add_patch(circle0)
    axs[0].set_title("Original Mixture Samples")
    axs[0].set_aspect('equal')

    axs[1].scatter(x_out_np[:,0], x_out_np[:,1], s=5, alpha=0.5)
    circle1 = plt.Circle((0,0),1.0,color='gray',alpha=0.1)
    axs[1].add_patch(circle1)
    axs[1].set_title("Flow Transformed Disk Samples")
    axs[1].set_aspect('equal')

    axs[2].scatter(x_recovered_np[:,0], x_recovered_np[:,1], s=5, alpha=0.5)
    circle2 = plt.Circle((0,0),1.0,color='gray',alpha=0.1)
    axs[2].add_patch(circle2)
    axs[2].set_title("Inverse-Recovered Disk Samples")
    axs[2].set_aspect('equal')

    plt.show()

    print("Sample log-determinants (first 10):", logdet[:10])

