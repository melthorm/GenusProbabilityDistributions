import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import flow
from mixture_sampler import MixtureSampler

# Hyperparameters
n_samples = 1000
num_layers, hidden_dim = 12, 64
lr = 1e-3
num_epochs = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"

# Target distribution
alpha, sigma, a, b = 0.5, 0.3, 5.0, 2.0
sampler = MixtureSampler(alpha, sigma, a, b)
data = sampler.sample(n_samples).to(device)

# Flow
my_flow = flow.Flow(num_layers=num_layers, hidden_dim=hidden_dim).to(device)
optimizer = optim.Adam(my_flow.parameters(), lr=lr)

def base_log_prob(z):
    return -0.5 * torch.sum(z**2, dim=1) - z.shape[1]/2 * torch.log(torch.tensor(2*torch.pi))

# Training
for epoch in range(num_epochs):
    optimizer.zero_grad()
    z, logdet = my_flow.forward(data)
    log_prob = base_log_prob(z) + logdet
    loss = -log_prob.mean()
    loss.backward()
    optimizer.step()

    # Visualization every 200 epochs
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: NLL = {loss.item():.4f}")
        with torch.no_grad():
            # Transform original data through flow
            flow_output, _ = my_flow.forward(data)
            # sample n_samples from flow 
            learned_samples = my_flow.sample(n_samples, device=device)

        # Convert to numpy
        data_np = data.cpu().numpy()
        flow_np = flow_output.cpu().numpy()
        learned_np = learned_samples.cpu().numpy()

        # Plot
        fig, axs = plt.subplots(1,3, figsize=(18,6))
        axs[0].scatter(data_np[:,0], data_np[:,1], s=5, alpha=0.5)
        axs[0].add_patch(plt.Circle((0,0),1.0,color='gray',alpha=0.1))
        axs[0].set_aspect('equal')
        axs[0].set_title("Original Mixture Samples")

        axs[1].scatter(flow_np[:,0], flow_np[:,1], s=5, alpha=0.5)
        axs[1].add_patch(plt.Circle((0,0),1.0,color='gray',alpha=0.1))
        axs[1].set_aspect('equal')
        axs[1].set_title("Flow Output of Original Samples")

        axs[2].scatter(learned_np[:,0], learned_np[:,1], s=5, alpha=0.5)
        axs[2].add_patch(plt.Circle((0,0),1.0,color='gray',alpha=0.1))
        axs[2].set_aspect('equal')
        axs[2].set_title("Samples from Learned Flow")

        plt.show()

