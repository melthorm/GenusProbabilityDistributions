# training.py
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from flow import Flow, flow_loss  # flow class and flow_loss function
from mixture_sampler import MixtureSampler

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base and target distributions
alpha = 0.5
sigma = 0.2
a = 2.0
b = 2.0

alpha2 = 0.5
sigma2 = 0.2
a2 = 100.0
b2 = 2.0


base = MixtureSampler(alpha, sigma, a, b)
target = MixtureSampler(alpha2, sigma2, a2, b2)

# Flow model
flow = Flow(num_layers=4, hidden_dim=32).to(device)

# Optimizer
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-1, weight_decay=0)
print(flow.parameters())
# Training parameters
n_samples = 512
num_steps = 5000

# Training loop
for step in range(1, num_steps + 1):
    optimizer.zero_grad()
    loss = flow_loss(flow, base, target, n_samples)
    loss.backward()
    if (step % 100 == 0):
        print(sum(p.grad.data.norm(2).item()**2 for p in flow.parameters())**0.5)
    optimizer.step()

    if step % 100 == 0:
        print()
        print(f"Step {step}, KL Loss: {loss.item():.6f}")


# --- Check training by sampling from the flow ---
flow.eval()
with torch.no_grad():
    # sample from base
    z = base.sample(1000).to(device)
    x_flow, _ = flow.forward(z)
    x_flow = x_flow.cpu()

    target_samples = target.sample(1000)

    # create figure with side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, samples, title in zip(axes, [target_samples, x_flow], ["Target Samples", "Flow Pushforward"]):
        # plot unit disk
        circle = plt.Circle((0, 0), 1.0, color='gray', alpha=0.1)
        ax.add_patch(circle)
        # scatter samples
        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.5)
        ax.set_aspect('equal')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(title)

    plt.suptitle("Comparison of Base and Flow Pushforward on Unit Disk")
    plt.show()


