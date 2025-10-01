# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %%
n_samples = 1000
x = torch.linspace(-2 * np.pi, np.pi, n_samples)

rng = torch.Generator().manual_seed(26)
x = x[torch.randperm(n_samples, generator=rng)]

rng_np = np.random.default_rng(seed=26)
noise_sin = rng_np.uniform(low=-0.01, high=0.01, size=n_samples)
noise_cos = rng_np.uniform(low=-0.01, high=0.01, size=n_samples)

y_sin = (torch.sin(x) + noise_sin).to(dtype=torch.float32)
y_cos = (torch.cos(x) + noise_cos).to(dtype=torch.float32)

# %%
plt.figure(figsize=(6, 4))
plt.scatter(x, y_sin, s=1, label="Sine")
plt.scatter(x, y_cos, s=1, label="Cosine")
plt.legend()
plt.show()

# %%
x = x.reshape(-1, 1)
y_sin = y_sin.reshape(-1, 1)
y_cos = y_cos.reshape(-1, 1)


# %%
class SinCosNet(nn.Module):
    def __init__(self, hidden_layers: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(1, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
        )

        self.model_sin = nn.Sequential(
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, 1),
        )

        self.model_cos = nn.Sequential(
            nn.Linear(hidden_layers, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, 1),
        )

    def forward(self, inputs: torch.Tensor) -> tuple:
        x = self.model(inputs)
        output_sin = self.model_sin(x)
        output_cos = self.model_cos(x)

        return output_sin, output_cos


# %%
hidden_layers = 64
net = SinCosNet(hidden_layers)

# %%
optimizer = optim.Adam(net.parameters(), lr=0.01)

# %%
n_epochs = 100
losses = []

for epoch in range(n_epochs):
    pred_sin, pred_cos = net(x)

    loss_sin = F.mse_loss(pred_sin, y_sin)
    loss_cos = F.mse_loss(pred_cos, y_cos)
    loss = loss_sin + loss_cos

    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")

# %%
plt.figure(figsize=(6, 4))
plt.plot(np.linspace(10, n_epochs, 10), losses[::10], "-o")
plt.ylabel("Losses")
plt.xlabel("Epochs")
plt.show()

# %%
net.eval()

with torch.no_grad():
    pred_sin, pred_cos = net(x)
    pred_sin = pred_sin.squeeze(dim=1)
    pred_cos = pred_cos.squeeze(dim=1)

# %%
plt.figure(figsize=(6, 4))
plt.scatter(x, pred_sin, s=1, label="Sine")
plt.scatter(x, pred_cos, s=1, label="Cosine")
plt.legend()
plt.show()

# %%
