# %%
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
from ipywidgets import interact

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']


# %%
def f(x):
    return x**4 + x**3 - 5 * x**2


# %%
def optimize(lr=0.01, momentum=0.0):
    x = torch.tensor(2.0, requires_grad=True)
    buffer = torch.zeros_like(x.detach())
    values = []

    for i in range(10):
        y = f(x)
        values.append((x.item(), y.item()))

        y.backward()
        d_p = x.grad.data

        if momentum != 0:
            d_p = buffer.mul_(momentum).add_(d_p)

        x.detach().add_(d_p, alpha=-lr)
        x.grad.zero_()

    return values


# %%
x = np.arange(-3, 2, 0.001)
y = f(x)

lr_slider = widgets.FloatLogSlider(
    value=0.01,
    base=10,
    min=-5,
    max=-1,
    step=0.01,
    description="Learning Rate",
    layout={"width": "80%"},
)

momentum_slider = widgets.FloatSlider(
    value=1.05,
    min=0.01,
    max=1.5,
    step=0.01,
    description="Momentum",
    layout={"width": "80%"},
)


@interact(lr=lr_slider, momentum=momentum_slider)
def plot_optimizer_steps(lr, momentum):
    values = optimize(lr, momentum)
    x_opt = [np.clip(v[0], -4, 3) for v in values]
    y_opt = [np.clip(v[1], -15, 15) for v in values]

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, linewidth=2)
    plt.plot(x_opt, y_opt, "r-X", linewidth=2, markersize=6)

    for i, (p, q) in enumerate(zip(x_opt, y_opt)):
        plt.text(p, q + 0.625, f"{i}", color="r", ha="center")

    plt.grid()
    plt.legend(["Square Function", "Optimizer Steps"])

    plt.show()


# %%
