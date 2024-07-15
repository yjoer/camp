# %%
import math
from typing import cast

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %%
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
state, info = env.reset(seed=26)

# %%
images = [cast(np.ndarray, env.render())]
actions = [2, 2, 1, 1, 1, 2]

for action in actions:
    state, reward, terminated, _, _ = env.step(action)
    images.append(cast(np.ndarray, env.render()))

    if terminated:
        print("You reached the goal!")

# %%
n_rows = math.ceil(len(images) / 3)
n_cols = 3

plt.figure(figsize=(6.4, 6.4), constrained_layout=True)

for i, image in enumerate(images):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.imshow(image)

plt.show()

# %%
