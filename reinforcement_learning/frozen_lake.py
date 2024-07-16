# %%
import math
from typing import Any
from typing import cast

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %% [markdown]
# ## Interacting with Environments

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
def render_images(images):
    n_rows = math.ceil(len(images) / 3)
    n_cols = 3

    plt.figure(figsize=(6.4, 6.4), constrained_layout=True)

    for i, image in enumerate(images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(image)

    plt.show()


# %%
render_images(images)

# %% [markdown]
# ## Markov Decision Processes

# %%
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
state, info = env.reset(seed=26)

# %%
lake_map = cast(Any, env.unwrapped).desc
terminal_states = []

for row in range(lake_map.shape[0]):
    for col in range(lake_map.shape[1]):
        if lake_map[row, col] == b"H" or lake_map[row, col] == b"G":
            terminal_states.append(4 * row + col)


# %% [markdown]
# ### Policy Iteration


# %%
def compute_state_value(policy, state, terminal_states, gamma):
    visited_states = []
    value = 0
    terminate = False

    while True:
        action = policy[state]
        _, next_state, reward, _ = env.unwrapped.P[state][action][0]

        if next_state in terminal_states:
            terminate = True

        if next_state == state or next_state in visited_states:
            terminate = True

        visited_states.append(state)
        value += reward + gamma * value

        if terminate:
            break

        state = next_state

    return value


# %%
def evaluate_policy(policy, n_states, terminal_states, gamma):
    return {
        state: compute_state_value(policy, state, terminal_states, gamma)
        for state in range(n_states)
    }


# %%
def compute_q_value(V, state, action, terminal_states, gamma):
    if state in terminal_states:
        return 0

    _, next_state, reward, _ = env.unwrapped.P[state][action][0]

    if next_state == state or next_state in [5, 7, 11, 12]:
        reward = -1

    return reward + gamma * V[next_state]


# %%
def improve_policy(V, n_states, n_actions, terminal_states, gamma):
    Q = {
        (state, action): compute_q_value(V, state, action, terminal_states, gamma)
        for state in range(n_states)
        for action in range(n_actions)
    }

    improved_policy = {}

    for state in range(n_states):
        max_action = max(range(n_actions), key=lambda action: Q[(state, action)])
        improved_policy[state] = max_action

    return improved_policy, Q


# %%
def iterate_policy():
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = {state: np.random.randint(n_actions) for state in range(n_states)}
    gamma = 0.99

    for i in range(10):
        V = evaluate_policy(policy, n_states, terminal_states, gamma)

        improved_policy, Q = improve_policy(
            V,
            n_states,
            n_actions,
            terminal_states,
            gamma,
        )

        if improved_policy == policy:
            break

        policy = improved_policy

    return policy, V, Q


# %%
policy, V, Q = iterate_policy()
print(f"V: {V}")
print(f"Q: {Q}")

# %%
images = [cast(np.ndarray, env.render())]
terminated = False

while not terminated:
    action = policy[state]
    state, reward, terminated, __, _ = env.step(action)
    images.append(cast(np.ndarray, env.render()))

    if terminated:
        print("You reached the goal!")

# %%
render_images(images)

# %%
