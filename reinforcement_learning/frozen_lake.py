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
def simulate_policy(policy):
    state, info = env.reset(seed=26)

    images = [cast(np.ndarray, env.render())]
    terminated = False

    for i in range(8):
        action = policy[state]
        state, reward, terminated, _, _ = env.step(action)
        images.append(cast(np.ndarray, env.render()))

        if terminated:
            print("You reached the goal!")
            break

    return images


# %%
images = simulate_policy(policy)
render_images(images)


# %% [markdown]
# ## Model-Free Learning

# %% [markdown]
# ### Monte Carlo


# %%
def generate_episode(i):
    state, info = env.reset(seed=i)
    episode = []

    j = 0
    terminated = False

    while not terminated:
        env.action_space.seed(int(str(i) + str(j)))
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward))

        j += 1
        state = next_state

    return episode


# %%
def first_visit_mc(n_episodes):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    returns_sum = np.zeros((n_states, n_actions))
    returns_count = np.zeros((n_states, n_actions))

    for i in range(n_episodes):
        episode = generate_episode(i)
        visited_states_actions = set()

        for j, (state, action, reward) in enumerate(episode):
            if (state, action) in visited_states_actions:
                continue

            returns_sum[state, action] += sum(x[2] for x in episode[j:])
            returns_count[state, action] += 1
            visited_states_actions.add((state, action))

    nonzero_counts = returns_count != 0

    Q = np.zeros((n_states, n_actions))
    Q[nonzero_counts] = returns_sum[nonzero_counts] / returns_count[nonzero_counts]

    return Q


# %%
def every_visit_mc(n_episodes):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    returns_sum = np.zeros((n_states, n_actions))
    returns_count = np.zeros((n_states, n_actions))

    for i in range(n_episodes):
        episode = generate_episode(i)

        for j, (state, action, reward) in enumerate(episode):
            returns_sum[state, action] += sum(x[2] for x in episode[j:])
            returns_count[state, action] += 1

    nonzero_counts = returns_count != 0

    Q = np.zeros((n_states, n_actions))
    Q[nonzero_counts] = returns_sum[nonzero_counts] / returns_count[nonzero_counts]

    return Q


# %%
def get_policy(Q):
    n_states = env.observation_space.n
    return {state: np.argmax(Q[state]) for state in range(n_states)}


# %%
Q = first_visit_mc(1155)
policy_first_visit = get_policy(Q)

images = simulate_policy(policy_first_visit)
render_images(images)

# %%
Q = every_visit_mc(915)
policy_every_visit = get_policy(Q)

images = simulate_policy(policy_every_visit)
render_images(images)


# %% [markdown]
# ### Temporal Difference Learning

# %% [markdown]
# #### SARSA


# %%
def update_q_table(Q, state, action, reward, next_state, next_action, alpha, gamma):
    old_value = Q[state, action]
    next_value = Q[next_state, next_action]

    Q[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_value)


# %%
def sarsa(n_episodes):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    alpha = 0.1
    gamma = 1

    for i in range(n_episodes):
        state, info = env.reset(seed=i)

        env.action_space.seed(i)
        action = env.action_space.sample()

        j = 0
        terminated = False

        while not terminated:
            next_state, reward, terminated, truncated, info = env.step(action)

            env.action_space.seed(int(str(i) + str(j)))
            next_action = env.action_space.sample()

            update_q_table(
                Q,
                state,
                action,
                reward,
                next_state,
                next_action,
                alpha,
                gamma,
            )

            j += 1
            state, action = next_state, next_action

    return Q


# %%
Q = sarsa(150)
policy_sarsa = get_policy(Q)

images = simulate_policy(policy_sarsa)
render_images(images)


# %% [markdown]
# #### Q-Learning


# %%
def update_q_table_v2(Q, state, action, reward, next_state, alpha, gamma):
    old_value = Q[state, action]
    next_max = max(Q[next_state])

    Q[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)


# %%
def q_learning(n_episodes):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    alpha = 0.1
    gamma = 1

    for i in range(n_episodes):
        state, info = env.reset(seed=i)

        j = 0
        terminated = False

        while not terminated:
            env.action_space.seed(int(str(i) + str(j)))
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)

            update_q_table_v2(Q, state, action, reward, next_state, alpha, gamma)

            j += 1
            state = next_state

    return Q


# %%
Q = q_learning(65)
policy_q = get_policy(Q)

images = simulate_policy(policy_q)
render_images(images)

# %%
