# %%
from io import BytesIO
from typing import SupportsFloat
from typing import cast

import gymnasium as gym
import imageio
import numpy as np
from gymnasium.core import ObsType
from IPython.display import Image

# %%
env = gym.make("Taxi-v3", render_mode="rgb_array")


# %%
def epsilon_greedy(
    Q: np.ndarray,
    state: int,
    epsilon: float,
    i: int,
    j: int,
) -> np.int64:
    seed = int(str(i) + str(j))
    env.action_space.seed(seed)
    rng = np.random.default_rng(seed)

    if rng.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])

    return action


# %%
def update_q_table_v2(
    Q: np.ndarray,
    state: int,
    action: np.int64,
    reward: SupportsFloat,
    next_state: ObsType,
    alpha: float,
    gamma: float,
) -> None:
    old_value = Q[state, action]
    next_max = np.max(Q[next_state])

    Q[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)


# %%
def get_policy(Q: np.ndarray) -> dict:
    n_states = env.observation_space.n
    return {state: np.argmax(Q[state]) for state in range(n_states)}


# %%
def simulate_policy(policy: dict) -> list[np.ndarray]:
    state, _info = env.reset(seed=26)

    frames = [cast("np.ndarray", env.render())]
    terminated = False

    for _ in range(100):
        action = policy[state]
        state, _reward, terminated, _, _ = env.step(action)
        frames.append(cast("np.ndarray", env.render()))

        if terminated:
            print("You reached the goal!")
            break

    return frames


# %%
def epsilon_greedy_decayed(n_episodes: int, max_actions: int) -> np.ndarray:
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    epsilon = 1.0
    epsilon_decay = 0.999
    min_epsilon = 0.01

    alpha = 0.1
    gamma = 1

    for i in range(n_episodes):
        state, _info = env.reset(seed=i)
        terminated = False

        for j in range(max_actions):
            if terminated:
                break

            action = epsilon_greedy(Q, state, epsilon, i, j)
            next_state, reward, terminated, _truncated, _info = env.step(action)
            update_q_table_v2(Q, state, action, reward, next_state, alpha, gamma)

            state = next_state

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return Q


# %%
Q = epsilon_greedy_decayed(1100, 100)
policy = get_policy(Q)

frames = simulate_policy(policy)

# %%
buffer = BytesIO()

with imageio.get_writer(
    buffer,
    format="gif",  # ty: ignore[invalid-argument-type]
    fps=2,
    loop=0,
) as writer:
    for frame in frames:
        writer.append_data(frame)  # ty: ignore[unresolved-attribute]

buffer.seek(0)

print(len(frames))
Image(data=buffer.getvalue())

# %%
