# %%
import os

os.environ["KERAS_BACKEND"] = "torch"

# %%
import random
from collections import deque
from io import BytesIO
from typing import SupportsFloat
from typing import cast

import gymnasium as gym
import imageio
import keras
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import Image
from torch import nn
from torch import optim

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %%
env = gym.make("LunarLander-v3", render_mode="rgb_array")


# %%
class ReplayBuffer:
  def __init__(self, capacity: int) -> None:
    self.buffer = deque([], maxlen=capacity)

  def __len__(self) -> int:
    return len(self.buffer)

  def push(
    self,
    state: torch.Tensor,
    action: int,
    reward: SupportsFloat,
    next_state: torch.Tensor,
    done: bool,
  ) -> None:
    self.buffer.append((state, action, reward, next_state, done))

  def sample(self, batch_size: int) -> tuple:
    batch = random.sample(self.buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch, strict=False)

    states_t = torch.stack(states, dim=0)
    actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    next_states_t = torch.stack(next_states, dim=0)
    dones_t = torch.tensor(dones, dtype=torch.float32)

    return states_t, actions_t, rewards_t, next_states_t, dones_t


# %%
class QNetwork(nn.Module):
  def __init__(self, n_states: int, n_actions: np.int64) -> None:
    super().__init__()

    self.fc1 = nn.Linear(n_states, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, n_actions)

  def forward(self, state: torch.Tensor) -> torch.Tensor:
    x = torch.relu(self.fc1(state))
    x = torch.relu(self.fc2(x))
    return self.fc3(x)


# %%
def select_action(q_network: QNetwork, state: torch.Tensor) -> int:
  q_values = q_network(state)
  return torch.argmax(q_values).item()


# %%
def select_action_epsilon_greedy(
  q_network: QNetwork,
  state: torch.Tensor,
  epsilon: np.float64,
) -> int:
  q_values = q_network(state)
  action = torch.argmax(q_values).item()

  rng = np.random.default_rng()
  sample = rng.random()

  if sample < epsilon:
    return rng.choice(range(len(q_values)))

  return action


# %%
def loss_barebone(
  q_network: QNetwork,
  state: torch.Tensor,
  action: int,
  reward: SupportsFloat,
  next_state: torch.Tensor,
  done: bool,
) -> torch.Tensor:
  q_values = q_network(state)
  q_value_current = q_values[action]

  q_value_next = q_network(next_state).max()
  q_value_target = reward + gamma * q_value_next * (1 - done)

  return F.mse_loss(q_value_current, q_value_target)


# %%
def loss_dqn(
  online_network: QNetwork,
  target_network: QNetwork,
  replay_buffer: ReplayBuffer,
  batch_size: int,
) -> torch.Tensor:
  states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
  q_values = online_network(states).gather(1, actions).squeeze(1)

  with torch.no_grad():
    q_values_next = target_network(next_states).amax(1)
    q_values_target = rewards + gamma * q_values_next * (1 - dones)

  return F.mse_loss(q_values_target, q_values)


# %%
def update_target_network(
  online_network: QNetwork,
  target_network: QNetwork,
  tau: float,
) -> None:
  state_online = online_network.state_dict()
  state_target = target_network.state_dict()

  for key in state_online:
    state_target[key] = tau * state_online[key] + (1 - tau) * state_target[key]

  target_network.load_state_dict(state_target)


# %%
n_states = cast("tuple[int, ...]", env.observation_space.shape)[0]
n_actions = cast("gym.spaces.Discrete", env.action_space).n
n_episodes = 1500

gamma = 0.99
batch_size = 64
start = 0.9
end = 0.05
total_steps = 0
decay = 1000

replay_buffer = ReplayBuffer(10000)
losses = []

online_network = QNetwork(n_states, n_actions)
target_network = QNetwork(n_states, n_actions)

optimizer = optim.Adam(online_network.parameters(), lr=1e-4)

# %%
for i in range(n_episodes):
  print(f"Episode: {i + 1}/{n_episodes}")
  pbar = keras.utils.Progbar(None, stateful_metrics=["loss"])

  state, info = env.reset(seed=26)
  state = torch.tensor(state, dtype=torch.float32)

  steps = 0
  done = False

  while not done:
    epsilon = end + (start - end) * np.exp(-total_steps / decay)
    action = select_action_epsilon_greedy(online_network, state, epsilon)

    next_state, reward, terminated, truncated, _ = env.step(action)
    next_state = torch.tensor(next_state, dtype=torch.float32)
    done = terminated or truncated

    replay_buffer.push(state, action, reward, next_state, done)

    if len(replay_buffer) < batch_size:
      steps += 1
      total_steps += 1
      state = next_state
      continue

    loss = loss_dqn(online_network, target_network, replay_buffer, batch_size)
    losses.append(loss.item())
    pbar.update(steps, values=[("loss", loss.item())])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    update_target_network(online_network, target_network, tau=0.005)

    steps += 1
    total_steps += 1
    state = next_state

  pbar.update(steps, values=None, finalize=True)

# %%
plt.plot(losses)

plt.title("Losses vs. Steps")
plt.ylabel("Losses")
plt.xlabel("Steps")

plt.show()


# %%
def play(q_network: QNetwork) -> list[np.ndarray]:
  state, _info = env.reset(seed=26)
  state = torch.tensor(state, dtype=torch.float32)

  frames = [cast("np.ndarray", env.render())]

  for _ in range(2000):
    action = torch.argmax(q_network(state)).item()

    state, _reward, terminated, truncated, _ = env.step(action)
    state = torch.tensor(state, dtype=torch.float32)

    frames.append(cast("np.ndarray", env.render()))

    if terminated:
      print("You reached the goal!")

    if terminated or truncated:
      break

  return frames


# %%
frames = play(online_network)

# %%
buffer = BytesIO()

with imageio.get_writer(
  buffer,
  format="gif",  # ty: ignore[invalid-argument-type]
  fps=30,
  loop=0,
) as writer:
  for frame in frames:
    writer.append_data(frame)  # ty: ignore[unresolved-attribute]

buffer.seek(0)

print(len(frames))
Image(data=buffer.getvalue())

# %%
