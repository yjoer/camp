# %%
import os

os.environ["KERAS_BACKEND"] = "torch"

# %%
import numpy as np
from keras.layers import LSTM
from keras.layers import SimpleRNN

# %%
rng = np.random.default_rng(seed=26)
X = rng.random((4, 3, 2)).astype(np.float32)  # (batch, time_step, feature)

# %% [markdown]
# ## RNN

# %%
simple_rnn = SimpleRNN(5)

output = simple_rnn(X)
print(output.shape)  # (batch, hidden_state)

# %%
simple_rnn = SimpleRNN(5, return_sequences=True, return_state=True)

whole_sequence_output, final_state = simple_rnn(X)
print(whole_sequence_output.shape)  # (batch, time_step, hidden_state)
print(final_state.shape)  # (batch, hidden_state)

# %% [markdown]
# ## LSTM

# %%
lstm = LSTM(5)

output = lstm(X)
print(output.shape)

# %%
lstm = LSTM(5, return_sequences=True, return_state=True)

whole_sequence_output, final_memory_state, final_carry_state = lstm(X)

print(whole_sequence_output.shape)
print(final_memory_state.shape)
print(final_carry_state.shape)

# %%
