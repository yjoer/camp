# %%
import numpy as np


# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# %%
print(sigmoid(0.1))
print(sigmoid(0.2))


# %%
def norm(x: np.ndarray):
    return x / np.sum(x)


# %%
def softmax(x: np.ndarray):
    return np.exp(x) / np.sum(np.exp(x))


# %%
print(norm(np.array([0.1, 0.2, 0.3, 0.4])))
print(norm(np.array([0.2, 0.4, 0.6, 0.8])))

# %%
print(softmax(np.array([0.1, 0.2, 0.3, 0.4])))
print(softmax(np.array([0.2, 0.4, 0.6, 0.8])))

# %%
