# %%
import numpy as np

# %%
p = np.array([1, 1])
q = np.array([5, 5])

euclidean = np.sqrt(np.sum((p - q) ** 2))
cityblock = np.sum(np.abs(p - q))
chessboard = np.max(np.abs(p - q))

(euclidean, cityblock, chessboard)

# %%
