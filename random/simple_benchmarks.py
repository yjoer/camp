# %%
import numpy as np

# %%
# %%timeit
total = 0

for i in range(1_000_000):
    total += 1

# %%
# %%timeit
sum(range(1_000_000))

# %%
# %%timeit
np.sum(np.arange(1_000_000))

# %%
