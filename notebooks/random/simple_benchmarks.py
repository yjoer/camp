# %%
import camp.rs
import numpy as np

# %%
# %%timeit
total = 0

for _ in range(1_000_000):
  total += 1  # noqa: SIM113

# %%
# %%timeit
sum(range(1_000_000))

# %%
# %%timeit
np.sum(np.arange(1_000_000))


# %%
def fibonacci(n: int) -> int:
  a = 0
  b = 1

  if n == 0:
    return a

  for _ in range(1, n):
    c = a + b
    a = b
    b = c

  return b


# %%
# %%timeit
fibonacci(50)

# %%
# %%timeit
camp.rs.fibonacci(50)

# %%
camp.rs.collatz_repeat(1_000_000_000)

# %%
