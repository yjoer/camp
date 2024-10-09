# %%
import numpy as np
from scipy.optimize import linprog

# %% [markdown]
# Plant 1 costs \\$60 per hour. \
# Plant 2 costs \\$50 per hour.
#
# Let: \
# $x_1 = \text{hours spent at plant 1}$ \
# $x_2 = \text{hours spent at plant 2}$
#
# Minimize $Z = 60x_1 + 50x_2$
#
# $5x_1 + 3x_2 >= 60$ \
# $2x_1 + 2x_2 >= 30$ \
# $7x_1 + 9x_2 >= 126$ \
# $x_1 >= 0$ \
# $x_2 >= 0$

# %% [markdown]
# As SciPy does not support greater than or equal to inequality constraints directly, we
# can flip the inequalities by multiplying both sides by -1.
#
# $-5x_1 - 3x_2 <= -60$ \
# $-2x_1 - 2x_2 <= -30$ \
# $-7x_1 - 9x_2 <= -126$

# %%
c = np.array([60, 50])
A = np.array([[-5, -3], [-2, -2], [-7, -9]])
b = np.array([-60, -30, -126])

# %%
solution = linprog(c, A_ub=A, b_ub=b, method="highs")
solution.fun

# %%
hours_spent = solution.x
hours_spent

# %%
units = np.sum(-A * hours_spent, axis=1)
units

# %%
