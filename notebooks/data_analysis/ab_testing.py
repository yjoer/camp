# %%
import numpy as np
import scipy.stats as st

# %% [markdown]
# ## Sample Ratio Mismatch

# %% [markdown]
# $H_0 = \text{The categorical data has the given frequencies}$

# %%
n_users = 10000
control_users = n_users * 0.485
exposed_users = n_users * 0.515
observed = [control_users, exposed_users]
expected = [n_users / 2, n_users / 2]

statistic, p_value = st.chisquare(observed, expected)

print(statistic, p_value)
print("SRM may be present") if p_value < 0.01 else print("SRM is likely not present")

# %% [markdown]
# ## Variance Estimation

# %% [markdown]
# ### Delta Method

# %%
outcomes = ["Fail to reject the null hypothesis", "Reject the null hypothesis"]


# %%
def var_ratio(x: np.ndarray, y: np.ndarray) -> np.float64:
  mean_x = np.mean(x)
  mean_y = np.mean(y)
  var_x = np.var(x, ddof=1)
  var_y = np.var(y, ddof=1)
  cov_xy = np.cov(x, y, ddof=1)[0][1]

  a_ = var_x / mean_x**2 + var_y / mean_y**2 - 2 * cov_xy / (mean_x * mean_y)
  b_ = mean_x**2 / mean_y**2

  return a_ * b_


# %%
def ttest_delta(
  n: int,
  mean_control: np.float64,
  mean_treatment: np.float64,
  var_control: np.float64,
  var_treatment: np.float64,
) -> tuple:
  diff = mean_treatment - mean_control
  stderr = np.sqrt(var_treatment / n + var_control / n)
  z = diff / stderr
  p_value = st.norm.sf(abs(z)) * 2

  return z, p_value


# %%
rng = np.random.default_rng(seed=26)

n = 100
clicks_control = rng.uniform(low=1, high=5, size=n)
clicks_treatment = rng.uniform(low=2, high=6, size=n)
views_control = rng.uniform(low=100, high=500, size=n)
views_treatment = rng.uniform(low=100, high=500, size=n)

mean_control = np.sum(clicks_control) / np.sum(views_control)
mean_treatment = np.sum(clicks_treatment) / np.sum(views_treatment)
var_control = var_ratio(clicks_control, views_control)
var_treatment = var_ratio(clicks_treatment, views_treatment)

print(mean_control, mean_treatment, var_control, var_treatment)

# %%
diff = mean_treatment - mean_control
stderr = np.sqrt(var_treatment / n + var_control / n)
lower, upper = diff - 1.96 * stderr, diff + 1.96 * stderr

print(f"Difference: {diff}")
print(f"95% CI: [{lower:.4f}, {upper:.4f}]")

# %%
alpha = 0.05

statistic, p_value = ttest_delta(
  n,
  mean_control,
  mean_treatment,
  var_control,
  var_treatment,
)

print(statistic, p_value)
print(outcomes[0]) if p_value > alpha else print(outcomes[1])

# %%
