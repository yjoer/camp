# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %% [markdown]
# ## Binomial Distribution

# %%
n_a = 1000
n_b = 1000
p_a = 0.80
p_b = 0.85

binom_x = np.arange(n_a * p_a - 50, n_b * p_b + 50)
binom_a = st.binom.pmf(binom_x, n=n_a, p=p_a)
binom_b = st.binom.pmf(binom_x, n=n_b, p=p_b)

norm_x = np.linspace(p_a - 0.05, p_b + 0.05, 500)
norm_a = st.norm.pdf(norm_x, loc=p_a, scale=np.sqrt(p_a * (1 - p_a) / n_a))
norm_b = st.norm.pdf(norm_x, loc=p_b, scale=np.sqrt(p_b * (1 - p_b) / n_b))

fig = plt.figure(figsize=(8, 4), constrained_layout=True)
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)

ax1.bar(binom_x, binom_a, alpha=0.5)
ax1.bar(binom_x, binom_b, alpha=0.5)

ax2.plot(norm_x, norm_a)
ax2.plot(norm_x, norm_b)

colors = ax2._get_lines._cycler_items  # ty: ignore[unresolved-attribute]
ax2.axvline(x=p_a, color=colors[0]["color"], linestyle="--")
ax2.axvline(x=p_b, color=colors[1]["color"], linestyle="--")

plt.show()

# %% [markdown]
# ## Central Limit Theorem

# %%
fig = plt.figure(figsize=(8, 2.67), constrained_layout=True)
ax1, ax2, ax3 = fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)
sns.histplot(st.norm.rvs(size=10, random_state=26), bins=25, kde=True, ax=ax1)
sns.histplot(st.norm.rvs(size=25, random_state=26), bins=25, kde=True, ax=ax2)
sns.histplot(st.norm.rvs(size=100, random_state=26), bins=25, kde=True, ax=ax3)
plt.show()

# %% [markdown]
# ## Confidence Interval

# %%
norm_dist = st.norm.rvs(size=100, random_state=26)

lower, upper = st.norm.interval(
    loc=np.mean(norm_dist),
    scale=np.std(norm_dist) / np.sqrt(100),
    confidence=0.95,
)

fig = plt.figure()
ax = fig.gca()

sns.histplot(norm_dist, bins=50, kde=True, ax=ax)
ax.annotate(text=f"{lower:.4f}", xy=(lower - 0.05, 6.05), ha="right")
ax.annotate(text=f"{upper:.4f}", xy=(upper + 0.05, 6.05))
ax.axvline(x=lower, color="orange")
ax.axvline(x=upper, color="orange")

plt.show()

# %%
(A_low, B_low), (A_up, B_up) = proportion_confint(
    count=[n_a * p_a, n_b * p_b],
    nobs=[n_a, n_b],
    alpha=0.05,
)

print(f"Group A 95% CI: [{A_low:.4f}, {A_up:.4f}]")
print(f"Group B 95% CI: [{B_low:.4f}, {B_up:.4f}]")

# %%
