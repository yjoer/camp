# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.graphics.gofplots import qqplot
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.proportion import proportions_ztest

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %% [markdown]
# ## Normal Distribution

# %%
dist_normal = st.norm.rvs(size=100, random_state=26)
dist_skewed = st.skewnorm.rvs(-25, size=100, random_state=26)

# %%
fig = plt.figure(figsize=(6, 3), constrained_layout=True)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, sharey=ax1)
ax2.tick_params(labelleft=False)

sns.kdeplot(data=dist_normal, ax=ax1)
sns.kdeplot(data=dist_skewed, ax=ax2)
plt.show()

# %%
fig = plt.figure(figsize=(6, 3), constrained_layout=True)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, sharey=ax1)
ax2.tick_params(labelleft=False)
ax2.yaxis.label.set_visible(False)

qqplot(dist_normal, line="s", dist=st.distributions.norm, ax=ax1)
qqplot(dist_skewed, line="s", dist=st.distributions.norm, ax=ax2)
plt.show()

# %% [markdown]
# ### Shapiro-Wilk Test

# %% [markdown]
# $H_0 = \text{Data is drawn from a normal distribution}$

# %%
outcomes = ["Fail to reject the null hypothesis", "Reject the null hypothesis"]

# %%
alpha = 0.05
test_statistic, p_value = st.shapiro(dist_normal)

print(test_statistic, p_value)
print(outcomes[0]) if p_value > alpha else print(outcomes[1])

# %%
alpha = 0.05
test_statistic, p_value = st.shapiro(dist_skewed)

print(test_statistic, p_value)
print(outcomes[0]) if p_value > alpha else print(outcomes[1])

# %% [markdown]
# ### Anderson-Darling Test

# %%
statistic, critical_values, significance_level = st.anderson(dist_normal, dist="norm")

print(statistic, critical_values, significance_level)
print(outcomes[0]) if np.all(statistic < critical_values) else print(outcomes[1])

# %%
statistic, critical_values, significance_level = st.anderson(dist_skewed, dist="norm")

print(statistic, critical_values, significance_level)
print(outcomes[0]) if np.all(statistic < critical_values) else print(outcomes[1])

# %% [markdown]
# ## Correlation

# %%
phi = 0.95
dist_ar = np.zeros(len(dist_skewed))

for i in range(1, len(dist_skewed)):
    dist_ar[i] = phi * dist_ar[i - 1] + dist_skewed[i]

# %%
fig = plt.figure(figsize=(6, 3), constrained_layout=True)

ax1 = fig.add_subplot(121)
ax1.plot(dist_skewed, label="Skewed Noises")
ax1.plot(dist_ar, label="AR(1)")
ax1.legend()

ax2 = fig.add_subplot(122, sharey=ax1)
ax2.tick_params(labelleft=False)
lag = 10
ax2.plot(np.arange(len(dist_ar))[:-lag], dist_ar[:-lag], label="Lagging")
ax2.plot(np.arange(len(dist_ar))[lag:], dist_ar[lag:], label="Leading", alpha=0.8)
ax2.legend()

plt.show()

# %%
r, p_value = st.pearsonr(dist_ar[:-lag], dist_ar[lag:])

print(r)
print(r**2)
print("Autocorrelated" if p_value < 0.05 else "No autocorrelation")

# %% [markdown]
# ## Means

# %% [markdown]
# ### Independent Samples t-Test

# %% [markdown]
# $H_0 = \text{Two independent samples have identical means}$

# %%
rng = np.random.default_rng(seed=26)
dist_x = rng.uniform(low=0, high=1, size=100)
dist_y = rng.uniform(low=1, high=2, size=100)
dist_z = rng.uniform(low=2, high=3, size=100)

# %%
alpha = 0.05
statistic, p_value = st.ttest_ind(dist_x, dist_y)

print(statistic, p_value)
print(outcomes[0]) if p_value > alpha else print(outcomes[1])

# %% [markdown]
# ### Mann-Whitney U Test

# %%
alpha = 0.05
statistic, p_value = st.mannwhitneyu(dist_x, dist_y)

print(statistic, p_value)
print(outcomes[0]) if p_value > alpha else print(outcomes[1])

# %% [markdown]
# ### ANOVA

# %% [markdown]
# $H_0 = \text{Two or more groups have the same population mean}$

# %%
dist_df = pd.DataFrame(
    {
        "blocks": np.hstack(
            (
                np.full((100), 1, dtype=np.uint8),
                np.full((100), 2, dtype=np.uint8),
                np.full((100), 3, dtype=np.uint8),
            )
        ),
        "scores": np.hstack((dist_x, dist_y, dist_z)),
        "treatment": rng.choice(["a", "b", "c"], size=300),
    }
)

dist_df = dist_df.sort_values(by=["blocks", "treatment"])
dist_df.groupby(["blocks", "treatment"]).first()

# %% [markdown]
# #### ANOVA Within Blocks

# %%
sns.boxplot(data=dist_df, x="blocks", y="scores", hue="treatment")
plt.show()

# %%
alpha = 0.05

dist_df.groupby("blocks").apply(
    lambda x: pd.Series(
        st.f_oneway(
            x[x["treatment"] == "a"]["scores"],
            x[x["treatment"] == "b"]["scores"],
            x[x["treatment"] == "c"]["scores"],
        )
    ),
    include_groups=False,
).rename({0: "statistic", 1: "p_value"}, axis=1).assign(
    outcome=lambda x: np.where(x["p_value"] > alpha, outcomes[0], outcomes[1])
)

# %% [markdown]
# #### ANOVA Between Blocks

# %%
sns.boxplot(data=dist_df, x="blocks", y="scores")
plt.show()

# %%
alpha = 0.05

statistic, p_value = st.f_oneway(
    dist_df[dist_df["blocks"] == 1]["scores"],
    dist_df[dist_df["blocks"] == 2]["scores"],
    dist_df[dist_df["blocks"] == 3]["scores"],
)

print(statistic, p_value)
print(outcomes[0]) if p_value > alpha else print(outcomes[1])

# %% [markdown]
# ### Kruskal-Wallis Test

# %%
alpha = 0.05
statistic, p_value = st.kruskal(
    dist_df[dist_df["blocks"] == 1]["scores"],
    dist_df[dist_df["blocks"] == 2]["scores"],
    dist_df[dist_df["blocks"] == 3]["scores"],
)

print(statistic, p_value)
print(outcomes[0]) if p_value > alpha else print(outcomes[1])

# %% [markdown]
# ## Proportions

# %% [markdown]
# ### Z-Test

# %%
alpha = 0.05
z_stat, p_value = proportions_ztest([800, 850], nobs=[1000, 1000])

print(z_stat, p_value)
print(outcomes[0]) if p_value > alpha else print(outcomes[1])

# %% [markdown]
# ## Post-Hoc Tests

# %% [markdown]
# ### Tukey's HSD

# %%
hsd = pairwise_tukeyhsd(dist_df["scores"], dist_df["blocks"], alpha=0.05)
print(hsd)

# %% [markdown]
# ### Bonferroni Correction

# %%
alpha = 0.05

x = np.linspace(1, 20, 20)
y = 1 - (1 - alpha) ** x
y_corr = 1 - (1 - alpha / x) ** x

fig = plt.figure()
ax = fig.gca()

ax.plot(x, y, marker="o", label="Uncorrected")
ax.plot(x, y_corr, marker="o", label="Bonferroni Correction")

(xmin, xmax), (ymin, ymax) = ax.get_xlim(), ax.get_ylim()
xmax_rel = (x[9] - xmin) / (xmax - xmin)
ymax_rel = (y[9] - ymin) / (ymax - ymin)

ax.axvline(x=x[9], ymin=0, ymax=ymax_rel, linestyle="--")
ax.axhline(y=y[9], xmin=0, xmax=xmax_rel, linestyle="--")
ax.annotate(f"{y[9]:.4f}", (x[9] - 0.01, y[9] + 0.01), ha="right")

ax.axhline(y=y_corr[9], xmax=xmax_rel, linestyle="--")
ax.annotate(f"{y_corr[9]:.4f}", (x[9] - 0.01, y_corr[9] + 0.01), ha="right")

ax.set_title("FWER vs. Number of Tests")
ax.set_ylabel("FWER")
ax.set_xlabel("Number of Tests")
ax.legend()
plt.show()

# %%
comparisons = [(1, 2), (1, 3), (2, 3)]
p_values = []

for c in comparisons:
    a = dist_df[dist_df["blocks"] == c[0]]["scores"]
    b = dist_df[dist_df["blocks"] == c[1]]["scores"]
    statistic, p_value = st.ttest_ind(a, b)
    p_values.append(p_value)

# %%
_, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method="bonferroni")

print(p_values)
print(p_values_corrected)


# %% [markdown]
# ## Power Analysis


# %%
def cohens_d(a, b):
    diff = np.mean(a) - np.mean(b)
    n1, n2 = len(a), len(b)
    var1, var2 = np.var(a), np.var(b)

    std_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = diff / std_pooled
    return d


# %%
def cohens_h(a, b):
    return 2 * np.arcsin(np.sqrt(a)) - 2 * np.arcsin(np.sqrt(b))


# %%
d = cohens_d(dist_x, dist_y)

power_analysis = TTestIndPower()
n_req = power_analysis.solve_power(effect_size=d, alpha=0.05, power=0.99, ratio=1)

print(d, n_req)

# %%
effect_size = proportion_effectsize(0.3, 0.7)
h = cohens_h(0.3, 0.7)

print(effect_size)
print(effect_size == h)

# %%
power_analysis.plot_power(nobs=np.arange(2, 100), effect_size=[0.2, 0.5, 0.8])
plt.show()

# %% [markdown]
# ## Variances

# %% [markdown]
# ### Levene Test

# %% [markdown]
# $H_0 = \text{All input samples are from populations with equal variances}$

# %%
alpha = 0.05
statistic, p_value = st.levene(dist_x, dist_y)

print(statistic, p_value)
print(outcomes[0]) if p_value > alpha else print(outcomes[1])

# %% [markdown]
# ## Covariances

# %% [markdown]
# ### ANCOVA

# %%
dist_df["covariate"] = rng.choice([1, 2, 3, 4, 5, 6, 7, 8], size=300)

# %%
model = ols("scores ~ treatment + covariate", data=dist_df).fit()
model.summary()

# %%
sns.lmplot(data=dist_df, x="covariate", y="scores", hue="treatment")
plt.show()

# %%
