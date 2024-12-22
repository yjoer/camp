# %%
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %% [markdown]
# ## Judgment List

# %%
df_labels = pd.DataFrame(
    [
        {"query_id": 1, "query": "red hats", "score": 0.9, "doc_id": 1},
        {"query_id": 1, "query": "red hats", "score": 0.8, "doc_id": 2},
        {"query_id": 1, "query": "red hats", "score": 0.1, "doc_id": 3},
        {"query_id": 2, "query": "green hats", "score": 1.0, "doc_id": 4},
        {"query_id": 2, "query": "green hats", "score": 0.9, "doc_id": 5},
        {"query_id": 2, "query": "green hats", "score": 0.8, "doc_id": 6},
        {"query_id": 2, "query": "green hats", "score": 0.7, "doc_id": 7},
        {"query_id": 2, "query": "green hats", "score": 0.1, "doc_id": 8},
    ]
)

df_labels

# %% [markdown]
# ## Search Results

# %%
df_results = pd.DataFrame(
    [
        {"query_id": 1, "rank": 1, "query": "red hats", "doc_id": 2},
        {"query_id": 1, "rank": 2, "query": "red hats", "doc_id": 3},
        {"query_id": 2, "rank": 1, "query": "green hats", "doc_id": 5},
        {"query_id": 2, "rank": 2, "query": "green hats", "doc_id": 9},
        {"query_id": 2, "rank": 3, "query": "green hats", "doc_id": 6},
    ]
)

df_results

# %%
df_labeled = df_results.merge(df_labels, how="left", on=["query_id", "query", "doc_id"])
df_labeled

# %% [markdown]
# ## Discounted Cumulative Gain

# %% [markdown]
# $$
# \begin{flalign}
# DCG_k = \sum_{i=1}^{k}\frac{r_i}{log_2(i + 1)} &&
# \end{flalign}
# $$
#
# An alternative with a strong emphasis on retrieving relevant documents:
#
# $$
# \begin{flalign}
# DCG_k = \sum_{i=1}^{k}\frac{2^{r_i} - 1}{log_2(i + 1)} &&
# \end{flalign}
# $$

# %%
x = np.arange(0, 1.01, 0.01)
y1 = x
y2 = 2**x - 1

plt.plot(x, y1, label="x")
plt.plot(x, y2, label="2**x - 1")
plt.axvline(x=0.5, alpha=0.5, linestyle="--")
plt.annotate(y1[50], (0.5 - 0.01, y1[50]), ha="right")
plt.annotate(f"{y2[50]:.4f}", (0.5 + 0.01, y2[50]), va="top")
plt.legend()
plt.show()

# %%
fig = plt.figure(figsize=(8, 4), constrained_layout=True)
ax1, ax3 = fig.add_subplot(121), fig.add_subplot(122)

x1 = np.arange(0.01, 5, 0.01)
y1 = np.log(x1)
y2 = np.divide(1, y1, where=y1 != 0, out=np.full_like(y1, np.inf))

ax2 = ax1.twinx()
line1 = ax1.plot(x1, y1, label="log(x)")
line2 = ax2.plot(x1, y2, color="tab:orange", label="1/log(x)")

asymp1 = ax1.axvline(x=0, alpha=0.08, color="red", linestyle="--", label="Asymptote 1")
asymp2 = ax1.axvline(x=1, alpha=0.08, color="red", linestyle="--", label="Asymptote 2")

lines = line1 + line2 + [asymp1, asymp2]
labels = cast(list[str], [l.get_label() for l in lines])
ax1.legend(lines, labels, loc="upper right")

x2 = np.arange(1, 10, 0.01)
y3 = np.log(x2)
y4 = np.divide(1, y3, where=y3 != 0, out=np.full_like(y3, np.inf))

ax4 = ax3.twinx()
line3 = ax3.plot(x2, y3, label="log(x)")
line4 = ax4.plot(x2, y4, color="tab:orange", label="1/log(x)")

lines = line3 + line4
labels = cast(list[str], [l.get_label() for l in lines])
ax3.legend(lines, labels, loc="upper right")

plt.show()

# %%
df_labeled["gain"] = 2 ** df_labeled["score"] - 1
df_labeled["discount"] = 1 / np.log(df_labeled["rank"] + 1)
df_labeled["discounted_gain"] = df_labeled["gain"] * df_labeled["discount"]
df_labeled

# %% [markdown]
# ### Zero Filling

# %%
df_labeled_z = df_labeled.fillna({"discounted_gain": 0})
df_labeled_z

# %%
dcg_z = df_labeled_z.groupby("query_id")["discounted_gain"].sum()
dcg_z

# %% [markdown]
# ### Deletion

# %%
df_labeled_d = df_labeled.dropna(subset="discounted_gain")
df_labeled_d

# %%
df_labeled_d.loc[:, "rank"] = df_labeled_d.groupby("query_id").cumcount() + 1
df_labeled_d.loc[:, "discount"] = 1 / np.log(df_labeled_d["rank"] + 1)
df_labeled_d.loc[:, "discounted_gain"] = df_labeled_d["gain"] * df_labeled_d["discount"]
df_labeled_d

# %%
dcg_d = df_labeled_d.groupby("query_id")["discounted_gain"].sum()
dcg_d

# %% [markdown]
# ## Normalized DCG

# %% [markdown]
# ### Local Ideal
#
# Ideal ranking over current search results measuring the precision within the result set.

# %%
df_labeled_local = df_labeled_d.sort_values(
    by=["query_id", "gain"],
    ascending=[True, False],
)

# %%
rank = df_labeled_local.groupby("query_id").cumcount() + 1
gain = df_labeled_local["gain"]
discount = 1 / np.log(rank + 1)

df_labeled_local.loc[:, "ideal_rank"] = rank
df_labeled_local.loc[:, "ideal_discount"] = discount
df_labeled_local.loc[:, "ideal_discounted_gain"] = gain * discount

df_labeled_local

# %%
idcg_local = df_labeled_local.groupby("query_id")["ideal_discounted_gain"].sum()
ndcg_local = dcg_d / idcg_local

pd.concat((dcg_d, idcg_local, ndcg_local.rename("ndcg")), axis=1)

# %% [markdown]
# ### Global Ideal
#
# Ideal ranking over all labels measuring the recall relative to the known ideal results.

# %%
df_labeled_global = df_labels.sort_values(
    by=["query_id", "score"],
    ascending=[True, False],
)

# %%
rank = df_labeled_global.groupby("query_id").cumcount() + 1
gain = 2 ** df_labeled_global["score"] - 1
discount = 1 / np.log(rank + 1)

df_labeled_global.loc[:, "ideal_rank"] = rank
df_labeled_global.loc[:, "ideal_discount"] = discount
df_labeled_global.loc[:, "ideal_discounted_gain"] = gain * discount

df_labeled_global

# %%
idcg_global = df_labeled_global.groupby("query_id")["ideal_discounted_gain"].sum()
ndcg_global = dcg_d / idcg_global

pd.concat((dcg_d, idcg_global, ndcg_global.rename("ndcg")), axis=1)

# %% [markdown]
# ### Max Ideal
#
# The ideal case is to have maximum relevance scores in all positions. This measures the recall and whether the system has relevant content for this query.

# %%
max_score = 1.0
max_gain = 2**max_score - 1

df_labeled_max = df_labeled_d.copy()
df_labeled_max.loc[:, "ideal_discounted_gain"] = df_labeled_max["discount"] * max_gain
df_labeled_max

# %%
idcg_max = df_labeled_max.groupby("query_id")["ideal_discounted_gain"].sum()
ndcg_max = dcg_d / idcg_max

pd.concat((dcg_d, idcg_max, ndcg_max.rename("ndcg")), axis=1)

# %% [markdown]
# ## NDCG@k

# %% [markdown]
# ### DCG

# %%
k = 10

df_labeled_k = df_labeled_d[df_labeled_d["rank"] < k]
dcg_k = df_labeled_k.groupby("query_id")["discounted_gain"].sum()

# %% [markdown]
# ### IDCG

# %% [markdown]
# #### Max Ideal

# %%
max_score = 1.0
max_gain = 2**max_score - 1
idcg_max_k_ = 0

for i in range(1, k + 1):
    discount = 1 / np.log(i + 1)
    idcg_max_k_ += max_gain * discount

idcg_max_k = dcg_k.copy().rename("ideal_discounted_gain")
idcg_max_k[:] = idcg_max_k_

ndcg_k = dcg_k / idcg_max_k
pd.concat((dcg_k, idcg_max_k, ndcg_k.rename("ndcg")), axis=1)

# %% [markdown]
# ## Jaccard Similarity

# %%
df_jaccard = (
    df_labels.groupby(["query_id", "query"])["doc_id"]
    .agg(set)
    .to_frame()
    .rename({"doc_id": "ideal"}, axis=1)
)

df_jaccard["results"] = df_results.groupby(["query_id", "query"])["doc_id"].agg(set)
df_jaccard["intersection"] = df_jaccard.apply(lambda x: x.ideal & x.results, axis=1)
df_jaccard["union"] = df_jaccard.apply(lambda x: x.ideal | x.results, axis=1)

iou = df_jaccard["intersection"].apply(len) / df_jaccard["union"].apply(len)
df_jaccard["similarity"] = iou

df_jaccard

# %%
