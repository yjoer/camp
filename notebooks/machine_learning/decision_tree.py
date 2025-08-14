# %%
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %% [markdown]
# ## Tennis

# %%
entropy = lambda p: -np.sum(p * np.log2(p[p > 0]))

positive_probabilities = np.linspace(0, 1, 50)
negative_probabilities = 1 - positive_probabilities
distributions = np.column_stack((positive_probabilities, negative_probabilities))

plt.plot(positive_probabilities, list(map(entropy, distributions)))
plt.ylabel("Entropy")
plt.xlabel("Proportion of Positive Samples")
plt.show()

# %%
play_tennis_csv = """
day,outlook,temp,humidity,wind,play
1,sunny,hot,high,weak,no
2,sunny,hot,high,strong,no
3,overcast,hot,high,weak,yes
4,rain,mild,high,weak,yes
5,rain,cool,normal,weak,yes
6,rain,cool,normal,strong,no
7,overcast,cool,normal,strong,yes
8,sunny,mild,high,weak,no
9,sunny,cool,normal,weak,yes
10,rain,mild,normal,weak,yes
11,sunny,mild,normal,strong,yes
12,overcast,mild,high,strong,yes
13,overcast,hot,normal,weak,yes
14,rain,mild,high,strong,no
"""

# %%
df_tennis = pd.read_csv(io.StringIO(play_tennis_csv))

# %%
play = df_tennis["play"] == "yes"
n_total = df_tennis.shape[0]
p_positive = df_tennis[play].shape[0] / n_total
p_negative = df_tennis[~play].shape[0] / n_total

p_positive, p_negative

# %%
h = entropy(np.array([p_positive, p_negative]))
h

# %%
wind_wk = df_tennis["wind"] == "weak"
wind_st = df_tennis["wind"] == "strong"

n_wind_wk_total = df_tennis[wind_wk].shape[0]
p_wind_wk_positive = df_tennis[wind_wk & play].shape[0] / n_wind_wk_total
p_wind_wk_negative = df_tennis[wind_wk & ~play].shape[0] / n_wind_wk_total

n_wind_st_total = df_tennis[wind_st].shape[0]
p_wind_st_positive = df_tennis[wind_st & play].shape[0] / n_wind_st_total
p_wind_st_negative = df_tennis[wind_st & ~play].shape[0] / n_wind_st_total

print(p_wind_wk_positive, p_wind_wk_negative)
print(p_wind_st_positive, p_wind_st_negative)

# %%
h_wind_wk = entropy(np.array([p_wind_wk_positive, p_wind_wk_negative]))
h_wind_st = entropy(np.array([p_wind_st_positive, p_wind_st_negative]))

h_wind_wk, h_wind_st

# %%
h_wind_wk_weighted = n_wind_wk_total / n_total * h_wind_wk
h_wind_st_weighted = n_wind_st_total / n_total * h_wind_st

ig_wind = h - h_wind_wk_weighted - h_wind_st_weighted
ig_wind

# %% [markdown]
# ## Iris

# %%
iris = load_iris()

# %%
X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.2,
    random_state=12345,
    stratify=iris.target,
)

# %% [markdown]
# **Default Parameters**

# %%
dt = DecisionTreeClassifier(random_state=12345)
dt.fit(X_train, y_train)

# %%
X_sample = [[5, 5.5, 4, 4.5]]
dt.predict(X_sample), dt.predict_proba(X_sample)

# %%
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
print(classification_report(y_test, y_pred))

# %%
fig = plt.figure(figsize=(4, 4))
ax = plt.gca()

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=iris.target_names,
    ax=ax,
)

plt.show()

# %%
fig = plt.figure(figsize=(12, 8.8))
ax = fig.gca()

plot_tree(
    dt,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    ax=ax,
)

plt.show()

# %% [markdown]
# ### Hyperparameter Tuning

# %% [markdown]
# #### Criterion = Entropy

# %%
dt = DecisionTreeClassifier(criterion="entropy", random_state=12345)
dt.fit(X_train, y_train)

# %%
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
print(classification_report(y_test, y_pred))

# %%
fig = plt.figure(figsize=(10, 7.2))
ax = fig.gca()

plot_tree(
    dt,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    ax=ax,
)

plt.show()

# %% [markdown]
# #### Max Depth = 3

# %%
dt = DecisionTreeClassifier(max_depth=3, random_state=12345)
dt.fit(X_train, y_train)

# %%
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
print(classification_report(y_test, y_pred))

# %%
fig = plt.figure(figsize=(8, 4.8))
ax = fig.gca()

plot_tree(
    dt,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    ax=ax,
)

plt.show()

# %% [markdown]
# #### Min Samples Leaf = 3

# %%
dt = DecisionTreeClassifier(min_samples_leaf=3, random_state=12345)
dt.fit(X_train, y_train)

# %%
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
print(classification_report(y_test, y_pred))

# %%
fig = plt.figure(figsize=(8, 5.6))
ax = fig.gca()

plot_tree(
    dt,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    ax=ax,
)

plt.show()

# %% [markdown]
# ### Grid Search

# %%
parameters = {
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ["best", "random"],
    "max_depth": [1, 2, 3, 4, 5],
    "max_features": [None, "sqrt", "log2"],
}


dt = DecisionTreeClassifier()
gs = GridSearchCV(dt, param_grid=parameters, n_jobs=-1, cv=5, scoring="accuracy")
gs.fit(X_train, y_train)
gs.best_params_

# %%
y_pred = gs.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
print(classification_report(y_test, y_pred))

# %%
