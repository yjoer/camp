# %%
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

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
# **Max Depth = 3**

# %%
dt = DecisionTreeClassifier(max_depth=3, random_state=12345)
dt.fit(X_train, y_train)

# %%
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)

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
# **Min Samples Leaf = 3**

# %%
dt = DecisionTreeClassifier(min_samples_leaf=3, random_state=12345)
dt.fit(X_train, y_train)

# %%
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)

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

# %%
