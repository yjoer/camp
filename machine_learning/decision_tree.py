# %%
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
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
)

# %%
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# %%
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
fig = plt.figure(figsize=(10, 8))
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
