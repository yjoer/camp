# %%
import altair as alt
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# %%
X, y = make_moons(n_samples=500, noise=0.30, random_state=12345)

# %%
df = pd.concat(
    objs=(
        pd.DataFrame(X, columns=["X1", "X2"]),
        pd.DataFrame(y, columns=["y"]),
    ),
    axis=1,
)

# %%
alt.Chart(df).mark_point().encode(
    x="X1:Q",
    y="X2:Q",
    color="y:N",
    tooltip=["X1", "X2", "y"],
)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=12345,
    stratify=y,
)

# %% [markdown]
# **Default Parameters**

# %%
rf = RandomForestClassifier(n_jobs=-1, random_state=12345)
rf.fit(X_train, y_train)

# %%
y_pred = rf.predict(X_test)
accuracy_score(y_test, y_pred)

# %% [markdown]
# **Grid Search**

# %%
param_grid = {
    "n_estimators": [100, 200, 300, 1000],
    "max_depth": [80, 90, 100, 110],
    "min_samples_split": [8, 10, 12],
    "min_samples_leaf": [3, 4, 5],
    "max_features": [1, 2],
    "bootstrap": [True],
}

if False:
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        n_jobs=-1,
        cv=3,
        verbose=2,
    )

    grid_search.fit(X_train, y_train)

# %%
best_params = {
    "n_estimators": 100,
    "max_depth": 80,
    "min_samples_split": 10,
    "min_samples_leaf": 3,
    "max_features": 2,
    "bootstrap": True,
}

rf = RandomForestClassifier(**best_params, n_jobs=-1, random_state=12345)
rf.fit(X_train, y_train)

# %%
y_pred = rf.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
