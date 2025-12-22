# %%
import io

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.datasets import make_moons
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %% [markdown]
# ## Linear Regression

# %% [markdown]
# ### Tollbooth

# %%
df_tollbooth = pd.DataFrame(
  {
    "age": [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6],
    "speed": [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86],
  },
)

# %%
plt.scatter(df_tollbooth["age"], df_tollbooth["speed"])
plt.ylabel("Speed")
plt.xlabel("Age")
plt.show()

# %%
slope, intercept, r, p, se = stats.linregress(
  df_tollbooth["age"],
  df_tollbooth["speed"],
)

# %%
pd.DataFrame([slope, intercept, r, p, se]).T

# %%
lr = lambda x: slope * x + intercept
y_pred = list(map(lr, df_tollbooth["age"]))

# %%
plt.scatter(df_tollbooth["age"], df_tollbooth["speed"])
plt.plot(df_tollbooth["age"], y_pred)

age = 10
speed = lr(age)

ymin, ymax = plt.ylim()
xmin, xmax = plt.xlim()
ymax_relative = (speed - ymin) / (ymax - ymin)
xmax_relative = (age - xmin) / (xmax - xmin)
y_margin = (ymax - ymin) * 0.01
x_margin = (xmax - xmin) * 0.01
plt.axvline(x=age, ymin=0, ymax=ymax_relative, color="orange", linestyle="--")
plt.axhline(y=speed, xmin=0, xmax=xmax_relative, color="orange", linestyle="--")
plt.annotate(speed, (age + x_margin, speed + y_margin))

plt.ylabel("Speed")
plt.xlabel("Age")
plt.show()

# %%
residuals = df_tollbooth["speed"] - y_pred
residuals

# %% [markdown]
# ## Multiple Linear Regression

# %% [markdown]
# ### Cars

# %%
cars_csv = """
Car,Model,Volume,Weight,CO2
Toyota,Aygo,1000,790,99
Mitsubishi,Space Star,1200,1160,95
Skoda,Citigo,1000,929,95
Fiat,500,900,865,90
Mini,Cooper,1500,1140,105
VW,Up!,1000,929,105
Skoda,Fabia,1400,1109,90
Mercedes,A-Class,1500,1365,92
Ford,Fiesta,1500,1112,98
Audi,A1,1600,1150,99
Hyundai,I20,1100,980,99
Suzuki,Swift,1300,990,101
Ford,Fiesta,1000,1112,99
Honda,Civic,1600,1252,94
Hyundai,I30,1600,1326,97
Opel,Astra,1600,1330,97
BMW,1,1600,1365,99
Mazda,3,2200,1280,104
Skoda,Rapid,1600,1119,104
Ford,Focus,2000,1328,105
Ford,Mondeo,1600,1584,94
Opel,Insignia,2000,1428,99
Mercedes,C-Class,2100,1365,99
Skoda,Octavia,1600,1415,99
Volvo,S60,2000,1415,99
Mercedes,CLA,1500,1465,102
Audi,A4,2000,1490,104
Audi,A6,2000,1725,114
Volvo,V70,1600,1523,109
BMW,5,2000,1705,114
Mercedes,E-Class,2100,1605,115
Volvo,XC70,2000,1746,117
Ford,B-Max,1600,1235,104
BMW,2,1600,1390,108
Opel,Zafira,1600,1405,109
Mercedes,SLK,2500,1395,120
"""

# %%
df_cars = pd.read_csv(io.StringIO(cars_csv))

# %%
df_cars.head()

# %%
X = df_cars[["Weight", "Volume"]].to_numpy()
y = df_cars["CO2"].to_numpy()

# %%
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.ylabel("Volume")
plt.xlabel("Weight")
plt.colorbar(label="CO2")
plt.show()

# %%
lr = LinearRegression()
lr.fit(X, y)

# %%
X_sample = np.array([[2300, 1300]])
lr.predict(X_sample)

# %%
lr.coef_, lr.intercept_

# %%
X_sample = np.array([[3300, 1300]])
lr.predict(X_sample)

# %%
np.sum(np.multiply(X_sample, lr.coef_)) + lr.intercept_

# %%
X1_range = np.min(X[:, 0]), np.max(X[:, 0])
X2_range = np.min(X[:, 1]), np.max(X[:, 1])
X1_mesh, X2_mesh = np.meshgrid(X1_range, X2_range)

X_mesh = np.hstack((X1_mesh.reshape(-1, 1), X2_mesh.reshape(-1, 1)))
y_pred_mesh = lr.predict(X_mesh).reshape(X1_mesh.shape)

fig = plt.figure()
ax: Axes3D = fig.add_subplot(projection="3d")

ax.scatter(X[:, 0], X[:, 1], y)
ax.plot_surface(X1_mesh, X2_mesh, y_pred_mesh, alpha=0.1)

ax.set_zlabel("CO2", rotation=90)
ax.set_ylabel("Volume")
ax.set_xlabel("Weight")
ax.view_init(elev=30, azim=135)
plt.show()

# %% [markdown]
# ## Polynomial Regression

# %%
_hours = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
_speed = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]

df_tollbooth = pd.DataFrame({"hours": _hours, "speed": _speed})

# %%
plt.scatter(df_tollbooth["hours"], df_tollbooth["speed"])
plt.ylabel("Speed")
plt.xlabel("Hours")
plt.show()

# %%
pr = np.poly1d(np.polyfit(df_tollbooth["hours"], df_tollbooth["speed"], deg=3))

# %%
X_sample = np.linspace(1, 22, 100)
y_pred = pr(X_sample)

# %%
plt.scatter(df_tollbooth["hours"], df_tollbooth["speed"])
plt.plot(X_sample, y_pred)

hour = 17
speed = pr(hour)

ymin, ymax = plt.ylim()
xmin, xmax = plt.xlim()
ymax_relative = (speed - ymin) / (ymax - ymin)
xmax_relative = (hour - xmin) / (xmax - xmin)
y_margin = (ymax - ymin) * 0.01
x_margin = (xmax - xmin) * 0.01
plt.axvline(x=hour, ymin=0, ymax=ymax_relative, color="orange", linestyle="--")
plt.axhline(y=speed, xmin=0, xmax=xmax_relative, color="orange", linestyle="--")
plt.annotate(speed, (hour + x_margin, speed + y_margin))

plt.ylabel("Speed")
plt.xlabel("Hours")
plt.show()

# %%
r2_score(df_tollbooth["speed"], pr(df_tollbooth["hours"]))

# %% [markdown]
# ## Logistic Regression

# %% [markdown]
# ### Moons

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

# %%
logreg = LogisticRegression(random_state=12345)
logreg.fit(X_train, y_train)

# %%
y_pred = logreg.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm

# %%
fig = plt.figure(figsize=(4, 4))
ax = fig.gca()

ConfusionMatrixDisplay(cm).plot(ax=ax)

plt.show()

# %%
accuracy_score(y_test, y_pred)

# %%
precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)

# %% [markdown]
# ### Iris

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

# %%
logreg = LogisticRegression(random_state=12345)
logreg.fit(X_train, y_train)

# %%
X_sample = np.array([[5, 5.5, 4, 4.5]])
logreg.predict(X_sample), logreg.predict_proba(X_sample)

# %%
logreg.coef_

# %%
logreg.intercept_

# %%
softmax = lambda x: np.exp(x) / np.exp(x).sum(axis=0)
softmax((np.matmul(X_sample, logreg.coef_.T) + logreg.intercept_).reshape(-1))

# %%
