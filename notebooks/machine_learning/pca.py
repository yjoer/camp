# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %%
points = make_regression(n_samples=100, n_features=1, noise=5, random_state=26)
points = np.append(points[0], points[1].reshape(-1, 1), axis=1)

ss = StandardScaler()
points = ss.fit_transform(points)

# %%
print(np.corrcoef(points[:, 0], points[:, 1])[0, 1])

plt.scatter(points[:, 0], points[:, 1])
plt.show()

# %%
pca = PCA()
points_t = pca.fit_transform(points)

# %%
print(np.corrcoef(points_t[:, 0], points_t[:, 1])[0, 1])

plt.scatter(points_t[:, 0], points_t[:, 1])
plt.axhline(0, linewidth=0.5)
plt.axvline(0, linewidth=0.5)
plt.show()

# %%
mean = pca.mean_
pcs = pca.components_

plt.scatter(points[:, 0], points[:, 1])
plt.arrow(mean[0], mean[1], pcs[0, 0], pcs[0, 1], color="red", width=0.025)
plt.arrow(mean[0], mean[1], pcs[1, 0], pcs[1, 1], color="red", width=0.025)
plt.axis("equal")
plt.show()

# %%
features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_)
plt.ylabel("Variance")
plt.xlabel("PCA Features")
plt.xticks(features)
plt.show()

# %%
