# %%
import os

os.environ["KERAS_BACKEND"] = "torch"

# %%
import numpy as np
from keras.layers import Dense
from keras.layers import Input
from keras.models import Sequential
from keras.optimizers import Adam

# %%
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32)
y = np.array([[0], [1], [1], [0]], np.float32)

# %%
nn = Sequential()
nn.add(Input((2,)))
nn.add(Dense(5, activation="relu", kernel_initializer="random_normal"))
nn.add(Dense(1, activation="sigmoid", kernel_initializer="random_normal"))

opt = Adam(learning_rate=0.01)
nn.compile(opt, loss="mean_squared_error", metrics=["binary_accuracy"])
nn.summary()

# %%
history = nn.fit(X, y, epochs=1000, verbose=0)

# %%
print(history.history["loss"][-1])
print(history.history["binary_accuracy"][-1])

# %%
y_pred = nn.predict(X)

for i in range(4):
    print(f"Input: {X[i]}, Output: {y_pred[i]}")

# %%
