# %%
import os

os.environ["KERAS_BACKEND"] = "torch"

# %%
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.layers import Softmax
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers import Adam

from camp.datasets.mnist import FashionMNIST

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %%
storage_options = {
    "endpoint_url": os.getenv("S3_ENDPOINT"),
    "key": os.getenv("S3_ACCESS_KEY_ID"),
    "secret": os.getenv("S3_SECRET_ACCESS_KEY"),
}

# %%
dataset = FashionMNIST.load("s3://datasets/fashion_mnist", storage_options)

X_train = dataset["train"] / 255
X_test = dataset["test"] / 255

y_train = dataset["train_labels"]
y_test = dataset["test_labels"]

# %%
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# %%
plt.imshow(dataset["train"][0].view(28, 28))
plt.colorbar()
plt.show()

# %%
fig = plt.figure(figsize=(8, 8), constrained_layout=True)

for i in range(25):
    fig.add_subplot(5, 5, i + 1)

    plt.imshow(X_train[i].view(28, 28), cmap=plt.get_cmap("binary"))
    plt.yticks([])
    plt.xticks([])
    plt.xlabel(class_names[y_train[i]])

plt.show()

# %%
nn = Sequential()
nn.add(Dense(256, activation="relu"))
nn.add(Dense(256, activation="relu"))
nn.add(Dense(10))

nn.compile(
    optimizer=Adam(learning_rate=0.01),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# %%
history = nn.fit(X_train, y_train, epochs=10, validation_split=0.1)

# %%
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Loss vs. Epoch")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["loss", "val_loss"])

plt.show()

# %%
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Accuracy vs. Epoch")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["accuracy", "val_accuracy"])

plt.show()

# %%
test_loss, test_acc = nn.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# %%
nn_proba = Sequential()
nn_proba.add(nn)
nn_proba.add(Softmax())

# %%
predictions = nn_proba.predict(X_test)
print(f"Prediction: {np.argmax(predictions[0])}, Target: {y_test[0]}")

# %%
mask = np.argmax(predictions, axis=1) != y_test.numpy()
incorrect_preds = np.argwhere(mask).reshape(-1)
idx = incorrect_preds[0]

fig = plt.figure(figsize=(6, 3), constrained_layout=True)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.imshow(dataset["test"][idx].view(28, 28), cmap=plt.get_cmap("binary"))

prediction = np.argmax(predictions[idx])
name, true_name = class_names[prediction], class_names[y_test[idx]]
confidence = np.max(predictions[idx]) * 100
color = "blue" if prediction == y_test[idx] else "red"

ax1.set_yticks([])
ax1.set_xticks([])
ax1.set_xlabel(f"{name} {confidence:2.0f}% ({true_name})", color=color)

bar = ax2.bar(range(10), predictions[idx], color="gray")

bar[prediction].set_color("red")
bar[y_test[idx]].set_color("blue")

ax2.set_ylim((0, 1))
ax2.set_xticks(range(10))

plt.show()

# %%
