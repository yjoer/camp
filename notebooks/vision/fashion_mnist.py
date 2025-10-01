# %%
import os

os.environ["KERAS_BACKEND"] = "torch"

# %%
import keras
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchmetrics import Accuracy
from torchmetrics import MetricCollection
from torchmetrics import Precision
from torchmetrics import Recall
from torchvision.transforms import v2

from camp.datasets.mnist import FashionMNIST

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Sample

# %%
storage_options = {
    "endpoint_url": os.getenv("S3_ENDPOINT"),
    "key": os.getenv("S3_ACCESS_KEY_ID"),
    "secret": os.getenv("S3_SECRET_ACCESS_KEY"),
}

# %%
dataset = FashionMNIST.load("s3://datasets/fashion_mnist", storage_options)

# %% [markdown]
# ## Modify

# %%
transforms = v2.Compose(
    [
        v2.Lambda(lambda x: x.view(-1, 1, 28, 28)),
        v2.ToDtype(torch.float32, scale=True),
    ],
)

train_images = transforms(dataset["train"])
test_images = transforms(dataset["test"])

# %%
train_dataset = TensorDataset(train_images, dataset["train_labels"])
test_dataset = TensorDataset(test_images, dataset["test_labels"])

# %%
batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size)

# %% [markdown]
# ## Model

# %%
n_batches = np.ceil(len(train_dataset) / batch_size).astype(np.int32)
n_epochs = 2

# %%
feature_extractor = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
)

classifier = nn.Sequential(nn.Linear(64 * 7 * 7, 10))

# %%
parameters = list(feature_extractor.parameters()) + list(classifier.parameters())
optimizer = optim.Adam(parameters, lr=1e-3)

# %%
for i in range(n_epochs):
    print(f"Epoch: {i + 1}/{n_epochs}")

    steps = 1
    pbar = keras.utils.Progbar(n_batches)

    for images, labels in train_dataloader:
        x = feature_extractor(images)
        x = classifier(x)

        loss = F.cross_entropy(x, labels)
        pbar.update(steps, values=[("loss", loss.item())])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        steps += 1

# %%
metrics = MetricCollection(
    [
        Accuracy(task="multiclass", num_classes=10),
        Precision(task="multiclass", num_classes=10, average=None),
        Recall(task="multiclass", num_classes=10, average=None),
    ],
)

feature_extractor.eval()
classifier.eval()

with torch.no_grad():
    for images, labels in test_dataloader:
        x = feature_extractor(images)
        x = classifier(x)

        _, predictions = torch.max(x, dim=1)
        metrics.update(predictions, labels)

metrics.compute()

# %%
