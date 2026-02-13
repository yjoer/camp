# %%
import os

os.environ["KERAS_BACKEND"] = "torch"

# %%
import datetime as dt
from datetime import datetime

import fsspec
import keras
import matplotlib.pyplot as plt
import psutil
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.models import ResNet50_Weights
from torchvision.models import resnet50

from camp.datasets.soccernet import SoccerNetLegibilityDataset
from camp.utils.jupyter_utils import is_notebook
from camp.utils.torch_utils import save_checkpoint
from solutions.sports.numbers.resnet_pipeline import collate_fn
from solutions.sports.numbers.resnet_pipeline import transforms
from solutions.sports.numbers.resnet_pipeline import validation_loop

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %load_ext autoreload
# %autoreload 2

# %%
OVERFITTING_TEST = False
VALIDATION_SPLIT = False
VALIDATION_SPLIT_TEST = False

DATASET_PATH = "s3://datasets/soccernet_legibility"
CHECKPOINT_PATH = "s3://models/soccernet_legibility/resnet_50"
DATALOADER_WORKERS = psutil.cpu_count(logical=False)

USE_AMP = False

if OVERFITTING_TEST:
  CHECKPOINT_PATH = "s3://models/soccernet_legibility/resnet_50_test"

if VALIDATION_SPLIT:
  CHECKPOINT_PATH = "s3://models/soccernet_legibility/resnet_50_val"

if VALIDATION_SPLIT_TEST:
  CHECKPOINT_PATH = "s3://models/soccernet_legibility/resnet_50_val_test"

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on CPU.") if device.type == "cpu" else print("Running on GPU.")

# %%
storage_options = {
  "endpoint_url": os.getenv("S3_ENDPOINT"),
  "key": os.getenv("S3_ACCESS_KEY_ID"),
  "secret": os.getenv("S3_SECRET_ACCESS_KEY"),
}

if not os.getenv("S3_ENDPOINT"):
  storage_options = {}

# %%
train_dataset: SoccerNetLegibilityDataset | Subset = SoccerNetLegibilityDataset(
  path=DATASET_PATH,
  storage_options=storage_options,
  transforms=transforms,
)

if hasattr(os, "register_at_fork") and hasattr(fsspec, "asyn"):
  os.register_at_fork(after_in_child=fsspec.asyn.reset_lock)

if OVERFITTING_TEST:
  train_dataset = Subset(train_dataset, indices=[0])

if VALIDATION_SPLIT_TEST:
  train_dataset = Subset(train_dataset, indices=list(range(100)))

n_images = len(train_dataset)

if VALIDATION_SPLIT:
  split_point = int(0.8 * n_images)
  train_idx = list(range(split_point))
  val_idx = list(range(split_point, n_images))

  val_dataset = Subset(train_dataset, indices=val_idx)
  train_dataset = Subset(train_dataset, indices=train_idx)

# %%
if is_notebook():
  train_image, train_label = train_dataset[0]

  print(train_label)
  plt.imshow(train_image.permute(1, 2, 0))
  plt.show()

# %%
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

fc_in_features = resnet.fc.in_features
n_classes = 2
resnet.fc = nn.Linear(fc_in_features, n_classes)

resnet = resnet.to(device)

dt_now = datetime.now(tz=dt.UTC)
train_started_at = dt_now.isoformat(timespec="seconds").replace(":", "-")
train_storage_path = f"{CHECKPOINT_PATH}/{train_started_at}"

batch_size = 16
n_epochs = 30
epoch = 0
save_epochs = 5

if OVERFITTING_TEST:
  n_epochs = 10
  save_epochs = 10

if VALIDATION_SPLIT_TEST:
  n_epochs = 5
  save_epochs = 5

# %%
train_dataloader = DataLoader(
  train_dataset,
  batch_size,
  shuffle=True,
  num_workers=DATALOADER_WORKERS,
  collate_fn=collate_fn,
  persistent_workers=True,
)

if VALIDATION_SPLIT:
  val_dataloader = DataLoader(
    val_dataset,
    batch_size,
    num_workers=DATALOADER_WORKERS,
    collate_fn=collate_fn,
    persistent_workers=True,
  )

# %%
params = [p for p in resnet.parameters() if p.requires_grad]
optimizer = optim.AdamW(params, lr=0.001, weight_decay=0.0001)

criterion = nn.BCEWithLogitsLoss()

scaler = torch.GradScaler(device.type, enabled=USE_AMP)

# %%
resnet.train()

for i in range(n_epochs):
  print(f"Epoch: {i + 1}/{n_epochs}")

  step = 1
  pbar = keras.utils.Progbar(len(train_dataloader))

  for batch_images, batch_targets in train_dataloader:
    images = torch.stack(batch_images, dim=0).to(device)

    targets = torch.stack(batch_targets, dim=0)
    targets = F.one_hot(targets, num_classes=n_classes)
    targets = targets.to(device, dtype=torch.float32)

    outputs = resnet(images)
    loss = criterion(outputs, targets)

    pbar.update(current=step, values=[("loss", loss.item())])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad()

    step += 1  # noqa: SIM113

  if ((epoch + 1) % save_epochs) == 0:
    save_checkpoint(
      path=train_storage_path,
      epoch=epoch,
      model=resnet,
      optimizer=optimizer,
      scaler=scaler,
      storage_options=storage_options,
    )

  if VALIDATION_SPLIT:
    metrics_dict = validation_loop(resnet, val_dataloader, device)

  epoch += 1  # noqa: SIM113

# %%
