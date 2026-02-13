# %%
import os

os.environ["KERAS_BACKEND"] = "torch"

# %%
import json
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2.functional as tvf
from safetensors.torch import load
from safetensors.torch import save
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchmetrics.detection import IntersectionOverUnion
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from tqdm.auto import tqdm

from camp.datasets.vitrox import VitroxBody1k
from camp.utils.torch_utils import load_optimizer
from camp.utils.torch_utils import save_optimizer

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %load_ext autoreload
# %autoreload 2

# %%
storage_options = {
  "endpoint_url": os.getenv("S3_ENDPOINT"),
  "key": os.getenv("S3_ACCESS_KEY_ID"),
  "secret": os.getenv("S3_SECRET_ACCESS_KEY"),
}

# %%
dataset = VitroxBody1k.load("s3://datasets/vitrox_body_1k", storage_options)


# %%
class CustomDataset(Dataset):
  def __init__(self, dataset: dict, subset: str) -> None:
    self.images = dataset[subset]
    self.labels = dataset[f"{subset}_labels"]
    self.boxes = dataset[f"{subset}_boxes"]

  def __len__(self) -> int:
    return len(self.images)

  def __getitem__(self, idx: int) -> tuple:
    max_size = 224
    output_size = (224, 224)

    image = self.images[idx]
    width, height = image.size

    labels = self.labels[idx]
    labels = torch.tensor(labels, dtype=torch.int64)

    boxes = torch.tensor(self.boxes[idx])
    boxes = box_convert(boxes, "cxcywh", "xyxy")

    if width > 224 or height > 224:
      image = tvf.resize(image, size=None, max_size=max_size)

      scale_factor = max_size / width if width > height else max_size / height

      boxes = torch.mul(boxes, scale_factor)
      width, height = image.size

    image = tvf.center_crop(image, output_size)

    output_width, output_height = output_size
    left, top = (width - output_width) // 2, (height - output_height) // 2
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top

    image = tvf.to_image(image)
    image = tvf.to_dtype(image, dtype=torch.float32, scale=True)

    return image, {"labels": labels, "boxes": boxes}


# %%
train_dataset = CustomDataset(dataset, subset="train")
test_dataset = CustomDataset(dataset, subset="test")

# %%
train_image, train_target = train_dataset[0]

train_image_preview = draw_bounding_boxes(
  image=train_image,
  boxes=train_target["boxes"],
  labels=[str(x.item()) for x in train_target["labels"]],
  colors="lime",
)

plt.imshow(train_image_preview.permute(1, 2, 0))
plt.show()

# %%
batch_size = 8


def collate_fn(batch: list) -> tuple:
  return tuple(zip(*batch, strict=False))


train_dataloader = DataLoader(
  train_dataset,
  batch_size,
  shuffle=True,
  collate_fn=collate_fn,
)

test_dataloader = DataLoader(test_dataset, batch_size, collate_fn=collate_fn)

# %%
model = fasterrcnn_mobilenet_v3_large_fpn(
  weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

# %%
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

n_batches = np.ceil(len(train_dataset) / batch_size).astype(np.int32)

n_epochs = 20
epochs = 0

# %%
model.train()

for i in range(epochs, n_epochs):
  print(f"Epoch: {i + 1}/{n_epochs}, Learning Rate: {lr_scheduler.get_last_lr()}")

  steps = 1
  pbar = keras.utils.Progbar(n_batches)

  for images, targets in train_dataloader:
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    pbar.update(steps, values=[("loss", losses.item())])

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    steps += 1  # noqa: SIM113

  lr_scheduler.step()
  epochs += 1

# %%
checkpoint_path = f"./checkpoint-{epochs}"
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

save_optimizer(checkpoint_path, optimizer)

Path(f"{checkpoint_path}/scheduler.json").write_text(
  json.dumps(lr_scheduler.state_dict()),
  encoding="utf-8",
)

Path(f"{checkpoint_path}/model.safetensors").write_bytes(save(model.state_dict()))

# %%
epochs = 20
checkpoint_path = f"./checkpoint-{epochs}"

load_optimizer(checkpoint_path, optimizer)

scheduler_json = Path(f"{checkpoint_path}/scheduler.json").read_text(encoding="utf-8")
lr_scheduler.load_state_dict(json.loads(scheduler_json))

model.load_state_dict(load(Path(f"{checkpoint_path}/model.safetensors").read_bytes()))

# %%
# %%time
metric = IntersectionOverUnion()

model.eval()
predictions = []

with torch.no_grad():
  for images, targets in tqdm(test_dataloader):
    batch_predictions = model(images)
    batch_predictions_max = []

    for prediction in batch_predictions:
      if prediction["scores"].numel() == 0:
        prediction_dict = {
          "boxes": torch.tensor([[0, 0, 0, 0]]),
          "labels": torch.tensor([0]),
          "scores": torch.tensor([0]),
        }
      else:
        idx = torch.argmax(prediction["scores"])

        prediction_dict = {
          "boxes": prediction["boxes"][idx : idx + 1],
          "labels": prediction["labels"][idx : idx + 1],
          "scores": prediction["scores"][idx : idx + 1],
        }

      batch_predictions_max.append(prediction_dict)
      predictions.append(prediction_dict)

    metric.update(batch_predictions_max, targets)

# %%
metric.compute()

# %%
test_idx = 0
test_image, test_target = test_dataset[test_idx]

test_image_preview = draw_bounding_boxes(
  image=test_image,
  boxes=test_target["boxes"],
  labels=[str(x.item()) for x in test_target["labels"]],
  colors="lime",
)

test_image_predicted = draw_bounding_boxes(
  image=test_image,
  boxes=predictions[test_idx]["boxes"],
  labels=[str(x.item()) for x in predictions[test_idx]["labels"]],
  colors="lime",
)

plt.figure(figsize=(8, 4.8))
plt.subplot(1, 2, 1)
plt.imshow(test_image_preview.permute(1, 2, 0))
plt.title("Ground Truth")

plt.subplot(1, 2, 2)
plt.imshow(test_image_predicted.permute(1, 2, 0))
plt.title("Prediction")
plt.show()

# %%
