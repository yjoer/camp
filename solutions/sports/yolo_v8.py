# %%
import os

os.environ["KERAS_BACKEND"] = "torch"

# %%
import json
from datetime import datetime
from typing import Any
from typing import Union

import fsspec
import keras
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2.functional as tvf
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.torch_utils import ModelEMA

from camp.datasets.ikcest import IKCESTDetectionDataset
from camp.datasets.utils import resize_image_and_boxes
from camp.models.yolo.yolo_utils import YOLOv8DetectionLoss
from camp.models.yolo.yolo_utils import YOLOv8DetectionPredictor
from camp.utils.torch_utils import save_checkpoint
from camp.utils.torch_utils import save_initial_weights

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %load_ext autoreload
# %autoreload 2

# %%
OVERFITTING_TEST = False
VALIDATION_SPLIT = False
VALIDATION_SPLIT_TEST = False

TRAIN_DATASET_PATH = "s3://datasets/ikcest_2024"
CHECKPOINT_PATH = "s3://models/ikcest_2024/yolo_v8"

if OVERFITTING_TEST:
    CHECKPOINT_PATH = "s3://models/ikcest_2024/yolo_v8_test"

# %%
storage_options = {
    "endpoint_url": os.getenv("S3_ENDPOINT"),
    "key": os.getenv("S3_ACCESS_KEY_ID"),
    "secret": os.getenv("S3_SECRET_ACCESS_KEY"),
}


# %%
def transforms(image, target):
    boxes = box_convert(target["boxes"], "xywh", "xyxy")
    max_size = 640
    output_size = (384, 640)

    image, boxes = resize_image_and_boxes(image, boxes, max_size, output_size)
    target["boxes"] = boxes

    image = tvf.to_image(image)
    image = tvf.to_dtype(image, dtype=torch.float32, scale=True)

    return image, target


# %%
train_dataset: Union[IKCESTDetectionDataset, Subset] = IKCESTDetectionDataset(
    path=TRAIN_DATASET_PATH,
    subset="train",
    storage_options=storage_options,
    transforms=transforms,
)

if OVERFITTING_TEST:
    train_dataset = Subset(train_dataset, indices=[0])

if VALIDATION_SPLIT_TEST:
    train_dataset = Subset(train_dataset, indices=list(range(20)))

if VALIDATION_SPLIT:
    n_images = len(train_dataset)
    split_point = int(0.8 * n_images)
    train_idx = list(range(split_point))
    val_idx = list(range(split_point, n_images))

    val_dataset = Subset(train_dataset, indices=val_idx)
    train_dataset = Subset(train_dataset, indices=train_idx)

# %%
train_image, train_target = train_dataset[0]

train_image_preview = draw_bounding_boxes(
    image=train_image,
    boxes=train_target["boxes"],
    labels=[str(x.int().item()) for x in train_target["labels"]],
    colors="lime",
)

plt.figure(figsize=(6.4, 4.8))
plt.imshow(train_image_preview.permute(1, 2, 0))
plt.show()

# %%
yolo = YOLO("yolov8n.pt")

yolo_head = yolo.model.model[-1]
i, f, t = yolo_head.i, yolo_head.f, yolo_head.type

# https://github.com/ultralytics/ultralytics/blob/v8.2.85/ultralytics/nn/modules/head.py#L21
# https://github.com/ultralytics/ultralytics/blob/v8.2.85/ultralytics/nn/tasks.py#L990
yolo.model.model[-1] = Detect(nc=1, ch=(64, 128, 256))
yolo.model.model[-1].i = i
yolo.model.model[-1].f = f
yolo.model.model[-1].type = t
yolo.model.model[-1].stride = yolo_head.stride

train_started_at = datetime.now().isoformat(timespec="seconds")
train_storage_path = f"{CHECKPOINT_PATH}/{train_started_at}"

save_initial_weights(train_storage_path, yolo.model, storage_options)

reg_max = yolo.model.model[-1].reg_max
n_classes = yolo.model.model[-1].nc
strides = yolo.model.model[-1].stride

batch_size = 8
n_batches = np.ceil(len(train_dataset) / batch_size).astype(np.int32)

n_epochs = 100
epochs = 0
save_epochs = 5

tal_top_k = 10

# https://github.com/ultralytics/ultralytics/blob/v8.2.87/ultralytics/cfg/default.yaml#L97
cls_gain = 0.5
box_gain = 7.5
dfl_gain = 1.5

if OVERFITTING_TEST:
    n_epochs = 50
    save_epochs = 50


# %%
def collate_fn(batch):
    return tuple(zip(*batch))


train_dataloader = DataLoader(
    train_dataset,
    batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)

if VALIDATION_SPLIT:
    val_dataloader = DataLoader(val_dataset, batch_size, collate_fn=collate_fn)

# %%
for param in yolo.model.parameters():
    param.requires_grad = True

params = [p for p in yolo.model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.01, momentum=0.937, weight_decay=0.0005)

lr_scheduler = optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.01,
    total_iters=n_epochs,
)

ema = ModelEMA(yolo.model, decay=0.9999)

criterion = YOLOv8DetectionLoss(
    reg_max,
    n_classes,
    strides,
    tal_top_k,
    cls_gain,
    box_gain,
    dfl_gain,
)

predictor = YOLOv8DetectionPredictor(
    yolo.model,
    reg_max,
    n_classes,
    strides,
    confidence_threshold=0.25,
    iou_threshold=0.7,
)


# %%
def validation_loop():
    metric = MeanAveragePrecision(box_format="xyxy")
    metric.warn_on_many_detections = False

    yolo.model.eval()
    yolo.model.model[-1].training = True

    steps = 1
    pbar = keras.utils.Progbar(len(val_dataloader))

    with torch.no_grad():
        for images, targets in val_dataloader:
            images = torch.stack(images, dim=0)
            feat_maps = yolo.model(images)

            losses = criterion(feat_maps, targets)
            pred_nms = predictor(feat_maps)

            preds = [
                {"boxes": p[:, :4], "scores": p[:, 4], "labels": p[:, 5].int()}
                for p in pred_nms
            ]

            targets = [
                {"boxes": t["boxes"], "labels": t["labels"].int()} for t in targets
            ]

            map_dict = metric.forward(preds, targets)

            pbar.update(
                current=steps,
                values=[
                    ("cls_loss", losses[0].item()),
                    ("box_loss", losses[1].item()),
                    ("dfl_loss", losses[2].item()),
                    *list(map_dict.items()),
                ],
            )

            steps += 1

    loss = {}
    loss["cls"] = pbar._values["cls_loss"][0]
    loss["box"] = pbar._values["box_loss"][0]
    loss["dfl"] = pbar._values["dfl_loss"][0]

    return loss, metric.compute()


# %%
yolo.model.train()

history: dict[str, list[Any]] = dict(train=[], val=[], val_metric=[])

for i in range(n_epochs):
    print(f"Epoch: {i + 1}/{n_epochs}, Learning Rate: {lr_scheduler.get_last_lr()}")

    steps = 1
    pbar = keras.utils.Progbar(n_batches)

    for images, targets in train_dataloader:
        images = torch.stack(images, dim=0)
        feat_maps = yolo.model(images)

        losses = criterion(feat_maps, targets)

        pbar.update(
            current=steps,
            values=[
                ("cls_loss", losses[0].item()),
                ("box_loss", losses[1].item()),
                ("dfl_loss", losses[2].item()),
            ],
        )

        optimizer.zero_grad()
        losses.sum().backward()

        nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()

        ema.update(yolo.model)
        steps += 1

    train_loss = {}
    train_loss["cls"] = pbar._values["cls_loss"][0]
    train_loss["box"] = pbar._values["box_loss"][0]
    train_loss["dfl"] = pbar._values["dfl_loss"][0]
    history["train"].append(train_loss)

    if ((epochs + 1) % save_epochs) == 0:
        save_checkpoint(
            path=train_storage_path,
            epoch=epochs,
            model=yolo.model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            storage_options=storage_options,
        )

    if VALIDATION_SPLIT:
        loss_dict, map_dict = validation_loop()
        history["val"].append(loss_dict)
        history["val_metric"].append(map_dict)

        yolo.model.train()

    lr_scheduler.step()
    epochs += 1

# %%
history_path = f"{CHECKPOINT_PATH}/{train_started_at}/history.json"

for metric in history["val_metric"]:
    for k, v in metric.items():
        if torch.is_tensor(v):
            metric[k] = v.item()

with fsspec.open(history_path, "w", **storage_options) as f:
    json.dump(history, f, indent=2)

# %%
