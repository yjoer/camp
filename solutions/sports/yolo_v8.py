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
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.torch_utils import ModelEMA

from camp.datasets.ikcest import IKCESTDetectionDataset
from camp.models.yolo.yolo_utils import YOLOv8DetectionLoss
from camp.models.yolo.yolo_utils import YOLOv8DetectionPredictor
from camp.utils.jupyter_utils import is_notebook
from camp.utils.torch_utils import load_checkpoint
from camp.utils.torch_utils import load_initial_weights
from camp.utils.torch_utils import save_checkpoint
from camp.utils.torch_utils import save_initial_weights
from solutions.sports.yolo_v8_pipeline import collate_fn
from solutions.sports.yolo_v8_pipeline import transforms

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %load_ext autoreload
# %autoreload 2

# %%
OVERFITTING_TEST = False
OVERFITTING_VIDEO_TEST = False
VALIDATION_SPLIT = False
VALIDATION_SPLIT_TEST = False

TRAIN_DATASET_PATH = "s3://datasets/ikcest_2024"
CHECKPOINT_PATH = "s3://models/ikcest_2024/yolo_v8"
DATALOADER_WORKERS = psutil.cpu_count(logical=False)

USE_AMP = False

RESUME_TRAIN_STARTED_AT = ""
RESUME_EPOCH = 0

if OVERFITTING_TEST or OVERFITTING_VIDEO_TEST or VALIDATION_SPLIT_TEST:
    CHECKPOINT_PATH = "s3://models/ikcest_2024/yolo_v8_test"

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
train_dataset: Union[IKCESTDetectionDataset, Subset] = IKCESTDetectionDataset(
    path=TRAIN_DATASET_PATH,
    subset="train",
    storage_options=storage_options,
    transforms=transforms,
)

# When multiple workers are initialized, the lock in the parent process is copied into
# the child process causing the data loader to wait endlessly for the lock to become
# available.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=fsspec.asyn.reset_lock)

if OVERFITTING_TEST:
    train_dataset = Subset(train_dataset, indices=[0])

if OVERFITTING_VIDEO_TEST:
    train_dataset = Subset(train_dataset, indices=list(range(750)))

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
if is_notebook():
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

yolo.model = yolo.model.to(device)

train_started_at = datetime.now().isoformat(timespec="seconds")
train_storage_path = f"{CHECKPOINT_PATH}/{train_started_at}"

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

if RESUME_EPOCH == 0:
    save_initial_weights(train_storage_path, yolo.model, storage_options)

if RESUME_EPOCH > 0:
    train_storage_path = f"{CHECKPOINT_PATH}/{RESUME_TRAIN_STARTED_AT}"

    # The initial weights must be restored before EMA is instantiated.
    load_initial_weights(train_storage_path, yolo.model, storage_options)

    epochs = RESUME_EPOCH + 1

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
    val_dataloader = DataLoader(val_dataset, batch_size, collate_fn=collate_fn)

# %%
for param in yolo.model.parameters():
    param.requires_grad = True

params = [p for p in yolo.model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.01, momentum=0.937, weight_decay=0.0005)

# Reduce the learning rate to 0.01x the initial value linearly for 100 epochs.
lr_scheduler = optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.01,
    total_iters=100,
)

ema = ModelEMA(yolo.model, decay=0.9999, updates=epochs)

criterion = YOLOv8DetectionLoss(
    reg_max,
    n_classes,
    strides,
    tal_top_k,
    cls_gain,
    box_gain,
    dfl_gain,
    device,
)

predictor = YOLOv8DetectionPredictor(
    yolo.model,
    reg_max,
    n_classes,
    strides,
    confidence_threshold=0.25,
    iou_threshold=0.7,
)

scaler = torch.GradScaler(device.type, enabled=USE_AMP)

if RESUME_EPOCH > 0:
    load_checkpoint(
        train_storage_path,
        RESUME_EPOCH,
        yolo.model,
        optimizer,
        lr_scheduler,
        storage_options,
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

for i in range(epochs, n_epochs):
    print(f"Epoch: {i + 1}/{n_epochs}, Learning Rate: {lr_scheduler.get_last_lr()}")

    steps = 1
    pbar = keras.utils.Progbar(n_batches)

    for images, targets in train_dataloader:
        images = torch.stack(images, dim=0).to(device)

        with torch.autocast(device.type, torch.float16, enabled=USE_AMP):
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

        scaler.scale(losses.sum()).backward()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(params, max_norm=10.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

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
history_path = f"{train_storage_path}/history.json"

for metric in history["val_metric"]:
    for k, v in metric.items():
        if torch.is_tensor(v):
            metric[k] = v.item()

with fsspec.open(history_path, "w", **storage_options) as f:
    json.dump(history, f, indent=2)

# %%
