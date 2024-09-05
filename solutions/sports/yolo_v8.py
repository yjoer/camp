# %%
import os

os.environ["KERAS_BACKEND"] = "torch"

# %%
from typing import cast

import keras
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2.functional as tvf
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.utils.torch_utils import ModelEMA

from camp.datasets.ikcest import IKCESTDetectionDataset
from camp.datasets.utils import resize_image_and_boxes
from camp.models.yolo.yolo_utils import decode_boxes
from camp.models.yolo.yolo_utils import decode_feature_maps
from camp.models.yolo.yolo_utils import make_anchors
from camp.models.yolo.yolo_utils import preprocess_targets
from camp.utils.torch_utils import save_checkpoint

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %load_ext autoreload
# %autoreload 2

# %%
OVERFITTING_TEST = False

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
train_dataset = IKCESTDetectionDataset(
    path=TRAIN_DATASET_PATH,
    subset="train",
    storage_options=storage_options,
    transforms=transforms,
)

if OVERFITTING_TEST:
    train_dataset = cast(IKCESTDetectionDataset, Subset(train_dataset, indices=[0]))

# %%
train_image, train_target = train_dataset[0]

train_image_preview = draw_bounding_boxes(
    image=train_image,
    boxes=train_target["boxes"],
    labels=[str(x.item()) for x in train_target["labels"]],
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

reg_max = yolo.model.model[-1].reg_max
n_coords = reg_max * 4
n_classes = yolo.model.model[-1].nc
n_total = n_coords + n_classes

tal_top_k = 10
strides = yolo.model.model[-1].stride

batch_size = 8
n_batches = np.ceil(len(train_dataset) / batch_size).astype(np.int32)

n_epochs = 100
epochs = 0
save_epochs = 5

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

# %%
assigner = TaskAlignedAssigner(tal_top_k, n_classes, alpha=0.5, beta=6.0)

# %%
bce = nn.BCEWithLogitsLoss(reduction="none")
bbox_loss = BboxLoss(reg_max)

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

# %%
yolo.model.train()

for i in range(n_epochs):
    print(f"Epoch: {i + 1}/{n_epochs}, Learning Rate: {lr_scheduler.get_last_lr()}")

    steps = 1
    pbar = keras.utils.Progbar(n_batches)

    for images, targets in train_dataloader:
        images = torch.stack(images, dim=0)
        feat_maps = yolo.model(images)

        pred_dist, pred_scores = decode_feature_maps(feat_maps, reg_max, n_classes)

        anchor_points, stride_tensors = make_anchors(
            feat_maps,
            strides,
            grid_cell_offset=0.5,
        )

        pred_boxes = decode_boxes(pred_dist, anchor_points, reg_max)

        target_labels, target_boxes = preprocess_targets(targets)
        mask = target_boxes.sum(dim=2, keepdim=True).gt_(0.0)

        _, target_boxes, target_scores, fg_mask, _ = assigner(
            pred_scores.detach().sigmoid(),
            pred_boxes.detach() * stride_tensors,
            anchor_points * stride_tensors,
            target_labels,
            target_boxes,
            mask,
        )

        losses = torch.zeros(3)
        target_scores_sum = max(target_scores.sum(), 1)
        losses[0] = bce(pred_scores, target_scores).sum() / target_scores_sum

        if fg_mask.sum():
            target_boxes /= stride_tensors

            losses[1], losses[2] = bbox_loss(
                pred_dist,
                pred_boxes,
                anchor_points,
                target_boxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        losses[0] *= cls_gain
        losses[1] *= box_gain
        losses[2] *= dfl_gain

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

    if ((epochs + 1) % save_epochs) == 0:
        save_checkpoint(
            path=CHECKPOINT_PATH,
            epoch=epochs,
            model=yolo.model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            storage_options=storage_options,
        )

    lr_scheduler.step()
    epochs += 1

# %%
