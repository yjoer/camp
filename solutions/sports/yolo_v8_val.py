# %%
import json
import os

import altair as alt
import fsspec
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms.v2.functional as tvf
from IPython.display import display
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

from camp.datasets.ikcest import IKCESTDetectionDataset
from camp.datasets.utils import resize_image_and_boxes
from camp.models.yolo.yolo_utils import YOLOv8DetectionPredictor
from camp.utils.torch_utils import load_model

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %load_ext autoreload
# %autoreload 2

# %%
OVERFITTING_TEST = False
VALIDATION_SPLIT = False

TRAIN_DATASET_PATH = "s3://datasets/ikcest_2024"
TRAIN_STARTED_AT = ""
CHECKPOINT_PATH = "s3://models/ikcest_2024/yolo_v8"

if OVERFITTING_TEST:
    CHECKPOINT_PATH = "s3://models/ikcest_2024/yolo_v8_test"

# %%
storage_options = {
    "endpoint_url": os.getenv("S3_ENDPOINT"),
    "key": os.getenv("S3_ACCESS_KEY_ID"),
    "secret": os.getenv("S3_SECRET_ACCESS_KEY"),
}

if not os.getenv("S3_ENDPOINT"):
    storage_options = {}

# %%
history_path = f"{CHECKPOINT_PATH}/{TRAIN_STARTED_AT}/history.json"

with fsspec.open(history_path, **storage_options) as f:
    history = json.load(f)

# %%
df_train_loss = pd.DataFrame(history["train"])
df_train_loss["sum"] = df_train_loss.sum(axis=1)
df_train_loss["epoch"] = df_train_loss.index
df_train_loss["subset"] = "train"

df_val_loss = pd.DataFrame(history["val"])
df_val_loss["sum"] = df_val_loss.sum(axis=1)
df_val_loss["epoch"] = df_val_loss.index
df_val_loss["subset"] = "validation"

df_loss = pd.concat((df_train_loss, df_val_loss), axis=0)

# %%
chart_nearest = alt.selection_point(
    nearest=True,
    on="pointerover",
    fields=["epoch"],
    empty=False,
)

chart_rules = (
    alt.Chart(df_loss)
    .mark_rule(color="gray")
    .encode(
        x="epoch:Q",
        opacity=alt.condition(chart_nearest, alt.value(0.2), alt.value(0)),
        tooltip=["epoch", "cls", "box", "dfl", "sum"],
    )
    .add_params(chart_nearest)
)

chart_loss = (
    alt.Chart(df_loss)
    .mark_line()
    .encode(x="epoch:Q", y=alt.Y("sum:Q").scale(type="log"), color="subset:N")
    .properties(width=640, height=320)
)

chart_loss + chart_rules

# %%
if VALIDATION_SPLIT:
    df_val_map = pd.DataFrame(history["val_metric"])
    df_val_map["epoch"] = df_val_map.index

    chart_rules = (
        alt.Chart(df_val_map)
        .mark_rule(color="gray")
        .encode(
            x="epoch:Q",
            opacity=alt.condition(chart_nearest, alt.value(0.2), alt.value(0)),
            tooltip=["epoch", "map", "map_50", "map_75"],
        )
        .add_params(chart_nearest)
    )

    chart_map = (
        alt.Chart(df_val_map)
        .mark_line()
        .encode(x="epoch:Q", y=alt.Y("map:Q"))
        .properties(width=640, height=320)
    )

    display(chart_map + chart_rules)


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

# %%
train_image, train_target = train_dataset[0]

# %%
yolo = YOLO("yolov8n.pt")

yolo_head = yolo.model.model[-1]
i, f, t = yolo_head.i, yolo_head.f, yolo_head.type

yolo.model.model[-1] = Detect(nc=1, ch=(64, 128, 256))
yolo.model.model[-1].i = i
yolo.model.model[-1].f = f
yolo.model.model[-1].type = t
yolo.model.model[-1].stride = yolo_head.stride

reg_max = yolo.model.model[-1].reg_max
n_classes = yolo.model.model[-1].nc
strides = yolo.model.model[-1].stride

epochs = 0

if OVERFITTING_TEST:
    epochs = 49

# %%
load_model(CHECKPOINT_PATH, epochs, yolo.model, storage_options)

# %%
predictor = YOLOv8DetectionPredictor(
    yolo.model,
    reg_max,
    n_classes,
    strides,
    confidence_threshold=0.25,
    iou_threshold=0.7,
)

# %%
yolo.model.eval()
yolo.model.model[-1].training = True

feat_maps = yolo.model(train_image.unsqueeze(0))
pred_nms = predictor(feat_maps)

# %%
test_image_preview = draw_bounding_boxes(
    image=train_image,
    boxes=pred_nms[0][:, :4],
    labels=[str(x.int().item()) for x in pred_nms[0][:, 5]],
    colors="lime",
)

plt.figure(figsize=(6.4, 4.8))
plt.imshow(test_image_preview.permute(1, 2, 0))
plt.show()

# %%
