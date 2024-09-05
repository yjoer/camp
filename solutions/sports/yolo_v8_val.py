# %%
import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2.functional as tvf
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.ops import non_max_suppression

from camp.datasets.ikcest import IKCESTDetectionDataset
from camp.datasets.utils import resize_image_and_boxes
from camp.models.yolo.yolo_utils import decode_boxes_eval
from camp.models.yolo.yolo_utils import decode_feature_maps
from camp.models.yolo.yolo_utils import make_anchors
from camp.utils.torch_utils import load_model

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
n_coords = reg_max * 4
n_classes = yolo.model.model[-1].nc
n_total = n_coords + n_classes

strides = yolo.model.model[-1].stride

epochs = 0

if OVERFITTING_TEST:
    epochs = 49

# %%
load_model(CHECKPOINT_PATH, epochs, yolo.model, storage_options)

# %%
yolo.model.eval()
yolo.model.model[-1].training = True

feat_maps = yolo.model(train_image.unsqueeze(0))
pred_dist, pred_scores = decode_feature_maps(feat_maps, reg_max, n_classes)

anchor_points, stride_tensors = make_anchors(feat_maps, strides, grid_cell_offset=0.5)

pred_dist = pred_dist.permute(0, 2, 1)
pred_scores = pred_scores.permute(0, 2, 1)

pred_boxes = decode_boxes_eval(
    yolo.model.model[-1],
    pred_dist,
    anchor_points.permute(1, 0),
    stride_tensors.permute(1, 0),
)

# %%
pred = torch.cat((pred_boxes, pred_scores.sigmoid()), dim=1)

pred_nms = non_max_suppression(
    pred,
    conf_thres=0.25,
    iou_thres=0.7,
    agnostic=False,
    max_det=300,
    classes=None,
    in_place=False,
)

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
