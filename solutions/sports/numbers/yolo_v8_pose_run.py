# %%
import os

import fsspec
import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_keypoints
from ultralytics import YOLO

from camp.datasets.soccernet import SoccerNetLegibilityDataset
from camp.models.yolo.yolo_utils import YOLOv8PosePredictor
from solutions.sports.numbers.yolo_v8_pose_pipeline import transforms

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %load_ext autoreload
# %autoreload 2

# %%
DATASET_PATH = "s3://datasets/soccernet_legibility"

# %%
storage_options = {
    "endpoint_url": os.getenv("S3_ENDPOINT"),
    "key": os.getenv("S3_ACCESS_KEY_ID"),
    "secret": os.getenv("S3_SECRET_ACCESS_KEY"),
}

if not os.getenv("S3_ENDPOINT"):
    storage_options = {}

# %%
train_dataset = SoccerNetLegibilityDataset(
    path=DATASET_PATH,
    storage_options=storage_options,
    transforms=transforms,
)

if hasattr(os, "register_at_fork") and hasattr(fsspec, "asyn"):
    os.register_at_fork(after_in_child=fsspec.asyn.reset_lock)

# %%
yolo = YOLO("yolov8n-pose.pt")

reg_max = yolo.model.model[-1].reg_max
n_classes = yolo.model.model[-1].nc
strides = yolo.model.model[-1].stride
keypoint_shape = yolo.model.model[-1].kpt_shape

# %%
predictor = YOLOv8PosePredictor(
    yolo.model,
    reg_max,
    n_classes,
    strides,
    keypoint_shape,
    confidence_threshold=0.25,
    iou_threshold=0.7,
)

# %%
yolo.model.eval()
yolo.model.model[-1].training = True

with torch.no_grad():
    feat_maps, keypoints = yolo.model(train_dataset[0][0].unsqueeze(0))
    pred_nms = predictor(feat_maps, keypoints)

preds = []
for p in pred_nms:
    p_keypoints = p[:, 6:].view(p.shape[0], 17, 3)

    preds.append(
        {
            "boxes": p[:, :4],
            "scores": p[:, 4],
            "labels": p[:, 5],
            "keypoints": p_keypoints[:, :, :2],
            "visibility": p_keypoints[:, :, 2],
        }
    )

# %%
image_preview = draw_bounding_boxes(
    image=train_dataset[0][0],
    boxes=preds[0]["boxes"],
    labels=[str(x.int().item()) for x in preds[0]["labels"]],
    colors="lime",
)

image_preview = draw_keypoints(
    image=image_preview,
    keypoints=preds[0]["keypoints"],
    colors="blue",
)

plt.figure()
plt.imshow(image_preview.permute(1, 2, 0))
plt.show()

# %%
