# %%
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_keypoints

from camp.datasets.soccernet import SoccerNetCalibrationDataset
from camp.utils.jupyter_utils import is_notebook
from solutions.sports.yolo_v8_calibration_pipeline import transforms

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %load_ext autoreload
# %autoreload 2

# %%
DATASET_PATH = "s3://datasets/soccernet_calibration_2023"

# %%
storage_options = {
    "endpoint_url": os.getenv("S3_ENDPOINT"),
    "key": os.getenv("S3_ACCESS_KEY_ID"),
    "secret": os.getenv("S3_SECRET_ACCESS_KEY"),
}

if not os.getenv("S3_ENDPOINT"):
    storage_options = {}

# %%
train_dataset = SoccerNetCalibrationDataset(
    DATASET_PATH,
    subset="train",
    storage_options=storage_options,
    transforms=transforms,
)

# %%
if is_notebook():
    train_image, train_target = train_dataset[0]

    train_image_preview = draw_bounding_boxes(
        image=train_image,
        boxes=train_target["boxes"],
        labels=[str(x.int().item()) for x in train_target["labels"]],
        colors="lime",
    )

    keypoints = []
    for sublist in train_target["keypoints"]:
        for item in sublist:
            keypoints.append(item)

    keypoints = torch.tensor(keypoints, dtype=torch.float32)
    keypoints[:, 0] *= 960
    keypoints[:, 1] *= 540

    train_image_preview = draw_keypoints(
        image=train_image_preview,
        keypoints=keypoints.unsqueeze(0),
        colors="blue",
    )

    plt.figure(figsize=(6.4, 4.8))
    plt.imshow(train_image_preview.permute(1, 2, 0))
    plt.show()

# %%
