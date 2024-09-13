# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import supervision as sv
import torch
import torchvision.transforms.v2.functional as tvf
from torchvision.ops import box_convert

from camp.datasets.ikcest import IKCESTDetectionTestDataset

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %%
OVERFITTING_TEST = False

TRAIN_DATASET_PATH = "s3://datasets/ikcest_2024"
CHECKPOINT_PATH = "s3://models/ikcest_2024/yolo_v8"
TRAIN_STARTED_AT = ""
EPOCH = 99

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
test_dataset = IKCESTDetectionTestDataset(
    path=TRAIN_DATASET_PATH,
    storage_options=storage_options,
)

# %%
tracking_sub_path = f"{CHECKPOINT_PATH}/{TRAIN_STARTED_AT}/checkpoint-{EPOCH}-sub"
video_name = ""
video_sub_file = f"{tracking_sub_path}/{video_name}.txt"

df = pd.read_csv(video_sub_file, header=None, storage_options=storage_options)

# %%
df

# %%
frame = 1
frame_offset = 0

detections = df[df[0] == frame]

boxes = detections.iloc[:, 2:6]
boxes = torch.from_numpy(boxes.to_numpy())
boxes = box_convert(boxes, in_fmt="xywh", out_fmt="xyxy")
boxes = boxes.numpy()

test_image, _ = test_dataset[frame + frame_offset - 1]
test_image = tvf.to_image(test_image)
test_image = test_image.permute(1, 2, 0).numpy()

detections = sv.Detections(
    xyxy=boxes,
    confidence=detections.iloc[:, 6].to_numpy(),
    class_id=detections.iloc[:, 7].to_numpy().astype(int),
    tracker_id=detections.iloc[:, 1].to_numpy().astype(int),
)

labels = [f"{c}/{t}" for c, t in zip(detections.class_id, detections.tracker_id)]

box_annotator = sv.BoxAnnotator(thickness=1)

label_annotator = sv.LabelAnnotator(
    text_scale=1,
    text_padding=2,
    color_lookup=sv.ColorLookup.TRACK,
)

test_image = box_annotator.annotate(test_image, detections)
test_image = label_annotator.annotate(test_image, detections, labels)

plt.figure(figsize=(8, 6))
plt.imshow(test_image)
plt.show()

# %%
