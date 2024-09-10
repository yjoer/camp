# %%
import os
from argparse import Namespace

import fsspec
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.nn.modules.head import Detect
from ultralytics.trackers.byte_tracker import BYTETracker

from camp.datasets.ikcest import IKCESTDetectionDataset
from camp.models.yolo.yolo_utils import YOLOv8DetectionPredictor
from camp.utils.torch_utils import load_model
from solutions.sports.yolo_v8_pipeline import collate_fn
from solutions.sports.yolo_v8_pipeline import transforms

# %%
OVERFITTING_VIDEO_TEST = False

TRAIN_DATASET_PATH = "s3://datasets/ikcest_2024"
CHECKPOINT_PATH = "s3://models/ikcest_2024/yolo_v8"
TRAIN_STARTED_AT = ""
DATALOADER_WORKERS = psutil.cpu_count(logical=False)

if OVERFITTING_VIDEO_TEST:
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
train_dataset: IKCESTDetectionDataset | Subset = IKCESTDetectionDataset(
    path=TRAIN_DATASET_PATH,
    subset="train",
    storage_options=storage_options,
    transforms=transforms,
)

if OVERFITTING_VIDEO_TEST:
    train_dataset = Subset(train_dataset, list(range(750)))

# %%
yolo = YOLO("yolov8n.pt")

yolo_head = yolo.model.model[-1]
i, f, t = yolo_head.i, yolo_head.f, yolo_head.type

yolo.model.model[-1] = Detect(nc=1, ch=(64, 128, 256))
yolo.model.model[-1].i = i
yolo.model.model[-1].f = f
yolo.model.model[-1].type = t
yolo.model.model[-1].stride = yolo_head.stride

yolo.model = yolo.model.to(device)

train_storage_path = f"{CHECKPOINT_PATH}/{TRAIN_STARTED_AT}"

reg_max = yolo.model.model[-1].reg_max
n_classes = yolo.model.model[-1].nc
strides = yolo.model.model[-1].stride

batch_size = 8
epochs = 99

# %%
load_model(train_storage_path, epochs, yolo.model, storage_options)

# %%
train_dataloader = DataLoader(
    train_dataset,
    batch_size,
    num_workers=DATALOADER_WORKERS,
    collate_fn=collate_fn,
    persistent_workers=True,
)

if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=fsspec.asyn.reset_lock)

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
args = Namespace()
args.track_high_thresh = 0.5
args.track_low_thresh = 0.1
args.new_track_thresh = 0.6
args.track_buffer = 30
args.match_thresh = 0.8
args.fuse_score = True


def save_tracking_arrays(
    path: str,
    epoch: int,
    video_id: int,
    tracklets_seq: list[np.ndarray],
    storage_options={},
):
    tracking_path = f"{path}/checkpoint-{epoch}-tracking/{video_id}.npz"

    with fsspec.open(tracking_path, "wb", **storage_options) as f:
        np.savez(f, *tracklets_seq)


# %%
yolo.model.eval()
yolo.model.model[-1].training = True

with torch.no_grad():
    frame_counter = 0
    video_id = 0
    tracklets_seq = []

    for images, targets in tqdm(train_dataloader):
        images = torch.stack(images, dim=0).to(device)
        feat_maps = yolo.model(images)

        pred_nms = predictor(feat_maps)

        for pred in pred_nms:
            # Run on every first frame.
            if (frame_counter % 750) == 0:
                tracker = BYTETracker(args, frame_rate=30)

            boxes = Boxes(pred, orig_shape=())
            tracklets = tracker.update(boxes)
            tracklets_seq.append(tracklets)

            # Run on every last frame.
            if ((frame_counter + 1) % 750) == 0:
                save_tracking_arrays(
                    train_storage_path,
                    epochs,
                    video_id,
                    tracklets_seq,
                    storage_options,
                )

                video_id += 1
                tracklets_seq = []

            frame_counter += 1

# %%
