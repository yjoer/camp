# %%
import os
from argparse import Namespace

import cv2
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import supervision as sv
import torch
import torchvision.transforms.v2.functional as tvf
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.ops import box_convert
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.nn.modules.head import Detect
from ultralytics.trackers.byte_tracker import BYTETracker

from camp.datasets.ikcest import IKCESTDetectionDataset
from camp.datasets.ikcest import IKCESTDetectionTestDataset
from camp.models.yolo.yolo_utils import YOLOv8DetectionPredictor
from camp.utils.torch_utils import load_model
from solutions.sports.yolo_v8_pipeline import collate_fn
from solutions.sports.yolo_v8_pipeline import inverse_transforms
from solutions.sports.yolo_v8_pipeline import transforms
from solutions.sports.yolo_v8_pipeline import transforms_test

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %load_ext autoreload
# %autoreload 2

# %%
OVERFITTING_VIDEO_TEST = False
TEST_VIDEOS_TEST = False

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

train_dataset_raw: IKCESTDetectionDataset | Subset = IKCESTDetectionDataset(
    path=TRAIN_DATASET_PATH,
    subset="train",
    storage_options=storage_options,
)

test_dataset: IKCESTDetectionTestDataset | Subset = IKCESTDetectionTestDataset(
    path=TRAIN_DATASET_PATH,
    storage_options=storage_options,
    transforms=transforms_test,
)

test_dataset_raw: IKCESTDetectionTestDataset | Subset = IKCESTDetectionTestDataset(
    path=TRAIN_DATASET_PATH,
    storage_options=storage_options,
)

if OVERFITTING_VIDEO_TEST:
    train_dataset = Subset(train_dataset, list(range(750)))

if TEST_VIDEOS_TEST:
    test_dataset = Subset(test_dataset, list(range(2250)))

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

test_dataloader = DataLoader(
    test_dataset,
    batch_size,
    num_workers=DATALOADER_WORKERS,
    collate_fn=collate_fn,
    persistent_workers=True,
)

if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=fsspec.asyn.reset_lock)

dataloader = test_dataloader

if OVERFITTING_VIDEO_TEST:
    dataloader = train_dataloader

if TEST_VIDEOS_TEST:
    dataloader = test_dataloader

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
    video_name: str,
    tracklets_seq: list[np.ndarray],
    storage_options={},
):
    tracking_path = f"{path}/checkpoint-{epoch}-tracking/{video_name}.npz"

    with fsspec.open(tracking_path, "wb", **storage_options) as f:
        np.savez(f, *tracklets_seq)


def load_tracking_arrays(path: str, epoch: int, video_name: str, storage_options={}):
    tracking_path = f"{path}/checkpoint-{epoch}-tracking/{video_name}.npz"

    with fsspec.open(tracking_path, **storage_options) as f:
        tracklets_seq = np.load(f)
        tracklets_seq = [tracklets_seq[key] for key in tracklets_seq]

    return tracklets_seq


# %%
yolo.model.eval()
yolo.model.model[-1].training = True

with torch.no_grad():
    frame_counter = 0
    video_name = "0"
    tracklets_seq: list[np.ndarray] = []

    pbar = tqdm(total=len(dataloader))

    for images, metadata in dataloader:
        images = torch.stack(images, dim=0).to(device)
        feat_maps = yolo.model(images)

        pred_nms = predictor(feat_maps)

        for idx, pred in enumerate(pred_nms):
            pred = pred.cpu().numpy()
            # Run on every first frame.
            if (frame_counter % 750) == 0:
                if "seq" in metadata[idx]:
                    video_name = metadata[idx]["seq"]["name"]

                if video_name.isdigit() and frame_counter != 0:
                    video_name = str(int(video_name) + 1)

                tracker = BYTETracker(args, frame_rate=25)

            boxes = Boxes(pred, orig_shape=())
            tracklets = tracker.update(boxes)
            tracklets_seq.append(tracklets)

            # Run on every last frame.
            if ((frame_counter + 1) % 750) == 0:
                save_tracking_arrays(
                    train_storage_path,
                    epochs,
                    video_name,
                    tracklets_seq,
                    storage_options,
                )

                tracklets_seq = []

            frame_counter += 1

        pbar.update()

# %%
tracking_path = f"{train_storage_path}/checkpoint-{epochs}-tracking"
tracking_sub_path = f"{train_storage_path}/checkpoint-{epochs}-sub"

protocol = fsspec.utils.get_protocol(tracking_path)
fs = fsspec.filesystem(protocol, **storage_options)
tracklets_npzs = fs.ls(tracking_path)
tracklets_npzs = [fs.unstrip_protocol(t) for t in tracklets_npzs]

for tracklets_npz in tqdm(tracklets_npzs):
    with fsspec.open(tracklets_npz, **storage_options) as f:
        tracklets_seq = np.load(f)
        tracklets_seq = [tracklets_seq[key] for key in tracklets_seq]

    tracklets_seq_df = []

    # Convert the detections to align with the submission format one frame at a time.
    for i, tracklets in enumerate(tracklets_seq):
        n_tracklets = tracklets.shape[0]

        # Skip the iteration if the frame has no tracklets.
        if n_tracklets == 0:
            continue

        frame_id = np.full((n_tracklets, 1), i + 1, dtype=np.int32)
        tracker_id = tracklets[:, [4]].astype(np.int32)

        boxes = torch.from_numpy(tracklets[:, :4])
        boxes = inverse_transforms(boxes)
        boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="xywh")
        boxes = boxes.numpy()

        confidence = tracklets[:, [5]]

        class_id = tracklets[:, [6]].astype(np.int32)
        extras = np.full((n_tracklets, 2), -1, dtype=np.int32)

        tracklets_df = [
            pd.DataFrame(x)
            for x in [frame_id, tracker_id, boxes, confidence, class_id, extras]
        ]

        tracklets_df = pd.concat(tracklets_df, axis=1)
        tracklets_seq_df.append(tracklets_df)

    tracklets_seq_df = pd.concat(tracklets_seq_df, axis=0)
    video_name = tracklets_npz.split("/")[-1].split(".")[0]
    video_sub_file = f"{tracking_sub_path}/{video_name}.txt"

    with fsspec.open(video_sub_file, "wb", **storage_options) as f:
        tracklets_seq_df.to_csv(f, header=False, index=False)

# %%
video_name = "0"
frame_offset = 0

tracklets_seq = load_tracking_arrays(
    train_storage_path,
    epochs,
    video_name,
    storage_options,
)

video_writer = cv2.VideoWriter(
    filename=f"{video_name}.mkv",
    fourcc=cv2.VideoWriter.fourcc(*"VP80"),
    fps=25,
    frameSize=(1920, 1080),
)

# %%
for i, tracklets in enumerate(tqdm(tracklets_seq)):
    if tracklets.size == 0:
        frame = test_dataset_raw[i + frame_offset][0]

        if OVERFITTING_VIDEO_TEST:
            frame = train_dataset_raw[i][0]

        if TEST_VIDEOS_TEST:
            frame = test_dataset_raw[i + frame_offset][0]

        frame = tvf.to_image(frame)
        frame = frame.permute(1, 2, 0).numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

        continue

    boxes = torch.from_numpy(tracklets[:, :4])
    boxes = inverse_transforms(boxes)
    boxes = boxes.numpy()

    detections = sv.Detections(
        xyxy=boxes,
        confidence=tracklets[:, 5],
        class_id=tracklets[:, 6].astype(int),
        tracker_id=tracklets[:, 4].astype(int),
    )

    labels = [str(tracker_id) for tracker_id in detections.tracker_id]

    box_annotator = sv.BoxAnnotator(thickness=1)

    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_padding=2,
        color_lookup=sv.ColorLookup.TRACK,
    )

    frame = test_dataset_raw[i + frame_offset][0]

    if OVERFITTING_VIDEO_TEST:
        frame = train_dataset_raw[i][0]

    if TEST_VIDEOS_TEST:
        frame = test_dataset_raw[i + frame_offset][0]

    frame = tvf.to_image(frame)
    frame = frame.permute(1, 2, 0).numpy()

    annotated_frame = box_annotator.annotate(frame, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    video_writer.write(annotated_frame)

video_writer.release()

# %%
annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
plt.imshow(annotated_frame / 255)
plt.show()

# %%
