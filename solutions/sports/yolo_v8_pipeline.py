from multiprocessing import Pool
from typing import TYPE_CHECKING

import fsspec
import motmetrics as mm
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2.functional as tvf
from PIL import Image
from torchvision.ops import box_convert
from tqdm.auto import tqdm

from camp.datasets.utils import resize_image_and_boxes

if TYPE_CHECKING:
  from torchvision.tv_tensors import Image as TVImage


def transforms(image: Image.Image, target: dict) -> tuple:
  boxes = box_convert(target["boxes"], "xywh", "xyxy")
  max_size = 640
  output_size = [384, 640]

  image, boxes = resize_image_and_boxes(image, boxes, max_size, output_size)
  target["boxes"] = boxes

  image: TVImage = tvf.to_image(image)
  image: torch.Tensor = tvf.to_dtype(image, dtype=torch.float32, scale=True)

  return image, target


def transforms_test(image: torch.Tensor) -> torch.Tensor:
  max_size = 640
  output_size = [384, 640]

  image = tvf.resize(image, size=None, max_size=max_size)

  width, height = image.size
  output_height, output_width = output_size

  if width != output_width or height != output_height:
    image = tvf.center_crop(image, output_size)

  image = tvf.to_image(image)
  return tvf.to_dtype(image, dtype=torch.float32, scale=True)


def inverse_transforms(boxes: torch.Tensor) -> torch.Tensor:
  size = (384, 640)
  output_size = (360, 640)

  height, width = size
  output_height, output_width = output_size

  left = (output_width - width) // 2
  top = (output_height - height) // 2

  boxes[:, [0, 2]] += left
  boxes[:, [1, 3]] += top

  max_size = 1920

  if output_width > output_height:
    scale_factor = max_size / output_width
  else:
    scale_factor = max_size / output_height

  return torch.mul(boxes, scale_factor)


def collate_fn(batch: list) -> tuple:
  return tuple(zip(*batch, strict=False))


def _compute_tracking_metrics(args: tuple[pd.DataFrame, pd.DataFrame]) -> tuple:
  df_gt, df_sub = args
  acc = mm.utils.compare_to_groundtruth(df_gt, df_sub, dist="iou", distth=0.5)

  acc_rew = mm.utils.compare_to_groundtruth_reweighting(
    df_gt,
    df_sub,
    dist="iou",
    distth=np.arange(0.05, 0.99, 0.05),
  )

  mh = mm.metrics.create()
  df_metrics = mh.compute(acc, metrics=["idf1", "mota"])
  idf1, mota = df_metrics.iloc[0]["idf1"], df_metrics.iloc[0]["mota"]

  df_hota = mh.compute_many(
    dfs=acc_rew,
    metrics=["hota_alpha"],
    generate_overall=True,
  )

  hota_avg = df_hota.loc["OVERALL"]["hota_alpha"]

  return idf1, hota_avg, mota


def evaluate_tracking(
  path: str,
  checkpoint_path: str,
  train_started_at: str,
  epoch: int,
  storage_options: dict | None = None,
) -> pd.DataFrame:
  if storage_options is None:
    storage_options = {}

  protocol = fsspec.utils.get_protocol(path)
  test_path = f"{path}/test"
  tracking_sub_path = f"{checkpoint_path}/{train_started_at}/checkpoint-{epoch}-sub"

  fs = fsspec.filesystem(protocol, **storage_options)
  video_names = fs.ls(test_path)
  video_names = [x.split("/")[-1] for x in video_names]
  video_names = sorted(video_names)

  df_gts = []
  df_subs = []

  for video_name in tqdm(video_names):
    video_gt_file = f"{test_path}/{video_name}/gt/gt.txt"
    video_sub_file = f"{tracking_sub_path}/{video_name}.txt"

    with fsspec.open(video_gt_file, "rb", **storage_options) as f:
      df_gt = mm.io.loadtxt(f)

    with fsspec.open(video_sub_file, "rb", **storage_options) as f:
      df_sub = mm.io.loadtxt(f)

    df_gts.append(df_gt)
    df_subs.append(df_sub)

  with Pool() as p:
    imap_it = p.imap(
      _compute_tracking_metrics,
      list(zip(df_gts, df_subs, strict=False)),
    )

    metrics = []
    for idf1, hota, mota in tqdm(imap_it, total=len(video_names)):
      metrics.append((idf1, hota, mota))

  df_video_names = pd.DataFrame(video_names, columns=["video_name"])
  df_metrics = pd.DataFrame(metrics, columns=["idf1", "hota", "mota"])
  df_metrics.loc["OVERALL"] = df_metrics.mean(axis=0)

  return pd.concat((df_video_names, df_metrics), axis=1)
