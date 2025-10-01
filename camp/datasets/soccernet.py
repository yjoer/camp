import json
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import fsspec
import numpy as np
import torch
from IPython.display import clear_output
from minio import Minio
from PIL import Image
from pyarrow import csv
from rich.console import Console
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import pandas as pd


def read_json_multi(
    files: list[str],
    protocol: str,
    storage_options: dict | None = None,
) -> list:
    if storage_options is None:
        storage_options = {}

    pool = ThreadPoolExecutor()

    if protocol == "s3":
        minio = Minio(
            endpoint=storage_options["endpoint_url"]
            .replace("http://", "")
            .replace("https://", ""),
            access_key=storage_options["key"],
            secret_key=storage_options["secret"],
            secure=False,
        )

        def read_json(f: str) -> dict:
            segments = f.split("/")
            bucket_name = segments[0]
            object_name = "/".join(segments[1:])

            response = minio.get_object(bucket_name, object_name)
            data = json.load(response)
            response.release_conn()

            return data
    elif protocol == "file":

        def read_json(filename: str) -> dict:
            with Path(filename).open("r") as f:
                return json.load(f)

    return list(pool.map(read_json, files))


class SoccerNetLegibilityDataset(Dataset):
    def __init__(
        self,
        path: str,
        storage_options: dict | None = None,
        transforms: Callable | None = None,
    ) -> None:
        if storage_options is None:
            storage_options = {}

        protocol = fsspec.utils.get_protocol(path)
        subset_path = f"{path}/train"

        fs = fsspec.filesystem(protocol, **storage_options)
        self.images = fs.ls(f"{subset_path}/images")
        self.images = [fs.unstrip_protocol(i) for i in self.images]
        self.images = sorted(self.images)

        read_options = csv.ReadOptions(autogenerate_column_names=True)

        with fsspec.open(f"{subset_path}/train_gt.txt", **storage_options) as f:
            df: pd.DataFrame = csv.read_csv(f, read_options).to_pandas()
            df = df.set_index(df.columns[0])
            self.labels = df

        self.storage_options = storage_options
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Image.Image, np.int64]:
        image_path = self.images[idx]
        image_filename = image_path.split("/")[-1]

        with fsspec.open(image_path, **self.storage_options) as f:
            image = Image.open(f).convert("RGB")

        label: np.int64 = self.labels.iloc[:, 0].loc[image_filename]

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label


CalibrationSubsetType = Literal["train", "val", "test"]


class SoccerNetCalibrationDataset(Dataset):
    def __init__(
        self,
        path: str,
        subset: CalibrationSubsetType,
        storage_options: dict | None = None,
        transforms: Callable | None = None,
    ) -> None:
        if storage_options is None:
            storage_options = {}

        console = Console()

        protocol = fsspec.utils.get_protocol(path)
        subset_path = f"{path}/{self._transform_subset(subset)}"
        fs = fsspec.filesystem(protocol, **storage_options)

        with console.status("Listing images"):
            self.images = fs.glob(f"{subset_path}/*.jpg")
            self.images = [fs.unstrip_protocol(i) for i in self.images]
            self.images = sorted(self.images)

        self.annotations = [
            fs._strip_protocol(i.replace(".jpg", ".json")) for i in self.images
        ]

        self.subset_segments = []
        self.subset_boxes = []
        self.subset_keypoints = []

        with console.status("Loading annotations"):
            annotation_dicts = read_json_multi(
                self.annotations,
                protocol,
                storage_options,
            )

        status = console.status("Processing annotations")
        status.start()

        for d in annotation_dicts:
            inst_segments = []
            inst_boxes = []
            inst_keypoints = []

            for segment, keypoints in d.items():
                x_min, y_min, x_max, y_max = math.inf, math.inf, -math.inf, -math.inf
                keypoints_t = []

                for k in keypoints:
                    x_min, y_min = min(x_min, k["x"]), min(y_min, k["y"])
                    x_max, y_max = max(x_max, k["x"]), max(y_max, k["y"])
                    keypoints_t.append((k["x"], k["y"]))

                inst_segments.append(segment)
                inst_boxes.append((x_min, y_min, x_max, y_max))
                inst_keypoints.append(keypoints_t)

            self.subset_segments.append(inst_segments)
            self.subset_boxes.append(inst_boxes)
            self.subset_keypoints.append(inst_keypoints)

        status.stop()
        clear_output()

        self.storage_options = storage_options
        self.transforms = transforms

    def _transform_subset(self, subset: str) -> str:
        mapping = {"val": "valid"}

        return mapping.get(subset, subset)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_path = self.images[idx]

        with fsspec.open(image_path, **self.storage_options) as f:
            image = Image.open(f).convert("RGB")

        boxes = self.subset_boxes[idx]

        boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes[:, [0, 2]] *= 960
        boxes[:, [1, 3]] *= 540

        target = {
            "boxes": boxes,
            "labels": torch.zeros(len(boxes), dtype=torch.int64),
            "keypoints": self.subset_keypoints[idx],
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
