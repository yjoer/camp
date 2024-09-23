from typing import Callable
from typing import Optional

import fsspec
import numpy as np
import pandas as pd
from PIL import Image
from pyarrow import csv
from torch.utils.data import Dataset


class SoccerNetLegibilityDataset(Dataset):
    def __init__(
        self,
        path: str,
        storage_options={},
        transforms: Optional[Callable] = None,
    ):
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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        image_filename = image_path.split("/")[-1]

        with fsspec.open(image_path, **self.storage_options) as f:
            image = Image.open(f).convert("RGB")

        label: np.int64 = self.labels.iloc[:, 0].loc[image_filename]

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label
