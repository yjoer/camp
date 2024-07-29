import gzip
import struct
from typing import Literal

import fsspec
import numpy as np
import torch

TensorType = Literal["np", "pt"]


class FashionMNIST:
    files = {
        "train": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    @staticmethod
    def load(path: str, storage_options={}, return_tensors: TensorType = "pt"):
        mnist = FashionMNIST()
        buffers = mnist._load(path, storage_options)
        arrays = mnist._parse(buffers)

        if return_tensors == "np":
            return arrays

        if return_tensors == "pt":
            return mnist._to_tensor(arrays)

    def _load(self, path: str, storage_options: dict):
        buffers = {}

        for k, v in self.files.items():
            with fsspec.open(f"{path}/{v}", **storage_options) as f:
                buffers[k] = bytearray(gzip.decompress(f.read()))

        return buffers

    def _parse(self, buffers: dict[str, bytes]):
        arrays = {}

        for subset in ["train", "test"]:
            header = struct.unpack(">IIII", buffers[subset][0:16])
            magic_number, n_items, n_rows, n_cols = header

            images = np.frombuffer(buffers[subset][16:], dtype=np.uint8)
            images = images.reshape(n_items, n_rows * n_cols)

            arrays[subset] = images

        for subset in ["train_labels", "test_labels"]:
            magic_number, n_items = struct.unpack(">II", buffers[subset][0:8])
            labels = np.frombuffer(buffers[subset][8:], dtype=np.uint8)

            arrays[subset] = labels

        return arrays

    def _to_tensor(self, arrays: dict[str, np.ndarray]):
        tensors = {}

        for k, v in arrays.items():
            tensors[k] = torch.from_numpy(v)

        return tensors
