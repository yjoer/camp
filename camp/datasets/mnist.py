import gzip
import struct
from typing import ClassVar
from typing import Literal

import fsspec
import numpy as np

try:
  import torch
except ImportError:
  torch = None  # ty: ignore[invalid-assignment]

TensorType = Literal["np", "pt"]


class FashionMNIST:
  files: ClassVar[dict[str, str]] = {
    "train": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
  }

  @staticmethod
  def load(
    path: str,
    storage_options: dict | None = None,
    return_tensors: TensorType = "pt",
  ) -> dict[str, np.ndarray] | dict[str, torch.Tensor] | None:
    if storage_options is None:
      storage_options = {}

    mnist = FashionMNIST()
    buffers = mnist._load(path, storage_options)
    arrays = mnist._parse(buffers)

    if return_tensors == "np":
      return arrays

    if return_tensors == "pt":
      return mnist._to_tensor(arrays)

    return None

  def _load(self, path: str, storage_options: dict) -> dict[str, bytes]:
    buffers = {}

    for k, v in self.files.items():
      with fsspec.open(f"{path}/{v}", **storage_options) as f:
        buffers[k] = bytearray(gzip.decompress(f.read()))

    return buffers

  def _parse(self, buffers: dict[str, bytes]) -> dict[str, np.ndarray]:
    arrays = {}

    for subset in ["train", "test"]:
      header = struct.unpack(">IIII", buffers[subset][0:16])
      _magic_number, n_items, n_rows, n_cols = header

      images = np.frombuffer(buffers[subset][16:], dtype=np.uint8)
      images = images.reshape(n_items, n_rows * n_cols)

      arrays[subset] = images

    for subset in ["train_labels", "test_labels"]:
      _magic_number, n_items = struct.unpack(">II", buffers[subset][0:8])
      labels = np.frombuffer(buffers[subset][8:], dtype=np.uint8)

      arrays[subset] = labels

    return arrays

  def _to_tensor(
    self,
    arrays: dict[str, np.ndarray],
  ) -> dict[str, torch.Tensor] | None:
    if torch is None:
      print("cannot convert to tensors because torch is not installed.")
      return None

    tensors = {}

    for k, v in arrays.items():
      tensors[k] = torch.from_numpy(v)

    return tensors
