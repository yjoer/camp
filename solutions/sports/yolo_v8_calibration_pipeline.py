from typing import TYPE_CHECKING

import torch
import torchvision.transforms.v2.functional as tvf
from PIL import Image

if TYPE_CHECKING:
    from torchvision.tv_tensors import Image as TVImage


def transforms(image: Image.Image, target: dict) -> tuple:
    image: TVImage = tvf.to_image(image)
    image: torch.Tensor = tvf.to_dtype(image, dtype=torch.float32, scale=True)

    return image, target
