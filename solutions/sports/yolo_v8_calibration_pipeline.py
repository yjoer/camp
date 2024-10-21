import torch
import torchvision.transforms.v2.functional as tvf


def transforms(image, target):
    image = tvf.to_image(image)
    image = tvf.to_dtype(image, dtype=torch.float32, scale=True)

    return image, target
