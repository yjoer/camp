import torch
import torchvision.transforms.v2.functional as tvf


def transforms(image, target):
    max_size = 256
    output_size = (256, 256)

    image = tvf.resize(image, size=None, max_size=max_size)
    image = tvf.center_crop(image, output_size)
    image = tvf.to_image(image)
    image = tvf.to_dtype(image, dtype=torch.float32, scale=True)

    return image, target
