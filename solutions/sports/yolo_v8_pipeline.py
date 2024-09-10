import torch
import torchvision.transforms.v2.functional as tvf
from torchvision.ops import box_convert

from camp.datasets.utils import resize_image_and_boxes


def transforms(image, target):
    boxes = box_convert(target["boxes"], "xywh", "xyxy")
    max_size = 640
    output_size = (384, 640)

    image, boxes = resize_image_and_boxes(image, boxes, max_size, output_size)
    target["boxes"] = boxes

    image = tvf.to_image(image)
    image = tvf.to_dtype(image, dtype=torch.float32, scale=True)

    return image, target


def transforms_test(image):
    max_size = 640
    output_size = (384, 640)

    image = tvf.resize(image, size=None, max_size=max_size)

    width, height = image.size
    output_height, output_width = output_size

    if width != output_width or height != output_height:
        image = tvf.center_crop(image, output_size)

    image = tvf.to_image(image)
    image = tvf.to_dtype(image, dtype=torch.float32, scale=True)

    return image


def collate_fn(batch):
    return tuple(zip(*batch))
