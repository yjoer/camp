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


def collate_fn(batch):
    return tuple(zip(*batch))
