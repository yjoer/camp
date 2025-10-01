import torch
import torchvision.transforms.v2.functional as tvf
from PIL import Image


def resize_image_and_boxes(
    image: Image.Image,
    boxes: torch.Tensor,
    max_size: int,
    output_size: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Match the longest edge of the image to the maximum size.
    width, height = image.size
    image: torch.Tensor = tvf.resize(image, size=None, max_size=max_size)

    # Scale the bounding boxes accordingly.
    scale_factor = max_size / width if width > height else max_size / height

    boxes = torch.mul(boxes, scale_factor)

    # Optionally pad the image and adjust the bounding boxes.
    width, height = image.size
    output_height, output_width = output_size

    if width == output_width and height == output_height:
        return image, boxes

    image = tvf.center_crop(image, output_size)

    left, top = (width - output_width) // 2, (height - output_height) // 2
    boxes[:, [0, 2]] -= left
    boxes[:, [1, 3]] -= top

    return image, boxes


class TestUtils:
    @staticmethod
    def test_resize_image_and_boxes() -> None:
        image = Image.new("RGB", (1920, 1080))
        boxes = torch.tensor([[120, 120, 240, 240], [240, 240, 360, 360]])
        max_size = 640
        output_size = [384, 640]

        image, boxes = resize_image_and_boxes(image, boxes, max_size, output_size)

        assert image.size == tuple(reversed(output_size))
        assert torch.all(boxes == torch.tensor([[40, 52, 80, 92], [80, 92, 120, 132]]))
