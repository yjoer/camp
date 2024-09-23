import torch
import torchvision.transforms.v2.functional as tvf


def transforms(image, target):
    max_size = 224
    output_size = (224, 224)

    image = tvf.resize(image, size=None, max_size=max_size)
    image = tvf.center_crop(image, output_size)
    image = tvf.to_image(image)
    image = tvf.to_dtype(image, dtype=torch.float32, scale=True)

    # One-hot encoding in PyTorch requires the target to have an int64 type.
    target = torch.tensor(target)

    return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
