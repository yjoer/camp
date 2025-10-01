import keras
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvf
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import AUROC
from torchmetrics import Accuracy
from torchmetrics import F1Score
from torchmetrics import MetricCollection
from torchmetrics import Precision
from torchmetrics import Recall


def transforms(image: Image.Image, target: np.int64) -> tuple:
    max_size = 224
    output_size = [224, 224]

    image: torch.Tensor = tvf.resize(image, size=None, max_size=max_size)
    image = tvf.center_crop(image, output_size)
    image = tvf.to_image(image)
    image = tvf.to_dtype(image, dtype=torch.float32, scale=True)

    # One-hot encoding in PyTorch requires the target to have an int64 type.
    target: torch.Tensor = torch.tensor(target)

    return image, target


def collate_fn(batch: list) -> tuple:
    return tuple(zip(*batch, strict=False))


def validation_loop(
    resnet: nn.Module,
    val_dataloader: DataLoader,
    device: torch.device,
) -> dict:
    criterion = nn.BCEWithLogitsLoss()

    metrics = MetricCollection(
        [
            Accuracy(task="binary"),
            Precision(task="binary"),
            Recall(task="binary"),
            F1Score(task="binary"),
            AUROC(task="binary"),
        ],
    )

    resnet.eval()

    steps = 1
    pbar = keras.utils.Progbar(len(val_dataloader))

    with torch.no_grad():
        for batch_images, batch_targets in val_dataloader:
            images = torch.stack(batch_images, dim=0).to(device)

            targets = torch.stack(batch_targets, dim=0)
            targets_hot = F.one_hot(targets)
            targets_hot = targets_hot.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            outputs = resnet(images)
            predictions = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, targets_hot)

            metrics_dict = metrics.forward(predictions, targets)

            pbar.update(
                current=steps,
                values=[("loss", loss.item()), *list(metrics_dict.items())],
            )

            steps += 1

    resnet.train()

    means = {}
    for k, v in pbar._values.items():
        acc = v[0].item() if torch.is_tensor(v[0]) else v[0]
        means[k] = acc / max(1, v[1])

    return means
