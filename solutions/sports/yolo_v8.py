# %%
import os

import matplotlib.pyplot as plt
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes

from camp.datasets.ikcest import IKCESTDetectionDataset

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %load_ext autoreload
# %autoreload 2

# %%
storage_options = {
    "endpoint_url": os.getenv("S3_ENDPOINT"),
    "key": os.getenv("S3_ACCESS_KEY_ID"),
    "secret": os.getenv("S3_SECRET_ACCESS_KEY"),
}

# %%
train_dataset = IKCESTDetectionDataset(
    path="s3://datasets/ikcest_2024",
    subset="train",
    storage_options=storage_options,
)

# %%
train_image, train_target = train_dataset[0]

train_image_preview = draw_bounding_boxes(
    image=train_image,
    boxes=box_convert(train_target["boxes"], "xywh", "xyxy"),
    labels=[str(x.item()) for x in train_target["labels"]],
    colors="lime",
)

plt.figure(figsize=(9, 6))
plt.imshow(train_image_preview.permute(1, 2, 0))
plt.show()

# %%
