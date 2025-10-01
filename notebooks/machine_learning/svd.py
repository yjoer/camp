# %%
import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact
from minio import Minio
from PIL import Image

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %%
minio = Minio(
    os.environ.get("S3_ENDPOINT", "").split("//")[-1],
    access_key=os.environ.get("S3_ACCESS_KEY_ID"),
    secret_key=os.environ.get("S3_SECRET_ACCESS_KEY"),
    secure=False,
)

# %%
file = minio.get_object(
    bucket_name="datasets",
    object_name="ethz/food-101.zip/images/takoyaki/326340.jpg",
    request_headers={"x-minio-extract": "true"},
)

# %%
im = Image.open(file)
im_array = np.array(im)

# %%
rank_slider = widgets.IntSlider(
    value=64,
    min=1,
    max=128,
    step=1,
    description="Rank",
    layout={"width": "50%"},
    style={"description_width": "initial"},
)


# %%
@interact(k=rank_slider)
def low_rank_approximation(k: int) -> None:
    channels = []

    for c in range(3):
        u, s, vt = np.linalg.svd(im_array[..., c], full_matrices=False)
        approx = (u[:, :k] * s[:k]) @ vt[:k, :]
        channels.append(np.clip(approx, 0, 255).astype(np.uint8))

    im_approx = np.stack(channels, axis=-1)
    plt.imshow(im_approx)
    plt.show()


# %%
