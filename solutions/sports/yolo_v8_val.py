# %%
import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2.functional as tvf
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

from camp.datasets.ikcest import IKCESTDetectionDataset
from camp.datasets.utils import resize_image_and_boxes
from camp.models.yolo.yolo_utils import YOLOv8DetectionPredictor
from camp.utils.torch_utils import load_model

# %matplotlib inline
# %config InlineBackend.figure_formats = ['retina']

# %load_ext autoreload
# %autoreload 2

# %%
OVERFITTING_TEST = False

TRAIN_DATASET_PATH = "s3://datasets/ikcest_2024"
CHECKPOINT_PATH = "s3://models/ikcest_2024/yolo_v8"

if OVERFITTING_TEST:
    CHECKPOINT_PATH = "s3://models/ikcest_2024/yolo_v8_test"

# %%
storage_options = {
    "endpoint_url": os.getenv("S3_ENDPOINT"),
    "key": os.getenv("S3_ACCESS_KEY_ID"),
    "secret": os.getenv("S3_SECRET_ACCESS_KEY"),
}


# %%
def transforms(image, target):
    boxes = box_convert(target["boxes"], "xywh", "xyxy")
    max_size = 640
    output_size = (384, 640)

    image, boxes = resize_image_and_boxes(image, boxes, max_size, output_size)
    target["boxes"] = boxes

    image = tvf.to_image(image)
    image = tvf.to_dtype(image, dtype=torch.float32, scale=True)

    return image, target


# %%
train_dataset = IKCESTDetectionDataset(
    path=TRAIN_DATASET_PATH,
    subset="train",
    storage_options=storage_options,
    transforms=transforms,
)

# %%
train_image, train_target = train_dataset[0]

# %%
yolo = YOLO("yolov8n.pt")

yolo_head = yolo.model.model[-1]
i, f, t = yolo_head.i, yolo_head.f, yolo_head.type

yolo.model.model[-1] = Detect(nc=1, ch=(64, 128, 256))
yolo.model.model[-1].i = i
yolo.model.model[-1].f = f
yolo.model.model[-1].type = t
yolo.model.model[-1].stride = yolo_head.stride

reg_max = yolo.model.model[-1].reg_max
n_classes = yolo.model.model[-1].nc
strides = yolo.model.model[-1].stride

epochs = 0

if OVERFITTING_TEST:
    epochs = 49

# %%
load_model(CHECKPOINT_PATH, epochs, yolo.model, storage_options)

# %%
predictor = YOLOv8DetectionPredictor(
    yolo.model,
    reg_max,
    n_classes,
    strides,
    confidence_threshold=0.25,
    iou_threshold=0.7,
)

# %%
yolo.model.eval()
yolo.model.model[-1].training = True

feat_maps = yolo.model(train_image.unsqueeze(0))
pred_nms = predictor(feat_maps)

# %%
test_image_preview = draw_bounding_boxes(
    image=train_image,
    boxes=pred_nms[0][:, :4],
    labels=[str(x.int().item()) for x in pred_nms[0][:, 5]],
    colors="lime",
)

plt.figure(figsize=(6.4, 4.8))
plt.imshow(test_image_preview.permute(1, 2, 0))
plt.show()

# %%
