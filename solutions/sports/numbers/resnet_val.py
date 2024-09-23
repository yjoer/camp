# %%
import os

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models import resnet50

from camp.datasets.soccernet import SoccerNetLegibilityDataset
from camp.utils.torch_utils import load_model
from solutions.sports.numbers.resnet_pipeline import transforms

# %%
OVERFITTING_TEST = False

DATASET_PATH = "s3://datasets/soccernet_legibility"
TRAIN_STARTED_AT = ""
CHECKPOINT_PATH = "s3://models/soccernet_legibility/resnet_50"

if OVERFITTING_TEST:
    CHECKPOINT_PATH = "s3://models/soccernet_legibility/resnet_50_test"

# %%
storage_options = {
    "endpoint_url": os.getenv("S3_ENDPOINT"),
    "key": os.getenv("S3_ACCESS_KEY_ID"),
    "secret": os.getenv("S3_SECRET_ACCESS_KEY"),
}

if not os.getenv("S3_ENDPOINT"):
    storage_options = {}

# %%
train_dataset = SoccerNetLegibilityDataset(
    path=DATASET_PATH,
    storage_options=storage_options,
    transforms=transforms,
)

# %%
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

fc_in_features = resnet.fc.in_features
n_classes = 2
resnet.fc = nn.Linear(fc_in_features, n_classes)

epoch = 9

# %%
load_model(f"{CHECKPOINT_PATH}/{TRAIN_STARTED_AT}", epoch, resnet, storage_options)

# %%
train_image, train_target = train_dataset[0]

resnet.eval()

with torch.no_grad():
    outputs = resnet(train_image.unsqueeze(0))
    predictions = torch.argmax(outputs, dim=1)

predictions

# %%
