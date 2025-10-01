# %%
from deepspeed.profiling.flops_profiler.profiler import get_model_profile
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# %%
resnet = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# %%
# %%capture resnet_profile
flops, macs, params = get_model_profile(resnet, input_shape=(1, 3, 224, 224))

# %%
(flops, macs, params)

# %%
mobilenet = fasterrcnn_mobilenet_v3_large_fpn(
    weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
)

# %%
# %%capture mobilenet_profile
flops, macs, params = get_model_profile(mobilenet, input_shape=(1, 3, 224, 224))

# %%
(flops, macs, params)

# %%
mobilenet_low = fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
)

# %%
# %%capture mobilenet_low_profile
flops, macs, params = get_model_profile(mobilenet_low, input_shape=(1, 3, 224, 224))

# %%
(flops, macs, params)

# %%
