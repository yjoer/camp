import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.utils.tal import dist2bbox


def decode_feature_maps(
    feat_maps: list[torch.Tensor],
    reg_max: int,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = feat_maps[0].shape[0]
    n_coords = reg_max * 4
    n_total = n_coords + n_classes

    # (b, c, h, w) -> (b, hw, c)
    feat_maps_cat = torch.cat(
        tensors=[x.view(batch_size, n_total, -1) for x in feat_maps],
        dim=2,
    )

    pred_dist, pred_scores = feat_maps_cat.split((n_coords, n_classes), dim=1)

    return pred_dist, pred_scores


def make_anchors(
    feat_maps: list[torch.Tensor],
    strides: torch.Tensor,
    grid_cell_offset=0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = feat_maps[0].device, feat_maps[0].dtype
    anchor_points = []
    stride_tensors = []

    for i, stride in enumerate(strides):
        _, _, h, w = feat_maps[i].shape

        sx = torch.arange(0, w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(0, h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")  # (h, w)

        # (h, w) -> (h, w, 2) -> (hw, 2)
        anchor_point = torch.stack((sx, sy), dim=-1).view(-1, 2)
        stride_tensor = torch.full((h * w, 1), stride, device=device, dtype=dtype)

        anchor_points.append(anchor_point)
        stride_tensors.append(stride_tensor)

    return torch.cat(anchor_points), torch.cat(stride_tensors)


def decode_boxes(pred_dist: torch.Tensor, anchor_points: torch.Tensor, reg_max: int):
    batch_size, anchors, channels = pred_dist.shape
    device = pred_dist.device

    # (b, hw, 4, 16) -> (b, hw, 4)
    projection = torch.arange(reg_max, device=device, dtype=torch.float)
    pred_dist = pred_dist.view(batch_size, anchors, 4, channels // 4).softmax(dim=3)
    pred_dist = pred_dist.matmul(projection)

    # (b, hw, 4)
    return dist2bbox(pred_dist, anchor_points, xywh=False)


def decode_boxes_eval(
    head: Detect,
    pred_dist: torch.Tensor,
    anchor_points: torch.Tensor,
    stride_tensors: torch.Tensor,
):
    # https://github.com/ultralytics/ultralytics/blob/v8.2.87/ultralytics/nn/modules/head.py#L87
    boxes_dist = head.dfl(pred_dist)

    boxes = dist2bbox(boxes_dist, anchor_points.unsqueeze(0), xywh=True, dim=1)
    boxes_scaled = boxes * stride_tensors

    return boxes_scaled


def preprocess_targets(targets: tuple[dict]):
    batch_size = len(targets)
    n_max_boxes = 0

    for target in targets:
        n_max_boxes = max(n_max_boxes, len(target["labels"]))

    labels = torch.zeros((batch_size, n_max_boxes, 1))
    boxes = torch.zeros((batch_size, n_max_boxes, 4))

    for i, target in enumerate(targets):
        n_boxes = len(target["labels"])

        labels[i, :n_boxes, 0] = target["labels"]  # (n)
        boxes[i, :n_boxes, :4] = target["boxes"]  # (n, 4)

    return labels, boxes


class YOLOv8DetectionLoss:
    def __init__(
        self,
        reg_max: int,
        n_classes: int,
        strides: torch.Tensor,
        tal_top_k: int,
        cls_gain: float,
        box_gain: float,
        dfl_gain: float,
        device: torch.device,
    ):
        self.reg_max = reg_max
        self.n_classes = n_classes
        self.strides = strides
        self.cls_gain = cls_gain
        self.box_gain = box_gain
        self.dfl_gain = dfl_gain
        self.device = device

        self.assigner = TaskAlignedAssigner(tal_top_k, n_classes, alpha=0.5, beta=6.0)

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BboxLoss(reg_max)

    def __call__(self, feat_maps: list[torch.Tensor], targets: tuple[dict]):
        pred_dist, pred_scores = decode_feature_maps(
            feat_maps,
            self.reg_max,
            self.n_classes,
        )

        # Permute the dimension here as the preceding transformer is shared with the
        # inference pipeline.
        pred_dist = pred_dist.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()

        anchor_points, stride_tensors = make_anchors(
            feat_maps,
            self.strides,
            grid_cell_offset=0.5,
        )

        pred_boxes = decode_boxes(pred_dist, anchor_points, self.reg_max)

        target_labels, target_boxes = preprocess_targets(targets)
        target_labels = target_labels.to(self.device)
        target_boxes = target_boxes.to(self.device)

        mask = target_boxes.sum(dim=2, keepdim=True).gt_(0.0)

        _, target_boxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_boxes.detach() * stride_tensors,
            anchor_points * stride_tensors,
            target_labels,
            target_boxes,
            mask,
        )

        losses = torch.zeros(3, device=self.device)
        target_scores_sum = max(target_scores.sum(), 1)
        losses[0] = self.bce(pred_scores, target_scores).sum() / target_scores_sum

        if fg_mask.sum():
            target_boxes /= stride_tensors

            losses[1], losses[2] = self.bbox_loss(
                pred_dist,
                pred_boxes,
                anchor_points,
                target_boxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        losses[0] *= self.cls_gain
        losses[1] *= self.box_gain
        losses[2] *= self.dfl_gain

        return losses


class YOLOv8DetectionPredictor:
    def __init__(
        self,
        model: nn.Module,
        reg_max: int,
        n_classes: int,
        strides: torch.Tensor,
        confidence_threshold: float,
        iou_threshold: float,
    ):
        self.model = model
        self.reg_max = reg_max
        self.n_classes = n_classes
        self.strides = strides
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def __call__(self, feat_maps: list[torch.Tensor]):
        pred_dist, pred_scores = decode_feature_maps(
            feat_maps,
            self.reg_max,
            self.n_classes,
        )

        anchor_points, stride_tensors = make_anchors(
            feat_maps,
            self.strides,
            grid_cell_offset=0.5,
        )

        pred_boxes = decode_boxes_eval(
            self.model.model[-1],
            pred_dist,
            anchor_points.permute(1, 0),
            stride_tensors.permute(1, 0),
        )

        pred = torch.cat((pred_boxes, pred_scores.sigmoid()), dim=1)

        pred_nms = non_max_suppression(
            pred,
            conf_thres=self.confidence_threshold,
            iou_thres=self.iou_threshold,
            agnostic=False,
            max_det=300,
            classes=None,
            in_place=False,
        )

        return pred_nms


class TestYOLOUtils:
    @staticmethod
    def test_decode_feature_maps():
        feat_maps = [
            torch.zeros((3, 144, 48, 80)),
            torch.zeros((3, 144, 24, 40)),
            torch.zeros((3, 144, 12, 20)),
        ]

        reg_max = 16
        n_classes = 80
        pred_dist, pred_scores = decode_feature_maps(feat_maps, reg_max, n_classes)

        assert pred_dist.shape == (3, 64, 5040)
        assert pred_scores.shape == (3, 80, 5040)

    @staticmethod
    def test_make_anchors():
        feat_maps = [
            torch.zeros((3, 144, 48, 80)),
            torch.zeros((3, 144, 24, 40)),
            torch.zeros((3, 144, 12, 20)),
        ]

        strides = torch.tensor([8, 16, 32], dtype=torch.float32)

        anchor_points, stride_tensors = make_anchors(
            feat_maps,
            strides,
            grid_cell_offset=0.5,
        )

        assert anchor_points.shape == (5040, 2)
        assert stride_tensors.shape == (5040, 1)

    @staticmethod
    def test_decode_boxes():
        anchor_points = torch.zeros(5040, 2)
        pred_dist = torch.zeros(3, 5040, 64)

        boxes = decode_boxes(pred_dist, anchor_points, reg_max=16)

        assert boxes.shape == (3, 5040, 4)

    @staticmethod
    def test_decode_boxes_eval():
        feat_maps = [
            torch.zeros((3, 144, 48, 80)),
            torch.zeros((3, 144, 24, 40)),
            torch.zeros((3, 144, 12, 20)),
        ]

        head = Detect(nc=80, ch=(64, 128, 256))
        strides = torch.tensor([8, 16, 32])

        reg_max = 16
        n_classes = 80
        pred_dist, _ = decode_feature_maps(feat_maps, reg_max, n_classes)

        anchor_points, stride_tensors = make_anchors(
            feat_maps,
            strides,
            grid_cell_offset=0.5,
        )

        anchor_points = anchor_points.permute(1, 0)
        stride_tensors = stride_tensors.permute(1, 0)

        pred_boxes = decode_boxes_eval(head, pred_dist, anchor_points, stride_tensors)

        assert pred_boxes.shape == (3, 4, 5040)

    @staticmethod
    def test_preprocess_targets():
        targets = [
            {"labels": torch.tensor([1, 2, 3]), "boxes": torch.ones((3, 4))},
            {"labels": torch.tensor([4, 5]), "boxes": torch.ones((2, 4))},
        ]

        labels, boxes = preprocess_targets(targets)

        assert labels.shape == (2, 3, 1)
        assert boxes.shape == (2, 3, 4)

        assert torch.equal(labels[0, :, 0], torch.tensor([1, 2, 3]))
        assert torch.equal(labels[1, :, 0], torch.tensor([4, 5, 0]))

        expected_a = torch.ones((3, 4))
        expected_b = torch.cat((torch.ones((2, 4)), torch.zeros((1, 4))), dim=0)
        assert torch.equal(boxes[0], expected_a)
        assert torch.equal(boxes[1], expected_b)
