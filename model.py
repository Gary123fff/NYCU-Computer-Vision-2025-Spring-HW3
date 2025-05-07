"""Custom Mask R-CNN model using Swin Transformer as backbone."""

from collections import OrderedDict
from torch import nn, relu
import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from timm import create_model


class CustomBoxPredictor(nn.Module):
    """Custom box predictor head for object classification and bounding box regression."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

    def forward(self, x):
        """Forward pass for box predictor."""
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        scores = self.cls_score(x)
        boxes = self.bbox_pred(x)
        return scores, boxes


class CustomMaskHead(nn.Sequential):
    """Custom mask head for pixel-level instance segmentation."""

    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(
            nn.Conv2d(in_channels, dim_reduced, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim_reduced, dim_reduced, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim_reduced, dim_reduced, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(dim_reduced, num_classes, 1)
        )


class SwinFPNBackbone(nn.Module):
    """Backbone combining Swin Transformer and Feature Pyramid Network."""

    def __init__(self, out_channels=384):
        super().__init__()
        self.body = create_model(
            model_name='swin_base_patch4_window12_384',
            pretrained=True,
            features_only=True
        )
        self.out_channels = out_channels

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[128, 256, 512, 1024],
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool()
        )

    def forward(self, x):
        """Forward pass through Swin Transformer and FPN."""
        feats = self.body(x)
        feats = [f.permute(0, 3, 1, 2) for f in feats]
        fpn_outs = self.fpn(OrderedDict(
            [(str(i), feat) for i, feat in enumerate(feats)]))
        return fpn_outs


def swin_model_fpn(num_classes, device):
    """
    Build a Mask R-CNN model with Swin Transformer + FPN backbone.

    Args:
        num_classes (int): Number of classes including background.
        device (torch.device): Device to put the model on.

    Returns:
        MaskRCNN: The constructed model.
    """
    swin_backbone = SwinFPNBackbone()

    anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    num_anchors = anchor_generator.num_anchors_per_location()[0]

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    mask_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2
    )

    model = MaskRCNN(
        backbone=swin_backbone,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        num_classes=num_classes,
        min_size=384,
        max_size=384
    )

    model.roi_heads.positive_fraction = 0.25
    model.rpn.head = RPNHead(in_channels=384, num_anchors=num_anchors)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = CustomBoxPredictor(
        in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = CustomMaskHead(
        in_channels=in_features_mask,
        dim_reduced=384,
        num_classes=num_classes
    )

    return model.to(device)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
