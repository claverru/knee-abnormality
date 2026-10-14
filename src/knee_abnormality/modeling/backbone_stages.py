import torch
from timm.models.convnext import ConvNeXt
from timm.models.resnet import ResNet
from torch import nn

from knee_abnormality.modeling.staged_backbones import StagedBackbone

CONVNEXT_IMAGE_COUNT_STEP = 16


class ResNetStages:
    def __init__(self, backbone: ResNet) -> None:
        self.backbone = backbone
        self.middle_width = backbone.layer3[-1].conv3.out_channels

    def to_stride_8(self, images: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        stem = backbone.maxpool(backbone.act1(backbone.bn1(backbone.conv1(images))))
        return backbone.layer2(backbone.layer1(stem))

    def deep(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        middle = self.backbone.layer3(features)
        return self.backbone.layer4(middle), middle


class StemStages:
    def __init__(
        self,
        backbone: nn.Module,
        final_norm: nn.Module,
        middle_width: int,
        count_step: int,
    ) -> None:
        self.backbone = backbone
        self.final_norm = final_norm
        self.middle_width = middle_width
        self.count_step = count_step

    def _padded(self, tensor: torch.Tensor) -> torch.Tensor:
        missing = -tensor.shape[0] % self.count_step
        if not missing:
            return tensor
        memory_format = (
            torch.channels_last
            if tensor.is_contiguous(memory_format=torch.channels_last)
            else torch.contiguous_format
        )
        padded = torch.cat((tensor, tensor.new_zeros(missing, *tensor.shape[1:])))
        return padded.contiguous(memory_format=memory_format)

    def to_stride_8(self, images: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        stem = backbone.stem(self._padded(images))
        return backbone.stages[1](backbone.stages[0](stem))[: images.shape[0]]

    def deep(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = features.shape[0]
        middle = self.backbone.stages[2](self._padded(features))
        maps = self.final_norm(self.backbone.stages[3](middle))
        return maps[:count], middle[:count]


BACKBONE_STAGES = {
    ResNet: ResNetStages,
    ConvNeXt: lambda backbone: StemStages(
        backbone,
        backbone.norm_pre,
        backbone.feature_info[-2]["num_chs"],
        CONVNEXT_IMAGE_COUNT_STEP,
    ),
    StagedBackbone: lambda backbone: StemStages(
        backbone, backbone.norm, backbone.stage_widths[1], 1
    ),
}


def backbone_stages(backbone: nn.Module) -> ResNetStages | StemStages:
    return BACKBONE_STAGES[type(backbone)](backbone)
