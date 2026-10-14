import torch
from timm.models.convnext import ConvNeXt
from timm.models.resnet import ResNet
from torch import nn

from knee_abnormality.modeling.spatial_tokens import backbone_mapping
from knee_abnormality.modeling.staged_backbones import StagedBackbone

CONVNEXT_IMAGE_COUNT_STEP = 16


class ResNetStages:
    def __init__(self, name: str, backbone: ResNet) -> None:
        self.backbone = backbone
        self.widths = tuple(
            getattr(stage[-1], "conv3", stage[-1].conv2).out_channels
            for stage in (backbone.layer2, backbone.layer3, backbone.layer4)
        )
        mapping = backbone_mapping(f"{name}.layer3", backbone)
        self.decoder_mapping = (*mapping[:2], *(step * 0.5 for step in mapping[2:]))

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
        name: str,
        backbone: nn.Module,
        final_norm: nn.Module,
        widths: tuple[int, int, int],
        count_step: int,
    ) -> None:
        self.backbone = backbone
        self.final_norm = final_norm
        self.widths = widths
        self.count_step = count_step
        self.decoder_mapping = backbone_mapping(f"{name}.layer2", backbone)

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
    ConvNeXt: lambda name, backbone: StemStages(
        name,
        backbone,
        backbone.norm_pre,
        tuple(stage["num_chs"] for stage in backbone.feature_info[-3:]),
        CONVNEXT_IMAGE_COUNT_STEP,
    ),
    StagedBackbone: lambda name, backbone: StemStages(
        name, backbone, backbone.norm, backbone.stage_widths, 1
    ),
}


def backbone_stages(name: str, backbone: nn.Module) -> ResNetStages | StemStages:
    return BACKBONE_STAGES[type(backbone)](name, backbone)
