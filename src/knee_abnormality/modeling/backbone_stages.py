from contextlib import contextmanager, nullcontext

import torch
from timm.layers import DropPath
from timm.models.convnext import ConvNeXt
from timm.models.efficientvit_mit import EfficientVit, ResidualBlock
from timm.models.focalnet import FocalNet
from timm.models.resnet import ResNet
from torch import nn
from torch.utils.checkpoint import checkpoint

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


class FocalNetPointwise(nn.Module):
    def __init__(self, convolution: nn.Conv2d) -> None:
        super().__init__()
        self.weight = convolution.weight
        self.bias = convolution.bias

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(
            features.permute(0, 2, 3, 1), self.weight[:, :, 0, 0], self.bias
        ).permute(0, 3, 1, 2)


class FocalNetStages(StemStages):
    def __init__(
        self,
        backbone: FocalNet,
        final_norm: nn.Module,
        middle_width: int,
        count_step: int,
    ) -> None:
        super().__init__(backbone, final_norm, middle_width, count_step)
        for module in backbone.layers.modules():
            for name, child in tuple(module.named_children()):
                if isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1):
                    if (
                        child.stride != (1, 1)
                        or child.padding != (0, 0)
                        or child.dilation != (1, 1)
                        or child.groups != 1
                    ):
                        raise ValueError("FocalNet pointwise projection geometry changed")
                    setattr(module, name, FocalNetPointwise(child))

    def _padded(self, tensor: torch.Tensor) -> torch.Tensor:
        missing = -tensor.shape[0] % self.count_step
        if not missing:
            return tensor
        memory_format = (
            torch.channels_last
            if tensor.is_contiguous(memory_format=torch.channels_last)
            else torch.contiguous_format
        )
        padding = torch.empty(
            (missing, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
            memory_format=memory_format,
        ).zero_()
        return torch.cat((tensor, padding))

    def to_stride_8(self, images: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        stem = backbone.stem(self._padded(images))
        return backbone.layers[1](backbone.layers[0](stem))[: images.shape[0]]

    def deep(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = features.shape[0]
        middle = self.backbone.layers[2](self._padded(features))
        maps = self.final_norm(self.backbone.layers[3](middle))
        return maps[:count], middle[:count]


class RealRowBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, normalization: nn.BatchNorm2d) -> None:
        if not normalization.affine or not normalization.track_running_stats:
            raise ValueError("EfficientViT requires affine tracked BatchNorm")
        if normalization.momentum is None:
            raise ValueError("EfficientViT requires fixed BatchNorm momentum")
        super().__init__(
            normalization.num_features,
            eps=normalization.eps,
            momentum=normalization.momentum,
            device=normalization.weight.device,
            dtype=normalization.weight.dtype,
        )
        self.weight = normalization.weight
        self.bias = normalization.bias
        self.running_mean = normalization.running_mean
        self.running_var = normalization.running_var
        self.num_batches_tracked = normalization.num_batches_tracked
        self.real_count = 0
        self.recomputing = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return super().forward(features)
        count = self.real_count
        if not 0 < count <= features.shape[0]:
            raise ValueError("EfficientViT BatchNorm requires the real image count")
        if self.recomputing:
            running_mean = self.running_mean.clone()
            running_var = self.running_var.clone()
        else:
            running_mean = self.running_mean
            running_var = self.running_var
            self.num_batches_tracked.add_(1)
        real, mean, inverse_std = torch.native_batch_norm(
            features[:count],
            self.weight,
            self.bias,
            running_mean,
            running_var,
            True,
            self.momentum,
            self.eps,
        )
        if count == features.shape[0]:
            return real
        with torch.no_grad():
            padded = features[count:].float()
            padded = (padded - mean[None, :, None, None]) * inverse_std[
                None, :, None, None
            ]
            padded = (
                padded * self.weight[None, :, None, None]
                + self.bias[None, :, None, None]
            )
            padded = padded.to(features.dtype)
        return torch.cat((real, padded)).contiguous(memory_format=torch.channels_last)


class EfficientVitResidualBlock(ResidualBlock):
    def __init__(self, block: ResidualBlock, rate: float) -> None:
        super().__init__(block.main, block.shortcut, block.pre_norm)
        self.drop_path = DropPath(rate)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.main(self.pre_norm(features))) + self.shortcut(
            features
        )


class EfficientVitStages(StemStages):
    def __init__(self, backbone: EfficientVit, stochastic_depth: float) -> None:
        super().__init__(
            backbone, nn.Identity(), backbone.feature_info[-2]["num_chs"], 16
        )
        backbone.head = nn.Identity()
        residuals = [
            (module, name, child)
            for stage in backbone.stages
            for module in stage.modules()
            for name, child in module.named_children()
            if isinstance(child, ResidualBlock) and child.shortcut is not None
        ]
        for index, (module, name, block) in enumerate(residuals):
            rate = stochastic_depth * (index / (len(residuals) - 1))
            setattr(module, name, EfficientVitResidualBlock(block, rate))
        for module in backbone.modules():
            for name, child in tuple(module.named_children()):
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, name, RealRowBatchNorm2d(child))
        self.normalizations = tuple(
            module
            for module in backbone.modules()
            if isinstance(module, RealRowBatchNorm2d)
        )
        self.checkpoint_normalizations = tuple(
            module
            for module in backbone.stages[0].modules()
            if isinstance(module, RealRowBatchNorm2d)
        )

    @contextmanager
    def _recompute(self, count: int):
        normalizations = self.checkpoint_normalizations
        previous = tuple(
            (normalization.real_count, normalization.recomputing)
            for normalization in normalizations
        )
        try:
            for normalization in normalizations:
                normalization.real_count = count
                normalization.recomputing = True
            yield
        finally:
            for normalization, (real_count, recomputing) in zip(
                normalizations, previous, strict=True
            ):
                normalization.real_count = real_count
                normalization.recomputing = recomputing

    def to_stride_8(self, images: torch.Tensor) -> torch.Tensor:
        count = images.shape[0]
        features = self.backbone.stem(self._padded(images))
        if self.backbone.training and torch.is_grad_enabled():
            features = checkpoint(
                self.backbone.stages[0],
                features,
                use_reentrant=False,
                preserve_rng_state=True,
                context_fn=lambda: (nullcontext(), self._recompute(count)),
            )
        else:
            features = self.backbone.stages[0](features)
        return self.backbone.stages[1](features)[:count]

    def _padded(self, tensor: torch.Tensor) -> torch.Tensor:
        for normalization in self.normalizations:
            normalization.real_count = tensor.shape[0]
        return super()._padded(tensor)


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
    FocalNet: lambda backbone: FocalNetStages(
        backbone, backbone.norm, backbone.feature_info[-2]["num_chs"], 16
    ),
}


def backbone_stages(
    backbone: nn.Module, stochastic_depth: float
) -> ResNetStages | StemStages:
    if isinstance(backbone, EfficientVit):
        return EfficientVitStages(backbone, stochastic_depth)
    return BACKBONE_STAGES[type(backbone)](backbone)
