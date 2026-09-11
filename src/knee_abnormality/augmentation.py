from dataclasses import dataclass

import torch
from kornia.augmentation import RandomGaussianNoise
from kornia.enhance import adjust_contrast, adjust_gamma
from kornia.geometry.transform import get_affine_matrix2d, warp_affine
from torch import nn

from knee_abnormality.config import DataConfig, IntensityAugmentationConfig


@dataclass(frozen=True)
class IntensityParameters:
    applied: torch.Tensor
    gamma: torch.Tensor
    contrast: torch.Tensor


@dataclass(frozen=True)
class AffineParameters:
    applied: torch.Tensor
    angle_degrees: torch.Tensor
    translation_fraction: torch.Tensor
    scale: torch.Tensor


@dataclass(frozen=True)
class NoiseParameters:
    applied: torch.Tensor
    std: torch.Tensor


@dataclass(frozen=True)
class AugmentationParameters:
    intensity: IntensityParameters
    affine: AffineParameters
    noise: NoiseParameters


def sample_intensity_parameters(
    settings: IntensityAugmentationConfig,
    series_count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> IntensityParameters:
    draws = torch.rand((series_count, 3), device=device, generator=generator)
    gamma_low, gamma_high = settings.gamma_range
    contrast_low, contrast_high = settings.contrast_range
    return IntensityParameters(
        draws[:, 0] < settings.probability,
        gamma_low + (gamma_high - gamma_low) * draws[:, 1],
        contrast_low + (contrast_high - contrast_low) * draws[:, 2],
    )


def apply_intensity_augmentation(
    images: torch.Tensor,
    series_index: torch.Tensor,
    parameters: IntensityParameters,
) -> torch.Tensor:
    selected = parameters.applied[series_index]
    if not selected.any():
        return images
    selected_series = series_index[selected]
    values = images[selected]
    background = values == 0
    values = adjust_gamma(values, parameters.gamma[selected_series])
    values = (
        adjust_contrast(
            values - 0.5, parameters.contrast[selected_series], clip_output=False
        )
        .add_(0.5)
        .clamp_(0, 1)
    )
    values.masked_fill_(background, 0)
    images[selected] = values
    return images


def sample_augmentation_parameters(
    settings: DataConfig,
    series_count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> AugmentationParameters:
    intensity = sample_intensity_parameters(
        settings.intensity_augmentation, series_count, device, generator
    )
    affine = settings.affine_augmentation
    draws = torch.zeros((series_count, 5), device=device)
    if affine.probability > 0:
        draws = torch.rand((series_count, 5), device=device, generator=generator)
    scale_low, scale_high = affine.scale_range
    affine_parameters = AffineParameters(
        draws[:, 0] < affine.probability,
        (2 * draws[:, 1] - 1) * affine.rotation_degrees,
        (2 * draws[:, 2:4] - 1) * affine.translation_fraction,
        scale_low + (scale_high - scale_low) * draws[:, 4],
    )
    noise = settings.noise_augmentation
    draws = torch.zeros((series_count, 2), device=device)
    if noise.probability > 0:
        draws = torch.rand((series_count, 2), device=device, generator=generator)
    std_low, std_high = noise.std_range
    return AugmentationParameters(
        intensity,
        affine_parameters,
        NoiseParameters(
            draws[:, 0] < noise.probability,
            std_low + (std_high - std_low) * draws[:, 1],
        ),
    )


def apply_affine_augmentation(
    images: torch.Tensor,
    series_index: torch.Tensor,
    parameters: AffineParameters,
) -> torch.Tensor:
    selected = parameters.applied[series_index]
    if not selected.any():
        return images
    selected_series = series_index[selected]
    height, width = images.shape[-2:]
    center = images.new_tensor([(width - 1) / 2, (height - 1) / 2])
    translations = parameters.translation_fraction * images.new_tensor([width, height])
    matrices = get_affine_matrix2d(
        translations,
        center.expand(len(parameters.applied), -1),
        parameters.scale[:, None].expand(-1, 2),
        parameters.angle_degrees,
    )
    images[selected] = warp_affine(
        images[selected],
        matrices[selected_series, :2],
        (height, width),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return images


class ImageAugmentation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.noise = RandomGaussianNoise(p=1.0)

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        series_index: torch.Tensor,
        parameters: AugmentationParameters,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if not self.training:
            return images
        with torch.autocast(device_type=images.device.type, enabled=False):
            images = images.to(torch.float32)
            images = apply_affine_augmentation(images, series_index, parameters.affine)
            background = images == 0
            images = apply_intensity_augmentation(
                images, series_index, parameters.intensity
            )
            selected = parameters.noise.applied[series_index]
            if selected.any():
                values = images[selected]
                noise = torch.randn(
                    values.shape,
                    device=values.device,
                    dtype=values.dtype,
                    generator=generator,
                )
                noise.mul_(
                    parameters.noise.std[series_index[selected], None, None, None]
                )
                images[selected] = self.noise(
                    values,
                    params={
                        "batch_prob": values.new_ones(len(values)),
                        "gaussian_noise": noise,
                    },
                ).clamp_(0, 1)
            return images.masked_fill_(background, 0)
