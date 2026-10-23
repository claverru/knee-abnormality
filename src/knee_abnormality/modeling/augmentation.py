from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, partial

import torch
import torch.nn.functional as F
from kornia.augmentation import RandomGaussianNoise
from kornia.enhance import adjust_contrast, adjust_gamma
from kornia.filters import get_gaussian_kernel2d
from kornia.geometry.transform import get_affine_matrix2d, warp_affine
from torch import nn

from knee_abnormality.config import (
    AffineAugmentationConfig,
    DataConfig,
    FeatureStatisticsAugmentationConfig,
    GinAugmentationConfig,
    IntensityAugmentationConfig,
    NoiseAugmentationConfig,
    ResolutionAugmentationConfig,
    SmoothingAugmentationConfig,
)

SMOOTHING_KERNEL_SIZE = 5
GIN_KERNEL_SIZE = 3
GIN_HIDDEN_CHANNELS = 2
GIN_CHANNEL_DIMS = (
    (1, GIN_HIDDEN_CHANNELS),
    (GIN_HIDDEN_CHANNELS, GIN_HIDDEN_CHANNELS),
    (GIN_HIDDEN_CHANNELS, GIN_HIDDEN_CHANNELS),
    (GIN_HIDDEN_CHANNELS, 1),
)
PLANE_NORMAL_AXIS = (2, 1, 0)

REFLECTION_IDENTITY = {
    "algorithm": "study_LPS_X_reflection_v3",
    "eligibility": "all_supplied_acquisitions_native_series_reliable_dominant_axes_margin_orientation_tolerance_either_sign_before_dropout",
    "axes": {"Axial": [0, 1, 2], "Coronal": [0, 2, 1], "Sagittal": [1, 2, 0]},
    "pixels": "exact_columns_axial_coronal_channels_sagittal_before_affine_and_normalization",
    "membership": "triplet_container_order_and_middle_slice_unchanged",
    "embeddings": "plane_and_contrast_unchanged",
    "knee_side": "side_head_target_and_routing_left_probability_flipped_for_mirrored_studies",
    "positions": "negate_sagittal_preserve_axial_coronal",
    "source_channels": "sagittal_2_minus_k_middle_unchanged_source_metadata_immutable",
    "validity": "F_times_U_original_rectangles_all_associated_channels",
    "targets": "unchanged_all_twelve_and_observation_masks",
    "rng": "one_torch_device_draw_per_study_training_forward_mask_eligibility_eval_no_draw",
    "parameters": "data.reflection_augmentation.probability",
}


@dataclass(frozen=True)
class ReflectionParameters:
    applied: torch.Tensor


def sample_reflection_parameters(
    probability: float, eligible: torch.Tensor, generator: torch.Generator | None = None
) -> ReflectionParameters:
    return ReflectionParameters(
        (
            torch.rand(eligible.shape, device=eligible.device, generator=generator)
            < probability
        )
        & eligible
    )


def reflection_channels(
    reflected: torch.Tensor, sagittal: torch.Tensor
) -> torch.Tensor:
    channels = torch.arange(3, device=reflected.device).expand(len(reflected), -1)
    return torch.where((reflected & sagittal)[:, None], 2 - channels, channels)


def apply_reflection(
    images: torch.Tensor, reflected: torch.Tensor, sagittal: torch.Tensor
) -> torch.Tensor:
    images = torch.where(
        (reflected & ~sagittal)[:, None, None, None], images.flip(-1), images
    )
    return torch.where(
        (reflected & sagittal)[:, None, None, None], images.flip(1), images
    )


@cache
def _reflection_matrices(
    width: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode(False):
        identity = torch.eye(3, dtype=torch.float32)
        reversal = identity.clone()
        reversal[0, 0] = -1
        reversal[0, 2] = width - 1
        return identity.to(device), reversal.to(device)


@cache
def _prepared_conversion(
    height: int, width: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode(False):
        conversion = torch.tensor(
            [
                [(width - 1) / width, 0, (width - 1) / (2 * width)],
                [0, (height - 1) / height, (height - 1) / (2 * height)],
                [0, 0, 1],
            ],
            device=device,
        )
        return torch.linalg.inv(conversion), conversion


@cache
def _plane_normal_axis(device: torch.device) -> torch.Tensor:
    with torch.inference_mode(False):
        return torch.tensor(PLANE_NORMAL_AXIS, dtype=torch.long).to(device)


@cache
def _normal_centre(height: int, width: int, device: torch.device) -> torch.Tensor:
    with torch.inference_mode(False):
        return torch.tensor([width / 2, height / 2, 1.0]).to(device)


@cache
def _normalized_to_pixel(height: int, width: int, device: torch.device) -> torch.Tensor:
    with torch.inference_mode(False):
        matrix = torch.eye(3, dtype=torch.float32)
        matrix[0, 0] = width / 2
        matrix[1, 1] = height / 2
        matrix[0, 2] = (width - 1) / 2
        matrix[1, 2] = (height - 1) / 2
        return matrix.to(device)


@cache
def contrast_pair_slots(
    pairs: tuple[tuple[int, int], ...], device: torch.device
) -> torch.Tensor:
    with torch.inference_mode(False):
        return torch.tensor(pairs, dtype=torch.long).to(device)


def reflection_inverse(
    inverse: torch.Tensor, reflected: torch.Tensor, sagittal: torch.Tensor, width: int
) -> torch.Tensor:
    with torch.autocast(device_type=inverse.device.type, enabled=False):
        identity, reversal = _reflection_matrices(width, inverse.device)
        return (
            torch.where((reflected & ~sagittal)[:, None, None], reversal, identity)
            @ inverse.float()
        )


def reflection_positions(
    positions: torch.Tensor, reflected: torch.Tensor, sagittal: torch.Tensor
) -> torch.Tensor:
    return torch.where(reflected & sagittal, -positions, positions)


def reflection_valid_rects(
    valid_rects: torch.Tensor,
    reflected: torch.Tensor,
    sagittal: torch.Tensor,
    width: int,
) -> torch.Tensor:
    rects = valid_rects.clone()
    rects[..., 1] = torch.where(
        (reflected & ~sagittal)[:, None],
        width - rects[..., 1] - rects[..., 3],
        rects[..., 1],
    )
    return torch.where((reflected & sagittal)[:, None, None], rects.flip(1), rects)


def uniform_draws(
    probability: float,
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if probability == 0:
        return torch.zeros(shape, device=device)
    return torch.rand(shape, device=device, generator=generator)


@dataclass(frozen=True)
class AugmentationActivity:
    affected_acquisitions: torch.Tensor
    total_acquisitions: int
    changed_elements: torch.Tensor
    mean_absolute_change: torch.Tensor


@dataclass(frozen=True)
class FeatureStatisticsActivity(AugmentationActivity):
    clamped_channels: torch.Tensor
    applied_clamped_channels: torch.Tensor
    total_channels: int


@dataclass(frozen=True)
class ContrastMixingParameters:
    eligible: torch.Tensor
    applied: torch.Tensor
    target_first: torch.Tensor


@dataclass(frozen=True)
class ContrastMixingActivity:
    acquisitions: torch.Tensor
    studies: torch.Tensor
    mean_lambda: torch.Tensor
    mean_pixel_change: torch.Tensor
    effective_probability: torch.Tensor
    mean_alignment_change: torch.Tensor


def sample_contrast_mixing_parameters(
    probability: float, eligible: torch.Tensor, generator: torch.Generator | None = None
) -> ContrastMixingParameters:
    draws = uniform_draws(probability, (*eligible.shape, 2), eligible.device, generator)
    return ContrastMixingParameters(
        eligible, (draws[..., 0] < probability) & eligible, draws[..., 1] < 0.5
    )


def sample_contrast_mixing_lambda(
    count: int,
    weight_range: tuple[float, float],
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    low, high = weight_range
    return low + (high - low) * torch.rand(count, device=device, generator=generator)


def aligned_contrast_donors(
    donors: torch.Tensor,
    target_maps: torch.Tensor,
    donor_maps: torch.Tensor,
    column_reflected: torch.Tensor,
) -> torch.Tensor:
    with torch.autocast(device_type=donors.device.type, enabled=False):
        height, width = donors.shape[-2:]
        identity, reversal = _reflection_matrices(width, donors.device)
        pixel_transform = torch.where(
            column_reflected[:, None, None], reversal, identity
        )
        target_maps = target_maps.float() @ pixel_transform
        donor_maps = donor_maps.float() @ pixel_transform
        donor_basis = donor_maps[..., :2]
        gram = donor_basis.transpose(1, 2) @ donor_basis
        a, b, c = gram[:, 0, 0], gram[:, 0, 1], gram[:, 1, 1]
        inverse = torch.stack((c, -b, -b, a), -1).view(-1, 2, 2)
        inverse = inverse / (a * c - b.square()).clamp_min(1e-12)[:, None, None]
        relative_maps = target_maps.clone()
        relative_maps[..., 2] -= donor_maps[..., 2]
        affine = inverse @ donor_basis.transpose(1, 2) @ relative_maps
        theta = affine @ _normalized_to_pixel(height, width, donors.device)
        theta[:, 0, 2] -= (width - 1) / 2
        theta[:, 1, 2] -= (height - 1) / 2
        theta[:, 0] *= 2 / width
        theta[:, 1] *= 2 / height
        grid = F.affine_grid(theta, donors.shape, align_corners=False)
        aligned = F.grid_sample(
            donors.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        covered = (grid[..., 0].abs() <= 1 - 1 / width + 1e-6) & (
            grid[..., 1].abs() <= 1 - 1 / height + 1e-6
        )
        return aligned.masked_fill(~covered[:, None], 0)


def apply_contrast_mixing(
    images: torch.Tensor,
    parameters: ContrastMixingParameters,
    pairs: tuple[tuple[int, int], ...],
    weight_range: tuple[float, float],
    acquisition_index: torch.Tensor,
    token_study: torch.Tensor,
    token_type: torch.Tensor,
    token_plane: torch.Tensor,
    patient_maps: torch.Tensor,
    geometry_valid: torch.Tensor,
    reflected: torch.Tensor,
) -> tuple[torch.Tensor, ContrastMixingActivity]:
    device = images.device
    zero = torch.zeros((), device=device)
    with torch.autocast(device_type=device.type, enabled=False):
        normal_axis = _plane_normal_axis(device)[token_plane]
        rows = len(images)
        normal_rows = patient_maps[torch.arange(rows, device=device), 1, normal_axis]
        height, width = images.shape[-2:]
        normal_position = normal_rows @ _normal_centre(height, width, device)
        studies, planes = parameters.applied.shape
        groups_total = studies * planes
        slots = contrast_pair_slots(pairs, device)
        first = parameters.target_first
        target_slot = torch.where(first, slots[:, 0], slots[:, 1])
        other_slot = torch.where(first, slots[:, 1], slots[:, 0])
        study_rows = (
            token_study == torch.arange(studies, device=device)[:, None]
        ) & geometry_valid
        candidate = study_rows[:, None] & parameters.applied[..., None]
        target = (candidate & (token_type == target_slot[..., None])).reshape(
            groups_total, rows
        )
        other = (candidate & (token_type == other_slot[..., None])).reshape(
            groups_total, rows
        )
        valid = target.any(-1) & other.any(-1)
        target = target & valid[:, None]
        rank = torch.cat(
            (
                acquisition_index.new_zeros(1),
                (acquisition_index[1:] != acquisition_index[:-1]).long(),
            )
        ).cumsum(0)
        acquired = (
            images.new_zeros((groups_total, rows)).index_add_(1, rank, target.float())
            > 0
        )
        unique_counts = acquired.sum(-1)
        row_counts = target.sum(-1)
        fetched = torch.cat(
            (
                valid.long(),
                unique_counts,
                row_counts,
                parameters.eligible.sum().reshape(1),
            )
        ).tolist()
        valid_groups = [
            group for group, flag in enumerate(fetched[:groups_total]) if flag
        ]
        group_acquisitions = fetched[groups_total : 2 * groups_total]
        group_rows = fetched[2 * groups_total : 3 * groups_total]
        eligible_count = fetched[-1]
        mixed_groups = len(valid_groups)
        mixed_acquisitions = sum(group_acquisitions[group] for group in valid_groups)
        mixed_studies = len({group // planes for group in valid_groups})
        lambdas = [
            sample_contrast_mixing_lambda(
                group_acquisitions[group], weight_range, device
            )
            for group in valid_groups
        ]
        pixel_change = alignment_change = zero
        if valid_groups:
            target_count = sum(group_rows[group] for group in valid_groups)
            positions = torch.nonzero_static(target.flatten(), size=target_count)[:, 0]
            group_of = positions // rows
            target_rows = positions % rows
            used = unique_counts * valid
            offset = used.cumsum(0) - used
            inverse = acquired.long().cumsum(-1)[group_of, rank[target_rows]] - 1
            row_lambda = torch.cat(lambdas)[offset[group_of] + inverse]
            distances = (
                (normal_position[target_rows][:, None] - normal_position[None, :])
                .abs()
                .masked_fill(~other[group_of], torch.inf)
            )
            matched_rows = distances.argmin(dim=1)
            group_index = valid.long().cumsum(0)[group_of] - 1
            before = images[target_rows]
            donors = images[matched_rows]
            other = aligned_contrast_donors(
                donors,
                patient_maps[target_rows, 1],
                patient_maps[matched_rows, 1],
                reflected[target_rows] & (token_plane[target_rows] != 2),
            )
            lam = row_lambda[:, None, None, None]
            alignment_change = (
                ((1 - lam) * (other - donors).abs())
                .masked_fill((before == 0) | (other == 0), 0)
                .mean()
            )
            mixed = torch.where(other == 0, before, lam * before + (1 - lam) * other)
            mixed = mixed.masked_fill(before == 0, 0)
            images[target_rows] = mixed
            row_changes = (mixed - before).abs().mean((1, 2, 3))
            group_changes = zero.new_zeros(mixed_groups).index_add_(
                0, group_index, row_changes
            )
            group_counts = zero.new_zeros(mixed_groups).index_add_(
                0, group_index, torch.ones_like(row_changes)
            )
            pixel_change = (group_changes / group_counts).mean()
    return images, ContrastMixingActivity(
        zero.new_full((), float(mixed_acquisitions)),
        zero.new_full((), float(mixed_studies)),
        row_lambda.mean() if valid_groups else zero,
        pixel_change,
        zero.new_full((), mixed_groups / max(eligible_count, 1)),
        alignment_change,
    )


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
    draws = uniform_draws(settings.probability, (series_count, 3), device, generator)
    gamma_low, gamma_high = settings.gamma_range
    contrast_low, contrast_high = settings.contrast_range
    return IntensityParameters(
        draws[:, 0] < settings.probability,
        gamma_low + (gamma_high - gamma_low) * draws[:, 1],
        contrast_low + (contrast_high - contrast_low) * draws[:, 2],
    )


def sample_affine_parameters(
    settings: AffineAugmentationConfig,
    series_count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> AffineParameters:
    draws = uniform_draws(settings.probability, (series_count, 5), device, generator)
    scale_low, scale_high = settings.scale_range
    return AffineParameters(
        draws[:, 0] < settings.probability,
        (2 * draws[:, 1] - 1) * settings.rotation_degrees,
        (2 * draws[:, 2:4] - 1) * settings.translation_fraction,
        scale_low + (scale_high - scale_low) * draws[:, 4],
    )


def sample_noise_parameters(
    settings: NoiseAugmentationConfig,
    series_count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> NoiseParameters:
    draws = uniform_draws(settings.probability, (series_count, 2), device, generator)
    std_low, std_high = settings.std_range
    return NoiseParameters(
        draws[:, 0] < settings.probability,
        std_low + (std_high - std_low) * draws[:, 1],
    )


def sample_augmentation_parameters(
    settings: DataConfig,
    series_count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> AugmentationParameters:
    return AugmentationParameters(
        sample_intensity_parameters(
            settings.intensity_augmentation, series_count, device, generator
        ),
        sample_affine_parameters(
            settings.affine_augmentation, series_count, device, generator
        ),
        sample_noise_parameters(
            settings.noise_augmentation, series_count, device, generator
        ),
    )


def selected_rows(selected: torch.Tensor, count: int) -> torch.Tensor:
    return torch.nonzero_static(selected, size=count)[:, 0]


def apply_intensity_augmentation(
    images: torch.Tensor,
    series_index: torch.Tensor,
    parameters: IntensityParameters,
    rows: torch.Tensor,
) -> torch.Tensor:
    if not len(rows):
        return images
    selected_series = series_index[rows]
    values = images[rows]
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
    images[rows] = values
    return images


def affine_matrices(
    parameters: AffineParameters, height: int, width: int
) -> torch.Tensor:
    values = parameters.translation_fraction.float()
    with torch.autocast(device_type=values.device.type, enabled=False):
        center = values.new_tensor([(width - 1) / 2, (height - 1) / 2])
        matrices = get_affine_matrix2d(
            values * values.new_tensor([width, height]),
            center.expand(len(parameters.applied), -1),
            parameters.scale.float()[:, None].expand(-1, 2),
            parameters.angle_degrees.float(),
        )
        return torch.where(
            parameters.applied[:, None, None],
            matrices,
            torch.eye(3, device=values.device, dtype=values.dtype)[None],
        )


def augmented_to_prepared(
    matrices: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    matrices = matrices.float()
    with torch.autocast(device_type=matrices.device.type, enabled=False):
        inverse_conversion, conversion = _prepared_conversion(
            height, width, matrices.device
        )
        return inverse_conversion @ torch.linalg.inv_ex(matrices)[0] @ conversion


def apply_affine_augmentation(
    images: torch.Tensor,
    series_index: torch.Tensor,
    matrices: torch.Tensor,
    rows: torch.Tensor,
) -> torch.Tensor:
    if not len(rows):
        return images
    selected_series = series_index[rows]
    height, width = images.shape[-2:]
    images[rows] = warp_affine(
        images[rows],
        matrices[selected_series, :2],
        (height, width),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return images


def sample_noise_values(
    images: torch.Tensor,
    series_index: torch.Tensor,
    parameters: NoiseParameters,
    rows: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    if not len(rows):
        return None
    noise = torch.randn(
        (len(rows), *images.shape[1:]),
        device=images.device,
        dtype=images.dtype,
        generator=generator,
    )
    return noise.mul_(parameters.std[series_index[rows], None, None, None])


def geometric_valid_mask(
    valid_rects: torch.Tensor,
    inverse_affine: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=valid_rects.device, dtype=torch.float32),
        torch.arange(width, device=valid_rects.device, dtype=torch.float32),
        indexing="ij",
    )
    prepared_x = (
        inverse_affine[:, 0, 0, None, None] * x
        + inverse_affine[:, 0, 1, None, None] * y
        + inverse_affine[:, 0, 2, None, None]
    )[:, None]
    prepared_y = (
        inverse_affine[:, 1, 0, None, None] * x
        + inverse_affine[:, 1, 1, None, None] * y
        + inverse_affine[:, 1, 2, None, None]
    )[:, None]
    top, left, rows, columns = valid_rects.float().unbind(-1)
    return (
        (prepared_x >= left[..., None, None])
        & (prepared_x <= (left + columns - 1)[..., None, None])
        & (prepared_y >= top[..., None, None])
        & (prepared_y <= (top + rows - 1)[..., None, None])
    )


def warped_valid_mask(
    valid_rects: torch.Tensor,
    matrices: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    channels = valid_rects.shape[1]
    top, left, rows, columns = valid_rects.float().unbind(-1)
    y = torch.arange(height, device=valid_rects.device)[None, None, :, None]
    x = torch.arange(width, device=valid_rects.device)[None, None, None, :]
    valid = (
        (y >= top[..., None, None])
        & (y < (top + rows)[..., None, None])
        & (x >= left[..., None, None])
        & (x < (left + columns)[..., None, None])
    )
    return (
        warp_affine(
            valid.reshape(-1, 1, height, width).float(),
            matrices.repeat_interleave(channels, dim=0)[:, :2],
            (height, width),
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(-1, channels, height, width)
        > 0.5
    )


@dataclass(frozen=True)
class SmoothingParameters:
    applied: torch.Tensor
    range_sigma: torch.Tensor
    spatial_sigma: float


def sample_smoothing_parameters(
    settings: SmoothingAugmentationConfig,
    series_count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> SmoothingParameters:
    draws = uniform_draws(settings.probability, (series_count, 2), device, generator)
    sigma_low, sigma_high = settings.range_sigma_range
    return SmoothingParameters(
        draws[:, 0] < settings.probability,
        sigma_low + (sigma_high - sigma_low) * draws[:, 1],
        settings.spatial_sigma,
    )


@cache
def _space_kernel(
    spatial_sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    with torch.inference_mode(False):
        return get_gaussian_kernel2d(
            (SMOOTHING_KERNEL_SIZE, SMOOTHING_KERNEL_SIZE),
            (spatial_sigma, spatial_sigma),
            device=device,
            dtype=dtype,
        ).reshape(1, 1, 1, 1, SMOOTHING_KERNEL_SIZE * SMOOTHING_KERNEL_SIZE)


def apply_bilateral_smoothing(
    values: torch.Tensor,
    mask: torch.Tensor,
    selected_series: torch.Tensor,
    parameters: SmoothingParameters,
) -> torch.Tensor:
    series_count, channels, height, width = values.shape
    pad = SMOOTHING_KERNEL_SIZE // 2
    support = mask.to(values.dtype)
    masked = values * support
    flattened = masked.reshape(series_count * channels, 1, height, width)
    flattened_support = support.reshape(series_count * channels, 1, height, width)
    padded_input = F.pad(flattened, (pad, pad, pad, pad), mode="constant", value=0.0)
    padded_support = F.pad(
        flattened_support, (pad, pad, pad, pad), mode="constant", value=0.0
    )
    unfolded_input = (
        padded_input.unfold(2, SMOOTHING_KERNEL_SIZE, 1)
        .unfold(3, SMOOTHING_KERNEL_SIZE, 1)
        .flatten(-2)
    )
    unfolded_support = (
        padded_support.unfold(2, SMOOTHING_KERNEL_SIZE, 1)
        .unfold(3, SMOOTHING_KERNEL_SIZE, 1)
        .flatten(-2)
    )
    diff = unfolded_input - flattened.unsqueeze(-1)
    color_sigma = (
        parameters.range_sigma[selected_series][:, None]
        .expand(-1, channels)
        .reshape(-1, 1, 1, 1, 1)
    )
    color_kernel = (-0.5 * diff.square() / color_sigma.square()).exp()
    space_kernel = _space_kernel(parameters.spatial_sigma, values.device, values.dtype)
    weights = space_kernel * color_kernel * unfolded_support
    denominator = weights.sum(-1).clamp_min(1e-6)
    numerator = (unfolded_input * weights).sum(-1)
    return (numerator / denominator).reshape(series_count, channels, height, width)


@dataclass(frozen=True)
class ResolutionParameters:
    applied: torch.Tensor
    columns: torch.Tensor
    keep_fraction: torch.Tensor


def sample_resolution_parameters(
    settings: ResolutionAugmentationConfig,
    series_count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> ResolutionParameters:
    draws = uniform_draws(settings.probability, (series_count, 3), device, generator)
    low, high = settings.keep_fraction_range
    return ResolutionParameters(
        draws[:, 0] < settings.probability,
        draws[:, 1] < 0.5,
        low + (high - low) * draws[:, 2],
    )


def apply_resolution_degradation(
    values: torch.Tensor,
    mask: torch.Tensor,
    selected_series: torch.Tensor,
    parameters: ResolutionParameters,
    axis_rows: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    values = values.masked_fill(~mask, 0)
    for dimension, on_axis in zip((2, 3), axis_rows, strict=True):
        if not len(on_axis):
            continue
        size = values.shape[dimension]
        frequency = torch.fft.rfft(values[on_axis], dim=dimension)
        cutoff = torch.floor(
            parameters.keep_fraction[selected_series[on_axis]] * (size // 2)
        )
        bands = torch.arange(size // 2 + 1, device=values.device)
        shape = [1, 1, 1, 1]
        shape[dimension] = size // 2 + 1
        frequency.masked_fill_(bands.reshape(shape) > cutoff[:, None, None, None], 0)
        values[on_axis] = torch.fft.irfft(frequency, n=size, dim=dimension).abs()
    return values


@dataclass(frozen=True)
class GinParameters:
    applied: torch.Tensor
    blend: torch.Tensor
    layer_weights: tuple[torch.Tensor, ...]


@cache
def _gin_center_only(device: torch.device) -> torch.Tensor:
    with torch.inference_mode(False):
        center = GIN_KERNEL_SIZE // 2
        center_only = torch.zeros(GIN_KERNEL_SIZE, GIN_KERNEL_SIZE)
        center_only[center, center] = 1.0
        return center_only.to(device)


def sample_gin_parameters(
    settings: GinAugmentationConfig,
    series_count: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> GinParameters:
    draws = uniform_draws(settings.probability, (series_count, 3), device, generator)
    layer_weights = ()
    if settings.probability > 0:
        center_only = _gin_center_only(device)
        kernel = torch.where(
            (draws[:, 1] < 0.5)[:, None, None],
            center_only,
            torch.ones(GIN_KERNEL_SIZE, GIN_KERNEL_SIZE, device=device),
        )
        layer_weights = tuple(
            torch.randn(
                (
                    series_count,
                    out_channels,
                    in_channels,
                    GIN_KERNEL_SIZE,
                    GIN_KERNEL_SIZE,
                ),
                device=device,
                generator=generator,
            )
            * kernel[:, None, None]
            for in_channels, out_channels in GIN_CHANNEL_DIMS
        )
    return GinParameters(
        draws[:, 0] < settings.probability,
        draws[:, 2] * settings.blend_max,
        layer_weights,
    )


def apply_gin_filter(
    values: torch.Tensor,
    mask: torch.Tensor,
    selected_series: torch.Tensor,
    parameters: GinParameters,
) -> torch.Tensor:
    triplets, channels, height, width = values.shape
    groups = triplets * channels
    hidden = (values * mask).reshape(groups, 1, height, width)
    layer_count = len(parameters.layer_weights)
    for index, weight in enumerate(parameters.layer_weights):
        selected_weight = weight[selected_series].repeat_interleave(channels, dim=0)
        out_channels = selected_weight.shape[1]
        in_channels = selected_weight.shape[2]
        hidden = F.conv2d(
            hidden.reshape(1, groups * in_channels, height, width),
            selected_weight.reshape(
                groups * out_channels, in_channels, GIN_KERNEL_SIZE, GIN_KERNEL_SIZE
            ),
            padding=GIN_KERNEL_SIZE // 2,
            groups=groups,
        ).reshape(groups, out_channels, height, width)
        if index < layer_count - 1:
            hidden = F.leaky_relu(hidden)
    network_output = hidden.reshape(triplets, channels, height, width)
    blend = parameters.blend[selected_series][:, None, None, None]
    blended = blend * network_output + (1 - blend) * values
    series_count = len(parameters.blend)
    valid_before = (values * mask).square().sum(dim=(1, 2, 3))
    valid_after = (blended * mask).square().sum(dim=(1, 2, 3))
    sum_before = values.new_zeros(series_count).index_add_(
        0, selected_series, valid_before
    )
    sum_after = values.new_zeros(series_count).index_add_(
        0, selected_series, valid_after
    )
    scale = (sum_before.clamp_min(1e-12).sqrt() / sum_after.clamp_min(1e-12).sqrt())[
        selected_series
    ]
    return blended * scale[:, None, None, None]


def filter_regions(
    images: torch.Tensor,
    background: torch.Tensor,
    series_index: torch.Tensor,
    parameters: SmoothingParameters | ResolutionParameters | GinParameters,
    rows: torch.Tensor,
    region_mask: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    region_filter: Callable[..., torch.Tensor],
) -> tuple[torch.Tensor, AugmentationActivity]:
    changed_elements = images.new_zeros((), dtype=torch.long)
    mean_change = images.new_zeros(())
    if len(rows):
        selected_series = series_index[rows]
        values = images[rows]
        mask = region_mask(rows, selected_series)
        result = region_filter(values, mask, selected_series, parameters).clamp(0, 1)
        result.masked_fill_(~mask, 0)
        support = mask & ~background[rows]
        changed_elements = support.sum()
        mean_change = ((result - values).abs() * support).sum() / (
            changed_elements.clamp_min(1)
        )
        images[rows] = result
        background[rows] |= ~mask
    return images, AugmentationActivity(
        parameters.applied.sum().detach(),
        len(parameters.applied),
        changed_elements.detach(),
        mean_change.detach(),
    )


class ImageAugmentation(nn.Module):
    def __init__(self, settings: DataConfig) -> None:
        super().__init__()
        self.noise = RandomGaussianNoise(p=1.0)
        self.smoothing_settings = settings.smoothing_augmentation
        self.resolution_settings = settings.resolution_augmentation
        self.gin_settings = settings.gin_augmentation
        self.activity: dict[str, AugmentationActivity] = {}

    def _add_noise(
        self, images: torch.Tensor, rows: torch.Tensor, noise: torch.Tensor | None
    ) -> torch.Tensor:
        if noise is None:
            return images
        values = images[rows]
        images[rows] = self.noise(
            values,
            params={
                "batch_prob": values.new_ones(len(values)),
                "gaussian_noise": noise,
            },
        ).clamp_(0, 1)
        return images

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        series_index: torch.Tensor,
        parameters: AugmentationParameters,
        generator: torch.Generator | None = None,
        matrices: torch.Tensor | None = None,
        valid_rects: torch.Tensor | None = None,
        inverse_affine: torch.Tensor | None = None,
        *,
        source_image_index: torch.Tensor,
        primary_image_index: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training:
            return images
        if (valid_rects is None or inverse_affine is None) and any(
            settings.probability > 0
            for settings in (
                self.smoothing_settings,
                self.resolution_settings,
                self.gin_settings,
            )
        ):
            raise ValueError("Region augmentations require geometric valid rectangles")
        with torch.autocast(device_type=images.device.type, enabled=False):
            images = images.to(torch.float32)
            height, width = images.shape[-2:]
            series_count = len(parameters.affine.applied)
            device = images.device
            if matrices is None:
                matrices = affine_matrices(parameters.affine, height, width)

            def prepared_mask(
                rows: torch.Tensor, selected_series: torch.Tensor
            ) -> torch.Tensor:
                return geometric_valid_mask(
                    valid_rects[rows], inverse_affine[rows], height, width
                )

            def warped_mask(
                rows: torch.Tensor, selected_series: torch.Tensor
            ) -> torch.Tensor:
                return warped_valid_mask(
                    valid_rects[rows], matrices[selected_series], height, width
                )

            primary_count = len(primary_image_index)
            affine_selected = parameters.affine.applied[series_index]
            intensity_selected = parameters.intensity.applied[series_index]
            noise_selected = parameters.noise.applied[series_index[:primary_count]]
            affine_count, intensity_count, noise_count = torch.stack(
                (affine_selected.sum(), intensity_selected.sum(), noise_selected.sum())
            ).tolist()
            images = apply_affine_augmentation(
                images,
                series_index,
                matrices,
                selected_rows(affine_selected, affine_count),
            )
            background = images == 0
            noise_rows = selected_rows(noise_selected, noise_count)
            noise = sample_noise_values(
                images[:primary_count],
                series_index[:primary_count],
                parameters.noise,
                noise_rows,
                generator,
            )
            smoothing = sample_smoothing_parameters(
                self.smoothing_settings, series_count, device, generator
            )
            resolution = sample_resolution_parameters(
                self.resolution_settings, series_count, device, generator
            )
            gin = sample_gin_parameters(
                self.gin_settings, series_count, device, generator
            )
            smoothing_selected = smoothing.applied[series_index]
            resolution_selected = resolution.applied[series_index]
            gin_selected = gin.applied[series_index]
            resolution_columns = resolution.columns[series_index]
            (
                smoothing_count,
                resolution_count,
                gin_count,
                rows_axis_count,
                columns_axis_count,
            ) = torch.stack(
                (
                    smoothing_selected.sum(),
                    resolution_selected.sum(),
                    gin_selected.sum(),
                    (resolution_selected & ~resolution_columns).sum(),
                    (resolution_selected & resolution_columns).sum(),
                )
            ).tolist()
            images, smoothing_activity = filter_regions(
                images,
                background,
                series_index,
                smoothing,
                selected_rows(smoothing_selected, smoothing_count),
                prepared_mask,
                apply_bilateral_smoothing,
            )
            resolution_rows = selected_rows(resolution_selected, resolution_count)
            resolution_axis = resolution_columns[resolution_rows]
            images, resolution_activity = filter_regions(
                images,
                background,
                series_index,
                resolution,
                resolution_rows,
                warped_mask,
                partial(
                    apply_resolution_degradation,
                    axis_rows=(
                        selected_rows(~resolution_axis, rows_axis_count),
                        selected_rows(resolution_axis, columns_axis_count),
                    ),
                ),
            )
            images = apply_intensity_augmentation(
                images,
                series_index,
                parameters.intensity,
                selected_rows(intensity_selected, intensity_count),
            )
            images, gin_activity = filter_regions(
                images,
                background,
                series_index,
                gin,
                selected_rows(gin_selected, gin_count),
                prepared_mask,
                apply_gin_filter,
            )
            original = images[:primary_count]
            noisy = self._add_noise(original.clone(), noise_rows, noise)
            delta = (noisy - original).index_select(0, source_image_index)
            primary = source_image_index == torch.arange(len(images), device=device)
            images = torch.where(
                primary[:, None, None, None],
                noisy.index_select(0, source_image_index),
                (images + delta).clamp(0, 1),
            )
            self.activity = {
                "smoothing": smoothing_activity,
                "resolution": resolution_activity,
                "gin": gin_activity,
            }
            return images.masked_fill_(background, 0)


def apply_feature_statistics_augmentation(
    features: torch.Tensor,
    acquisition_index: torch.Tensor,
    acquisition_type: torch.Tensor,
    settings: FeatureStatisticsAugmentationConfig,
    primary_image: torch.Tensor,
    primary_count: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, FeatureStatisticsActivity]:
    dtype = features.dtype
    with torch.autocast(device_type=features.device.type, enabled=False):
        values = features.float()
        _, group = torch.unique(acquisition_index, return_inverse=True)
        group_count = int(group.max()) + 1
        channel_count = values.shape[1]
        element_count = values.shape[2] * values.shape[3]
        channel_sum = values.sum(dim=(2, 3))
        channel_square_sum = values.square().sum(dim=(2, 3))
        acquisition_triplets = values.new_zeros(group_count).index_add_(
            0, group, values.new_ones(len(group))
        )
        acquisition_elements = acquisition_triplets * element_count
        acquisition_mean = (
            values.new_zeros(group_count, channel_count).index_add_(
                0, group, channel_sum
            )
            / acquisition_elements[:, None]
        )
        acquisition_square_mean = (
            values.new_zeros(group_count, channel_count).index_add_(
                0, group, channel_square_sum
            )
            / acquisition_elements[:, None]
        )
        acquisition_std = (
            (acquisition_square_mean - acquisition_mean.square()).clamp_min(1e-8).sqrt()
        )

        acquisition_type_value = torch.zeros(
            group_count, dtype=acquisition_type.dtype, device=values.device
        ).scatter_(0, group, acquisition_type)
        primary_group = torch.zeros(
            group_count, dtype=torch.bool, device=values.device
        ).scatter_(0, group, primary_image)
        primary_weight = primary_group.float()
        _, type_group = torch.unique(acquisition_type_value, return_inverse=True)
        type_count = int(type_group.max()) + 1
        type_totals = (
            values.new_zeros(type_count)
            .index_add_(0, type_group, primary_weight)
            .clamp_min(1)
        )
        type_mean_of_mean = (
            values.new_zeros(type_count, channel_count).index_add_(
                0, type_group, acquisition_mean * primary_weight[:, None]
            )
            / type_totals[:, None]
        )
        type_mean_of_std = (
            values.new_zeros(type_count, channel_count).index_add_(
                0, type_group, acquisition_std * primary_weight[:, None]
            )
            / type_totals[:, None]
        )
        spread_mean = (
            values.new_zeros(type_count, channel_count)
            .index_add_(
                0,
                type_group,
                (acquisition_mean - type_mean_of_mean[type_group]).square()
                * primary_weight[:, None],
            )
            .div_(type_totals[:, None])
            .clamp_min_(1e-12)
            .sqrt_()[type_group]
        )
        spread_std = (
            values.new_zeros(type_count, channel_count)
            .index_add_(
                0,
                type_group,
                (acquisition_std - type_mean_of_std[type_group]).square()
                * primary_weight[:, None],
            )
            .div_(type_totals[:, None])
            .clamp_min_(1e-12)
            .sqrt_()[type_group]
        )
        eligible = (
            primary_group
            & (type_totals[type_group] >= 2)
            & (
                nn.functional.pad(
                    torch.rand(
                        primary_count, device=values.device, generator=generator
                    ),
                    (0, group_count - primary_count),
                    value=1.0,
                )
                < settings.probability
            )
        )
        resampled_mean = (
            acquisition_mean
            + nn.functional.pad(
                torch.randn(
                    primary_count,
                    channel_count,
                    device=values.device,
                    generator=generator,
                ),
                (0, 0, 0, group_count - primary_count),
            )
            * spread_mean
        )
        resampled_std = (
            acquisition_std
            + nn.functional.pad(
                torch.randn(
                    primary_count,
                    channel_count,
                    device=values.device,
                    generator=generator,
                ),
                (0, 0, 0, group_count - primary_count),
            )
            * spread_std
        )
        clamped_channels = resampled_std.detach() < 0
        resampled_std.clamp_min_(0)

        per_triplet_apply = eligible[group][:, None, None, None]
        per_triplet_mean = acquisition_mean[group][:, :, None, None]
        per_triplet_std = acquisition_std[group][:, :, None, None]
        per_triplet_resampled_mean = resampled_mean[group][:, :, None, None]
        per_triplet_resampled_std = resampled_std[group][:, :, None, None]
        restyled = (
            (values - per_triplet_mean) / per_triplet_std * per_triplet_resampled_std
        )
        restyled = restyled + per_triplet_resampled_mean
        result = torch.where(per_triplet_apply, restyled, values)

        changed_elements = per_triplet_apply.expand_as(values).sum()
        mean_change = (result - values).abs().sum() / changed_elements.clamp_min(1)
    return result.to(dtype), FeatureStatisticsActivity(
        eligible.sum().detach(),
        group_count,
        changed_elements.detach(),
        mean_change.detach(),
        clamped_channels.sum(),
        (clamped_channels & eligible[:, None]).sum(),
        resampled_std.numel(),
    )
