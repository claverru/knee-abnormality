from collections import defaultdict

import numpy as np
import torch

from knee_abnormality.data.preprocessing.intensity import (
    normalize_pixels,
    quantile_bounds,
)
from knee_abnormality.data.preprocessing.preparation import shared_empty
from knee_abnormality.data.preprocessing.spatial import resample_block, resample_plan

RAW_KEYS = (
    "frames",
    "frame_buffers",
    "series_frames",
    "image_positions",
    "image_sources",
)

_PLANS: dict[tuple, tuple[torch.Tensor, ...]] = {}


def _device_plan(key: tuple, device: torch.device) -> tuple[torch.Tensor, ...]:
    cached = _PLANS.get((key, device))
    if cached is None:
        plan = resample_plan(*key)
        cached = tuple(
            torch.from_numpy(values).to(device)
            for values in (
                plan.vertical_index,
                plan.vertical_weight,
                plan.horizontal_index,
                plan.horizontal_weight,
            )
        )
        _PLANS[(key, device)] = cached
    return cached


def _resample(blocks: torch.Tensor, plan: tuple[torch.Tensor, ...]) -> torch.Tensor:
    vertical_index, vertical_weight, horizontal_index, horizontal_weight = plan
    middle = blocks.new_zeros(
        (blocks.shape[0], vertical_index.shape[0], blocks.shape[2])
    )
    for tap in range(vertical_index.shape[1]):
        middle = middle + vertical_weight[:, tap, None] * blocks.index_select(
            1, vertical_index[:, tap]
        )
    region = blocks.new_zeros(
        (blocks.shape[0], middle.shape[1], horizontal_index.shape[0])
    )
    for tap in range(horizontal_index.shape[1]):
        region = region + horizontal_weight[:, tap] * middle.index_select(
            2, horizontal_index[:, tap]
        )
    return region


def _native_blocks(
    buffers: list[torch.Tensor],
    members: list[tuple[int, int, int]],
    height: int,
    width: int,
    columns: tuple[int, int],
    signed: bool,
) -> torch.Tensor:
    size = height * width
    blocks = torch.stack(
        [buffers[buffer][offset : offset + size] for _, buffer, offset in members]
    ).view(len(members), height, width)[:, :, columns[0] : columns[1]]
    if signed:
        return blocks.to(torch.float32)
    return (blocks.to(torch.int32) & 0xFFFF).to(torch.float32)


def _region_values(canvas: torch.Tensor, frames: list, start: int, count: int):
    rects = {frames[index][10] for index in range(start, start + count)}
    if len(rects) == 1:
        top, left, rows, columns = next(iter(rects))
        return canvas[
            start : start + count, top : top + rows, left : left + columns
        ].reshape(-1)
    return torch.cat(
        [
            canvas[index, top : top + rows, left : left + columns].reshape(-1)
            for index in range(start, start + count)
            for top, left, rows, columns in (frames[index][10],)
        ]
    )


def _region(frame: tuple, buffers: list[torch.Tensor]) -> np.ndarray:
    native, signed, buffer, offset, height, width, first, last = frame[:8]
    slope, intercept, (_, _, rows, columns), plan_key = frame[8:]
    values = buffers[buffer].numpy()
    if not native:
        return values[offset : offset + rows * columns].reshape(rows, columns)
    block = values[offset : offset + height * width]
    block = block.view(np.int16 if signed else np.uint16).reshape(height, width)
    region = resample_block(block[:, first:last], resample_plan(*plan_key))
    if slope != 1.0 or intercept != 0.0:
        region *= np.float32(slope)
        region += np.float32(intercept)
    return region


def _prepare_on_cpu(batch: dict) -> dict:
    buffers = batch["frame_buffers"]
    frames = batch["frames"]
    size = batch["image_size"]
    normalized = np.zeros((len(frames), size, size), dtype=np.uint8)
    constant = torch.zeros(len(batch["series_frames"]), dtype=torch.bool)
    for row, (start, count, (lower, upper, fraction)) in enumerate(
        batch["series_frames"]
    ):
        rects = np.asarray([frame[10] for frame in frames[start : start + count]])
        canvas = np.zeros((count, size, size), dtype=np.float32)
        for index, (top, left, rows, columns) in enumerate(rects):
            canvas[index, top : top + rows, left : left + columns] = _region(
                frames[start + index], buffers
            )
        low, high = quantile_bounds(
            np.concatenate(
                [
                    canvas[index, top : top + rows, left : left + columns].reshape(-1)
                    for index, (top, left, rows, columns) in enumerate(rects)
                ]
            ),
            np.asarray(lower),
            np.asarray(upper),
            np.asarray(fraction, dtype=np.float32),
        )
        constant[row] = bool(
            not np.isfinite(low) or not np.isfinite(high) or high <= low
        )
        normalized[start : start + count] = normalize_pixels(canvas, rects, low, high)
    slice_mask = batch["slice_mask"]
    images = shared_empty(slice_mask.numel() * 3 * size * size, torch.uint8)
    images = images.view(*slice_mask.shape, 3, size, size)
    images.view(-1, size, size)[batch["image_positions"]] = torch.from_numpy(
        normalized
    )[batch["image_sources"]]
    return {
        **{key: value for key, value in batch.items() if key not in RAW_KEYS},
        "images": images,
        "slice_mask": slice_mask & ~constant[:, None],
        "constant_series": constant,
    }


def prepare_images(batch: dict) -> dict:
    if batch["slice_mask"].device.type == "cpu":
        return _prepare_on_cpu(batch)
    buffers = batch["frame_buffers"]
    device = batch["slice_mask"].device
    frames = batch["frames"]
    size = batch["image_size"]
    canvas = torch.zeros((len(frames), size, size), dtype=torch.float32, device=device)
    rescale = torch.tensor(
        [(frame[8], frame[9]) for frame in frames], dtype=torch.float32
    ).to(device, non_blocking=True)
    groups = defaultdict(list)
    for index, frame in enumerate(frames):
        native, signed, buffer, offset, height, width, first, last = frame[:8]
        rect, plan_key = frame[10], frame[11]
        if native:
            groups[(signed, plan_key, height, width, (first, last), rect)].append(
                (index, buffer, offset)
            )
        else:
            top, left, rows, columns = rect
            canvas[index, top : top + rows, left : left + columns] = buffers[buffer][
                offset : offset + rows * columns
            ].view(rows, columns)
    for (signed, plan_key, height, width, columns, rect), members in groups.items():
        indices = torch.tensor([index for index, _, _ in members]).to(
            device, non_blocking=True
        )
        region = _resample(
            _native_blocks(buffers, members, height, width, columns, signed),
            _device_plan(plan_key, device),
        )
        factors = rescale[indices]
        region = region * factors[:, 0, None, None]
        region = region + factors[:, 1, None, None]
        top, left, rows, columns = rect
        canvas[indices, top : top + rows, left : left + columns] = region
    series = batch["series_frames"]
    values = [
        _region_values(canvas, frames, start, count) for start, count, _ in series
    ]
    padded = torch.full(
        (len(values), max(1, *(len(value) for value in values))),
        torch.inf,
        dtype=torch.float32,
        device=device,
    )
    for row, value in enumerate(values):
        padded[row, : len(value)] = value
    ordered = padded.sort(dim=1).values
    ranks = torch.tensor(
        [(*lower, *upper) for _, _, (lower, upper, _) in series], dtype=torch.int64
    ).to(device, non_blocking=True)
    fraction = torch.tensor(
        [tuple(fraction) for _, _, (_, _, fraction) in series], dtype=torch.float32
    ).to(device, non_blocking=True)
    statistics = ordered.gather(1, ranks)
    lower_values = statistics[:, :2]
    bounds = lower_values + (statistics[:, 2:] - lower_values) * fraction
    low, high = bounds[:, 0], bounds[:, 1]
    constant = ~torch.isfinite(low) | ~torch.isfinite(high) | (high <= low)
    gain = torch.reciprocal(high - low) * 255.0
    owner = torch.tensor(
        [row for row, (_, count, _) in enumerate(series) for _ in range(count)]
    ).to(device, non_blocking=True)
    frame_low = low[owner][:, None, None]
    normalized = torch.minimum(
        torch.maximum(canvas, frame_low), high[owner][:, None, None]
    )
    normalized = normalized - frame_low
    normalized = normalized * gain[owner][:, None, None]
    normalized = torch.round(normalized).to(torch.uint8)
    rects = torch.tensor([frame[10] for frame in frames]).to(device, non_blocking=True)
    lines = torch.arange(size, device=device)
    tops, lefts = rects[:, 0, None], rects[:, 1, None]
    inside_rows = (lines >= tops) & (lines < tops + rects[:, 2, None])
    inside_columns = (lines >= lefts) & (lines < lefts + rects[:, 3, None])
    normalized = normalized * (inside_rows[:, :, None] & inside_columns[:, None, :])
    normalized = normalized * ~constant[owner][:, None, None]
    slice_mask = batch["slice_mask"]
    images = torch.zeros(
        (*slice_mask.shape, 3, size, size), dtype=torch.uint8, device=device
    )
    images.view(-1, size, size)[batch["image_positions"]] = normalized[
        batch["image_sources"]
    ]
    return {
        **{key: value for key, value in batch.items() if key not in RAW_KEYS},
        "images": images,
        "slice_mask": slice_mask & ~constant[:, None],
        "constant_series": constant,
    }


def with_images(batch: dict) -> dict:
    return batch if "images" in batch else prepare_images(batch)
