from pathlib import Path

import numpy as np

LOWER_QUANTILE = 0.005
UPPER_QUANTILE = 0.995
ACQUISITIONS = 40
SLICES_PER_ACQUISITION = 22
IMAGE_SIZE = 224
REPEATS = 5
HISTOGRAM_BINS = 4096
SPACING_MM = 0.625
REAL_PIXELS = True


def _linear_quantiles(values, lower_quantile, upper_quantile):
    quantiles = np.asarray([lower_quantile, upper_quantile], dtype=np.float32)
    ranks = quantiles * np.float32(values.size - 1)
    lower = np.floor(ranks).astype(np.int64)
    upper = np.ceil(ranks).astype(np.int64)
    values.partition(np.unique(np.concatenate((lower, upper))))
    lower_values = values[lower]
    upper_values = values[upper]
    bounds = lower_values + (upper_values - lower_values) * (ranks - lower).astype(
        np.float32
    )
    return float(bounds[0]), float(bounds[1])


def _histogram_quantiles(values, lower_quantile, upper_quantile):
    quantiles = np.asarray([lower_quantile, upper_quantile], dtype=np.float32)
    ranks = quantiles * np.float32(values.size - 1)
    lower = np.floor(ranks).astype(np.int64)
    upper = np.ceil(ranks).astype(np.int64)
    wanted = np.unique(np.concatenate((lower, upper)))
    low = float(values.min())
    high = float(values.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return _linear_quantiles(values, lower_quantile, upper_quantile)
    edges = np.linspace(low, high, HISTOGRAM_BINS + 1, dtype=np.float64)
    counts = np.histogram(values, bins=edges)[0]
    cumulative = np.cumsum(counts)
    selected = np.empty(wanted.size, dtype=np.float32)
    for position, rank in enumerate(wanted):
        index = int(np.searchsorted(cumulative, rank, side="right"))
        index = min(index, HISTOGRAM_BINS - 1)
        start = 0 if index == 0 else int(cumulative[index - 1])
        left = np.float32(edges[index])
        right = np.float32(edges[index + 1])
        if index == HISTOGRAM_BINS - 1:
            inside = values[values >= left]
        else:
            inside = values[(values >= left) & (values < right)]
        inside.partition(int(rank) - start)
        selected[position] = inside[int(rank) - start]
    table = {int(rank): selected[position] for position, rank in enumerate(wanted)}
    lower_values = np.asarray([table[int(value)] for value in lower], dtype=np.float32)
    upper_values = np.asarray([table[int(value)] for value in upper], dtype=np.float32)
    bounds = lower_values + (upper_values - lower_values) * (ranks - lower).astype(
        np.float32
    )
    return float(bounds[0]), float(bounds[1])


def _normalize_pixels(pixels, rects, low, high):
    low_value = np.float32(low)
    high_value = np.float32(high)
    if (
        not np.isfinite(low_value)
        or not np.isfinite(high_value)
        or high_value <= low_value
    ):
        return np.zeros(pixels.shape, dtype=np.uint8)
    gain = (np.float32(1.0) / (high_value - low_value)) * np.float32(255.0)
    np.clip(pixels, low_value, high_value, out=pixels)
    np.subtract(pixels, low_value, out=pixels)
    np.multiply(pixels, gain, out=pixels)
    np.rint(pixels, out=pixels)
    np.clip(pixels, np.float32(0.0), np.float32(255.0), out=pixels)
    output = pixels.astype(np.uint8)
    for index, (top, left, rows, columns) in enumerate(rects):
        output[index, :top] = 0
        output[index, top + rows :] = 0
        output[index, top : top + rows, :left] = 0
        output[index, top : top + rows, left + columns :] = 0
    return output


def _fused_normalize(pixels, rects, low, high):
    low_value = np.float32(low)
    high_value = np.float32(high)
    if (
        not np.isfinite(low_value)
        or not np.isfinite(high_value)
        or high_value <= low_value
    ):
        return np.zeros(pixels.shape, dtype=np.uint8)
    gain = (np.float32(1.0) / (high_value - low_value)) * np.float32(255.0)
    offset = np.float32(low_value * gain)
    np.clip(pixels, low_value, high_value, out=pixels)
    np.multiply(pixels, gain, out=pixels)
    np.subtract(pixels, offset, out=pixels)
    np.rint(pixels, out=pixels)
    np.clip(pixels, np.float32(0.0), np.float32(255.0), out=pixels)
    output = pixels.astype(np.uint8)
    for index, (top, left, rows, columns) in enumerate(rects):
        output[index, :top] = 0
        output[index, top + rows :] = 0
        output[index, top : top + rows, :left] = 0
        output[index, top : top + rows, left + columns :] = 0
    return output


def make_acquisition(generator):
    slices = SLICES_PER_ACQUISITION
    pixels = np.zeros((slices, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    rects = []
    for index in range(slices):
        rows = int(generator.integers(150, IMAGE_SIZE + 1))
        columns = int(generator.integers(150, IMAGE_SIZE + 1))
        top = (IMAGE_SIZE - rows) // 2
        left = (IMAGE_SIZE - columns) // 2
        body = generator.gamma(2.0, 240.0, size=(rows, columns)).astype(np.float32)
        pixels[index, top : top + rows, left : left + columns] = body
        rects.append((top, left, rows, columns))
    scratch = np.concatenate(
        [
            pixels[index, top : top + rows, left : left + columns].ravel()
            for index, (top, left, rows, columns) in enumerate(rects)
        ]
    )
    return pixels, np.asarray(rects), np.ascontiguousarray(scratch)


def build(count):
    generator = np.random.default_rng(SEED)
    return [make_acquisition(generator) for _ in range(count)]


def _axis_filter(source, target, first, count):
    import math

    from scipy.sparse import csr_matrix

    key = (source, target, first, count)
    if key in _FILTERS:
        return _FILTERS[key]
    scale = source / target
    support = max(1.0, scale)
    rows, columns, values = [], [], []
    for output_index in range(first, first + count):
        center = (output_index + 0.5) * scale - 0.5
        weights = {}
        for index in range(
            math.floor(center - support), math.ceil(center + support) + 1
        ):
            weight = max(0.0, 1.0 - abs(center - index) / support)
            if weight == 0 or index < 0 or index >= source:
                continue
            weights[index] = weight
        total = sum(weights.values())
        for index, weight in weights.items():
            rows.append(output_index - first)
            columns.append(index)
            values.append(weight / total)
    _FILTERS[key] = csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, columns)),
        shape=(count, source),
        dtype=np.float32,
    )
    return _FILTERS[key]


_FILTERS: dict = {}


def _resample(pixels, row_spacing, column_spacing, slope, intercept):
    target_height = max(1, round(pixels.shape[0] * row_spacing / SPACING_MM))
    target_width = max(1, round(pixels.shape[1] * column_spacing / SPACING_MM))
    source_top = max(0, (target_height - IMAGE_SIZE) // 2)
    source_left = max(0, (target_width - IMAGE_SIZE) // 2)
    output_top = max(0, (IMAGE_SIZE - target_height) // 2)
    output_left = max(0, (IMAGE_SIZE - target_width) // 2)
    rows = min(target_height, IMAGE_SIZE)
    columns = min(target_width, IMAGE_SIZE)
    vertical = _axis_filter(pixels.shape[0], target_height, source_top, rows)
    horizontal = _axis_filter(pixels.shape[1], target_width, source_left, columns)
    region = horizontal.dot(vertical.dot(pixels).T).T
    if slope != 1.0 or intercept != 0.0:
        region = region * np.float32(slope) + np.float32(intercept)
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    image[output_top : output_top + rows, output_left : output_left + columns] = region
    return image, (output_top, output_left, rows, columns)


def real_acquisitions(count):
    import os

    import pydicom

    acquisitions = []
    for study in series_directories():
        if len(acquisitions) >= count:
            break
        for series in sorted(os.scandir(study), key=lambda entry: entry.name):
            if not series.is_dir() or len(acquisitions) >= count:
                continue
            paths = sorted(Path(series.path).glob("*.dcm"))[:SLICES_PER_ACQUISITION]
            if len(paths) < SLICES_PER_ACQUISITION:
                continue
            images, rects = [], []
            for path in paths:
                try:
                    dataset = pydicom.dcmread(path)
                    frame = np.asarray(dataset.pixel_array, dtype=np.float32)
                    spacing = dataset.PixelSpacing
                    if frame.ndim != 2 or len(spacing) != 2:
                        continue
                    image, rect = _resample(
                        frame,
                        float(spacing[0]),
                        float(spacing[1]),
                        float(getattr(dataset, "RescaleSlope", 1.0)),
                        float(getattr(dataset, "RescaleIntercept", 0.0)),
                    )
                    images.append(image)
                    rects.append(rect)
                except Exception:
                    continue
            if len(images) < SLICES_PER_ACQUISITION:
                continue
            pixels = np.stack(images)
            scratch = np.ascontiguousarray(
                np.concatenate(
                    [
                        pixels[index, top : top + rows, left : left + columns].ravel()
                        for index, (top, left, rows, columns) in enumerate(rects)
                    ]
                )
            )
            acquisitions.append((pixels, np.asarray(rects), scratch))
    return acquisitions


def measure_quantiles(name, function, acquisitions):
    best = None
    for _ in range(REPEATS):
        scratches = [scratch.copy() for _, _, scratch in acquisitions]
        with Timer() as timer:
            for scratch in scratches:
                function(scratch, LOWER_QUANTILE, UPPER_QUANTILE)
        best = timer.wall if best is None else min(best, timer.wall)
    values = sum(scratch.size for _, _, scratch in acquisitions)
    return {
        "stage": "quantiles",
        "implementation": name,
        "acquisitions": len(acquisitions),
        "million_values": round(values / 1e6, 1),
        "wall_seconds": round(best, 4),
        "milliseconds_per_acquisition": round(1000 * best / len(acquisitions), 3),
    }


def measure_normalize(name, function, acquisitions, bounds):
    best = None
    for _ in range(REPEATS):
        copies = [pixels.copy() for pixels, _, _ in acquisitions]
        with Timer() as timer:
            for copy, (_, rects, _), (low, high) in zip(copies, acquisitions, bounds):
                function(copy, rects, low, high)
        best = timer.wall if best is None else min(best, timer.wall)
    return {
        "stage": "normalize",
        "implementation": name,
        "acquisitions": len(acquisitions),
        "wall_seconds": round(best, 4),
        "milliseconds_per_acquisition": round(1000 * best / len(acquisitions), 3),
    }


def audit(acquisitions):
    exact_bounds = 0
    worst_bound = 0.0
    exact_pixels = 0
    differing = 0
    for pixels, rects, scratch in acquisitions:
        expected = _linear_quantiles(scratch.copy(), LOWER_QUANTILE, UPPER_QUANTILE)
        actual = _histogram_quantiles(scratch.copy(), LOWER_QUANTILE, UPPER_QUANTILE)
        if expected == actual:
            exact_bounds += 1
        worst_bound = max(
            worst_bound, max(abs(expected[0] - actual[0]), abs(expected[1] - actual[1]))
        )
        left = _normalize_pixels(pixels.copy(), rects, *expected)
        right = _fused_normalize(pixels.copy(), rects, *expected)
        if np.array_equal(left, right):
            exact_pixels += 1
        else:
            differing += int(np.count_nonzero(left != right))
    return {
        "acquisitions": len(acquisitions),
        "histogram_bounds_exact": exact_bounds,
        "histogram_worst_bound_difference": worst_bound,
        "fused_normalize_exact": exact_pixels,
        "fused_normalize_differing_pixels": differing,
    }


def main() -> None:
    acquisitions = (
        real_acquisitions(ACQUISITIONS) if REAL_PIXELS else build(ACQUISITIONS)
    )
    bounds = [
        _linear_quantiles(scratch.copy(), LOWER_QUANTILE, UPPER_QUANTILE)
        for _, _, scratch in acquisitions
    ]
    result = {
        "source": "real resampled slices" if REAL_PIXELS else "synthetic",
        "audit": audit(acquisitions),
    }
    result["cells"] = (
        []
        if AUDIT_ONLY
        else [
            measure_quantiles("partition", _linear_quantiles, acquisitions),
            measure_quantiles("histogram", _histogram_quantiles, acquisitions),
            measure_normalize("production", _normalize_pixels, acquisitions, bounds),
            measure_normalize("fused", _fused_normalize, acquisitions, bounds),
        ]
    )
    report("quantiles", result)


main()
