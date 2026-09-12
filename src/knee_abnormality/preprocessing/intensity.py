import numpy as np


def _linear_quantiles(
    values: np.ndarray, lower_quantile: float, upper_quantile: float
) -> tuple[float, float]:
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


def _normalize_pixels(
    pixels: np.ndarray, rects: np.ndarray, low: float, high: float
) -> np.ndarray:
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
