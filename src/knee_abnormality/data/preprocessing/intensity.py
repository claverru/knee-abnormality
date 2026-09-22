import numpy as np


def quantile_ranks(
    count: int, lower_quantile: float, upper_quantile: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quantiles = np.asarray([lower_quantile, upper_quantile], dtype=np.float32)
    ranks = quantiles * np.float32(max(count, 1) - 1)
    lower = np.floor(ranks).astype(np.int64)
    upper = np.ceil(ranks).astype(np.int64)
    return lower, upper, (ranks - lower).astype(np.float32)


def quantile_bounds(
    values: np.ndarray, lower: np.ndarray, upper: np.ndarray, fraction: np.ndarray
) -> tuple[np.float32, np.float32]:
    l0, h0 = int(lower[0]), int(lower[1])
    l1, h1 = int(upper[0]), int(upper[1])
    if l1 < h0:
        values.partition(h0)
        h_lo = values[h0]
        h_hi = values[h0 + 1 :].min() if h1 == h0 + 1 else h_lo
        prefix = values[:h0]
        prefix.partition(l1)
        l_hi = prefix[l1]
        l_lo = prefix[:l1].max() if l0 == l1 - 1 else l_hi
        lower_values = np.asarray([l_lo, h_lo], dtype=np.float32)
        upper_values = np.asarray([l_hi, h_hi], dtype=np.float32)
    else:
        values.partition(np.unique(np.concatenate((lower, upper))))
        lower_values = values[lower]
        upper_values = values[upper]
    bounds = lower_values + (upper_values - lower_values) * fraction
    return bounds[0], bounds[1]


def normalize_pixels(
    pixels: np.ndarray, rects: np.ndarray, low: np.float32, high: np.float32
) -> np.ndarray:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(pixels.shape, dtype=np.uint8)
    gain = (np.float32(1.0) / (high - low)) * np.float32(255.0)
    np.clip(pixels, low, high, out=pixels)
    np.subtract(pixels, low, out=pixels)
    np.multiply(pixels, gain, out=pixels)
    np.rint(pixels, out=pixels)
    output = pixels.astype(np.uint8)
    for index, (top, left, rows, columns) in enumerate(rects):
        output[index, :top] = 0
        output[index, top + rows :] = 0
        output[index, top : top + rows, :left] = 0
        output[index, top : top + rows, left + columns :] = 0
    return output
