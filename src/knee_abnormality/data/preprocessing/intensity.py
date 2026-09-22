import numpy as np


def quantile_ranks(
    count: int, lower_quantile: float, upper_quantile: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quantiles = np.asarray([lower_quantile, upper_quantile], dtype=np.float32)
    ranks = quantiles * np.float32(max(count, 1) - 1)
    lower = np.floor(ranks).astype(np.int64)
    upper = np.ceil(ranks).astype(np.int64)
    return lower, upper, (ranks - lower).astype(np.float32)
