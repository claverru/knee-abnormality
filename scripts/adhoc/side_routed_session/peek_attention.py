import numpy as np

data = np.load("outputs/analysis/side-routing-attention-shape-20261006/attention.npz", allow_pickle=True)
for key in data.files:
    value = data[key]
    print(key, value.shape, value.dtype, value[:3] if value.ndim == 1 and value.shape[0] < 60 else "")
