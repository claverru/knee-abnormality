import numpy as np

old = np.load("outputs/analysis/s/side-routing-old-vs-reference-20261006/old.npz", allow_pickle=True)
new = np.load("outputs/analysis/s/side-routing-old-vs-reference-20261006/reference.npz", allow_pickle=True)
print("knees", len(old["study"]), len(new["study"]), "same token counts", int((old["tokens"] == new["tokens"]).sum()), "same valid counts", int((old["valid_count"] == new["valid_count"]).sum()))
print("image sums equal", int(np.isclose(old["image_sum"], new["image_sum"], rtol=1e-6).sum()))
corr_a, corr_b = [], []
for k in range(0, len(old["study"]), 10):
    if old["tokens"][k] != new["tokens"][k]:
        continue
    a, b = old["axis_a"][k], new["axis_a"][k]
    keep = np.isfinite(a) & np.isfinite(b) & (np.abs(a) + np.abs(b) > 0)
    if keep.sum() > 10:
        corr_a.append(np.corrcoef(a[keep], b[keep])[0, 1])
        corr_b.append(np.corrcoef(old["axis_b"][k][keep], new["axis_b"][k][keep])[0, 1])
print("medial axis correlation: median", np.median(corr_a).round(3), "min", np.min(corr_a).round(3), " anterior axis: median", np.median(corr_b).round(3), "min", np.min(corr_b).round(3))
