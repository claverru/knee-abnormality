# Next changes: sampling and physical coverage

Planned from the 2026-09-11 discussion. This document records the proposed
implementation and comparisons; neither change has been implemented.
Continue iterating on fold 0. Label-supervision experiments remain in
[future work](docs/future-work.md); pooling and augmentation changes are excluded.

## 1. Replace triangular sampling weights with Gaussian weights

Use a Gaussian over the physical positions of triplet centers. Preserve the
current nominal **3:1 center-to-edge weight ratio** for the first comparison,
so the experiment changes the curve shape rather than its endpoint ratio.

For normalized position `u` spanning `[-1, 1]` from the first to last slice:

```text
u = (triplet_center_position - series_midpoint) / series_half_span
weight(u) = exp(log(3) * (1 - u²))
```

This is proportional to a Gaussian with standard deviation about 0.675 of the
half-span, or 0.337 of the full series extent. Weights remain positive at the
edges. These are selection weights, not exact inclusion probabilities after
the non-overlap constraints.

Keep complete, non-overlapping triplets; training selection remains random and
seeded, while validation/submission selection remains deterministic. Keep slice
budgets and pooled acquisition allocation fixed. Replace the triangular code
and update affected callers; do not add a triangle/Gaussian selector or a legacy
compatibility path.

### How often the budget requires subsampling

Read-only counts across all 24,371 training series, covering all folds, from
`data/metadata/train_series_image_counts.parquet`, with the two known bad files
subtracted. Each percentage is the share of individual series whose available
file count exceeds that plane–contrast type's nominal budget.

| Plane | Fluid-sensitive / fat suppression | 1× raw-slice budget | Above 1× | Above 2× | Above 3× |
|---|---|---:|---:|---:|---:|
| Axial | 0 / 0 | 18 | 99.6% | 36.6% | 7.6% |
| Axial | 1 / 1 | 18 | 96.5% | 26.5% | 6.7% |
| Coronal | 0 / 0 | 15 | 97.4% | 29.1% | 0.1% |
| Coronal | 1 / 1 | 15 | 99.7% | 33.9% | 0.2% |
| Sagittal | 0 / 0 | 15 | 98.2% | 40.4% | 7.1% |
| Sagittal | 1 / 1 | 15 | 99.8% | 34.6% | 5.3% |
| Overall | | | 98.4% | 33.3% | 4.3% |

The actual budget is shared across acquisitions within a study/plane/contrast.
An individual series below the nominal cap can therefore still be subsampled.
For example, at 2×, 36.6% of individual axial 0/0 series exceed 36 slices, while
61.7% of corresponding study/type groups exceed 36 complete-triplet slices in
total. The table is not an exact count of series affected after allocation or
of predictions changed by the new weighting. File counts do not replace a
complete usable-pixel audit. See [metadata provenance](docs/metadata.md#images-per-series).

## 2. Increase shared in-plane spacing from 0.67 to 0.72 mm

Keep 224 × 224 inputs and use **0.72 mm in every plane**, giving a centered
field of view of **161.28 × 161.28 mm**, compared with 150.08 × 150.08 mm at
0.67 mm. This changes physical coverage while keeping the CNN input dimensions
and slice budget fixed. The tradeoff is slightly less spatial detail per mm.

Do not introduce separate spacing values or switches for individual planes.
A read-only sample of 768 series, 128 per plane–contrast combination, found a
median acquired field of view of approximately 160 × 160 mm in all six groups.
That larger sample did not support the earlier 48-header sample's suggestion
that sagittal acquisitions generally need a larger field of view.

Sampling method: sort the count inventory by study and series UID, sample 128
series within each type using Polars seed 42, and read the lexicographically
first DICOM header per selected series. All 768 headers supplied valid geometry.
Extent is rows × row spacing and columns × column spacing. Native image extent
does not establish useful anatomy coverage; inspect crop boundaries before
interpreting an AUC change as evidence that anatomy was previously excluded.

## Comparison sequence

1. Compare Gaussian sampling at the existing 0.67 mm spacing against the
   recorded triangular baseline.
2. Compare 0.67 versus 0.72 mm while holding the selected sampling recipe fixed.

Keep fold-0 membership, seed 42, ReXNet-200, training budget, labels, slice
limits and validation image multiplier fixed except for the factor under
comparison. Retain full 903-study validation predictions, macro/per-target AUC,
gold diagnostics, runtime and peak VRAM. Keep validation and test inference at
2× image limits, with inference batch size 2, following the
[completed density comparison](docs/inference-efficiency.md#inference-image-count-and-memory).
Do not mix a change in inference image multiplier into either comparison.
Gains from Gaussian sampling and spacing remain unverified until measured;
no additional folds, ensemble, or routine submission parity pass is required.
