# Field-of-view selection history

On 2026-09-10, the user authorized choosing a larger spacing for 224-pixel
inputs when 0.38 mm would crop useful information. The selected initial spacing
is **0.80 × 0.80 mm**, giving a **179.2 × 179.2 mm** centered field of view.
At 0.38 mm the same pixel dimensions cover only 85.12 mm per side. Increasing
spacing preserves more context at a lower spatial resolution; images smaller
than the target field are padded after resampling.

Baseline3 later selected **0.67 × 0.67 mm**, covering
**150.08 × 150.08 mm** at 224 pixels. The earlier
0.80 mm previews remain useful evidence about the cost of an 85.12 mm crop,
but they do not validate 0.67 mm as optimal.

The completed [140-mm crop comparison](../validation/results/crops-and-views/2026-09-12-140-mm-crop-result.md)
subsequently selected **0.625 × 0.625 mm** at 224 pixels (140 mm), the default until 2026-10-01, when the 120 mm field (0.5357 mm) and later per-type views replaced it (see [current preparation](current-preparation.md)).
It improved local and public AUC over 150.08 mm. The earlier proposal to expand
from 0.67 to 0.72 mm (161.28 mm at 224 pixels) is retired; 0.72 mm itself was
not evaluated, and the crop result does not isolate coverage from pixel spacing.

A larger header survey recorded on 2026-09-11 sampled 768 series, 128 per
plane–contrast type, with Polars seed 42 after sorting the count inventory by
study and series UID. It read the lexicographically first DICOM header for each
selected series; all 768 supplied valid geometry. Extents used rows × row spacing
and columns × column spacing. Median acquired fields were approximately
160 × 160 mm in all six groups, which did not support giving sagittal images a
larger crop than other planes. Native extent alone does not establish useful
anatomy coverage. This survey was originally recorded in the future-work notes
without a linked artifact identity; its summary does not establish an optimal crop.

A local header sample used eight shortest-selected acquisitions per type,
sampled with seed 42 after stable sorting. Across these 48 series, per-type
median image heights were about 155–175 mm, with sampled heights ranging from
136.8 to 200.0 mm. One sampled coronal image was 135 mm wide. Source image
extent alone does not establish which parts contain useful anatomy.

Six visual previews used the middle physical slice from the largest sampled
image extent in each type. Overlaid 85.12 mm and 179.2 mm centered windows show
that the former excludes substantial surrounding image content, while the
latter retains much more of the acquired field in these examples. This is a
limited geometry inspection, not an exhaustive anatomy audit or validation of
either spacing choice.

Local artifacts:

- `outputs/archive/planning/spacing-audit.json`: source hash, versions, sample rule,
  all 48 headers' dimensions/spacing and the six preview paths.
- `outputs/archive/planning/crop-field-of-view.png`: native-image comparison with
  both crop boundaries. Its display uses the 0.5th–99.5th percentiles only for
  visualization. Training now uses the same percentile levels jointly over
  selected images per series, as specified in the [implemented input recipe](../modeling/triplet-inputs-and-sampling.md#adjacent-triplet-sampling).

The input is the [saved image-count table](../metadata/acquisition-and-series-profiles.md#images-per-series),
SHA-256 `75c5f7b77f69d6c522529d42cb5e3dddfa0eff2120f501a6b298d2a2a6a6ce55`.
The acquisition journal supplies paths without a corpus directory scan.

On 2026-09-10, the implemented deterministic path prepared example-test study
`1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191` in 0.71
seconds on the development host. Its four supported series contributed 126
slices, all decoded successfully, and the transient batch spanned the full
uint8 range. Visual inspection of the four middle prepared slices showed
centered axial, coronal and sagittal knee anatomy with useful surrounding
context. This checks one study and does not replace broader training-data review.

A second review prepared one shortest acquisition from each of the six training
types in 0.60 seconds, covering 114 selected slices without decode failures or
constant intensity ranges. Their middle slices retained centered knee anatomy
in every plane and both contrast groups. Both studies containing the confirmed
truncated payloads also completed: each bad header was logged and excluded,
while five usable series and 151 selected slices remained for each study.
These executions verify the known failure path and representative types, while
loader concurrency and rare unobserved failures still require runtime monitoring.
