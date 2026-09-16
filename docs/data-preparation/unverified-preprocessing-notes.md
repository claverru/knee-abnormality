# Unverified preprocessing notes and open questions

## Data shape and redundancy

- One inspected study contained 284 DICOM slices across five series, about
  436 MB. This is an example study, not an average.
- Studies contain multiple planes and contrasts, including acquisitions with
  and without fat suppression. Five canonical scan categories were described,
  but their names and mapping rules are not recorded.
- Slice spacing varies by acquisition: about 3 mm was mentioned initially,
  and about 0.5 mm for some axial acquisitions later. Neither is a universal
  spacing for the dataset.
- About 50 series were described as literal duplicates. Separately, about
  100 retakes were reported at matching nominal coordinates, acquired in
  another scan minutes later and not perfectly aligned. These observations
  should not be collapsed into one duplicate count.
- When multiple series map to the same canonical category, the reported
  approach deduplicates and samples from both rather than assuming a fixed
  number of input series.

## Intensity and geometry

| Component | Reported setting | Qualification |
|---|---|---|
| Training cache | `uint8` arrays | Neither PNGs nor repeated DICOM decoding during training; container format unspecified |
| Intensity clipping | Percentiles 0.5 and 99.5 before conversion | Bounds may have been computed per series and applied to every slice; not confirmed from code |
| Physical field of view | 140 mm center crop | Exact resampling/cropping order is unavailable |
| Usual training size | 384 pixels | Described together with physical-scale normalization |
| Target spacing | 0.38 mm/pixel | Reported separately the next day |
| Preparation time | About 30 minutes offline | Hardware and processing scope unspecified |

The settings are approximate descriptions, not a complete executable recipe.
In particular, `140 / 384` is about 0.365 mm/pixel, which does not exactly match
the separately reported 0.38. Confirm whether these reflect different runs,
rounding, or an additional crop/resize step before combining them.

## Resolution tradeoffs

Reported on 2026-09-08, relative to the usual 384-pixel setup:

| Resolution | Reported AUC decrease | Other observation |
|---|---|---|
| 320 | About 0.001 | Suggested as a practical smaller setting |
| 292 | About 0.002 | No separate throughput figure |
| 224 | About 0.003 | About 8× throughput; suggested for fast iteration |

A roughly 440-pixel, 160 mm configuration reportedly performed similarly to
the usual setup. The score's evaluation split, exact model, and benchmark
conditions were not specified, so these are experiment leads rather than
guaranteed tradeoffs.

## Ordering and unanswered preprocessing questions

The maintained pipeline sorts by the DICOM orientation/location projection,
falling back to `SliceLocation`, then index metadata. The current code always uses
float32 millimeter positions relative to the midpoint of the acquisition's
physical extent; the former `data.position_encoding` and `data.position_reference`
fields are gone. This reference remains fixed when different slices are sampled.
Physical mode rejects candidates whose sampled slices only have index metadata,
using the existing alternative-acquisition fallback. Baseline1's ordinal (`index`)
positions need its archived sources. The [correction record](../history/baseline1-review.md#implemented-corrections)
documents compatibility and real-data execution.

The following uncertainties concern the historical pipeline:

- The notes warn against ordering by sequence IDs and indicate use of patient
  orientation/coordinate information. Exact tags, sorting logic, and handling
  of inconsistent geometry are unavailable.
- Whether the pipeline applies rescale slope/intercept is unresolved.
- Photometric inversion was raised during the DICOM review; the actual
  pipeline's handling was not established.
- The exact intensity mapping into `uint8`, padding policy, interpolation,
  series classification, and duplicate-detection criteria are missing.
