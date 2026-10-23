# Competitor-reported preprocessing notes (unverified)

What other competitors and the early working notes reported about preprocessing, kept as unverified leads. None of it describes this project's pipeline, which is in [current preparation](current-preparation.md).

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

## Ordering and the questions these notes left open

This project's pipeline sorts slices by the DICOM orientation/location projection, falling back to
`SliceLocation` and then index metadata (`data/preprocessing/dicom.py`). It uses float32 millimeter
positions relative to the midpoint of the acquisition's physical extent, fixed when different slices
are sampled; the former `data.position_encoding` and `data.position_reference` fields are gone, and
physical mode rejects candidates whose sampled slices only have index metadata, using the
alternative-acquisition fallback. Baseline1's ordinal (`index`) positions need its archived sources
([correction record](../history/baseline1-review.md#implemented-corrections)).

Questions the reported pipelines left open, answered for this project where the source shows it:

- Ordering by patient orientation and coordinates: implemented as above; handling of inconsistent
  geometry is the instance guard and study cleaning in [current preparation](current-preparation.md).
- Rescale slope/intercept: applied when not 1 and 0 (`data/preprocessing/device.py`).
- Photometric inversion: no handling exists in `data/preprocessing/` (no `PhotometricInterpretation`
  logic). Whether any competition study needed it was not established here.
- The reported pipelines' exact `uint8` mapping, padding, interpolation, series classification and
  duplicate criteria remain unknown; this project's are quantile bounds of 0.005 and 0.995 on valid
  crop pixels, `uint8` normalization, scale-aware antialiased resampling, and the cleaning rules in
  [current preparation](current-preparation.md).
