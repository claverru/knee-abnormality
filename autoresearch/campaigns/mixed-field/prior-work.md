# Mixed-field prior work

Field, resolution and multi-stream evidence the seeds build on. Sources:
[input resolution](../../../docs/research/geometry-and-views/input-resolution.md),
[crop-selection investigation](../../../docs/history/crop-selection-investigation.md),
[plane-specific backbones](../../../docs/history/plane-backbones.md), the
[RadImageNet ResNet-18 campaign](../radimagenet-rn18/findings.md) and the `revisits` campaign
(branch `revisits`, read on 2026-10-01 while it was running).

## Precedent table

No run has changed the field on the current labels, split or backbone. Deltas are against each
run's own control and are not comparable across rows.

| Mechanism | Setting | Run or source | Macro delta | Notes |
|---|---|---|---:|---|
| Narrower single field | 150 → 140 mm at 224 px, ReXNet-200 | input resolution | +0.003 local, +0.007 public | Also sharpened 0.67 → 0.625 mm; field and resolution confounded |
| Narrower single field | 130 mm at 224 px, ReXNet-200, `all_5fold_v3` | `crop130` | −0.0003 (tie) | Lost on MCL, Baker's cyst, effusion and medial meniscus |
| 0.5 mm crops on part of the slices | 112 mm on 75% of slices | per-label crops | −0.0011 | ACL +0.0111, synovitis +0.0077, medial OA +0.0057; lateral meniscus −0.0102, fracture −0.0085 |
| Coarse context on every slice | 0.875 mm, 140 mm | learned zoom | −0.0228 | Lateral meniscus −0.106, MCL −0.067, ACL −0.022; effusion +0.006 |
| Learned crop placement | scout value map | crop-selection runs | at most +0.005 over fixed placement | Crop centres differed by at most 2 mm between positive and negative studies |
| One backbone per plane, dense readouts | 3 × ReXNet-200 | plane backbones | +0.0020, 80% [−0.0003, +0.0044] | Tied publicly (0.945 against 0.946); +35% training time, 2× Kaggle time |
| One backbone per plane, dense readouts | 3 × ResNet-18 | `rn18-plane-backbones-dense-fusion` | −0.0033 | RadImageNet ResNet-18 campaign |
| One backbone per plane, joint pool | 3 × ResNet-18 | `rn18-plane-backbones-joint` (revisits) | +0.00005 | Tie |
| Dense per-plane readouts, shared backbone | ResNet-18 | `dense-plane-readouts` (revisits) | −0.0052 | Each plane pooled alone with its own BCE |
| Per-type readout weights in the joint pool | ResNet-18 | `plane-contrast-readouts` (revisits) | +0.0009, 80% [+0.0001, +0.0017] | Accepted there; weights initialized from the shared readout |
| Tokens from two backbone stages | ResNet-18 | `rn18-multiscale-tokens` | −0.0023 | Two depths of one view, not two fields |
| ResNet-10t as the only backbone | incumbent recipe | `rn10t-incumbent-recipe` (revisits) | −0.0053 | Weak alone; seeds 9 and 10 use it only for the wide context |
| More slices per study | 36 against 26 | `rn18-more-triplets` | +0.0018 | Deferred to the endgame; excluded here |

## Geometry

All from the 300-series sample of
[input resolution](../../../docs/research/geometry-and-views/input-resolution.md#field-of-view-and-joint-position)
(middle slice, about 97 series per plane).

- **Stored images:** field of view median 160 mm (p5 140), pixel median 0.31 mm, about 2×
  interpolated. Acquired resolution: finer axis median 0.49 mm, coarser 0.625 mm. Share of series
  whose full acquired band survives: 59% at 0.5 mm, 23% at 0.625 mm, 12% at 0.7 mm.
- **Tissue more than 10 mm past an image-centred field:**

  | Field | 120 | 130 | 140 | 150 | 160 |
  |---|---|---|---|---|---|
  | Sagittal (almost all posterior) | 81% | 64% | 34% | 8% | 2% |
  | Coronal | 49% | 25% | 9% | 1% | 0% |
  | Axial | 59% | 29% | 16% | 10% | 6% |
- **What 120 mm cuts**
  ([measured 2026-10-01](../../../docs/research/geometry-and-views/input-resolution.md#what-a-narrow-field-cuts)):
  the condyles, plateau, menisci, cruciates and patellar cartilage stay inside. The periphery does
  not: more than 20 mm of popliteal tissue is cut in 34% of sagittal series (1% at 140 mm), more
  than 10 mm on a coronal side in 30% (5–6%), and less than 40 mm of tibia remains below the joint
  line in 22–28% (3–4%).
- **Label groups this implies (anatomical reading, not measured per label):** periphery, served by
  160 mm: Baker's cyst, MCL, fracture below the plateau, a distended suprapatellar recess. Detail,
  served by 120 mm: ACL, both menisci, medial, lateral and patellofemoral cartilage.
- **Narrow field fixed at 120 mm** for every seed (2026-10-01), so the seeds share one field pair.
  130 mm halves most of the cuts for 8% coarser pixels and is the fallback for designs that show a
  slice at one field only, if their per-field AUC shows the periphery labels losing.
- **Acquired resolution by field** (header audit of 23,173 series,
  `outputs/analysis/i/input-resolution-research-20260915/acquisition/series_acquisition.parquet`,
  counted 2026-10-01): the share of series whose finer acquired axis is at or above the pixel size
  is 15% at 140 mm, 30% at 120 mm, 50% at 112 mm (0.5 mm exactly, the most common value, 13% of
  series), 70% at 100 mm and 91% at 84 mm. T1/PD series are acquired finer than fluid ones: 36%
  against 61% resolved at 112 mm, 11% against 19% at 140 mm.
- **Label reliance by contrast** (leave-one-type-out on three ReXNet controls,
  `outputs/analysis/t/type-importance-20260917/label_type_importance.csv`; older labels, indicative):
  the labels that gained at 120 mm rely 37–62% on T1/PD series (PF OA 42%, ACL 37%, lateral OA
  62%, medial OA 45%, lateral meniscus 61%); those that lost rely 76–100% on fluid series
  (synovitis 76%, Baker's cyst 94%, medial meniscus 78%, contusion 100%).
- **Placement stays image-centred.** The joint line sits 13–15 mm below the image centre, which
  leaves a 120 mm window 73–75 mm above it and 45–47 mm below.

## Code facts the seeds rely on

- `position_2d` is computed from normalized image coordinates (`modeling/spatial_tokens.py`,
  `feature_geometry`), so the same token position is a different distance from the centre at each
  field. The medial and anterior axes used by the OA detector and the half-knee masks are already
  in millimetres.
- `data/dataset.py` derives one crop radius from `data.image_size * data.spacing_mm / 2` and passes
  one spacing to `prepare_series`; the affine augmentation rescales the already-cropped image, so
  it cannot add the ring outside 140 mm.
- The reference peaks at 10.1 GiB of GPU memory at batch 8 and trains in 1,222 s (about 122 s per
  epoch).

## Noise yardstick

Paired Δ training noise about 0.0006–0.0010 on top of a bootstrap standard error of 0.0015
([calibration](../../literature-mechanisms.md#3-calibration-against-repeat-pairs-keep-the-rule));
a fresh seed moves absolute ×4 macro by up to about 0.006. One run per seed is weak evidence.

## Blocked directions

Image sizes above 224, more slices per study, ensembling and test-time augmentation wait for the
endgame. Foveated resampling, rule-anchored nested crops and a learned scout are kept for a
separate investigation with the user.
