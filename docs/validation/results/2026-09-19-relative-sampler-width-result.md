# 2026-09-19 relative sampler width

**Kept, on a weak positive.** `data.sampling_sigma_fraction: 0.867` replaces the fixed
`data.sampling_sigma_mm: 40`, setting each candidate pool's Gaussian width to a fraction of its own
limiting reach. Fold 0 at ×1 scores **0.88649**, between its two total-26 controls (0.88477 and
0.88848), and **+0.00016** at ×4. It is not a loss at any multiplier, it costs nothing at runtime,
and the new model beats `control-check` in all four cells of the sampler decomposition below.

The effect that exists is on the **training** side, not the drawing side, which is the opposite of
what the [cache measurement](2026-09-18-per-plane-sigma-cache.md) predicted.

## Setup

- **Run:** `sigma-fraction-0867`, fold 0, seed 42, ×1 validation, ten epochs, ReXNet-200,
  `study_triplet_total: 26`. Best 0.88649 at epoch 5, then 0.88609 / 0.88574 / 0.88645 / 0.88640 —
  a plateau, not a single lucky epoch.
- **Controls:** the two ReXNet ×1 runs carrying `study_triplet_total: 26` — `control-check`
  (0.88477, seed 42, identical config) and `04-total-topup-26` (0.88848, differs only by seed).
  Mean 0.88662, spread 0.0037 on seed alone.
- **Rule:** `sigma = 0.867 × limiting_reach(pool)`, where limiting reach is the distance from the
  study centre to the nearer edge of the pool's union-support envelope, floored at 1 mm. Measured
  over 4,218 physical pools on the fold-0 token cache: min −10.24, p1 22.29, p50 44.70, max 112.54.
  Two pools (0.05%) do not straddle the study centre at all and take the floor.
- **Realized width:** axial 45.1 mm, coronal 35.9, sagittal 39.0, against the old fixed 40
  everywhere. The fraction was anchored to hold the mean at the validated 40, so the arm changes
  the distribution across planes and series, not the overall reach.
- **Geometry behind it:** usable half-reach is axial 61.5, coronal 45.6 and sagittal 51.0 mm, a
  ratio of 1.35 : 1.00 : 1.12 ([acquisition geometry](../../data-preparation/peripheral-coverage.md#acquisition-geometry)).
- **Scripts:** `outputs/analysis/p/per-plane-sigma-20260918/`, plus the scratch cell harness described
  under [decomposition](#training-versus-drawing).

## Count multipliers

Against `control-check` at the same multiplier, both scored through the production prediction path.

| Multiplier | σ 0.867 | control-check | Δ |
|---|---:|---:|---:|
| ×1 | 0.88648 | 0.88472 | +0.00175 |
| ×2 | 0.88849 | 0.88846 | +0.00002 |
| ×3 | 0.89033 | 0.88994 | +0.00039 |
| ×4 | 0.89027 | 0.89010 | +0.00016 |

The gap is a ×1 phenomenon. At the ×4 the submissions use, it is +0.00016.

## Training versus drawing

The width applies to validation drawing as well as training, so the run differs from its controls on
both sides at once. Each checkpoint was therefore scored under **both** samplers, in one in-process
harness that reproduces the production path to 1e-5. Pinning the width to a constant changes source
bytes and so breaks `preprocessing_identity`; the harness patches `limiting_reach` at runtime
instead, leaving the files untouched.

**×1**

| | relative | fixed σ40 |
|---|---:|---:|
| new model | 0.88647 | 0.88588 |
| control model | 0.88411 | 0.88472 |

**×4**

| | relative | fixed σ40 |
|---|---:|---:|
| new model | 0.89027 | 0.89022 |
| control model | 0.88987 | 0.89011 |

- **Training effect** (new − control, same sampler): +0.00236 relative and +0.00116 fixed at ×1,
  +0.00039 and +0.00012 at ×4. Positive in all four.
- **Sampler effect** (relative − fixed, same model): +0.00059 for the new model, **−0.00061** for
  the control. Each model prefers the sampler it was trained under, which is a matched-condition
  effect rather than a better draw.
- **This contradicts the prior.** The cache put a comparable coronal narrowing at +0.00135 as a
  pure inference-time gain on frozen checkpoints. Retrained, the drawing side is worth ±0.0006 and
  the model side carries the rest. Inference-side cache measurements do not forecast what training
  under the same change will do.

## Per label

Paired bootstrap over studies at ×4, 400 resamples, new against `control-check`. Only two of twelve
exclude zero, in opposite directions:

| Label | Δ | 95% CI |
|---|---:|---|
| ACL | +0.0163 | [+0.0040, +0.0297] |
| Baker's | **−0.0175** | [−0.0291, −0.0042] |
| Contusion | −0.0109 | [−0.0210, +0.0012] |
| Effusion | +0.0088 | [−0.0012, +0.0195] |
| Fracture | −0.0077 | [−0.0161, +0.0007] |

The other seven move by 0.006 or less with intervals crossing zero. At twelve labels tested, two
hits is barely above chance, so the per-label profile is **not** established as different — an
earlier reading of these deltas as a real redistribution was wrong.

Baker's is worth watching if the width moves again: it keeps 80% of its coronal evidence beyond
±20 mm and coronal is the plane this arm narrowed.

## What the evidence does not cover

- **The second control was never scored at ×4.** `04-total-topup-26` carries
  `robust_center_outlier_mm`, removed in `df4b6cd`, so the production path rejects its saved config.
  It scores 0.88848 at ×1, above this run. Every ×4 comparison here is therefore against one
  control, not against the control spread.
- **One run, one fold, one seed**, against a cohort whose seed spread is 0.0037 — four times the
  measured ×4 difference.
- **Peripheral coverage did not improve as the geometry predicted.** Correlation of the per-label
  deltas with peripheral evidence share is −0.155 and with the map's flank preference −0.289, both
  slightly the wrong way.

## Follow-ups

The training side carrying the effect is what made the remaining arms worth a GPU night rather
than a cache query. Both are now closed: the fraction sweep settled the scale
([sweep](2026-09-19-fold-ensembles-and-sigma-sweep.md)), and splitting the widening from the
narrowing was trained once by the autoresearch campaign as `geometry-informed-plane-widths`,
inconclusive at +0.00041
([what is settled](../../data-preparation/peripheral-coverage.md#what-is-settled-and-what-is-not)).
