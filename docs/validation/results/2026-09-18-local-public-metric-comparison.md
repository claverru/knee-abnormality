# 2026-09-18 local metrics against the public leaderboard

**Over the nineteen comparable submissions, fold-0 macro AUC tracks the leaderboard (Pearson
+0.907, Spearman +0.620, 76% of ranked pairs) and soft-target BCE tracks it about as well
(+0.833, +0.607). Gold-only AUC does not track it at all: it is negative in every cohort and
agrees on 37% of ranked pairs. Gold-only BCE is pure noise.** Since 2026-09-13 no local metric
ranks the submissions, because their local differences are smaller than one run's training noise.

Reproduce with the [comparison script](../../../scripts/adhoc/compare_local_and_public_scores.py);
artifacts in `outputs/analysis/local-public-metrics-20260918/`. It reads saved artifacts and
`scripts/adhoc/submission_public_scores.json` on CPU, refuses an existing output directory, and
fails if a recomputed macro AUC differs from the run's saved metrics:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= uv run --locked python -m scripts.adhoc.compare_local_and_public_scores --output-dir outputs/analysis/local-public-metrics-new
```

## Which submissions are comparable

Metrics come from each run's selected epoch, chosen by the highest saved macro AUC exactly as
training selects its checkpoint. Eleven of the thirty leaderboard rows are excluded because
something other than the trained model differs:

| Excluded | Why |
|---|---|
| `baseline3`, `baseline4` ×1/×2/×3 | split `all_5fold_v1`, and ×1/×3 also change test density |
| `tonylin-soft-v3` | trained and validated against `tonylin_soft`, so its BCE is on another scale |
| `reflection-sigma40` without 3D, and its FP16 rerun | the same weights as the kept `reflection-sigma40` row |
| `ranked-filtered-seed42` τ0.75 ×3 and ×4 | attention temperature 0.75 and higher test density |
| `07-series-share-1.0` | ×1 validation multiplier, so its local score is not on the ×2 scale |
| `robust-center-20-topup-26` | ×1 validation, τ0.75 and ×4 test images |

The remaining cohort is fold 0 of `all_5fold_v3` with `steven_v4` labels, validated at ×2 and
submitted at temperature 1.0 with ×2 test images. Batch size, loader count, FP16 and the
one-versus-two-GPU split are kept, because `reflection-sigma40` scored 0.936 through all three
runtime variants.

| Model | Public | Macro AUC | Report AUC | Gold AUC | Soft BCE | Gold BCE |
|---|---:|---:|---:|---:|---:|---:|
| reflection-sigma40 | 0.936 | 0.892360 | 0.891725 | 0.9222 | 0.45185 | 0.3849 |
| overnight 00-baseline | 0.934 | 0.892763 | 0.892083 | 0.9320 | 0.44767 | 0.3649 |
| ranked-filtered-seed42 | 0.933 | 0.889573 | 0.888831 | 0.9281 | 0.44893 | 0.3620 |
| spatial-tokens-reflection | 0.931 | 0.890976 | 0.890228 | 0.9228 | 0.44931 | 0.3643 |
| overnight 02-central-sampling | 0.931 | 0.892538 | 0.891933 | 0.9172 | 0.44824 | 0.3832 |
| overnight 01-control-seed-43 | 0.931 | 0.889837 | 0.889165 | 0.9339 | 0.44801 | 0.3648 |
| overnight 02-control-seed-44 | 0.931 | 0.891627 | 0.890765 | 0.9300 | 0.44759 | 0.3598 |
| 3dsampling | 0.930 | 0.889964 | 0.889051 | 0.9367 | 0.45257 | 0.3592 |
| meniscus-classifier (1A) | 0.930 | 0.891192 | 0.890516 | 0.9147 | 0.44858 | 0.3779 |
| spatial-tokens-five-embeddings | 0.929 | 0.889506 | 0.888626 | 0.9311 | 0.45063 | 0.3592 |
| overnight 03-ema-0.999 | 0.929 | 0.893155 | 0.892273 | 0.9334 | 0.44562 | 0.3491 |
| disease-compartment-attention (1C) | 0.928 | 0.889430 | 0.888592 | 0.9374 | 0.44846 | 0.3676 |
| overnight 00-control | 0.928 | 0.892995 | 0.892217 | 0.9419 | 0.44628 | 0.3450 |
| crop140mm | 0.925 | 0.886122 | 0.885200 | 0.9270 | 0.45296 | 0.3829 |
| meniscus-oa-classifiers (1B) | 0.925 | 0.892022 | 0.891061 | 0.9420 | 0.45358 | 0.3690 |
| spatial-fusion | 0.920 | 0.884169 | 0.883542 | 0.9225 | 0.45306 | 0.3649 |
| overnight 04-higher-lr | 0.918 | 0.882938 | 0.882080 | 0.9176 | 0.45404 | 0.3895 |
| overnight 02-longer | 0.913 | 0.883536 | 0.882605 | 0.9418 | 0.46144 | 0.3889 |
| overnight 05-faster-head | 0.903 | 0.877244 | 0.876090 | 0.9399 | 0.46091 | 0.3663 |

Gold AUC and gold BCE are the eleven gold studies of fold 0; report AUC is the other 893. BCE
uses the frozen soft targets, which are hard 0/1 for gold studies.

## Correlation with the public score

Permutation p-values over 20,000 shuffles; concordance counts ranked pairs, ignoring ties in
either column. `spread` is the cohort standard deviation, `noise` the pooled within-recipe
standard deviation below, and `reliability` the share of the spread that is not that noise.

| Cohort | Metric | Pearson | Spearman | Concordant | Spread | Reliability |
|---|---|---:|---:|---:|---:|---:|
| All 19 | macro AUC | **+0.907** (p<0.001) | +0.620 (p=0.006) | 122/161 | 0.00431 | 0.84 |
| All 19 | report AUC | +0.913 | +0.625 | 123/161 | 0.00439 | 0.83 |
| All 19 | soft BCE | +0.833 (p<0.001) | +0.607 (p=0.006) | 119/161 | 0.00437 | 0.90 |
| All 19 | gold AUC | −0.304 (p=0.21) | −0.360 (p=0.13) | **59/161** | 0.00879 | 0.62 |
| All 19 | gold BCE | +0.235 (p=0.34) | +0.224 (p=0.36) | 88/161 | 0.01273 | 0.00 |
| 09-12 (n=5) | macro AUC | +0.950 | +0.900 | 9/10 | 0.00333 | 0.72 |
| 09-12 (n=5) | soft BCE | +0.853 | +0.900 | 9/10 | 0.00431 | 0.89 |
| 09-12 (n=5) | gold AUC | −0.691 | −0.500 | 4/10 | 0.01071 | 0.75 |
| 09-13 on (n=14) | macro AUC | +0.072 | +0.062 | 44/82 | **0.00140** | 0.00 |
| 09-13 on (n=14) | soft BCE | +0.079 | +0.020 | 42/82 | 0.00231 | 0.63 |
| 09-13 on (n=14) | gold AUC | −0.549 (p=0.04) | −0.651 (p=0.01) | 19/82 | 0.00846 | 0.59 |
| 09-13 on (n=14) | gold BCE | −0.391 | −0.339 | 27/82 | 0.01133 | 0.00 |

The cohort is split by submission date, not by score, so no metric is range-restricted by its own
value. Over all nineteen, public ≈ 1.69 × macro AUC with a residual standard deviation of 0.0035.

## Run-to-run noise per metric

Pooled within-recipe standard deviation over the three recorded same-recipe groups — the
2026-09-16 seed controls 42/43/44, `ranked-filtered` seeds 42–44, and the two unseeded 2026-09-18
controls — eight runs, five degrees of freedom.

| Metric | Replicate SD | Spread across the 09-13-on cohort |
|---|---:|---:|
| macro AUC | 0.00175 | 0.00140 |
| report AUC | 0.00181 | 0.00143 |
| soft BCE | 0.00141 | 0.00231 |
| gold AUC | 0.00541 | 0.00846 |
| gold BCE | 0.01306 | 0.01133 |

## Reading

- **Gold-only AUC is not a selection signal.** It is flat-to-negative wherever macro AUC
  correlates at +0.9, and it agrees with the leaderboard on 37% of ranked pairs across the whole
  cohort. Eleven studies give it three times macro AUC's run-to-run noise. The −0.65 rank
  correlation after 09-13 is not significant once the twenty tests here are accounted for
  (Bonferroni threshold 0.0033) and it shrinks to −0.17 when the last three epochs are averaged
  instead of taking the best one, so read it as absence of signal rather than an inverted one.
- **Gold-only BCE is noise:** its replicate standard deviation exceeds its entire between-model
  spread, in both halves of the cohort.
- **Soft BCE adds nothing over macro AUC**, though it is the most reliable single-run quantity
  (lowest noise relative to spread, 0.90). It is comparable only inside one label table.
- **After 09-13 the submitted models differ locally by less than one run's noise** (spread 0.00140
  against noise 0.00175), so no local metric could rank them; this is a resolution limit, not
  evidence that the local set stopped predicting the leaderboard. The three seed replicas of one
  recipe make it concrete: local 0.892995 / 0.889837 / 0.891627 → public 0.928 / 0.931 / 0.931.
- **Inference settings moved the leaderboard more than any recent training change.** On fixed
  weights, τ0.75 with ×3 images was worth +0.005
  ([submissions](../../inference/submissions/2026-09-18-temperature-and-count-submissions.md)),
  while the entire 09-13-on cohort of different models spans 0.011 unpredictably.
- **Averaging the last three epochs beats the best-epoch value** as a local statistic: over the
  nineteen it raises Pearson from +0.907 to +0.931 and Spearman from +0.620 to +0.677, by removing
  best-epoch selection noise. It does not rescue the 09-13-on cohort. This variant was measured in
  the session and is not part of the script's saved output.

## Limits

Public scores are rounded to three decimals, and each row is one fold and one unrepeated training
run, so training variability is inside every point. The replicate noise comes from five degrees of
freedom, giving its own standard deviation a roughly ×0.62–×2.45 interval; the gold reliability
figures in particular could be entirely noise at the upper end. Correlations describe submitted
models, which were selected during development, and no private-leaderboard behaviour is implied.
