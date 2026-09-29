# Local metrics against the public leaderboard

**2026-09-28 update, S23–S77 (41 single-checkpoint submissions in 8 comparable cohorts).** Within
a cohort, the public score's spread (SD 0.0012–0.0023) is about the size of pure seed noise, so
only differences of 0.003 or more say much. On those decisive pairs, scoring the saved validation
predictions against `rebuild_state_v2` agrees with the leaderboard on 78–83% of pairs, against
71–73% for `combined_v1`, the labels most of these runs were validated on. The ×4 and ×1
predictions track equally well on the same submissions. Gold-only AUC is below chance (38–46%),
again. Fixed labels cannot judge a change of training labels: each label set favors the models
trained on labels like itself by about ±0.005, more than the effects being measured.

## 2026-09-28 update

[`correlate_submission_metrics.py`](../../../scripts/adhoc/correlate_submission_metrics.py) reads
the machine ledger (`outputs/submission/kaggle-ledger.json`), resolves each bundle's checkpoint
to its run, fold and epoch, and scores that checkpoint's saved predictions against every frozen
label set: `steven_v4` (`all_5fold_v3`), `combined_v1` (`all_5fold_scanner_v1`) and
`rebuild_state_v2` (`all_5fold_scanner_v2`). It also scores against their consensus (per label, only
the studies where all three agree on the hard label) and against the official labels of the
fold's gold studies. CPU only; artifacts in `outputs/analysis/local-public-metrics-20260928/`.

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= uv run --locked python -m scripts.adhoc.correlate_submission_metrics --output-dir outputs/analysis/local-public-metrics-new
```

**Sources and cohorts.** `val_x<m>` is the per-epoch validation prediction of the submitted epoch
at the run's validation multiplier; `pred_x4` is the ×4 prediction saved for the selected
checkpoint, available for 28 submissions from S41 on; `own` is the metric recorded at training,
against the run's own labels. A cohort shares split, fold and inference setting (image count,
temperature, TTA); `own` metrics also share training labels. Ensembles (S35–S37, S71), the
temperature-folded checkpoint (S28, S29, S31) and S01–S22, whose run directories are gone, are
excluded. Cohorts: `all_5fold_v3` fold 0 at ×2 τ1 (6) and ×4 τ0.75 (5); scanner v1 fold 0 (19),
fold 1 (3), fold 2 (3), fold 3 (6), fold 4 (5); S47 (×5 τ0.7) stands alone.

**Statistics.** Public and metric are centered within each cohort before a pooled Pearson
correlation, with a permutation test that shuffles public scores within cohorts (20,000 draws).
Concordance counts within-cohort pairs ordered the same way, ignoring ties; decisive pairs differ
publicly by at least 0.003, the observed seed swing.

| Metric | n | Pearson (p) | Concordance | Decisive |
|---|---:|---:|---:|---:|
| ×4 against `rebuild_state_v2` | 28 | **+0.48 (0.008)** | 0.70 (86) | **0.78 (41)** |
| ×4 against `steven_v4` | 28 | +0.43 (0.040) | 0.66 (86) | 0.80 (41) |
| ×4 against the consensus | 28 | +0.48 (0.013) | 0.65 (86) | 0.76 (41) |
| ×4 against `combined_v1` | 28 | +0.37 (0.062) | 0.63 (86) | 0.73 (41) |
| ×4 soft AUC, `rebuild_state_v2` | 28 | +0.40 (0.053) | 0.65 (86) | 0.78 (41) |
| ×4 gold-only AUC | 28 | +0.16 (0.45) | 0.40 (86) | 0.46 (41) |
| ×1 soft AUC, `rebuild_state_v2` | 41 | +0.30 (0.075) | 0.64 (181) | 0.75 (80) |
| ×1 against `rebuild_state_v2` | 41 | +0.11 (0.52) | 0.57 (181) | 0.68 (80) |
| ×1 against `combined_v1` | 41 | −0.05 (0.79) | 0.48 (181) | 0.54 (80) |
| Recorded ×1 macro AUC (own labels) | 44 | −0.16 (0.39) | 0.41 (153) | 0.43 (70) |
| Recorded ×1 gold AUC | 44 | −0.06 (0.75) | 0.38 (152) | 0.38 (69) |

On the same 28 submissions, ×1 against `rebuild_state_v2` agrees on 0.66 of pairs and 0.83 of
decisive ones, ×1 against `combined_v1` on 0.58 and 0.71, and the recorded ×1 macro AUC on 0.55
and 0.63. The weaker ×1 rows in the table come from the earlier fold-0 submissions (S40–S47),
not from the multiplier.

**Label switches** (×4 Δ against each label set; public Δ in thousandths):

| Pair | Training labels | Public | `combined_v1` | `rebuild_state_v2` | `steven_v4` | Consensus | Gold |
|---|---|---:|---:|---:|---:|---:|---:|
| S73→S75 | combined → rebuild mean | +0.000 | −0.0010 | +0.0051 | +0.0017 | +0.0035 | +0.0088 |
| S73→S76 | combined → rebuild state | +0.001 | −0.0060 | +0.0039 | −0.0055 | +0.0014 | −0.0110 |
| S48→S50 | combined → image blend | +0.003 | +0.0035 | +0.0026 | +0.0081 | +0.0032 | −0.0057 |
| S51→S52 | combined → image blend | +0.001 | +0.0014 | +0.0017 | +0.0043 | +0.0011 | −0.0140 |

**Reading.**

- **Validate on `rebuild_state_v2`.** It tracks the leaderboard best of the three label sets at
  both multipliers, which supports the 2026-09-28 switch of the shared evaluation. Moderate-weak:
  28–41 submissions, about 30 metrics compared, and the rebuild was chosen partly on S75/S76.
- **The multiplier does not matter for tracking;** ×4 stays the adoption score for its lower
  prediction noise, not for better agreement.
- **The recorded ×1 macro AUC, which picks the checkpoint, tracks worst** (0.43 of decisive
  pairs overall, 0.63 on the recent 28). It is scored on each run's own training labels, mostly
  `combined_v1`.
- **Gold-only AUC stays below chance** with 11–12 fold studies; never use it to choose.
- **Label changes need a submission.** Each label set rates the S73→S76 switch in its own favor
  (`combined_v1` −0.0060, `rebuild_state_v2` +0.0039) while the public score moved +0.001. The
  consensus came closest in these four pairs, but only S48→S50 is decisive, where every label set
  except gold agreed with the public gain.

## 2026-09-18 analysis

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
