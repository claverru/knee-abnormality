# 2026-09-23 validation AUC variants against the public leaderboard

**Soft-label macro AUC tracks the public leaderboard better than the current hard-threshold macro
AUC. The gain comes entirely from pairs of studies on the same side of 0.5, which the current
metric ignores.** Across 85 variants, including machine, field-strength and acquisition strata, every one that counts those pairs correlates positively in
all three cohorts, and every one that ignores them is negative in both recent cohorts. Plain soft
AUC orders 168 of 213 within-cohort submission pairs like the leaderboard; the current metric
orders 140. The best variants reach 173: soft AUC with scanner-balanced weights, and soft AUC over
same-sex pairs only, which has no plausible mechanism. No variant beats plain soft AUC by more
than 5 pairs, which is noise. Against the current
metric, soft AUC's gain gives a permutation p=0.023, or p=0.21 after correcting for picking the
best of the 84 alternatives. The consistency across cohorts and the gold-label evidence below
support the result more than the corrected p-value does.

Reproduce with [`score_validation_auc_variants`](../../../scripts/adhoc/score_validation_auc_variants.py);
artifacts in `outputs/analysis/validation-auc-variants-20260923/` (`table.txt` holds the full
ranking).

## Definition

The current metric thresholds each frozen evaluation target at 0.5 and averages the 12 per-label
ROC AUCs. Soft AUC keeps the frozen soft targets. For each label, every ordered pair of distinct
studies (i, j) gets weight y_i(1 − y_j), and the score is the weighted share of those pairs where
the model ranks i above j, with ties counting half. When each study's true label is Bernoulli(y),
this is the ratio-of-expectations estimate of the expected hard AUC. The current metric instead
assumes y=0.4 is certainly negative and y=0.6 certainly positive. Soft AUC stays per-label and
rank-based, so it is insensitive to calibration. Gold studies have 0/1 targets and count the same
way in both metrics.

## Why soft targets are closer to the truth

On the 58 gold studies (696 label cells), report-derived targets compare with the radiologist
labels as follows:

| Report-derived y | Cells (Steven v4) | Gold-positive rate | Cells (combined v1) | Gold-positive rate |
|---|---:|---:|---:|---:|
| [0, 0.1) | 195 | 0.04 | 190 | 0.03 |
| [0.1, 0.3) | 150 | 0.07 | 164 | 0.08 |
| [0.3, 0.5) | 60 | **0.33** | 73 | **0.32** |
| [0.5, 0.7) | 11 | 0.18 | 16 | 0.38 |
| [0.7, 0.9) | 20 | 0.40 | 47 | 0.68 |
| [0.9, 1] | 260 | 0.74 | 206 | 0.78 |

Hard thresholding scores the [0.3, 0.5) band as 0% positive, while about a third of it is
gold-positive, and it scores the top band as 100% positive, while about a quarter is not. A single
logistic map fitted over all labels (gold ~ a + b·logit(y)) gives b=0.65 to 0.81, which flattens
the targets, and a<0, which lowers them.

## Cohorts

Each row is a distinct submitted checkpoint, scored with the predictions of its own selected
epoch. Within a cohort, the split, the evaluation targets and the inference settings are fixed.

| Cohort | Submissions | Validation | Kaggle inference |
|---|---|---|---|
| `v3_x2` (n=20) | S05–S15, S17–S20, S23–S27 | `all_5fold_v3`, x2, τ1 | x2, τ1 |
| `v3_x1_tau075` (n=5) | S32–S34, S38, S39 | `all_5fold_v3`, x1, τ1 | x4, τ0.75 |
| `scanner_x1` (n=8) | S40–S46, S48 | `all_5fold_scanner_v1`, x1, τ1 | x4, τ0.75 |

Excluded: `all_5fold_v1` baselines; S16 (unscored); S21, S22, S28, S29, S31 and S47, which reuse
already-included weights; S30, the only x1-validation row with x2 τ1 inference; and ensembles
S35–S37, which have no fold-0 validation predictions. S11 (Tonylin labels) is included because AUC
uses the frozen targets.

## Results

Pearson r, Spearman ρ and concordant/ranked pairs (pairs with a tie on either side are skipped).
p is a within-cohort permutation of public scores testing the concordance gain over the current
metric (20,000 draws). p_max here corrects over the 42 alternatives run before the acquisition
strata were added. Counting all 84 alternatives, plain soft AUC's p_max is 0.21. Selected rows:

| Variant | `v3_x2` | `v3_x1_tau075` | `scanner_x1` | Pairs | p / p_max |
|---|---|---|---|---:|---|
| hard 0.5 (current) | +0.856 / +0.566 / 131 | −0.672 / −0.667 / 2 | −0.379 / −0.540 / 7 | 140 | – |
| **soft** | +0.740 / +0.709 / 142 | +0.804 / +0.872 / 8 | +0.655 / +0.589 / 18 | **168** | 0.023 / 0.11 |
| soft, scanner-balanced weights | +0.714 / +0.773 / 148 | +0.762 / +0.872 / 8 | +0.440 / +0.454 / 17 | 173 | 0.009 / 0.057 |
| gap-weighted, max(y_i − y_j, 0) | +0.720 / +0.717 / 143 | +0.813 / +0.975 / 9 | +0.647 / +0.589 / 18 | 170 | 0.017 / 0.086 |
| soft, report-group deduplicated | +0.753 / +0.725 / 143 | +0.833 / +0.975 / 9 | +0.623 / +0.589 / 18 | 170 | 0.016 / 0.086 |
| soft, weighted by report length | +0.829 / +0.791 / 149 | +0.627 / +0.462 / 6 | +0.435 / +0.233 / 15 | 170 | 0.008 / 0.086 |
| soft, gold-calibrated targets | +0.836 / +0.740 / 144 | +0.513 / +0.462 / 6 | +0.568 / +0.552 / 18 | 168 | 0.012 / 0.11 |
| soft, same-side pairs only | +0.359 / +0.620 / 135 | +0.778 / +0.975 / 9 | +0.601 / +0.675 / 19 | 163 | 0.10 / 0.20 |
| soft, straddling pairs only | +0.898 / +0.647 / 139 | −0.660 / −0.667 / 2 | −0.323 / −0.540 / 7 | 148 | 0.047 / 0.79 |
| soft targets sharpened (T=0.5) | +0.905 / +0.745 / 146 | −0.337 / −0.308 / 3 | +0.013 / +0.086 / 13 | 162 | 0.010 / 0.23 |
| hard 0.3 | +0.919 / +0.774 / 151 | −0.489 / −0.667 / 2 | −0.232 / −0.123 / 10 | 163 | 0.009 / 0.20 |
| micro (pooled labels) | +0.900 / +0.691 / 142 | +0.675 / +0.564 / 7 | +0.757 / +0.491 / 17 | 166 | 0.006 / 0.14 |
| hard, gold-propensity weights | +0.859 / +0.574 / 131 | −0.628 / −0.667 / 2 | −0.461 / −0.589 / 7 | 140 | 0.53 / 1.0 |
| hard, same-scanner pairs only | +0.851 / +0.483 / 124 | −0.734 / −0.667 / 2 | −0.422 / −0.442 / 8 | 134 | 0.87 / 1.0 |

Ranked pairs per cohort are 180, 9 and 24. The other variants behave the same way, and none
changes the conclusion. They include thresholds of 0.7, dropping uncertain targets, partial AUC at
FPR ≤ 0.2 and report-only studies. They also include consensus-10 and nine-independent-extractor
targets (hard and soft), extractor-agreement weights, same-series-count pairs, gold ×10,
median-over-labels, Spearman against the target, soft BCE, and the mean or ensemble of the last
three epochs.

## Machine, field strength and acquisition strata

Each grouping axis is applied to hard and soft pair weights in three ways. The axes are scanner
group (vendor × field), vendor, field strength (1.5 T, 3 T, low, unknown), machine model (46
models), slice thickness (<3, 3–3.5, >3.5 mm), in-plane pixel spacing (<0.3, 0.3–0.45, ≥0.45 mm),
patient sex and series count.

- same-group pairs only, which removes ranking that could come from machine or site shortcuts;
- different-group pairs only;
- group-balanced study weights, which give every group equal total weight.

This gives 48 variants.

- **Hard-based (24):** none gets more than 3 of 9 pairs right in `v3_x1_tau075`. Only the
  vendor-balanced (18/24) and field-balanced (15/24) versions get at least half the pairs right in
  `scanner_x1`, and both lose pairs in `v3_x2`.
- **Soft-based (24):** they range from 149 to 173 pairs, against 168 for plain soft AUC. No axis
  stays above plain soft AUC in every cohort. Same-machine soft AUC reaches 21/24 and ρ +0.82 in
  `scanner_x1` but only 5/9 in `v3_x1_tau075`.

The acquisition mix of the hidden test (about 1,300 studies) is not recorded, so these axes
cannot be weighted toward the test distribution.

## Reading

- **The decomposition locates the signal.** Soft weights restricted to pairs that straddle 0.5
  behave like the current metric. Pairs on the same side of 0.5 (for example y=0.4 against
  y=0.05) carry all of the recent-cohort agreement. That same-side AUC spans 0.593–0.630 across
  the recent checkpoints, and its seed noise is 0.00106 (reliability 0.98 in `scanner_x1`). The
  current metric scores the other side of the same trade-off. The class-balance exponent sweep
  moves the two in opposite directions: k00 has the highest same-side AUC, and k075 and
  combined-scanner have the highest hard AUC. The leaderboard sits between them, which soft AUC
  captures.
- **Study, series and acquisition properties do not help.** Scanner balancing, report deduplication,
  report-length or gold-propensity weights, and within-scanner or within-series-count pairs
  leave every hard-based variant negative in the recent cohorts. On soft AUC they move the count
  by 5 pairs or fewer in either direction. Gold studies do differ: their reports are longer
  (median 1206 vs 974 characters), contain twice as many hedging terms and come with slightly more
  series, while their scanner mix matches the rest within noise. Test-like weighting built on that
  difference made both metrics worse.
- **Gold calibration is the principled variant but not a better one.** It ties plain soft AUC
  (168) with a stronger large cohort and a weaker five-submission cohort.
- **The inference mismatch does not explain the failure.** On the x4 τ0.75 predictions that exist
  (S40, S46), hard AUC still ranks S40 (public 0.940) above S46 (0.944): 0.90788 against 0.90467.
  Soft AUC ranks them correctly: 0.79617 against 0.79754.
- **Seed noise.** The pooled within-recipe SD over the three replicate groups of the
  [2026-09-18 comparison](2026-09-18-local-public-metric-comparison.md#run-to-run-noise-per-metric)
  (df=5) is 0.00126 for soft AUC, 0.00141 for gap AUC and 0.00175 for hard AUC. Micro AUC's SD of
  0.00279 exceeds its recent spread, so its concordance is not reliable.
- **No single label drives it.** With each label left out in turn, soft AUC still wins 39–53 of
  the pairs where it disagrees with the current metric, against 15–19.

## Checkpoint-selection check (S49)

Soft AUC would select a different epoch than hard AUC in 17 of the 33 runs, usually a later one.
The hard AUC of the soft-selected epoch was at most 0.001 lower. To test the selection itself,
`k_05` epoch 9 (`last.ckpt`, the soft-AUC pick) was submitted with S41's inference settings, from
the run's archived source (bundle `f96c8f4e`). Its weights are the only difference from S41,
which was the hard-AUC pick, epoch 7.

| Submission | Epoch | Hard AUC | Soft AUC pick | Public |
|---|---:|---:|---|---:|
| S41 | 7 | 0.89904 | no | **0.945** |
| S49 | 9 | 0.89872 | yes | 0.944 |

The soft-AUC pick scored 0.001 lower, as hard AUC's own ordering predicted. This is one pair, a
single rounding step apart, so it is inconclusive in either direction. Seed replicates alone
spread the public score by 0.003. The epochs where the two metrics disagree differ by at most
0.001 of hard AUC, so the choice is low-stakes, and settling it would need many paired
submissions. Checkpoint selection stays on hard AUC by default, not on evidence. Only `k_025`
(S44) has its soft-AUC epoch saved for a second pair without retraining.

## Later evidence (S51)

S51, the oblique group embedding, is a data point against both metrics when comparing runs. At
×4 it led `k05-clean` by +0.0024 hard and +0.0021 soft, but both scored 0.942 publicly
([result](2026-09-22-oblique-groups-result.md#group-embedding-on-the-cleaned-source-s51)). Soft
AUC did no better than hard AUC on this pair.

## Adopted use

Since 2026-09-23, `validation_metrics` also records `soft_macro_auc` and `per_target_soft_auc`,
and training logs `val_soft_macro_auc`. The implementation is sort-based and excludes
self-pairs; on `k_05` epoch 7 it matches the pairwise definition to within 1e-10.
[`plot_validation_metrics`](../plot-validation-metrics.md) plots it for every run, including runs
trained before it was logged. Soft AUC is used to compare runs and choose submissions.
Checkpoint selection (`val_macro_auc`) is unchanged.

## Limits

The pairs share submissions, so only the permutation p-values are valid tests, and the corrected
one for plain soft AUC is 0.21 once all 84 alternatives are counted. Only 13 submissions come after 2026-09-19, public scores are
rounded to 0.001, and each point is one fold and one training run. Validation ran at x1 or x2 and
τ1. The gold-calibration and property analyses rest on 58 gold studies. That test labels resemble
the gold set is an assumption. This analysis compared metrics on checkpoints that were already
selected. Selecting checkpoints by soft AUC would pick different epochs, which is untested.
