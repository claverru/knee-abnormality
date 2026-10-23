# Grouped AUC variants against the public leaderboard (2026-09-29)

**No AUC variant averaged over acquisition groups is a proven better tracker than the current one.** ("Current" throughout this page is the selection metric: pooled hard macro AUC against `rebuild_state_v2`, which stayed the metric to the end.)
The best narrow candidate, soft AUC averaged over the 8 scanner groups, orders the leaderboard's
pairs better in-sample (weighted concordance 0.761 against 0.625), but it depends on an arbitrary
group-size threshold: requiring 10 positives and 10 negatives per group instead of 5 drops it to
0.604. A wide search over 784 variants and their pairwise averages overfits: its pick scores
0.810 in-sample and 0.670 out of sample, no better than plain pooled soft AUC (0.671). The scanner
adversarial and confusion runs rank the same way on every metric.

[`correlate_grouped_auc.py`](../../../../../scripts/adhoc/correlate_grouped_auc.py) scores the saved
predictions of the submitted checkpoints: the selected epoch's validation predictions (`val`, 50
sets, S23–S79) and the ×4 predictions (`x4`, 30 sets, S41–S79). It pairs submissions within the
cohorts of the [2026-09-28 update](2026-09-28-label-sets-and-multipliers.md), 317 pairs in all. CPU only, about 20
minutes; artifacts in `outputs/analysis/g/grouped-auc-search-20260929/`.

```bash
env CUDA_VISIBLE_DEVICES= uv run --locked python -m scripts.adhoc.correlate_grouped_auc --output-dir outputs/analysis/grouped-auc-search-new
```

**Variants.** A grouped AUC is computed per label: the mean of the within-group AUCs over groups
with at least *m* positives and *m* negatives. The labels are then averaged. The search crosses:

- **Label sets:** `rebuild_state_v2`, `steven_v4`, `combined_v1`, and their mean (`mean3`).
- **Targets:** hard (0.5 threshold) or soft AUC.
- **Groupings:** pooled; major vendors only; site; site without the minor pool; scanner group;
  scanner group without the minor pool; field strength; raw vendor (7); scanner model; scanner
  model on major vendors only. "Site" is vendor with the four minor vendors pooled, because no
  DICOM site tag survives de-identification and the minor pool is one site.
- **Group weights:** equal, √n or n.
- **Threshold *m*:** 3, 5, 10 or 20.

The wide search also averages each pair of the 20 best variants, after scaling each variant by
its within-cohort standard deviation.

**Criterion.** Weighted concordance: the share of within-cohort pairs ordered like the public
score, each pair weighted by its public gap, so pairs with tied public scores count for nothing.
Out-of-sample estimate: leave one cohort out, pick on the rest, and score the pick on the held-out
cohort. The max-statistic p repeats the whole search on 2,000 within-cohort shuffles of the
public scores and compares its best gain over the current metric with the observed gain.

| Candidates | Pick on all cohorts | In-sample | Leave-one-cohort-out | Max-statistic p |
|---|---|---:|---:|---:|
| Wide: 784 variants + pairwise averages | `rebuild soft scanner_group m5` + `combined soft scanner √n m20` | 0.810 | 0.670 | 0.27 |
| Wide, single variants | `mean3 soft scanner m20` | 0.787 | 0.678 | 0.35 |
| Grouping only (rebuild, soft, equal, *m*=5; 10 variants) | `rebuild soft scanner_group m5` | 0.761 | 0.740 | 0.081 |
| Current: rebuild hard pooled | fixed | 0.625 | | |
| Rebuild soft pooled | fixed | 0.671 | | |
| Rebuild hard, major vendors only | fixed | 0.615 | | |
| Rebuild soft scanner group, *m*=10 | fixed | 0.604 | | |

Pairs ordered like the leaderboard, for the current metric → rebuild soft scanner group *m*=5:
- `x4` decisive pairs (public gap ≥ 0.003): 32 → 34 of 41.
- `val` decisive pairs: 54 → 69 of 84.

Against pooled soft AUC, the scanner-group variant's gain in weighted concordance is +0.090, with
single-comparison permutation p=0.019.

**Where the scanner-group gain comes from** (agreeing / non-tied pairs; current → scanner group
soft *m*=5):

| Cohort | `val` | `x4` |
|---|---|---|
| scanner v1 fold 0 (S40–S59) | 85 → 113 of 146 | 42 → 43 of 61 |
| scanner v1 fold 3 (S66–S72) | 7 → 9 of 13 | 8 → 10 of 13 |
| scanner v1 fold 4 (S73–S77) | 6 → 4 of 9 | 7 → 4 of 9 |
| folds 1, 2 and `all_5fold_v3` | 5 → 7 of 22 | 3 → 3 of 3 |

Most of the gain is the fold-0 ×1 validation predictions; on fold 0's ×4 predictions, the
adoption score, the two metrics tie. The fold-4 cohort, which is all confusion-head runs, goes the
other way.

**Sensitivity.** Rebuild soft scanner-group weighted concordance by threshold:

| Weights | *m*=3 | *m*=5 | *m*=10 | *m*=20 |
|---|---:|---:|---:|---:|
| equal | 0.691 | 0.761 | 0.604 | 0.625 |
| √n | 0.697 | 0.725 | 0.631 | 0.634 |
| n | 0.685 | 0.711 | 0.611 | 0.634 |

The *m*=5 peak drops below pooled soft AUC one step either side. The threshold decides which of
the small groups (minor vendors, unknown field strength) count for the rarer labels, so the
signal rides on a few noisy small-group AUCs. Averaging over scanner models, which gives about 10
small groups per label, is worse than pooling at every setting.

### Scanner adversarial runs

Fold 3, seed 2617320043, Δ against `half-stack-control`, ×4 predictions, 80% report-group
bootstrap in brackets (300 draws). S numbers follow the
[auxiliary-signals campaign](../../campaigns-and-queues/2026-09-27-auxiliary-signals-campaign.md); the machine ledger numbers
these five submissions one lower (S66–S70), as does the cohort table of the [2026-09-28 update](2026-09-28-label-sets-and-multipliers.md).

| Run | Public | Current (hard pooled) | Soft pooled | Hard, major vendors | Soft scanner group *m*=5 |
|---|---:|---:|---:|---:|---:|
| Confusion head (adopted) | 0.948 (S67) | −0.0002 [−0.0019, +0.0020] | −0.0004 | **+0.0007** | −0.0008 [−0.0027, +0.0011] |
| Per-site bias | 0.947 (S69) | −0.0003 | −0.0005 | +0.0003 | −0.0012 |
| Confusion over 31 scanner models | 0.944 (S70) | −0.0014 | −0.0012 | −0.0008 | −0.0013 |
| Stronger confusion head | 0.943 (S71) | −0.0024 | −0.0032 | −0.0017 | −0.0036 |
| Per-site bias + model confusion | – | −0.0008 | −0.0007 | −0.0002 | −0.0007 |
| Confusion on backbone features | – | −0.0015 | −0.0019 | −0.0009 | −0.0031 |
| Per-model bias | – | −0.0031 | −0.0021 | −0.0023 | −0.0023 |
| Gradient reversal, head at full strength | – | −0.0043 | −0.0028 | −0.0025 | −0.0035 |
| Gradient reversal | – | −0.0048 [−0.0074, −0.0022] | −0.0035 | −0.0034 | −0.0040 [−0.0052, −0.0011] |

- **Every metric orders the four submitted runs as the leaderboard did.** Only
  major-vendor hard AUC, the major-vendor-only score from [site bias](../../../../modeling/site-bias.md),
  puts the confusion head and the per-site bias above the control (+0.0007, +0.0003), as the
  leaderboard did. Those gaps are inside the bootstrap interval, and the public gap is one
  rounding step.
- **Grouped AUC does not reward site removal.** Averaging within scanner groups discards the
  between-site ranking a site prior could buy, yet the debiasing runs lose slightly more on it
  than on pooled AUC (confusion head −0.0008 against −0.0002). Within a group, removing the
  scanner signal does not improve ranking.
- **The confusion head's major-vendor gain is "no cost", not a measured gain.** It is +0.0007
  [−0.0015, +0.0030] at ×4 and −0.0019 [−0.0042, +0.0007] at ×1 against `rebuild_state_v2`, and
  +0.0002 against `combined_v1` in [site bias](../../../../modeling/site-bias.md). About 0.0009 of its ×4
  pooled loss comes from the minor site, where the labels reward the site prior it removes. That
  part matches its site-signature shift. Across all 317 submission pairs, the minor-site component
  (major-vendor minus pooled hard AUC) does not predict the public order: weighted concordance
  0.443, permutation p 0.55. This was measured in the session from `submission_metrics.parquet`
  and is not part of the script's saved output.
- **No rejected run is rescued.** Gradient reversal stays about −0.004 on every metric, and the
  unsubmitted variants stay negative. The `val` predictions give the same order.

**Reading.**

- Keep ×4 pooled hard AUC against `rebuild_state_v2` as the adoption score. Report soft AUC over
  the scanner groups (*m*=5) beside it as a diagnostic only. Its out-of-sample edge (0.740 against
  0.625) rests mostly on fold-0 ×1 predictions and on a threshold picked before any result but
  fragile to ±1 step. Weak evidence: family max-statistic p 0.081 over about 30 submissions.
- Do not use any wide-search combination; its in-sample edge vanishes out of sample.

### Same-group, confidence-weighted pair AUC

This recipe was seen outside the project and tested on 2026-09-29.
- **Pairs:** only positive–negative pairs within one scanner grouping count.
- **Weights:** each study gets w = |2s − 1|² from its soft target, and each pair w_pos × w_neg,
  summed across groups.
- **Score:** the mean over the 12 targets.

The original used a four-reader average label; `mean3` (the mean of the three frozen label
sets) is the closest analog here and was fixed as primary before the run.
[`correlate_confidence_pair_auc.py`](../../../../../scripts/adhoc/correlate_confidence_pair_auc.py)
scores 40 variants: 4 label sets × {pooled, vendor, vendor × field, scanner model,
model × field} × {unweighted, confidence}. It uses the same 317 pairs and 5,000 permutations.
Artifacts are in `outputs/analysis/c/confidence-pair-auc-20260929/`.

| Variant | Weighted concordance | ×4 decisive | Folds 1–4 |
|---|---:|---:|---:|
| Current (rebuild, unweighted, pooled) | 0.625 | 32/41 | 0.546 |
| Primary: `mean3`, confidence, vendor × field | 0.614 | 33/41 | 0.530 |
| Primary: `mean3`, confidence, vendor | 0.629 | 33/41 | 0.519 |
| Rebuild, confidence, pooled | 0.675 | 35/41 | 0.503 |
| Family best: `combined_v1`, confidence, pooled | 0.684 | 37/41 | 0.525 |

**It does not reproduce here.**
- **Primary variants:** both are level with the current metric (gains −0.011 and +0.004,
  permutation p 0.60 and 0.49).
- **Whole family:** the best variant's gain is chance (max-statistic p 0.61). Picked out of
  sample, it scores 0.558, below the current metric's 0.625.
- **Same-group restriction:** it lowers every label set, because it discards the cross-group
  pairs.
- **Confidence weight:** it helps pooled AUC in-sample (+0.05, in the direction of the soft-AUC
  finding of 2026-09-23), but not on folds 1–4.
