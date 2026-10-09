# 2026-09-21 loss-campaign ensemble simulation

**Averaging runs is the big gain; picking which losses to include is not.** Three control
seeds averaged reach 0.900994, +0.0044 over the single control. Swapping the seeds for
non-rejected loss variants adds only about +0.0009 at the same size and member quality. Once
optimism is removed, subset selection does no better than averaging every run. Those tied
held-out scores are from the split-half check below. The most complementary losses were
`shared-bce-ranking`, `shared-shift-projected-bce`, `shared-target-logit-margin`,
`shared-evidence-margin-bce` and `shared-boundary-proximity-weighting`.

## Cohort and method

Members: the 14 trained [losses campaign](../../../autoresearch/campaigns/losses/findings.md)
runs plus three control-recipe seeds, `13-control-repeat` (anchor, the campaign reference),
`00-control` and `balanced-bce`. All use control B (ReXNet-200, class-balanced BCE,
`all_5fold_v3` fold 0, 904 studies). Each run has its own random seed. Configs differ from the
anchor only in seed, output path and `trainer.loss.*`, and input identities match. The one
exception is `balanced-bce`, whose `preprocessing_identity.sources_sha256` differs because it
comes from an older source revision. `shared-group-variance-compression` never trained and is
excluded.

The ensemble is an unweighted mean of probabilities, as in exported bundles. The metric is
macro AUC on hard labels (target ≥ 0.5), computed from each run's saved selected-checkpoint
validation predictions. Every one of the 131,071 subsets was scored. Checkpoints were
[pruned](../../architecture/configs-and-experiment-artifacts.md#checkpoint-pruning), so the
ensembles cannot be rebuilt as bundles. Each run's epoch was already selected on this fold.

Script: [`simulate_loss_ensembles.py`](../../../scripts/adhoc/simulate_loss_ensembles.py), run on
the `a960863` tree, with its source copied under the output's `source/`. Output: `outputs/analysis/l/loss-ensembles-20260921/` (`summary.json`, `subsets.parquet`,
`pairs.csv`, `marginal.csv`, `best_by_size.csv`, `top_frequency.csv`, `greedy.csv`,
`selection_splits.csv`). It reads saved predictions on CPU, about 90 s at 16 workers, and refuses
an existing output directory:

```bash
rtk proxy env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked python -m scripts.adhoc.simulate_loss_ensembles --output-dir outputs/analysis/loss-ensembles-new
```

## Seed diversity versus loss diversity

This comparison holds size fixed. The seed-only and non-rejected-loss cohorts have the same mean
member AUC (0.8950 vs 0.8949).

| Size | Cohort | Subsets | Mean macro AUC | Gain vs mean member | Beats seed-only mean |
|---:|---|---:|---:|---:|---:|
| 2 | Seed only | 3 | 0.899558 | +0.0045 | — |
| 2 | Non-rejected losses | 36 | 0.899994 | +0.0051 | 69% |
| 2 | All losses | 91 | 0.897945 | +0.0054 | 30% |
| 3 | Seed only | 1 | 0.900994 | +0.0060 | — |
| 3 | Non-rejected losses | 84 | 0.901849 | +0.0070 | 85% |
| 3 | All losses | 364 | 0.899909 | +0.0073 | 32% |

On average, loss pairs are a little less correlated than seed pairs. The mean per-target
Spearman correlation is 0.906 for loss–loss pairs and 0.924 for control–control pairs. The
seed-only baseline has only three runs, so this extra +0.0004 to +0.0009 is weak evidence.
The non-rejected filter also comes from fold-0 decisions. Rejected losses lower the mean
because they are weaker members, not because they are more similar to the others.

## Which members complement

The table shows each member's mean marginal gain: the average change in AUC when it joins every
subset that lacks it.

| Member | Alone | Joining 1 run | Joining 2 runs | Joining 3 runs |
|---|---:|---:|---:|---:|
| `shared-bce-ranking` | 0.8973 | +0.0081 | +0.0039 | +0.0025 |
| `shared-target-logit-margin` | 0.8967 | +0.0071 | +0.0031 | +0.0019 |
| `shared-evidence-margin-bce` | 0.8965 | +0.0070 | +0.0029 | +0.0017 |
| `shared-boundary-proximity-weighting` | 0.8955 | +0.0067 | +0.0029 | +0.0017 |
| `13-control-repeat` | 0.8966 | +0.0068 | +0.0027 | +0.0015 |
| `shared-shift-projected-bce` | 0.8925 | +0.0060 | +0.0026 | +0.0016 |
| `balanced-bce` | 0.8953 | +0.0063 | +0.0025 | +0.0014 |
| Rejected losses | 0.8847–0.8922 | +0.0015 to +0.0047 | −0.0004 to +0.0014 | −0.0007 to +0.0005 |

`shared-shift-projected-bce` scores below the control on its own. It still ranks in the top
six because it is the least correlated member: mean Spearman 0.895 against the others,
range 0.883–0.907. It appears in every best subset from size 3 to 7. The remaining
inconclusive losses (`balanced-brier`, `balanced-focal`, `centroid-margin`,
`draw-consistency-anchor`) behave like an extra seed.

Best subsets on the full fold. These are selected on the same predictions they are scored on:

| Size | Macro AUC | Members |
|---:|---:|---|
| 2 | 0.902293 | `balanced-bce`, `bce-ranking` |
| 3 | 0.903445 | `13-control-repeat`, `bce-ranking`, `shift-projected-bce` |
| 4 | 0.904285 | size 3 + `boundary-proximity-weighting` |
| 5 | 0.904600 | size 4 + `balanced-bce` |
| 7 (greedy stop) | 0.904958 | size 5 + `evidence-margin-bce`, `target-logit-margin` |
| 17 (all) | 0.903375 | every run |

## Selection optimism

For the split-half check, report groups were split into random halves 200 times. Each subset
rule chose its members on one half and was scored on the other:

| Strategy | Held-out macro AUC | Optimism | vs seed trio | Mean size |
|---|---:|---:|---:|---:|
| Single control | 0.896587 | — | −0.0043 | 1 |
| Best single | 0.894896 | +0.0031 | −0.0060 | 1 |
| Seed trio | 0.900905 | — | 0 | 3 |
| Top 3 singles | 0.901516 | +0.0020 | +0.0006 | 3 |
| Top 5 singles | 0.903129 | +0.0012 | +0.0022 | 5 |
| Best subset ≤ 5 | 0.903186 | +0.0017 | +0.0023 | 5.0 |
| Greedy forward | 0.903479 | +0.0015 | +0.0026 | 6.3 |
| All 17 runs | 0.903298 | — | +0.0024 | 17 |
| Non-rejected 12 | 0.904523 | — | +0.0036 | 12 |

In the table above:

- Best-subset search on one half overstates its score by 0.0015–0.0017 and then ties the plain
  all-run average. The same-fold "best subset" scores in the previous section carry this
  optimism.
- The non-rejected set scores best of the fixed rules, but it was chosen using fold-0
  decisions, so its advantage is optimistic by an unknown amount.
- The strategies differ in size, so only rows of the same size compare loss diversity with seed
  diversity.

Paired report-group bootstrap intervals are in `summary.json`, relative to the anchor and to
the seed trio. They ignore selection.

## Implications

- The losses are not in main. They exist only as campaign diffs and snapshots. The recipe has
  since moved to learned strength, so any member would need re-implementing and retraining.
- For the [second-round ensemble](../../backlog/endgame.md#ensembling), plain seed averaging
  gives most of the gain. On this evidence, the loss variants worth retraining for extra
  diversity are `shared-bce-ranking` and `shared-shift-projected-bce`, followed by
  `boundary-proximity-weighting`, `target-logit-margin` or `evidence-margin-bce`. The expected
  extra gain over seeds is about +0.001 per trio.
- On held-out halves, searched subsets did no better than averaging every run. Choosing
  members by search on fold-0 predictions is therefore not supported. Selection on full
  out-of-fold predictions has not been tested.
