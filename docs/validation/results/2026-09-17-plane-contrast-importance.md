# 2026-09-17 plane/contrast importance

**Fluid-sensitive series and axial/coronal planes carry more than their token share; single-plane
or single-contrast checkpoints lose 0.03–0.07 macro AUC.** Leave-one-type-out and keep-one removal
on studies that have the type, averaged over the three 2026-09-17 control checkpoints
(`00-control`, `01-control-seed-43`, `02-control-seed-44`,
[overnight results](2026-09-17-overnight-results.md)), fold-0 validation, series dropout 0.2 as
trained. Scripts: [snapshots](../../history/series-and-sampling-sources/README.md#type-importance) (`infer.py`, `analyze.py`,
`capacity.py`, `numerics.py`); outputs: `outputs/analysis/type-importance-20260917/` (`results.json`,
`deltas.csv`, `label_type_importance.csv`, `attention_share.csv`, `availability_vs_prevalence.csv`,
`intact_auc_by_type_presence.csv`, `validation_triplet_capacity*.csv`).

## Leave-one-type-out (macro AUC ×1000, "all_defined" studies)

| Type removed | Studies | Δ macro ×1000 | 95% CI ×1000 | Sign agreement (of 3) |
|---|---:|---:|---|---:|
| Axial T1/PD | 179 | +1.1 | [−1.7, +3.8] | 2 |
| Axial fluid | 904 | −6.4 | [−8.8, −4.0] | 3 |
| Coronal T1/PD | 701 | −3.3 | [−4.7, −1.8] | 3 |
| Coronal fluid | 877 | −7.6 | [−9.7, −5.7] | 3 |
| Sagittal T1/PD | 865 | −2.7 | [−4.0, −1.4] | 3 |
| Sagittal fluid | 842 | −4.2 | [−6.0, −2.4] | 3 |

Axial T1/PD is the only type without full sign agreement and the only tie with zero; it is also
the rarest type (179 of 904 studies). Contusion is the one target with a positive, CI-excluding-zero
removal effect for axial T1/PD (+0.0093).

## Keep-one-plane / keep-one-contrast (macro AUC ×1000, "all_defined")

| Kept subset | Δ macro ×1000 |
|---|---:|
| Axial only | −67 |
| Coronal only | −33 |
| Sagittal only | −33 |
| T1/PD only | −62 |
| Fluid only | −6 |

Fluid-only is far less damaging than T1/PD-only or any single-plane-only checkpoint: fluid series
alone recover most of the macro score, consistent with fluid types' larger per-type removal
effects above.

## Attention already over-weights fluid types

Mean attention-share ÷ token-share ("enrichment") across all targets and all three checkpoints,
studies with all six types present (`attention_share.csv`, group `all_six_types`, n=48 each):

| Type | Mean enrichment |
|---|---:|
| Axial T1/PD | 0.63 |
| Axial fluid | 1.18 |
| Coronal T1/PD | 0.73 |
| Coronal fluid | 1.17 |
| Sagittal T1/PD | 0.86 |
| Sagittal fluid | 1.46 |

Every fluid-sensitive type is enriched (>1) and every T1/PD type is depleted (<1) relative to its
token share, before any acquisition-count normalization. This bears on the acquisition-count-normalized
attention proposal in [small changes](../../backlog/small-changes.md#attention-pooling): the model
already spends more attention per token on fluid series than its budget alone would predict.

## Availability and prevalence shift

Across the 4,407 studies used for the prevalence/availability check (`availability_patterns.csv`,
`availability_vs_prevalence.csv`), 566 have all six types, 2,516 have all but axial T1/PD, and 745
lack both axial and coronal T1/PD. Studies with axial T1/PD differ in target prevalence from those
without:

| Target | With axial T1/PD | Without | 
|---|---:|---:|
| ACL | 26.3% | 19.5% |
| Synovitis | 18.4% | 11.5% |
| Contusion | 23.0% | 16.0% |
| Baker's | 16.0% | 26.7% |

and, on the 904-study fold-0 validation split, in intact-model macro AUC: 0.8855 (with axial
T1/PD, n=179) vs 0.8920 (without, n=725). Axial
T1/PD availability is not missing at random, so its small/mixed-sign removal effect is confounded
with an easier study population, not evidence that the type itself is unimportant. MCL and lateral
meniscus prevalence also shift with availability (`availability_vs_prevalence.csv`); not tabulated
here.

## Proposed per-type budgets

Two candidate `max_images` budgets against the current 18/18/15/15/15/15 (axial T1/PD, axial fluid,
coronal T1/PD, coronal fluid, sagittal T1/PD, sagittal fluid):

| Budget | Ax T1/PD | Ax fluid | Cor T1/PD | Cor fluid | Sag T1/PD | Sag fluid | Total (all 6 types) | Total (5-type study, no ax T1/PD) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 9 | 18 | 15 | 21 | 15 | 18 | 96 | 87 |
| B | 9 | 18 | 12 | 21 | 12 | 15 | 87 | 78 |

Both shrink axial T1/PD (rarest, weakest evidence) and grow fluid types (largest removal effects,
already over-attended); B also shrinks coronal/sagittal T1/PD. Series capacity supports both:
`validation_triplet_capacity_summary.csv` shows the p10 capacity per type is 16–26 triplets at the
current budgets, well above either proposal, with 0% of slots short.

**Missing-type fallback**, when a study lacks a type its budget would otherwise fill: use the same
plane's other contrast first, then coronal fluid, then axial fluid, then sagittal fluid, in that
order, until a series runs out of positions to fill.

## Caveats

- Removal on a trained checkpoint measures reliance under the current budget and series dropout
  0.2; it is not the marginal value of adding slices to that type.
- These checkpoints have no reallocation on series dropout: a dropped series' images are not
  redistributed to other types, unlike the proposed budget reallocation.
- Keep-one-plane/contrast checkpoints evaluate a model trained on all six types with five types
  zeroed out; this is off-distribution and likely overstates the loss a model trained on the
  smaller budget would show.
- Removal happens after preparation: the study's reference geometry (used to place triplets) still
  used every series, so these numbers do not capture any geometry shift from a genuinely smaller
  acquisition set.
