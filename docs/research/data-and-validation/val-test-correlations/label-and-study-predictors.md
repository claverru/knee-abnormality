# Which labels and studies predict public performance

## Do individual labels predict public performance?

| Label | Pearson, all 19 | Pearson, plateau 14 | Plateau, mean of last three epoch AUCs |
|---|---:|---:|---:|
| ACL | +0.812 | +0.114 | −0.214 |
| MCL | +0.760 | +0.263 | +0.334 |
| Medial meniscus | +0.424 | −0.551 | −0.327 |
| Lateral meniscus | +0.831 | −0.014 | +0.193 |
| Medial OA | +0.719 | +0.053 | +0.473 |
| Lateral OA | −0.015 | −0.120 | +0.333 |
| PF OA | +0.385 | −0.230 | −0.198 |
| Effusion | +0.128 | +0.168 | −0.008 |
| Synovitis | −0.170 | −0.162 | −0.072 |
| Baker's cyst | +0.793 | +0.708 | +0.195 |
| Contusion | +0.702 | −0.618 | −0.475 |
| Fracture | +0.345 | +0.314 | +0.666 |

No label is a defensible standalone selector. Baker's has the strongest selected-epoch plateau
association, but its permutation p-value becomes 0.055 across twelve labels and 0.087 across
selected/last-three variants. The champion's Baker's AUC spikes to 0.922952 at epoch 7, then falls
near 0.908–0.910; five positive cases account for 0.01429 of its 0.01519 advantage over the Sep 17
control. Their Baker's and Contusion differences largely cancel in macro despite public 0.936 vs 0.928.

Across eight historical replicate runs, pooled within-recipe macro SD is 0.00175 versus only
0.00140 between-model SD in the plateau. The eleven-study gold subset has no stable selection
signal. Negative label correlations do **not** imply that deliberately worsening a label helps:
the cached inference intervention below improves Contusion while the corresponding submission improves.

## Hard, large, ambiguous and acquisition-defined studies

Hardness is fixed before comparing submitted models: average predictions from three unsubmitted
anchors, then per study average the fraction of opposite-class examples ranked incorrectly for
each label. This defines label-relative difficulty, not independently assessed clinical difficulty.

| Fixed subgroup | Studies | Mean plateau macro AUC | Pearson to public, plateau |
|---|---:|---:|---:|
| Images ≤163 / >163 | 454 / 450 | 0.8909 / 0.8902 | +0.246 / −0.234 |
| Series ≤5 / >5 | 595 / 309 | 0.8874 / 0.8951 | +0.502 / −0.464 |
| No reference in [0.4,0.6] / at least one | 628 / 276 | 0.8951 / 0.8734 | +0.432 / −0.348 |
| Easier half / harder half | 452 / 452 | 0.9585 / 0.8117 | −0.194 / +0.291 |
| Hardest quarter | 226 | 0.7661 | +0.296 |

Hard studies are not larger: median images 160 versus 164 in the easy half. Difficulty's Spearman
association is −0.082 with image count, +0.056 with series count, +0.425 with positive-label burden
and +0.475 with soft-label entropy. Hard studies average 3.88 positive labels versus 2.45;
37.6% versus 23.5% contain a reference in [0.4,0.6]. Those label-derived features partly share
the difficulty definition's source and must not be mistaken for independent disease measurements.

Control-B improves hard-half AUC by 0.02288 versus 0.00413 on the easy half; across five anchor
definitions the corresponding ranges are 0.0178–0.0234 and 0.0032–0.0062. **Improving hard studies
alone is therefore not a transfer guarantee.** Hardness mixes difficult images and uncertain references.

The ≤5-series minus >5-series AUC gap looks promising retrospectively: plateau r=0.663,
unadjusted p=0.0093, leave-one-model-out r=0.484–0.721. It falls to 0.417 after removing the
champion and 1B, and to 0.150 in the newer five. Few-series Fracture AUC gives plateau r=0.767
but newer-five r=−0.699. These fail as stable selectors.

A 160-diagnostic family covered 46 metadata, coverage, label and report features, median groups,
contrasts, difficulty quartiles and per-label series splits. Across 50,000 public-score permutations,
maximum-absolute-correlation adjustment gives p=0.1213 for the strongest few-series Fracture result
and p=0.5149 for the series-gap result. None survives this family; the wider adaptive exploration
adds further uncertainty. Small subgroups can also contain very few positives for rare labels.

Other useful dataset properties and negative checks:

- Fold 0 contains 64 studies in twelve repeated-report groups; 50 are all-negative across twelve
  frozen labels. They average 0.438 positives versus 3.371 for the other 840 studies and have
  shorter reports (median 205 versus 1,011 characters). Across all folds there are 54 repeated
  groups containing 204 studies, largest 37. These are repeated normalized reports, not proof of
  duplicate images. Existing split grouping keeps each report group together.
- Giving each report group total weight one lowers plateau mean AUC by about 0.0055 to 0.885817,
  but public correlation only moves 0.072 → 0.129. Repeated easy negatives inflate this endpoint
  without explaining the leaderboard order.
- Metadata-only predictions, estimated on other folds without image inputs, achieve fold-0 macro
  AUC 0.5748 from vendor, 0.5863 from vendor/field strength and 0.6175 from scanner model.
  Scanner-model ACL reaches 0.6839. This reveals cohort/report-label structure, not scanner causality;
  see [vendor-label associations](../../../metadata/vendor-label-associations.md). Equal training-group
  means use soft labels and fall back to the training global mean for groups with fewer than ten rows.
- Within-vendor and between-vendor pair AUCs have plateau/public correlations 0.077 and 0.066.
  Excluding 31 poor-geometry studies gives 0.085; excluding 40 studies with crop offsets >10 mm
  gives 0.042. Removing the hardest 10%, carrying 24.4% of ranking errors, gives 0.037.
- A generic average of twelve model outputs attains mean target AUC 0.7605; averaging the other
  eleven attains 0.7227. Broad abnormality burden is strong locally, but its plateau/public
  correlation is −0.253. Neither burden nor the checked geometry exclusions rescues selection.

## An intervention supported by both cache predictions and submissions

On fixed ranked-filtered seed-42 weights, changing ×2/τ1 to ×4/τ0.75 moved public **0.933 → 0.939**
([records](../../../validation/results/pooling-and-attention/2026-09-18-temperature-and-count-submissions.md)). Re-pooling
existing token scores/contributions with NumPy, averaged across ranked-filtered seeds 42–44:

| Fixed validation subset | ×2, τ1 | ×2, τ0.75 | ×4, τ1 | ×4, τ0.75 |
|---|---:|---:|---:|---:|
| All 904 | 0.890726 | 0.893040 | 0.892167 | 0.895374 |
| Easier half | 0.958827 | 0.958058 | 0.960025 | 0.960160 |
| Harder half | 0.809921 | 0.815849 | 0.811230 | 0.818012 |
| ≤5 series | 0.886632 | 0.889094 | 0.888440 | 0.892054 |
| >5 series | 0.894782 | 0.896983 | 0.895360 | 0.898266 |

The combined gain is +0.004648 overall, +0.008091 hard and +0.001333 easy. Largest per-label
gains are Contusion +0.013394, Lateral Meniscus +0.012768 and Baker's +0.012348; MCL, PF OA
and Effusion slightly worsen. This demonstrates useful ranking changes despite some negative
retrospective label correlations. Cache preprocessing uses an approximate whole-series window;
seed-42 baseline differs from its actual validation by about 0.00012. These are cache estimates,
not newly executed full inference. The cache does not cover control-B or the newer ordinary ReXNet.

Existing inference/source checks found no evidence that precision/runtime variants explain the
sigma40 score: they all reported 0.936. Newer ordinary/control-B evaluation sampling is deterministic
and unseeded; training seed does not change their evaluation selection. That statement should not
be projected onto older snapshots with a different sampler identity. Export metadata does not
indicate double application of τ for the newer bundles; checkpoint tensors were not loaded here.

Ensemble evidence is stronger than a new retrospective metric: B0 single 0.934 → five folds
0.940 → TTA 0.941, and mixed ten-model TTA 0.943
([results](../../../validation/results/validation-and-ensembles/2026-09-19-ensemble-submissions.md)). Saved seed predictions
have rank correlations near 0.95 yet averaging gains about 0.0034–0.0035 over their mean single
AUC (agreement artifacts (`outputs/analysis/p/prediction-agreement-20260917/summary.json`, deleted)).
