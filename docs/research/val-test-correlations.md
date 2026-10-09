# Validation–Kaggle correlations and dataset diagnostics

Measured 2026-09-20 from saved artifacts, using CPU only. Investigation was read-only; this report
and its index entry are the only requested documentation changes. No training, image inference,
checkpoint modification, submission, or test suite was run.

## Findings and decisions

**The strongest new finding is that a large validation gain can concentrate on report-derived
negatives whose evidence is weak or unaddressed.** Ordinary ReXNet → balanced-BCE control-B gains
0.012911 validation macro AUC but moves public 0.936 → 0.933. Of that local gain, 92.8% involves
negative references between 0.1 and 0.5; 80.8% comes from addressed positives beating negatives
that Steven's original extractor marked unaddressed. These are two overlapping decompositions.

This is a plausible explanation for poor transfer, not proof of hidden-test errors or a causal
loss effect: seeds differ, validation uses fewer images than submission, and Kaggle exposes only
aggregate scores. We cannot identify which hidden-test label actually raised or lowered a score.
The older plateau also has a separate resolution problem: model differences are smaller than
measured run-to-run noise. Neither explanation alone covers every submission.

Practical actions, in priority order; these are recommendations, not implemented changes:

1. **Add a label-evidence audit alongside full-fold AUC.** Preserve original report coverage,
   label-source agreement and extreme-value diagnostics separately. Require a prospective gain
   to be inspected for dependence on unaddressed negatives. Keep full-fold AUC as the primary
   recorded metric; none of the alternative endpoints below is validated as a replacement.
2. **Resolve balanced BCE with matched comparisons.** Use ordinary/balanced pairs with the same
   source snapshot, seeds, exposure, optimizer budget and inference settings, replicated across
   seeds or folds. Existing repeated runs support endpoint sensitivity, but do not isolate cause.
3. **Validate finalists at submission settings.** Compare the same checkpoint at the intended
   image budget and attention temperature before attributing a public change to training.
   Existing token caches permit a useful CPU approximation for their three compatible models.
4. **Audit labels with the official image rubric.** Prioritize unaddressed negatives, contusion,
   MCL, Baker's cyst and fracture; distinguish report silence, explicit denial and severity.
   More independent image-adjudicated validation would be more informative than selecting a
   metric against this small submission history. Keep the current eleven gold cases diagnostic.
5. **Retain diversity and uncertainty in selection.** Inspect neighboring epochs and repeated
   runs; evaluate complementary predictions in ensembles. Do not select a label, subgroup or
   loss weight merely because its retrospective leaderboard correlation looks favorable.

Blanket removal of unaddressed training labels was already
[tried and rejected](../history/sharing/2026-09-13-label-masking-result.md), costing 0.011455 local
AUC. The finding here motivates better diagnostics and controlled experiments, not that retry.

## Cohorts and checkpoint identity

All new calculations align predictions by `StudyInstanceUID` to fold 0 of `all_5fold_v3`:
**904 studies, 893 report-derived references and 11 official gold studies**, twelve targets.
Canonical AUC binarizes frozen targets at `q >= 0.5`. Checkpoints are selected by maximum saved
epoch macro AUC; alternative diagnostics do not reselect the checkpoint.

| Cohort | Models | Validation / submission | Macro AUC vs public, Pearson / Spearman |
|---|---:|---|---:|
| Historical comparable | 19 | ×2 / ×2, τ=1 | +0.907 / +0.620 |
| Historical plateau, submitted from Sep 13 | 14 of 19 | ×2 / ×2, τ=1 | +0.072 / +0.062 |
| Newer single-model set | 5 | ×1 / ×4, τ=0.75 | −0.672 / not used for selection |

The historical membership and exclusions are in the
[score manifest](../../scripts/adhoc/submission_public_scores.json) and
[earlier comparison](../validation/results/2026-09-18-local-public-metric-comparison.md).
Do not pool these inference regimes into one correlation. The newer five are a small descriptive
comparison, not a held-out statistical validation set.

| Newer model | Seed | Completed epoch | Saved validation AUC | Public | Submission record |
|---|---:|---:|---:|---:|---|
| Robust center 20, top-up 26 | 3167394480 | 10 | 0.888021 | 0.938 | [record](../../outputs/submission/records/knee-rc20-topup26-tau075-4x-20260918-v2.json) |
| EfficientNet-B0 fold 0 | 886913563 | 11 | 0.886354 | 0.934 | [record](../../outputs/submission/records/knee-b0-fold0-tau075-4x-20260919.json) |
| Ordinary ReXNet fold 0 | 471048644 | 10 | 0.883671 | 0.936 | [record](../../outputs/submission/records/knee-rexnet-fold0-tau075-4x-20260919.json) |
| Balanced control-B | 2263738815 | 9 | 0.896582 | 0.933 | [record](../../outputs/submission/records/control-b-20260920-x4-tau075.json) |
| Balanced, stratified sampling | 4061686476 | 10 | 0.897515 | 0.933 | [record](../../outputs/submission/records/stratified-sampling-20260920-x4-tau075.json) |

Records resolve to export manifests and exact source checkpoint paths. Completed epochs above
are one-based; prediction filenames/checkpoint epochs are zero-based. Control-B means
`overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat`, **with balancing enabled**.
The seed-42 `balanced-bce` run is a separate model; its failed submission attempts have no score.

## Where the balanced validation gain comes from

For each label, AUC averages correctly ordered positive–negative pairs, with half credit for ties.
Partitioning those pairs gives an exact accounting of the ordinary → control-B macro difference.
Here, “extreme” describes the stored reference value, not independently established confidence.

| Positive reference | Negative reference | Share of macro pair weight | Contribution to macro gain | Share of total gain |
|---|---|---:|---:|---:|
| q ≥ 0.9 | q ≤ 0.1 | 51.09% | −0.000020 | −0.2% |
| q ≥ 0.9 | 0.1 < q < 0.5 | 31.80% | +0.008068 | 62.5% |
| 0.5 ≤ q < 0.9 | q ≤ 0.1 | 11.76% | +0.000952 | 7.4% |
| 0.5 ≤ q < 0.9 | 0.1 < q < 0.5 | 5.35% | +0.003911 | 30.3% |

Pairs with extreme references on both sides contribute essentially no aggregate improvement.
Separately normalizing AUC within extreme cells gives ordinary **0.930556**, control-B
**0.930325**, stratified **0.933304**; these normalized scores use different label weights from
the contribution table and should not be substituted into its accounting.

The original Steven v1 table provides a more interpretable indicator: `q_v1 == 0.5` means the
report extractor marked that finding unaddressed. Gold cells are always treated as addressed.

| Original report coverage of pair | Share of macro pair weight | Contribution to macro gain | Share of total gain |
|---|---:|---:|---:|
| Both addressed | 67.30% | +0.002488 | 19.3% |
| Addressed positive, unaddressed negative | 32.47% | +0.010427 | 80.8% |
| Unaddressed positive, addressed negative | 0.17% | −0.000060 | −0.5% |
| Both unaddressed | 0.05% | +0.000056 | 0.4% |

Requiring both addressed evidence and v1/v4 binary agreement retains 7,792 of 10,848 cells.
Pairs where both cells qualify carry 62.55% of pair weight but only **8.1% of the gain**;
qualifying positives against other negatives account for 90.6%. This condition is a source
consistency check, not a clinical correctness guarantee.

### Per-label AUC and sensitivity to reference strength

The full-fold gain spans eleven labels, yet several gains collapse or reverse on extreme cells.
This is more informative than the full-fold macro alone. Positives refer to the full 904 studies.

| Label | Positives | Ordinary AUC | Control-B AUC | Full-fold Δ | Extreme-cell Δ |
|---|---:|---:|---:|---:|---:|
| ACL | 184 | 0.907428 | 0.917806 | +0.010379 | −0.002023 |
| MCL | 134 | 0.832099 | 0.862192 | +0.030093 | +0.001762 |
| Medial meniscus | 356 | 0.915120 | 0.926587 | +0.011467 | +0.004332 |
| Lateral meniscus | 138 | 0.873557 | 0.897803 | +0.024246 | +0.014570 |
| Medial OA | 329 | 0.898369 | 0.907763 | +0.009393 | +0.001675 |
| Lateral OA | 241 | 0.855010 | 0.868603 | +0.013593 | −0.003544 |
| PF OA | 406 | 0.868973 | 0.869275 | +0.000302 | −0.007978 |
| Effusion | 525 | 0.869235 | 0.864832 | −0.004403 | −0.002441 |
| Synovitis | 115 | 0.896148 | 0.899631 | +0.003483 | −0.000644 |
| Baker's cyst | 216 | 0.885518 | 0.913874 | +0.028356 | +0.004023 |
| Contusion | 153 | 0.883336 | 0.895521 | +0.012184 | −0.011955 |
| Fracture | 63 | 0.919257 | 0.935092 | +0.015835 | −0.000547 |

Among frozen negatives, original v1 marked 96.2% of Synovitis, 66.3% of Baker's and 44.4% of
Fracture cells unaddressed. Restricting to addressed cells leaves only 30 Synovitis negatives,
so its restricted AUC is particularly fragile. Control-B also raises mean prediction
0.328378 → 0.390193 and worsens soft BCE 0.453629 → 0.485663. This is not merely calibration:
a monotonic per-label probability shift cannot change AUC, whereas these rankings changed.

### Alternative label sources and repeated runs

Each endpoint below uses the same selected predictions, official gold overrides and an equal
mean over twelve label AUCs. Masks are per cell, not a single common study subset.

| Endpoint | Ordinary | Control-B | Stratified | Pearson to public, newer five |
|---|---:|---:|---:|---:|
| Frozen v4, all cells | 0.883671 | 0.896582 | 0.897515 | −0.672 |
| Frozen v4, v1-addressed cells | 0.906905 | 0.909593 | 0.910632 | −0.746 |
| Frozen v4, addressed and v1-agreeing | 0.916119 | 0.917182 | 0.918587 | — |
| Frozen v4, extreme cells | 0.930556 | 0.930325 | 0.933304 | −0.624 |
| Original Steven v1 labels, addressed cells | 0.904597 | 0.901975 | 0.904745 | +0.645 |
| Pilkwang labels, verdict other than UNK | 0.898216 | 0.895757 | 0.897696 | +0.282 |
| TonyLin soft labels, all cells | 0.879398 | 0.891273 | 0.892957 | −0.656 |
| Lixin labels, all cells | 0.884878 | 0.897745 | 0.899472 | −0.691 |

Changing the reference labels is different from masking frozen labels: masking alone does not
restore public ordering. Steven v1 addressed and Pilkwang known endpoints reverse the loss
comparison, but are still report-derived and were explored after seeing public results.
Lixin is an ingredient of v4, so agreement there is not independent corroboration.

A broader saved-run check compares two ordinary runs with four balanced runs of the same
numerical recipe, allowing seed and the balancing flag to differ. The six selected epochs are:

| Run beneath `outputs/training/` | Seed | Completed epoch | Full AUC |
|---|---:|---:|---:|
| `sigma-fraction-0867` | 42 | 6 | 0.886486 |
| `overnight-20260918T232636Z-mt7ouybk/runs/rexnet-fold0` | 471048644 | 10 | 0.883671 |
| `balanced-bce` | 42 | 8 | 0.895282 |
| `overnight-20260919T120130Z-i2oqlnls/runs/00-control` | 927831896 | 9 | 0.894037 |
| `overnight-20260919T235434Z-8ol_1c30/runs/00-control` | 1133376617 | 7 | 0.893222 |
| `overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat` | 2263738815 | 9 | 0.896582 |

Balanced-minus-ordinary group means: **+0.009702 full AUC**, +0.000718 extreme-cell AUC,
+0.000768 addressed-and-agreeing AUC, **−0.001834 v1-addressed-label AUC** and
**−0.002491 Pilkwang-known-label AUC**. Thus endpoint sensitivity is not unique to control-B.
These are descriptive group means, not paired effect estimates. The seed-42 references have
different preprocessing digests; no exact same-source, same-seed ordinary control was found.
See the [source limitations](../validation/results/2026-09-19-balanced-bce-result.md#existing-references-checked-2026-09-19).

### Why the coverage distinction matters

[The label audit](../public-labels/uncertainty-and-confidence-encoding.md) establishes that v4 is
half Steven v2 and half Lixin. Of 13,438 original unaddressed cells across all 4,407 studies,
10,085 become 0.25. Steven v2 also infers unaddressed Synovitis from Effusion. Distance from 0.5
therefore does not recover report coverage or calibrated confidence. Pilkwang's UNK is broader,
including unmentioned and explicitly uncertain findings; its scores are described as ranking scores.

The [archived host criteria](../metadata/official-label-criteria.md) describe image-based
adjudication, confirmed absence for negatives and severity thresholds. Better ranking against a
report-derived negative can therefore fail to improve image-defined discrimination. That is the
mechanistic hypothesis supported by this audit; identifying actual mislabels needs image review.
Moving the binary threshold alone did not solve the issue: at thresholds 0.3–0.9 the balanced
advantage persisted and the newer-five macro/public correlation remained negative (about −0.49 to −0.69).

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
  see [vendor-label associations](../metadata/vendor-label-associations.md). Equal training-group
  means use soft labels and fall back to the training global mean for groups with fewer than ten rows.
- Within-vendor and between-vendor pair AUCs have plateau/public correlations 0.077 and 0.066.
  Excluding 31 poor-geometry studies gives 0.085; excluding 40 studies with crop offsets >10 mm
  gives 0.042. Removing the hardest 10%, carrying 24.4% of ranking errors, gives 0.037.
- A generic average of twelve model outputs attains mean target AUC 0.7605; averaging the other
  eleven attains 0.7227. Broad abnormality burden is strong locally, but its plateau/public
  correlation is −0.253. Neither burden nor the checked geometry exclusions rescues selection.

## An intervention supported by both cache predictions and submissions

On fixed ranked-filtered seed-42 weights, changing ×2/τ1 to ×4/τ0.75 moved public **0.933 → 0.939**
([records](../inference/submissions/2026-09-18-temperature-and-count-submissions.md)). Re-pooling
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
([results](../inference/submissions/2026-09-19-ensemble-submissions.md)). Saved seed predictions
have rank correlations near 0.95 yet averaging gains about 0.0034–0.0035 over their mean single
AUC ([agreement artifacts](../../outputs/analysis/p/prediction-agreement-20260917/summary.json)).

## Reproduction and limits

Data and calculations: NumPy/Polars/SciPy/scikit-learn, `CUDA_VISIBLE_DEVICES=''`, no torch/model
execution. Join frozen references and per-epoch CSVs by ID, assert identical 904-study membership,
then compute per-label ROC AUC and its unweighted mean. Pair-block contribution is
`mean_j[(P_block,j * N_block,j)/(P_j * N_j) * (AUC_new,block,j - AUC_old,block,j)]`.
Coverage-based blocks use the original v1 indicator; alternative-source scores replace gold cells
with frozen official labels. Subgroup scores retain all twelve labels and require both classes.

Fixed difficulty anchors are selected predictions from `ranked-filtered-seed43`,
`ranked-filtered-seed44`, and `min_lr_01`. Difficulty is the average fraction of opposite-class
rank errors per label, half credit for ties; its median is approximately 0.0997. Subgroup
permutations shuffle the same 14 public scores jointly across all diagnostics, seed 20260920.
Cached pooling uses attention `softmax(score / τ)`, weighted saved contributions, saved readout
bias and sigmoid; per-type ×2 budgets are (12,12,10,10,10,10), doubled for ×4 and capped by availability.

Primary artifact locations, relative to repository root:

- `data/folds/all_5fold_v3/{studies.csv,targets.csv,protocol.json}`; SHA-256 respectively
  `c709cd27ff54dd10d833abfa966622f9a5ecadef5d37670da658d81c26f7b3ed`,
  `0e48fbc11f2824db2d04c83a55f8a2d0067f58ebd43cf4142d8166c2ace91a7c`,
  `facd4494eeab7d5ff58f0f5b5440a634256bc64c12f99175e02f27e3cf6b6aab`.
- Historical runs and scores: `scripts/adhoc/submission_public_scores.json`, SHA-256
  `cc2137685e10632fd3c5a57f4cf7b8b5b10cb5f31156114f240367fc1fdb9cc3`.
  Every run supplies `effective_config.json`, `fold_0/epochs/metrics_epoch_*.json` and aligned
  `validation_epoch_*.csv`; use saved source snapshots/input identities rather than current source.
- Main ordinary CSV: `outputs/training/overnight-20260918T232636Z-mt7ouybk/runs/rexnet-fold0/fold_0/epochs/validation_epoch_009.csv`,
  SHA-256 `c34a26ba28bbca1b770f363b658aa1075901f4d9790f0707a8f45833f0d654a8`.
  Control-B CSV: `outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat/fold_0/epochs/validation_epoch_008.csv`,
  SHA-256 `c523e3397cdd070446df68be2b122b377108bf53dd8a07562236181768a4652b`.
- Alternative labels: `data/labels/public/stevenleehans/version-6/llm_labels_full.csv`,
  `data/labels/public/pilkwang/version-1/report_labels_v2.csv`,
  `data/labels/public/lixin73/version-1/labels_llm_gpt56sol.csv`,
  `data/labels/prepared/tonylin-soft-v3/labels.csv`; source identities in
  [public-label configuration](../../configs/labels/public-label-sources.json).
- Study headers: `outputs/analysis/s/sharing-disagreement-20260914/study_headers.parquet`;
  counts: `data/metadata/train_series_image_counts.parquet`; geometry/slabs:
  `outputs/analysis/p/peripheral-coverage-20260918/`; tokens/selections/series:
  `outputs/analysis/t/token-cache-20260917/`. Headers represent one series per study, which limits
  scanner/geometry interpretations. Report summaries use `data/raw/train.csv`; no raw text is reproduced.

Public scores are rounded to three decimals; models, checkpoints, labels and diagnostics were
adaptively selected and are not independent observations. The newer cohort has only five models.
Masked endpoints alter the case mix and rare-label sample sizes; alternative report labels are
not independent image truth. Downloaded notebook submission CSVs contain only three example
studies, not hidden-test predictions, so neither hidden-test calibration nor per-label attribution
can be measured. No private-leaderboard behavior is established. The useful result is a concrete
local failure mode to test prospectively, plus evidence against replacing validation with a
retrospectively favorable label or study subset.
