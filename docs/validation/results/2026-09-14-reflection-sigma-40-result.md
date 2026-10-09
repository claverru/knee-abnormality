# 2026-09-14 reflection sigma-40 result

`outputs/training/reflection-sigma40` completed ten epochs and selected **epoch 7
(index 6), 0.892360168 full-fold AUC**. Tightening reflection's sampling from
80 to 40 mm improves AUC by **0.001384348**. It remains **0.000177938 below
1C with sigma 40**, with worse soft-target BCE. Its subsequent
[public submission](../../inference/submissions/2026-09-14-reflection-sigma-40-submission.md)
**56230573 completed at 0.936**, beating both reflection sigma 80 and 1C sigma
40 by **0.005**. Reflection sigma 40 is the best submitted model and the
recommended next development baseline. This supersedes the pre-submission
recommendation to keep 1C sigma 40 as the provisional champion. At promotion
commit `ea19772`, `configs/baseline.yaml` adopted this sigma-40 recipe;
the run artifacts preserve its exact original configuration and source.

The subsequent `meniscus-soft-coupling` experiment regressed and its training
changes were removed. The working training recipe again matches `ea19772`;
its [historical objective and diagnostics](../../modeling/retired-side-and-compartment-variants.md#meniscus-classifier-coupling)
remain separate from this champion result.

During promotion on 2026-09-14, loading both configurations through the current
Pydantic schema and resolving paths produced identical effective settings.
The CoAtNet source addition was reviewed statically; it did not produce the
recorded checkpoint or a new validation score.

| Model | Sigma, mm | Selected epoch | Validation AUC | Report-derived AUC | Gold AUC | BCE, lower is better | Public AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reflection | 80 | 9 | 0.890976 | 0.890228 | 0.922784 | 0.449310 | 0.931 |
| Reflection | 40 | 7 | 0.892360 | 0.891725 | 0.922156 | 0.451855 | **0.936** |
| 1C | 80 | 10 | 0.889430 | 0.888592 | 0.937368 | 0.448458 | 0.928 |
| 1C | 40 | 10 | **0.892538** | **0.891933** | 0.917229 | **0.448244** | 0.931 |

All rows use the same 904 validation studies, including 893 report-derived
and eleven gold studies. Public values come from the completed local submission
records, at their reported precision. The gold endpoint is diagnostic only.

Public AUC now favors the independent reflection heads at both widths:
**0.931 versus 0.928 at sigma 80**, and **0.936 versus 0.931 at sigma 40**.
Tightening sigma improves public AUC by **0.005 for reflection** and **0.003
for 1C**. At sigma 40, the local AUC lead of 0.000178 and better BCE favor
1C, while the public result favors reflection by 0.005. Neither aggregate
local metric resolved this architecture choice correctly on the public subset.

This supports returning to independent heads for subsequent development.
It does not establish that classifier sharing, compartment attention or a
particular target caused the loss: the two architectures change both sharing
and attention, hidden target scores are unavailable, and the public subset
contains only 30% of the test set. The result does not resolve private-set
performance or prove the OA-label/acquisition hypotheses. The submitted
inference schedules also differ; both use FP32 and 2× images, but these were
not paired runtime-parity experiments.

## What the completed comparison establishes

Sigma 40 improves the observed macro AUC for both architectures: **+0.001384**
for reflection and **+0.003109** for 1C. The benefit is therefore not unique to
1C. At sigma 80, 1C trails reflection by **0.001546**; at sigma 40, 1C leads by
**0.000178**. The difference between the architectures' measured sigma gains is
**+0.001724** in favor of 1C. This is the four-cell, single-seed comparison of
the selected checkpoints; it does not isolate classifier sharing from 1C's
attention factorization or establish a repeatable architecture interaction.
Sigma governs triplet selection during both training and validation, so this
compares complete sampling recipes rather than isolating a training-only effect.

Reflection sigma 40 improves six targets and worsens six versus reflection
sigma 80. The largest gains are Baker's **+0.014811**, fracture **+0.007682**,
MCL **+0.006532** and ACL **+0.005933**. Both meniscus targets worsen, as do
medial and lateral OA. Mean meniscus AUC falls from **0.906272 to 0.901105**;
mean OA AUC falls from **0.882292 to 0.880014**. Higher overall AUC should not
be interpreted as resolving the meniscus/OA generalization concern.

At sigma 40, 1C wins eight of twelve targets, including all three OA targets
and medial meniscus. Reflection wins ACL, lateral meniscus, synovitis and
Baker's; its **+0.020773 Baker's** advantage substantially offsets its other
losses in the equal-target mean.

| Target | Reflection 80 | Reflection 40 | 1C 40 | Reflection 40 minus reflection 80 | Reflection 40 minus 1C 40 |
|---|---:|---:|---:|---:|---:|
| ACL | 0.913949 | 0.919882 | 0.914070 | +0.005933 | +0.005812 |
| MCL | 0.836247 | 0.842780 | 0.846666 | +0.006532 | -0.003886 |
| Medial Meniscus | 0.922563 | 0.915115 | 0.918898 | -0.007448 | -0.003783 |
| Lateral Meniscus | 0.889980 | 0.887095 | 0.885411 | -0.002885 | +0.001684 |
| Medial OA | 0.903951 | 0.902693 | 0.905749 | -0.001258 | -0.003055 |
| Lateral OA | 0.868678 | 0.862288 | 0.864535 | -0.006390 | -0.002247 |
| PF OA | 0.874246 | 0.875062 | 0.876016 | +0.000816 | -0.000955 |
| Effusion | 0.868662 | 0.869662 | 0.875839 | +0.001000 | -0.006177 |
| Synovitis | 0.896302 | 0.894418 | 0.893084 | -0.001885 | +0.001334 |
| Baker's | 0.908141 | 0.922952 | 0.902179 | +0.014811 | +0.020773 |
| Contusion | 0.888732 | 0.888436 | 0.897957 | -0.000296 | -0.009521 |
| Fracture | 0.920257 | 0.927939 | 0.930053 | +0.007682 | -0.002114 |

Reflection sigma 40 peaks at epoch 7. Epochs 8–10 score 0.891332, 0.891727
and 0.891347; final-epoch BCE improves to 0.447700. The selected epoch-7
checkpoint remains the comparison/submission candidate under the existing
AUC selection rule. Continued BCE improvement alongside lower AUC does not
by itself identify the cause of the ranking decline.

## Paired uncertainty and acquisition diagnostics

Two thousand paired bootstrap draws resample the **852 frozen report groups**
with seed **20260918**, preserving all studies within each sampled group and
using identical draws across models. All draws retain both classes for all
targets. Percentile 95% intervals for macro-AUC differences are:

| Comparison | Observed delta | Paired interval |
|---|---:|---:|
| Reflection 40 minus reflection 80 | +0.001384 | [-0.002015, +0.004864] |
| Reflection 40 minus 1C 40 | -0.000178 | [-0.003537, +0.003292] |
| 1C 40 minus 1C 80 | +0.003109 | [-0.000459, +0.006742] |
| Difference between 1C's and reflection's sigma gains | +0.001724 | [-0.002914, +0.006634] |

These intervals condition on the saved selected checkpoints. They exclude
training-seed, hyperparameter-search and checkpoint-selection uncertainty;
they neither establish equivalence nor estimate which model will win on Kaggle.

Reusing the prior audit's representative-series manufacturer metadata gives
the following exploratory subgroup macro AUCs. Manufacturer is not site, and
the hidden cohort's composition is unknown.

| Manufacturer | Studies | Reflection 80 | Reflection 40 | 1C 40 |
|---|---:|---:|---:|---:|
| GE | 201 | 0.921487 | 0.918223 | 0.921080 |
| Philips | 255 | 0.872926 | 0.876463 | 0.875951 |
| Siemens | 386 | 0.881197 | 0.882679 | 0.882564 |
| Toshiba | 48 | 0.844626 | 0.855128 | 0.853564 |

The sigma-40 candidates exchange advantages across acquisition groups.
Reflection's Philips macro gain also does not mean every OA target improves:
its Philips lateral-OA and PF-OA AUCs fall versus reflection sigma 80. These
results illustrate cohort-dependent tradeoffs without explaining the observed
public lead through a proven acquisition mechanism.

## Reproduction and provenance

The run's [effective configuration](../../../outputs/training/reflection-sigma40/effective_config.json)
records the recipe then adopted in the baseline (`configs/baseline.yaml`):
seed 42, Steven v4, fold 0 of `all_5fold_v3`, reflection probability 0.5,
ten epochs, batch size 8, 16 loader workers and 2× validation images. Its
effective configuration differs from the original reflection run only in
sigma and name. All 25 archived package Python files match reflection exactly.
The training package matches reflection at `9ba042d`; 1C remains preserved
on branch `1C` at `3b8c2cc`. The later CoAtNet adapter changes source identities,
so use the archived training source or frozen export for this checkpoint.

Across all four runs, label source/manifest, frozen split, study membership,
training/validation rows, series metadata/counts and initialized backbone
hashes match. Recomputed AUCs from the complete selected validation prediction
CSVs exactly match their selected-epoch AUCs. No GPU operations, fresh
inference, training or tests ran during this comparison.

Reflection sigma 40 trained for **28m 20s**, or **28m 52s** including setup and
selected-checkpoint prediction. Recorded peak CUDA allocation is **19.054 GiB**.
Those saved timings are descriptive; they are not a controlled inference
efficiency comparison.

Run roots are `outputs/training/spatial-tokens-reflection`, `outputs/training/reflection-sigma40`,
`outputs/training/disease-compartment-attention` and
`outputs/training/overnight-20260913T214822Z-74udtekr/runs/02-central-sampling`.
The new selected checkpoint is
`outputs/training/reflection-sigma40/fold_0/best-epoch=06-val_macro_auc=0.89236.ckpt`.
The [comparison artifact](../../../outputs/analysis/r/reflection-sigma40-20260914/comparison.json)
records full metrics, epoch histories, configuration differences, input checks,
checkpoint/source/prediction fingerprints and completed public-submission
records. Adjacent `per_target.csv`, `manufacturer_subgroups.csv` and
`bootstrap_auc.parquet` retain the tabular results and every resampled AUC.
Bootstrap AUC uses the existing
`scripts.adhoc.compare_public_labels.bootstrap_auc` helper; its source hash and
the metadata input hash are recorded in the artifact.
