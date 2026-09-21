# 2026-09-20 meniscus compartment adapter

**Rejected.** The shared nonlinear bottleneck on the pooled meniscus features
([candidate 2](../../research/medial-lateral-transfer.md#candidate-2-shared-disease-features-independent-decisions))
left macro AUC inside the control spread and moved the intended meniscus family the wrong way:
−0.0114 [−0.0200, −0.0030] against `13-control-repeat`, with within-study compartment
discrimination down −0.0191 [−0.0392, −0.0034]. Both intervals exclude zero and both are negative.
The research note's stopping rule requires a useful result to improve the meniscus family; this
degrades it, so the candidate fails on its own predefined criterion. `model.compartment_adapter` and its implementation were removed on 2026-09-21;
there is no pending removal or coefficient/width sweep.

## Runs

Fold 0 of `all_5fold_v3`, ten epochs, class-balanced BCE, `sampling_sigma_fraction: 0.867`,
`study_triplet_total: 26`, `validation_image_multiplier: 1`, 904 validation studies.

| Run | Change | Selected epoch | Macro AUC |
|---|---|---:|---:|
| `00-control` | none | 6 | 0.893222 |
| `13-control-repeat` | none | 8 | 0.896582 |
| `compartment-control` | `model.compartment_adapter: shared`, seed 2685821520 | 7 | 0.895099 |

**The candidate's run directory is misnamed.** `outputs/training/compartment-control` is the
`shared` arm, not a control: `configs/training.yaml` still carried `compartment_adapter: shared`
when it was launched, and its `effective_config.json` records `"compartment_adapter": "shared"`.
The live config was reverted to `null` after the run. Both controls come from the
[2026-09-20 queue](2026-09-20-overnight-hyperparameter-results.md)
(`outputs/training/overnight-20260919T235434Z-8ol_1c30/runs/`), which is the same recipe on the
prior commit `d5f352a`.

Against the mean of the two controls (0.894902) the candidate gains **+0.000197**, against a
control-to-control spread of **0.003360**. The sign of the macro delta depends on which control is
used, which is the definition of an unresolved comparison.

## Paired bootstrap

`scripts/tools/compare_fold_runs.py`, 2,000 paired report-group bootstrap draws over 852 groups.
Artifacts in `outputs/analysis/compartment-adapter-vs-00-control/` and
`outputs/analysis/compartment-adapter-vs-13-control-repeat/`.

| Family | vs `00-control` | vs `13-control-repeat` |
|---|---|---|
| Macro | +0.0018 [−0.0023, +0.0060] | −0.0015 [−0.0054, +0.0022] |
| Meniscus pair | −0.0006 [−0.0091, +0.0080] | **−0.0114 [−0.0200, −0.0030]** |
| OA pair | −0.0026 [−0.0101, +0.0046] | −0.0015 [−0.0089, +0.0054] |
| Other eight | +0.0036 [−0.0017, +0.0090] | +0.0009 [−0.0043, +0.0057] |
| Meniscus margin | −0.0110 [−0.0290, +0.0024] | **−0.0191 [−0.0392, −0.0034]** |
| OA margin | +0.0051 [−0.0279, +0.0391] | −0.0125 [−0.0427, +0.0142] |

The meniscus margin is the medial-minus-lateral logit margin on studies with exactly one positive
meniscus target (356 medial positives, 138 lateral). It measures *within-study* compartment
discrimination, which per-target AUC cannot see, and it is the diagnostic this candidate was
designed to move. It moves down against both controls.

The comparisons allow `preprocessing_identity` to differ because `modeling/model.py` is in the
hashed source set (`src/knee_abnormality/identity.py`) and the adapter edited it. The files that
determine decoded pixels — `data/preprocessing/*`, `modeling/augmentation.py`,
`modeling/spatial_tokens.py` — are untouched, so the decoded inputs are identical and the pairing
is valid.

## Per-target AUC at the selected epochs

| Target | `00-control` | `13-control-repeat` | adapter | Δ vs control mean |
|---|---:|---:|---:|---:|
| ACL | 0.9133 | 0.9178 | 0.9220 | +0.0065 |
| MCL | 0.8631 | 0.8622 | 0.8637 | +0.0011 |
| Medial Meniscus | 0.9176 | 0.9266 | 0.9146 | −0.0074 |
| Lateral Meniscus | 0.8852 | 0.8978 | 0.8871 | −0.0045 |
| Medial OA | 0.9106 | 0.9078 | 0.9022 | −0.0070 |
| Lateral OA | 0.8678 | 0.8686 | 0.8711 | +0.0029 |
| PF OA | 0.8617 | 0.8693 | 0.8678 | +0.0023 |
| Effusion | 0.8637 | 0.8648 | 0.8747 | +0.0104 |
| Synovitis | 0.8921 | 0.8996 | 0.8934 | −0.0025 |
| Baker's | 0.9161 | 0.9139 | 0.9160 | +0.0010 |
| Contusion | 0.8946 | 0.8955 | 0.8957 | +0.0007 |
| Fracture | 0.9331 | 0.9351 | 0.9329 | −0.0012 |

Both labels the adapter touches are the wrong side of zero, and they are the only pair whose
family interval excludes zero. The gains sit on labels the adapter cannot reach except through the
shared backbone.

## Why this is a mechanism result, not a tuning result

Medial meniscus is 40.7% positive in the training fold and lateral 15.8%
(`outputs/training/compartment-control/fold_0/class_balance.json`), so the gradient into the shared
`V`/`U` is dominated by medial. Both compartments then read their pooled features through a block
fitted to the majority side, and the margin diagnostic — the direct measure of telling one side
from the other — is what degrades most. That is the predicted cost of tying two compartments with
unequal prevalence, and it reproduces the direction of
[candidate 1](2026-09-14-meniscus-soft-coupling-result.md), which coupled the classifier vectors
instead and also lost the meniscus pair (−0.0041). Two different sharing mechanisms have now
regressed the same family.

## Do not repeat

- **No bottleneck-width or coefficient sweep.** The research note forbids it on a failed result,
  and the failure is in the margin diagnostic, which width does not address.
- **`independent` is not worth a slot.** It is a reparametrization of the rejected
  `model.residual_readout`: expanding the adapter gives
  `Δz_c = (U_c^T w_c) · GELU(V_c · pooled_c + b_V)` against residual readout's
  `Δz_c = r_c · GELU(H_c · pooled_c + b_c)`, the same function class at the same width 32 and the
  same zero init. `09-residual-readout` measured that at −0.0005
  ([result](2026-09-18-overnight-results.md)). The capacity control was already run, and the arm
  only existed to attribute a `shared` gain that did not occur.
- **Mid-run per-label readings are not evidence.** At epoch 2 this run showed Lateral Meniscus
  −0.0277 and MCL +0.0319 against the same controls; by the selected epoch these were −0.0045 and
  +0.0011. Per-label deltas before the selected epoch carry no information about the outcome.

## Empirical null distribution for per-label deltas

Computed while assessing this run, from the 16 candidate arms of the
[2026-09-19](2026-09-19-untried-hyperparameter-queue.md) and
[2026-09-20](2026-09-20-overnight-hyperparameter-results.md) queues — every one a null result on
this recipe — as each arm's delta against its own queue's control mean at its selected epoch.

| Label | Null range | sd |
|---|---|---:|
| Medial OA | −0.0082 … +0.0029 | 0.0026 |
| Effusion | −0.0033 … +0.0105 | 0.0037 |
| Lateral Meniscus | −0.0061 … +0.0072 | 0.0041 |
| Lateral OA | −0.0102 … +0.0067 | 0.0045 |
| PF OA | −0.0080 … +0.0113 | 0.0046 |
| ACL | −0.0122 … +0.0053 | 0.0047 |
| Medial Meniscus | −0.0103 … +0.0085 | 0.0047 |
| Synovitis | −0.0069 … +0.0083 | 0.0047 |
| Baker's | −0.0058 … +0.0108 | 0.0049 |
| Contusion | −0.0096 … +0.0071 | 0.0051 |
| Fracture | −0.0146 … +0.0178 | 0.0072 |
| MCL | −0.0237 … +0.0169 | 0.0100 |

Macro delta over the same 16 arms: min −0.0050, median −0.0011, **max +0.0011**. No change on this
recipe has ever exceeded +0.001 against its control mean, and the two controls of the 2026-09-20
queue differ by 0.0034 — more than 15 of the 16 candidate deltas in absolute value.

Two consequences for reading any future queue. The per-label noise floor differs by 4× across
labels, so identical numeric deltas mean different things on Medial OA and on MCL. And the paired
run-to-run spread is far smaller than the standard error of an absolute AUC on 904 studies
(±0.023 for Lateral Meniscus by Hanley–McNeil), because every run scores the same studies and the
study-sampling term cancels; the absolute standard error is the wrong yardstick for a paired
comparison and will hide real per-label movement.

## Caveats

One unseeded run against two controls. The comparison resolves roughly 0.005 on macro AUC, not
0.001. The intervals condition on these selected predictions and exclude training-seed and
checkpoint-selection variability. No public score was measured. The meniscus and margin intervals
exclude zero against only one of the two controls; what carries the rejection is that both
diagnostics are negative against both controls and the candidate's own predefined criterion
required them to be positive.
