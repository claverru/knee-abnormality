# 2026-09-21 anatomical attention

**Learned family strength adopted by user decision.** The recipe keeps position-rich pooling and
independent heads and adds a side-aware anatomical bias to the medial/lateral meniscus and OA
attention, with two learned positive family strengths and a fixed 20 mm spatial scale. The
learned-strength run scores 0.897443 against the repaired-data, matched-seed control's 0.895112:
+0.002331, 80% interval [+0.000103, +0.004518], 95% interval [−0.001034, +0.005645]. The fixed
bias and learned widths were not supported. The 80% adoption rule was chosen after seeing this
result, so the adoption is a working-recipe decision, not confirmation.

## Mechanism

Each token's feature-grid anchor maps back through the inverse augmentation to prepared-image
pixels, then through the middle slice's patient map to patient LPS X in millimetres:

```text
medial_mm = side_sign * (patient_x_mm - study_center_x_mm)
side_sign = -1 for left knee; +1 for right knee; 0 when unknown
medial attention addition = tanh(medial_mm / 20)
lateral attention addition = -tanh(medial_mm / 20)
```

- **Scope:** only the four compartment labels get the bias; the other eight get zero. No
  auxiliary loss, classifier sharing, sampling change or token removal. At unit strength the bias
  allows at most an exp(2) attention ratio between opposite extremes before the learned scores.
- **Inputs:** side comes from the [audited side policy](../../../metadata/knee-side/knee-side-policy-audit.md)
  (explicit laterality and coordinate consensus; abstains on bilateral, conflicting or text-only
  evidence). The centre is the median of per-series image centres, not a learned landmark. Unknown
  side, absent centre or invalid triplet geometry gives zero bias.
- **Reflection:** training's inverse transform maps tokens to original content coordinates;
  reflected inference composes the flip into the patient maps, with no second sign flip.
- **Learned strength:** per family (meniscus, OA), `alpha = exp(log_strength)` initialized at
  zero log, so both start at exactly one and stay positive. Head learning rate, zero weight decay.
- **Learned width (rejected):** per family, `width_mm = 20 * exp(log_width_scale)`, initialized
  at 20 mm, on top of learned strength.

Unlike the [retired attention prior](2026-09-15-attention-prior-result.md), which learned broad
Gaussian locations for all targets and stayed nearly flat, this is an active, fixed,
side-conditioned monotonic preference with no gate that can stay dormant.

## Runs

All runs: seed 2263738815, fold 0 of `all_5fold_v3`, Steven v4 labels, ten epochs, 26 study
triplets, ReXNet with class-balanced BCE, validation ×1, 904 studies. Intervals are paired
report-group bootstraps (852 report groups, 2,000 draws, seed 20260914); they exclude
training-seed, checkpoint-selection and multiple-comparison uncertainty.

| Run | Path under `outputs/training/` | Selected epoch | Macro AUC |
|---|---|---:|---:|
| Historical control B | `overnight-20260919T235434Z-8ol_1c30/runs/13-control-repeat` | 9 | 0.896582 |
| Control repeat (repaired data) | `anatomical-attention/control-repeat` | 10 | 0.895112 |
| Fixed bias | `anatomical-attention/candidate` | 10 | 0.895529 |
| Learned strength | `anatomical-attention/learned-strength` | 10 | 0.897443 |
| Learned width | `anatomical-attention/learned-width` | 10 | 0.894302 |

## Fixed bias against control B

| Metric | Control B | Fixed bias | Difference |
|---|---:|---:|---:|
| Macro AUC | 0.896582 | 0.895529 | −0.001053 |
| Medial Meniscus | 0.926587 | 0.923235 | −0.003352 |
| Lateral Meniscus | 0.897803 | 0.900045 | +0.002242 |
| Medial OA | 0.907763 | 0.912023 | +0.004261 |
| Lateral OA | 0.868603 | 0.865511 | −0.003092 |
| PF OA | 0.869275 | 0.874622 | +0.005347 |
| Meniscus margin AUC | 0.971248 | 0.970062 | −0.001186 |
| OA margin AUC | 0.866116 | 0.872219 | +0.006103 |

Macro 95% [−0.004640, +0.002449]. Meniscus family −0.000555, OA family +0.000584; the OA margin
interval is wide, [−0.034409, +0.045071]. Lateral false positives at the prevalence cutoff moved
little (meniscus 44 to 42, OA 82 to 81). Other-eight AUC fell 0.001587 (MCL −0.014799,
Synovitis −0.009522). Not adopted. It beat untied pooling's 0.894777
([untied position](2026-09-21-untied-position-result.md)) by 0.000752 without its OA-margin loss.

## Bias-off ablation

Inference-only on the fixed-bias checkpoint, no retraining. Intact replay matched the saved AUCs
exactly and raw probabilities within 2.98e-8.

| Metric | Bias on | Bias off | Off minus on |
|---|---:|---:|---:|
| Macro AUC | 0.895529 | 0.891910 | −0.003619 |
| Medial Meniscus | 0.923235 | 0.919718 | −0.003516 |
| Lateral Meniscus | 0.900045 | 0.872602 | −0.027444 |
| Medial OA | 0.912023 | 0.903936 | −0.008088 |
| Lateral OA | 0.865511 | 0.861130 | −0.004381 |
| Meniscus margin AUC | 0.970062 | 0.958764 | −0.011298 |
| OA margin AUC | 0.872219 | 0.817928 | −0.054291 |

- **95% intervals (off minus on):** macro [−0.005213, −0.002120]; meniscus family
  [−0.023588, −0.007764]; OA family [−0.010689, −0.002098]; meniscus margin
  [−0.021439, −0.002010]; OA margin [−0.093806, −0.018079].
- **Cross-talk:** meniscus lateral false positives rise 42 to 56 (medial-positive subset 27 to
  47); OA 81 to 87 (subset 50 to 63). The eight untargeted labels are unchanged.
- **Reading:** the trained checkpoint relies on the prior for compartment discrimination. It does
  not show that training with the prior beats training without it: the weights co-adapted to it.

Artifacts: `outputs/analysis/a/anatomical-bias-ablation/` (`paired_report_group.json`,
`intact.csv`, `without_anatomical_bias.csv`).

## Learned strength against fixed bias

| Metric | Fixed bias | Learned strength | Difference |
|---|---:|---:|---:|
| Macro AUC | 0.895529 | 0.897443 | +0.001914 |
| Meniscus family AUC | 0.911640 | 0.911947 | +0.000307 |
| OA family AUC | 0.888767 | 0.892716 | +0.003949 |
| Medial Meniscus | 0.923235 | 0.922620 | −0.000615 |
| Lateral Meniscus | 0.900045 | 0.901275 | +0.001230 |
| Medial OA | 0.912023 | 0.910633 | −0.001390 |
| Lateral OA | 0.865511 | 0.874799 | +0.009288 |
| Meniscus margin AUC | 0.970062 | 0.969327 | −0.000734 |
| OA margin AUC | 0.872219 | 0.861157 | −0.011062 |

- Macro 95% [−0.000926, +0.004925]; Lateral OA [+0.001267, +0.017286]; OA margin
  [−0.037622, +0.013255]. OA lateral false positives fall 81 to 74; meniscus rise 42 to 45.
- Against control B: +0.000861, 95% [−0.002575, +0.004456].
- Selected strengths: meniscus **1.46533**, OA **1.13049** (`model.anatomical_log_strength`
  [0.38208091, 0.12264736]). Meniscus rose fast (1.060, 1.247, 1.360 over epochs 1–3); both
  plateaued by epochs 8–10, neither collapsed nor diverged.
- Training 1,554.72 s, final prediction 20.27 s, 4,380 steps.

**Strengths reset to one at inference** (same checkpoint): macro 0.897443 → 0.897213, −0.000230,
95% [−0.000580, +0.000106]. Lateral Meniscus −0.003481 (95% [−0.007012, +0.000063]); Lateral OA
only −0.000169. The Lateral OA gain over the fixed-bias run lives in the trained weights, not in
the final coefficient's direct effect. Other-eight predictions unchanged.

Artifacts: `outputs/analysis/a/anatomical-learned-vs-fixed/`,
`outputs/analysis/a/anatomical-learned-vs-control-b/`, `outputs/analysis/l/learned-strength-reset/`.

## Learned width against learned strength

| Metric | Learned strength | Learned width | Width minus strength |
|---|---:|---:|---:|
| Macro AUC | 0.897443 | 0.894302 | −0.003141 |
| Medial Meniscus | 0.922620 | 0.923301 | +0.000682 |
| Lateral Meniscus | 0.901275 | 0.898248 | −0.003027 |
| Medial OA | 0.910633 | 0.909528 | −0.001105 |
| Lateral OA | 0.874799 | 0.869204 | −0.005595 |
| Meniscus margin | 0.969327 | 0.963904 | −0.005423 |
| OA margin | 0.861157 | 0.860394 | −0.000763 |

Macro 95% [−0.006254, −0.000153]. Other-eight mean fell 0.003581, so the loss is not confined to
the biased labels. Against B: −0.002280, 95% [−0.005798, +0.001127]. Selected widths 15.32794 mm
(meniscus) and 16.72296 mm (OA), strengths 1.47983 and 1.12414: a sharper transition did not help.
Rejected; no further training on this refinement.

Artifacts: `outputs/analysis/a/anatomical-width-vs-strength/`,
`outputs/analysis/a/anatomical-width-vs-control-b/`.

## Control repeat on repaired data

Historical control B excluded two DICOM slices that later preprocessing repaired and restores
([data repair record](../../../data-preparation/public-data-contract-and-decoding.md)), and the code
changed after B. B's seed matches every run here, but it is not a clean same-source control. The
control repeat (`configs/experiments/anatomical-attention/control.yaml` (removed), all anatomical options
off) retrains it on the repaired data and current code.

| Run | Macro AUC | Difference from control repeat |
|---|---:|---:|
| Historical B | 0.896582 | +0.001470 |
| Control repeat | 0.895112 | 0 |
| Fixed bias | 0.895529 | +0.000417 |
| Learned strength | 0.897443 | +0.002331 |
| Learned width | 0.894302 | −0.000810 |

- **Control repeat minus B:** 95% [−0.004217, +0.001340]. B's score was not reproduced, but the
  gap mixes the repaired data, intervening code and nondeterministic training, so it does not show
  B's seed was lucky. Earlier comparisons against B stay valid under their own conditions; failures
  against B do not prove those implementations useless.
- **Learned strength minus control repeat:** +0.002331, 95% [−0.001034, +0.005645]. Meniscus
  family +0.000499, OA family +0.007352 (95% [+0.000820, +0.014445], exploratory). Medial
  Meniscus −0.001604, Lateral Meniscus +0.002602, Medial OA +0.003732, Lateral OA +0.010971.
  Both margins declined (meniscus −0.001186, OA −0.011697, intervals include zero), so better
  compartment assignment is not established.
- Preprocessing code matches between learned strength and the control repeat; input identities
  differ only in source hash.

Artifacts: `outputs/analysis/c/control-repeat-vs-control-b/`,
`outputs/analysis/a/anatomical-strength-vs-control-repeat/`.

## Choice of the 80% interval

The same 2,000 draws for learned strength minus the control repeat, at three confidence levels:

| Level | Interval |
|---|---|
| 80% | [+0.000103, +0.004518] |
| 90% | [−0.000518, +0.005072] |
| 95% | [−0.001034, +0.005645] |

Gain ≥ 0.001 with a positive 80% lower bound passes this run; the 90% and 95% gates would not.
The user then fixed gain ≥ 0.001 plus a positive 80% lower bound as the adoption rule for all
later work ([shared evaluation](../../../../autoresearch/protocol.md#shared-evaluation-and-adoption)).
The confidence level was picked after observing this result, so this run must not be presented as
confirming the rule or the mechanism. The production assessment returns `provisional_accept` from
the saved predictions (`outputs/analysis/a/anatomical-strength-vs-control-repeat-ci-levels/`).

## Adoption and campaign transition

- **Reference:** `outputs/training/anatomical-attention/learned-strength`, fold 0, epoch 10,
  macro 0.8974429314502609 against the control repeat's 0.8951120901611253, gain
  +0.0023308412891356323, 80% [+0.0001034177, +0.0045179746]. A user-directed baseline
  replacement, not a retrospective acceptance against B.
- **Adopted source:**
  `outputs/autoresearch/adopted/user-adopt-learned-strength-20260921T115849Z-b19e375d/source`, with
  `adoption.json` in the same folder pinning the source manifest, trained source archive,
  prediction hash, configuration, input-identity transition and assessment, and
  `trained-to-adopted.diff` limited to config, model and Lightning (accepted behavior made
  mandatory, rejected alternatives removed). No score is attributed to this cleanup. Historical
  inference uses the trained run's archived sources.
- **Campaigns:** triplet-sampling, losses and medial-lateral-transfer took
  `learned-strength-20260921` as incumbent while stopped (at revisions 181, 259 and 351). Each keeps its original sealed
  reference, limits and verdicts; the manual replacement is excluded from campaign gain. Decision
  records: `outputs/autoresearch/<campaign>/decisions/user-adopt-learned-strength-20260921T115849Z-b19e375d/`
  (`state-before.json`, `pending-events.json`, `decision.json`).
- **Preprocessing-identity exception:** `model.py` is part of the preprocessing identity, so the
  cleanup changed the source hash. Losses forbids any preprocessing change; its supervisor accepts
  only this exact reviewed identity against this exact adopted reference, and the allowance ends
  when a new incumbent replaces it. Sampling and transfer kept their own declared permissions.
- **Stale work:** the cancelled extra `family-shared-local-lesion-evidence` run was finalized
  invalid (no evidence against the mechanism); the unlaunched loss preparation
  `shared-group-variance-compression-20260920T181146Z-04a138a1` was archived, its idea left queued.
- **Removed from the code:** appearance-only pooling, learned widths, the fixed-strength-only path
  and the pooled `compartment_adapter`.
