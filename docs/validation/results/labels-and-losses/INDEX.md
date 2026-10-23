# Label and loss results

| Read when | Document |
|---|---|
| Reviewing the two-family out-of-fold distillation: `oof_blend_v2` labels (silver plus ResNet-50 and ReXNet-200 out-of-fold predictions), the children against their teachers per fold, attention temperature at ×4, ×5 and ×6 per family and for the blend, family weights, and the ten-child bundle | [2026-10-04 two-family out-of-fold distillation](2026-10-04-two-family-oof-distillation.md) |
| Reviewing the report-attribute auxiliary-heads runs (JevK5 v3 and cleaned, reliability-weighted 3.1 targets) against `oa-detector-control`: their public pair, per-label gains and losses, attribute loss and per-question KL | [2026-09-25 report-attribute heads](2026-09-25-report-attribute-heads.md) |
| Checking whether compartment labels read the other compartment (medial/lateral sibling cross-talk): inference-only token masking and pixel blanking on `k05-clean`, AUC against sibling-positive and sibling-clean negatives, and the ceiling of the effect | [2026-09-23 sibling masking](2026-09-23-sibling-masking.md) |
| Reviewing prediction-driven label-noise training (bootstrapped and running-average targets, early-learning regularization, contradiction down-weighting, the 1→0 class-balance schedule): why the model never memorizes its labels and the schedule's public score | [2026-09-23 noisy-labels campaign](2026-09-23-noisy-labels-campaign.md) |
| Choosing how to compute validation AUC against the public leaderboard: soft-label against hard-threshold, same-side pairs, gold-calibrated targets, consensus targets, machine/field/acquisition strata, report weights | [2026-09-23 validation AUC variants](2026-09-23-validation-auc-variants.md) |
| Blending out-of-fold image predictions into training labels (self-distillation): the `image_blend_v1` gold check, fold-0 hard and soft AUC, the public tie and the teacher-recipe mismatch | [2026-09-23 image-blended labels](2026-09-23-image-blend-labels-result.md) |
| Judging label or weighting changes without submissions: the gold-calibrated expected-AUC judge, its eligibility rules, its agreement with past submissions against fold-0 AUC, and seed noise | [2026-09-22 calibrated judge](2026-09-22-calibrated-judge.md) |
| Choosing class-weighting strength, and why fold-0 validation and the public leaderboard disagreed on weighting (exponent 1, 0.5 and unweighted) | [2026-09-21 class-balance exponent](2026-09-21-class-balance-exponent.md) |
| Reviewing the first run on combined labels and the scanner split: per-label AUC, the tie with learned strength on shared studies, and the attention-temperature sweep | [2026-09-21 combined-labels first run](2026-09-21-combined-labels-first-run.md) |
| Choosing loss variants or seeds for an ensemble: the loss-campaign subset search and selection optimism | [2026-09-21 loss ensemble simulation](2026-09-21-loss-ensemble-simulation.md) |
| Reviewing fixed class-balanced BCE | [2026-09-19 balanced BCE](2026-09-19-balanced-bce-result.md) |
| Reviewing meniscus soft coupling | [2026-09-14 meniscus soft-coupling result](2026-09-14-meniscus-soft-coupling-result.md) |
| Comparing Tonylin and Steven labels | [2026-09-13 Tonylin label comparison](2026-09-13-tonylin-label-comparison.md) |
