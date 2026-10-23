# Prior work for the student-distillation campaign

The summary the researcher checks proposals against. The measurements are in
[two-family out-of-fold distillation](../../../docs/validation/results/labels-and-losses/2026-10-04-two-family-oof-distillation.md)
unless another source is named.

## The first round

2026-10-04, on `all_5fold_scanner_v2`. Teachers: five Refined ResNet-50 and five ReXNet-200, one
per fold. Labels `oof_blend_v2`: 0.5 `rebuild_state_v2` + 0.25 + 0.25 of the two families'
out-of-fold ×4 predictions (τ 1, no TTA), each study from the teacher that held it out. Students:
the same two recipes retrained on those labels, one run per fold, each with its own seed.

| | Teacher, pooled out of fold | Student | Δ |
|---|---:|---:|---:|
| Refined ResNet-50 | 0.931720 | 0.935119 | +0.0034 |
| ReXNet-200 | 0.925785 | 0.934825 | +0.0090 |

- Every student beat its teacher on every fold. The weaker family gained more.
- **The comparison was not clean.** Each student's validation fold had been trained on by four
  of its five teachers per family, and student and teacher differ in seed.
- Public: five ResNet-50 students alone 0.957 (S106), the ten students 0.955 (S103), one
  ResNet-50 teacher 0.954 (S102).

## What is settled

- **Teachers stay on the Refined-RadImageNet weights** (user decision, 2026-10-04). The ReXNet
  students add +0.0026 to the ResNet-50 students out of fold and cost 0.002 on the public set,
  and labels built with the ResNet-50 teacher alone agree better with the 58 gold studies than
  labels with the ReXNet teacher added (0.9394 against 0.9358; 80% [−0.0070, −0.0002] for the
  difference).
- **A ResNet-18 beside a ResNet-50 adds nothing on the public set** (S107 0.954, the ResNet-50
  alone 0.954), so the ResNet-18 is this campaign's fast model and not an ensemble member.
- **The teacher's inference setting moves its quality by little.** ResNet-50 students scored as
  labels, pooled out of fold: ×4, τ 1, no TTA 0.935119; ×6, τ 1, TTA 0.935548; ×6, τ 0.8, TTA
  0.936272. The campaign's table uses ×6, TTA and τ 0.85 (user decision, 2026-10-10); 0.8 was the value measured on the ResNet-50,
  and neither is measured on the ResNet-18.
- **A noisier teacher is a worse teacher here.** The table is written once, so the error of a
  ×1 prediction is frozen for every epoch and acts as a label error.

## Fold teachers are levelled differently

Each study's teacher value comes from a different model. The mean prediction of a label differs
between the five ResNet-50 teachers by 0.01 to 0.04. Out-of-fold ResNet-50 teacher predictions
scored as a label:

| Teacher predictions | Macro AUC |
|---|---:|
| Mean of the five fold scores | 0.932043 |
| Pooled, raw | 0.931720 |
| Pooled, each teacher reranked per label | 0.932046 |
| Pooled, each teacher recalibrated per label (logistic, two numbers) | 0.932594 |
| Pooled, rounded to 0.25 | 0.9099 |

- Reranking recovers exactly the gap to the fold mean. Recalibration is fitted on the targets it
  is scored against here, so part of its extra gain is optimistic.
- On the 58 gold studies: raw 0.9354, reranked 0.9309, recalibrated 0.9330; too few to choose.
- Rounding to 0.25 turns 47% of the cells into an exact 0 and costs the label 0.022. Closed.
- No student has been trained on reranked or recalibrated labels.

## Single-plane readings

With the plane-contrast readouts a label's logit is a sum over tokens of attention times the
evidence of the token's series type, and `pool_encoded(encoded, view=...)` pools one view alone.
No run has saved single-plane readings; nothing is known yet about their AUC per label or about
whether disagreement between planes marks the cells where the silver label is wrong.

## Combinations that do not help

Measured on the ResNet-50 students at ×6, out of fold:

| Combination | Result against the plain mean |
|---|---|
| Mean of ranks across models | −0.0003 |
| Mean of logits | identical |
| A second stage predicting each label from all twelve outputs, linear | −0.0008 (gold −0.0028) |
| The same with boosted trees | −0.003 to −0.010 |
| Mean over multipliers ×4, ×5, ×6 | −0.00004 |

## The adoption rule

Noise of a difference between two campaign runs of one seed, from the saved predictions of 37
runs of three campaigns on fold 1 (219 pairs, bootstrap over studies, scaled to 735 studies):
standard deviation 0.0009 at ×4 and 0.0012 at ×1, correlated at 0.6. Gold noise: 0.0056 on the
58 gold studies between a teacher and its student of 2026-10-04, which differ more than two
campaign runs; scaled by the ratio seen on the overall AUC, about 0.0031, the value used.

What the rule of the [contract](contract.md#adoption) does (simulated; chance that a run is
adopted):

| Run | Fold 0 of 735 (one sixth) | Fold 0 of 881 (one fifth, the contract's) | Fold 0 of 1,100 (one quarter) |
|---|---:|---:|---:|
| ×4 threshold that adopts a worthless run 10% of the time | +0.0010 | +0.0009 | +0.0008 |
| True gain +0.0005 | 23% | 24% | 27% |
| True gain +0.001 | 43% | 46% | 52% |
| True gain +0.0015 | 64% | 68% | 74% |
| True gain +0.002 | 79% | 82% | 85% |
| True gain +0.003 | 93% | 94% | 94% |
| Better on silver, worse on gold (+0.001 overall, −0.004 gold) | 16% | 17% | 19% |
| Share of the data each teacher trains on, with five teacher folds | 67% | 64% | 60% |
| The same with four teacher folds, the contract's | 62% | 60% | 56% |

Rules compared on the same noise, each set to adopt a worthless run one time in ten, fold of 735:

| Rule | +0.001 | +0.002 | Better on silver, worse on gold |
|---|---:|---:|---:|
| Better on ×1, ×4 and gold (adopts a worthless run 21% of the time) | 51% | 71% | 9% |
| ×4 ≥ +0.0011 alone | 44% | 85% | 44% |
| 0.7 ×4 + 0.3 ×1 ≥ +0.0011 | 45% | 86% | 45% |
| At least 9 of 12 labels up and ×4 > 0 | 43% | 79% | 43% |
| ×4 ≥ +0.0010, ×1 ≥ 0, gold ≥ −0.002 (the contract's form, at this fold size) | 43% | 79% | 16% |
| ×4 ≥ +0.0008, ×1 ≥ 0, gold ≥ 0 | 39% | 68% | 7% |

No rule adopts a +0.001 run reliably from one run; the size of fold 0 is what moves it.

## Precedents

| Mechanism | Setting | Run | Macro delta | Source |
|---|---|---|---:|---|
| 0.5 silver + 0.5 out-of-fold teacher, one family | ResNet-18, fold 0, leaky | `rn18-oof-blend` | +0.0015 (95% [+0.0001, +0.0031]) | [endgame backlog](../../../docs/validation/results/labels-and-losses/2026-10-04-two-family-oof-distillation.md) |
| 0.5 silver + 0.25 + 0.25, two families | Refined ResNet-50, five folds, leaky | `*-rn50-oof-blend-v2-f0..4` | +0.0034 pooled | first round |
| The same | ReXNet-200, five folds, leaky | `*-rexnet-200-oof-blend-v2-f0..4` | +0.0090 pooled | first round |
| 0.7 silver + 0.3 teacher of older models | ReXNet-200, fold 0 | image-blend labels, 2026-09-23 | +0.0035, a public tie | [image-blend labels](../../../docs/validation/results/labels-and-losses/2026-09-23-image-blend-labels-result.md) |
| Targets changed inside the training step from the model's own predictions (bootstrapped, running average, early-learning regularisation) | ReXNet-200, fold 0 | noisy-labels campaign | none adopted | [noisy-labels campaign](../../../docs/validation/results/labels-and-losses/2026-09-23-noisy-labels-campaign.md) |

## Open leads

- The share, never varied on a clean split.
- Reranking or recalibrating the fold teachers before mixing.
- What the single-plane readings hold, and the two uses of them: a share per study and label
  from plane agreement, and a target per plane.
- A second round with the students as teachers, outside this campaign.
